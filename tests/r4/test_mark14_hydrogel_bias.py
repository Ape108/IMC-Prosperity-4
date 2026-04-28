"""Tests for Mark14HydrogelBiasStrategy and HydrogelStrategy refactor."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r4"))

import pytest
from datamodel import Order, OrderDepth, Trade, TradingState  # noqa
from strategy_h import HydrogelStrategy, Mark14HydrogelBiasStrategy


HYDROGEL = "HYDROGEL_PACK"


def make_baseline_state(
    timestamp: int = 100,
    position: int = 0,
    market_trades: dict[str, list[Trade]] | None = None,
) -> TradingState:
    """Symmetric book around 10000: bid=9999 (vol 50), ask=10001 (vol 50).
    With position=0, HydrogelStrategy emits buy(9998, 40) and sell(10002, 40).
    """
    od = OrderDepth()
    od.buy_orders = {9999: 50}
    od.sell_orders = {10001: -50}
    return TradingState(
        traderData="",
        timestamp=timestamp,
        listings={},
        order_depths={HYDROGEL: od},
        own_trades={},
        market_trades=market_trades or {},
        position={HYDROGEL: position},
        observations=None,  # type: ignore
    )


def orders_to_tuples(orders: list[Order]) -> list[tuple[str, int, int]]:
    """Normalize for set comparison: (symbol, price, quantity)."""
    return sorted((o.symbol, o.price, o.quantity) for o in orders)


# ── HydrogelStrategy regression (refactor must be bit-identical) ─────────────


def test_hydrogel_baseline_orders_unchanged_under_refactor():
    """Locks the current HydrogelStrategy quote behavior on a fixed state.
    After Task 1 refactor adds a `_compute_signal_bias` hook returning 0.0,
    this test must still pass — the hook adds 0.0 to skewed_value, so all
    intermediate values and emitted orders are identical.
    """
    s = HydrogelStrategy(HYDROGEL, 200)
    state = make_baseline_state()
    orders, _ = s.run(state)
    assert orders_to_tuples(orders) == [
        (HYDROGEL, 9998, 40),
        (HYDROGEL, 10002, -40),
    ]


# ── Mark14HydrogelBiasStrategy: constructor + state ──────────────────────────


def make_bias_strategy(**kwargs) -> Mark14HydrogelBiasStrategy:
    defaults = dict(symbol=HYDROGEL, limit=200)
    defaults.update(kwargs)
    return Mark14HydrogelBiasStrategy(**defaults)


def test_constructor_default_params():
    s = make_bias_strategy()
    assert s.symbol == HYDROGEL
    assert s.limit == 200
    assert s.bias_k == 1.0
    assert s.scale == 10.0
    assert s.window_ticks == 5000
    assert s.history == []


def test_constructor_custom_params():
    s = make_bias_strategy(bias_k=2.5, scale=15.0, window_ticks=3000)
    assert s.bias_k == 2.5
    assert s.scale == 15.0
    assert s.window_ticks == 3000


def test_save_returns_dict_with_history():
    s = make_bias_strategy()
    s.history = [(100, 5), (200, -3)]
    saved = s.save()
    assert "history" in saved
    assert saved["history"] == [(100, 5), (200, -3)]


def test_load_restores_history():
    s = make_bias_strategy()
    s.load({"history": [(100, 5), (200, -3)]})
    assert s.history == [(100, 5), (200, -3)]


def test_load_normalizes_lists_back_to_tuples():
    """After a JSON round-trip, history elements arrive as lists. load() must
    coerce them back to tuples so consumers can rely on the declared type."""
    s = make_bias_strategy()
    s.load({"history": [[100, 5], [200, -3]]})
    assert s.history == [(100, 5), (200, -3)]


def test_load_with_missing_history_key_defaults_to_empty():
    s = make_bias_strategy()
    s.load({})
    assert s.history == []


def test_save_load_roundtrip_via_json():
    s = make_bias_strategy()
    s.history = [(100, 5), (200, -3), (4500, 7)]
    saved = s.save()
    serialized = json.dumps(saved)
    deserialized = json.loads(serialized)

    s2 = make_bias_strategy()
    s2.load(deserialized)
    # Sum should be preserved through the round-trip.
    assert sum(qty for _, qty in s2.history) == 9  # 5 - 3 + 7


def test_skeleton_without_signal_override_matches_baseline():
    """Before Task 3 lands the bias override, the skeleton inherits
    `_compute_signal_bias` returning 0.0 from HydrogelStrategy → behavior
    is identical to HydrogelStrategy."""
    s = make_bias_strategy()
    state = make_baseline_state()
    orders, _ = s.run(state)
    assert orders_to_tuples(orders) == [
        (HYDROGEL, 9998, 40),
        (HYDROGEL, 10002, -40),
    ]


# ── _compute_signal_bias unit tests ──────────────────────────────────────────


def make_trade(
    qty: int,
    buyer: str,
    seller: str,
    timestamp: int,
    symbol: str = HYDROGEL,
    price: int = 10000,
) -> Trade:
    return Trade(symbol, price, qty, buyer, seller, timestamp=timestamp)


def test_bias_empty_history_returns_zero():
    s = make_bias_strategy()
    state = make_baseline_state(timestamp=100)
    assert s._compute_signal_bias(state) == 0.0


def test_bias_single_mark14_buy_proportional():
    """qty=5, scale=10 → bias = 1.0 * (5/10) = 0.5."""
    s = make_bias_strategy()
    trade = make_trade(qty=5, buyer="Mark 14", seller="Mark 38", timestamp=100)
    state = make_baseline_state(timestamp=200, market_trades={HYDROGEL: [trade]})
    bias = s._compute_signal_bias(state)
    assert bias == pytest.approx(0.5)


def test_bias_saturation_positive():
    """qty=20, scale=10 → bias clipped to +1.0 (= bias_k)."""
    s = make_bias_strategy()
    trade = make_trade(qty=20, buyer="Mark 14", seller="Mark 38", timestamp=100)
    state = make_baseline_state(timestamp=200, market_trades={HYDROGEL: [trade]})
    assert s._compute_signal_bias(state) == pytest.approx(1.0)


def test_bias_saturation_negative():
    """Mark 14 sells qty=20 → net=-20, bias clipped to -1.0."""
    s = make_bias_strategy()
    trade = make_trade(qty=20, buyer="Mark 38", seller="Mark 14", timestamp=100)
    state = make_baseline_state(timestamp=200, market_trades={HYDROGEL: [trade]})
    assert s._compute_signal_bias(state) == pytest.approx(-1.0)


def test_bias_mark14_seller_negative():
    """Mark 14 selling 5 lots → signed_qty=-5 → bias=-0.5."""
    s = make_bias_strategy()
    trade = make_trade(qty=5, buyer="Mark 38", seller="Mark 14", timestamp=100)
    state = make_baseline_state(timestamp=200, market_trades={HYDROGEL: [trade]})
    assert s._compute_signal_bias(state) == pytest.approx(-0.5)


def test_bias_mark38_only_no_history_change():
    """Mark 38 trades a third party (Mark 14 NOT involved) → no signal."""
    s = make_bias_strategy()
    trade = make_trade(qty=5, buyer="Mark 38", seller="Mark 22", timestamp=100)
    state = make_baseline_state(timestamp=200, market_trades={HYDROGEL: [trade]})
    bias = s._compute_signal_bias(state)
    assert bias == 0.0
    assert s.history == []


def test_bias_mixed_buy_sell_within_window():
    """Mark 14 buys 5 then sells 3 → net=+2 → bias = 1.0 * 0.2 = 0.2."""
    s = make_bias_strategy()
    trades = [
        make_trade(qty=5, buyer="Mark 14", seller="Mark 38", timestamp=100),
        make_trade(qty=3, buyer="Mark 38", seller="Mark 14", timestamp=200),
    ]
    state = make_baseline_state(timestamp=300, market_trades={HYDROGEL: trades})
    assert s._compute_signal_bias(state) == pytest.approx(0.2)


def test_bias_trim_drops_old_entries():
    """Entries older than `state.timestamp - window_ticks` are dropped before
    computing net. window_ticks=5000, ts=10000 → drop entries with ts<5000."""
    s = make_bias_strategy(window_ticks=5000)
    s.history = [(2000, 10)]  # already too old; should be trimmed
    state = make_baseline_state(timestamp=10000)  # no new trades
    bias = s._compute_signal_bias(state)
    assert bias == 0.0
    assert s.history == []


def test_bias_trim_keeps_in_window_entries():
    """ts=10000, window=5000 → keep entries with ts >= 5000."""
    s = make_bias_strategy(window_ticks=5000)
    s.history = [(2000, 10), (6000, 5)]
    state = make_baseline_state(timestamp=10000)
    bias = s._compute_signal_bias(state)
    # After trim: [(6000, 5)] → net=5 → bias = 1.0 * 0.5 = 0.5
    assert bias == pytest.approx(0.5)
    assert s.history == [(6000, 5)]


def test_bias_day_boundary_clears_history():
    """When state.timestamp < latest history ts, day rolled over → clear."""
    s = make_bias_strategy()
    s.history = [(900_000, 10)]  # late in day N
    trade = make_trade(qty=3, buyer="Mark 14", seller="Mark 38", timestamp=50)
    state = make_baseline_state(timestamp=100, market_trades={HYDROGEL: [trade]})
    # Day boundary clears old entries; only the new trade remains.
    bias = s._compute_signal_bias(state)
    assert s.history == [(50, 3)]
    assert bias == pytest.approx(0.3)


def test_bias_custom_bias_k_and_scale():
    """Verify bias_k and scale are applied correctly."""
    s = make_bias_strategy(bias_k=2.0, scale=20.0)
    trade = make_trade(qty=10, buyer="Mark 14", seller="Mark 38", timestamp=100)
    state = make_baseline_state(timestamp=200, market_trades={HYDROGEL: [trade]})
    # net=10, scale=20 → ratio=0.5, bias = 2.0 * 0.5 = 1.0
    assert s._compute_signal_bias(state) == pytest.approx(1.0)


def test_bias_history_persists_across_ticks():
    """Multiple ticks accumulate trades in history."""
    s = make_bias_strategy()
    t1 = make_trade(qty=5, buyer="Mark 14", seller="Mark 38", timestamp=100)
    t2 = make_trade(qty=5, buyer="Mark 14", seller="Mark 38", timestamp=2000)

    # Tick 1
    state1 = make_baseline_state(timestamp=200, market_trades={HYDROGEL: [t1]})
    s._compute_signal_bias(state1)

    # Tick 2: history should still hold t1, plus t2 from this tick
    state2 = make_baseline_state(timestamp=2100, market_trades={HYDROGEL: [t2]})
    bias = s._compute_signal_bias(state2)
    # net = 5 + 5 = 10, ratio = 1.0, saturated → bias = 1.0
    assert bias == pytest.approx(1.0)
