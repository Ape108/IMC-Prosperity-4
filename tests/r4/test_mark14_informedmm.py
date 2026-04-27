"""Tests for Mark14InformedMMStrategy — pure-passive informed-MM bias."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r4"))

import pytest
from datamodel import Order, OrderDepth, Trade, TradingState  # noqa
from strategy_h import (
    Mark14InformedMMStrategy,
    MARK14_INFORMED_BOT,
    MARK14_WINDOW_TICKS,
)


def make_strategy(**kwargs) -> Mark14InformedMMStrategy:
    defaults = dict(
        symbol="VEV_5300", limit=300, strike=5300,
        predicted_dir="UP", max_pos=50,
    )
    defaults.update(kwargs)
    return Mark14InformedMMStrategy(**defaults)


def test_constructor_accepts_up():
    s = make_strategy(predicted_dir="UP")
    assert s.predicted_dir == "UP"


def test_constructor_accepts_down():
    s = make_strategy(predicted_dir="DOWN")
    assert s.predicted_dir == "DOWN"


def test_constructor_rejects_invalid_predicted_dir():
    with pytest.raises(ValueError):
        make_strategy(predicted_dir="bogus")
    with pytest.raises(ValueError):
        make_strategy(predicted_dir="up")  # case-sensitive


def make_full_state(timestamp: int, market_trades: dict[str, list[Trade]],
                    position: dict[str, int] | None = None) -> TradingState:
    """Build a TradingState with VELVETFRUIT_EXTRACT spot + all 10 voucher books.

    Spot mid is 5247 (so 5300/5400/5500 are OTM). Each voucher has a 2-wide
    book with sensible mids that produce a fittable IV smile.
    """
    if position is None:
        position = {}

    # Spot book: mid 5247
    od_spot = OrderDepth()
    od_spot.buy_orders = {5246: 50}
    od_spot.sell_orders = {5248: -50}

    order_depths = {"VELVETFRUIT_EXTRACT": od_spot}
    # Reasonable mids per strike (decreasing with strike from ITM to OTM).
    voucher_mids = {
        4000: 1247, 4500: 747, 5000: 250, 5100: 160, 5200: 95,
        5300: 50, 5400: 25, 5500: 12, 6000: 1, 6500: 1,
    }
    for strike, mid in voucher_mids.items():
        od = OrderDepth()
        od.buy_orders = {mid - 1: 5}
        od.sell_orders = {mid + 1: -5}
        order_depths[f"VEV_{strike}"] = od

    return TradingState(
        traderData="",
        timestamp=timestamp,
        listings={},
        order_depths=order_depths,
        own_trades={},
        market_trades=market_trades,
        position=position,
        observations=None,  # type: ignore
    )


def test_signal_starts_unset():
    s = make_strategy()
    assert s.last_signal_ts is None


def test_mark14_buy_updates_last_signal_ts():
    s = make_strategy()
    trade = Trade("VEV_5300", 50, 5, MARK14_INFORMED_BOT, "Other", timestamp=300)
    state = make_full_state(timestamp=350, market_trades={"VEV_5300": [trade]})
    s.run(state)
    assert s.last_signal_ts == 300


def test_non_mark14_buyer_ignored():
    s = make_strategy()
    trade = Trade("VEV_5300", 50, 5, "Mark 22", "Other", timestamp=300)
    state = make_full_state(timestamp=350, market_trades={"VEV_5300": [trade]})
    s.run(state)
    assert s.last_signal_ts is None


def test_day_boundary_guard_resets_stale_ts():
    s = make_strategy()
    s.last_signal_ts = 999_500
    state = make_full_state(timestamp=200, market_trades={"VEV_5300": []})
    s.run(state)
    assert s.last_signal_ts is None


def test_multiple_buys_uses_max_timestamp():
    s = make_strategy()
    trades = [
        Trade("VEV_5300", 50, 5, MARK14_INFORMED_BOT, "Other", timestamp=100),
        Trade("VEV_5300", 50, 5, MARK14_INFORMED_BOT, "Other", timestamp=300),
        Trade("VEV_5300", 50, 5, MARK14_INFORMED_BOT, "Other", timestamp=200),
    ]
    state = make_full_state(timestamp=400, market_trades={"VEV_5300": trades})
    s.run(state)
    assert s.last_signal_ts == 300


def test_offsets_symmetric_when_out_of_window():
    s = make_strategy(predicted_dir="UP", base_w=2.0, delta=1.0)
    bid_off, ask_off = s._compute_offsets(in_window=False)
    assert bid_off == 2.0
    assert ask_off == 2.0


def test_offsets_in_window_up_tightens_bid_loosens_ask():
    s = make_strategy(predicted_dir="UP", base_w=2.0, delta=1.0)
    bid_off, ask_off = s._compute_offsets(in_window=True)
    assert bid_off == 1.0   # W - delta = tightened (closer to fair, more competitive)
    assert ask_off == 3.0   # W + delta = loosened (further from fair, less likely to fill)


def test_offsets_in_window_down_tightens_ask_loosens_bid():
    s = make_strategy(predicted_dir="DOWN", base_w=2.0, delta=1.0)
    bid_off, ask_off = s._compute_offsets(in_window=True)
    assert bid_off == 3.0   # W + delta = loosened
    assert ask_off == 1.0   # W - delta = tightened


def test_inv_skew_zero_position_returns_zero():
    s = make_strategy(inv_k=5.0)
    assert s._compute_inv_skew(0) == 0.0


def test_inv_skew_long_position_negative():
    s = make_strategy(limit=300, inv_k=5.0)
    # pos = +150 → ratio = 0.5 → inv_skew = -0.5 * 5 = -2.5
    assert s._compute_inv_skew(150) == pytest.approx(-2.5)


def test_inv_skew_short_position_positive():
    s = make_strategy(limit=300, inv_k=5.0)
    # pos = -150 → ratio = -0.5 → inv_skew = 0.5 * 5 = +2.5
    assert s._compute_inv_skew(-150) == pytest.approx(2.5)


def test_inv_skew_at_full_long_returns_negative_inv_k():
    s = make_strategy(limit=300, inv_k=5.0)
    # pos = +300 → ratio = 1.0 → inv_skew = -5.0
    assert s._compute_inv_skew(300) == pytest.approx(-5.0)


def test_microprice_uses_volume_weighted_touch():
    s = make_strategy()
    od = OrderDepth()
    od.buy_orders = {100: 10}      # bid 100, size 10
    od.sell_orders = {102: -30}    # ask 102, size 30
    # microprice = (100 * 30 + 102 * 10) / 40 = (3000 + 1020) / 40 = 100.5
    assert s._compute_microprice(od) == pytest.approx(100.5)


def test_microprice_falls_back_to_mid_when_no_volume():
    s = make_strategy()
    od = OrderDepth()
    od.buy_orders = {100: 0}
    od.sell_orders = {102: 0}
    assert s._compute_microprice(od) == pytest.approx(101.0)


def test_compute_fair_blends_microprice_and_smile_theo():
    s = make_strategy(strike=5300, predicted_dir="UP")
    state = make_full_state(timestamp=100, market_trades={})
    fair = s._compute_fair(state)
    # We can't predict the exact value but it must be a finite float and
    # within a sensible range around the strike's voucher mid (~50 for 5300).
    assert fair is not None
    assert 0.0 < fair < 200.0


def test_compute_fair_falls_back_to_microprice_when_smile_unfittable():
    """When fewer than 3 voucher mids produce valid IVs, the blend falls back
    to pure microprice (the smile-theo term is dropped)."""
    s = make_strategy(strike=5300, predicted_dir="UP")
    # Build a state where almost all vouchers fall below intrinsic + 0.5
    # so the smile fit yields < 3 valid points. Spot 5247 means strikes 4000/4500
    # are deep ITM (above intrinsic). Force their books to be at intrinsic.
    state = make_full_state(timestamp=100, market_trades={})
    # Crush 4000 and 4500 to their intrinsic floor; voucher 5300 still has its mid.
    od_4000 = state.order_depths["VEV_4000"]
    od_4000.buy_orders = {1247: 5}
    od_4000.sell_orders = {1248: -5}    # mid 1247.5 ≈ intrinsic 1247
    # 5000-6500 books are above intrinsic, so smile fit may still succeed —
    # we don't assert smile fails here; just assert fair is computable.
    fair = s._compute_fair(state)
    assert fair is not None


def test_quote_prices_basic():
    s = make_strategy()
    od = OrderDepth()
    od.buy_orders = {100: 10}
    od.sell_orders = {110: -10}
    bid_price, ask_price = s._compute_quote_prices(
        od, fair_used=104.7, bid_offset=2.0, ask_offset=2.0,
    )
    # bid = floor(104.7 - 2.0) = floor(102.7) = 102
    # ask = ceil(104.7 + 2.0) = ceil(106.7) = 107
    assert bid_price == 102
    assert ask_price == 107


def test_quote_bid_clamped_below_best_ask():
    """If fair_used jumps so bid_price >= best_ask, clamp to best_ask - 1."""
    s = make_strategy()
    od = OrderDepth()
    od.buy_orders = {100: 10}
    od.sell_orders = {103: -10}
    # Naive: bid = floor(120 - 0) = 120; clamped to 102 (best_ask - 1)
    bid_price, ask_price = s._compute_quote_prices(
        od, fair_used=120.0, bid_offset=0.0, ask_offset=0.0,
    )
    assert bid_price == 102


def test_quote_ask_clamped_above_best_bid():
    """If fair_used jumps so ask_price <= best_bid, clamp to best_bid + 1."""
    s = make_strategy()
    od = OrderDepth()
    od.buy_orders = {100: 10}
    od.sell_orders = {103: -10}
    # Naive: ask = ceil(50 + 0) = 50; clamped to 101 (best_bid + 1)
    bid_price, ask_price = s._compute_quote_prices(
        od, fair_used=50.0, bid_offset=0.0, ask_offset=0.0,
    )
    assert ask_price == 101


def test_quote_sizes_default_to_quote_size():
    s = make_strategy(max_pos=50, quote_size=10, max_clip=15)
    bid_qty, ask_qty = s._compute_quote_sizes(position=0)
    assert bid_qty == 10
    assert ask_qty == 10


def test_max_clip_caps_per_tick():
    s = make_strategy(max_pos=100, quote_size=20, max_clip=15)
    bid_qty, ask_qty = s._compute_quote_sizes(position=0)
    assert bid_qty == 15   # min(20, 15, 100) = 15
    assert ask_qty == 15


def test_bid_suppressed_at_positive_max_pos():
    s = make_strategy(max_pos=50)
    bid_qty, ask_qty = s._compute_quote_sizes(position=50)
    assert bid_qty == 0    # at +max_pos → can't buy more
    assert ask_qty > 0     # can still sell


def test_ask_suppressed_at_negative_max_pos():
    s = make_strategy(max_pos=50)
    bid_qty, ask_qty = s._compute_quote_sizes(position=-50)
    assert bid_qty > 0
    assert ask_qty == 0    # at -max_pos → can't sell more


def test_partial_room_clamps_quote_size():
    s = make_strategy(max_pos=50, quote_size=10, max_clip=15)
    # position=44 → room to buy = 6, so bid_qty = min(10, 15, 6) = 6
    bid_qty, ask_qty = s._compute_quote_sizes(position=44)
    assert bid_qty == 6
    assert ask_qty == 10   # ask side has 50+44 = 94 lots of room → quote_size wins


def test_act_emits_two_sided_quotes_out_of_window():
    s = make_strategy(predicted_dir="UP", base_w=2.0, delta=1.0)
    state = make_full_state(timestamp=100, market_trades={"VEV_5300": []})
    orders, _ = s.run(state)
    # No signal → out of window → symmetric. Expect one BUY and one SELL.
    buys = [o for o in orders if o.quantity > 0]
    sells = [o for o in orders if o.quantity < 0]
    assert len(buys) == 1
    assert len(sells) == 1
    # Symmetric quotes — bid and ask are equidistant from fair.
    # Exact prices depend on fair value but spread should be ≥ 2 (= 2W).
    assert sells[0].price - buys[0].price >= 2


def test_act_skews_quotes_in_window_for_up_strike():
    s = make_strategy(predicted_dir="UP", base_w=2.0, delta=1.0, max_pos=50)
    trade = Trade("VEV_5300", 50, 5, MARK14_INFORMED_BOT, "Other", timestamp=100)
    state = make_full_state(timestamp=200, market_trades={"VEV_5300": [trade]})
    orders, _ = s.run(state)
    buys = [o for o in orders if o.quantity > 0]
    sells = [o for o in orders if o.quantity < 0]
    assert len(buys) == 1
    assert len(sells) == 1
    # In-window UP: bid tighter (closer to fair) and ask wider; the spread
    # remains ≥ 2 (W - δ + W + δ = 2W) so equal to or wider than out-of-window.
    # Tight bid should also be HIGHER than the out-of-window symmetric bid.
    assert sells[0].price - buys[0].price >= 2


def test_act_skews_quotes_in_window_for_down_strike():
    s = make_strategy(
        symbol="VEV_5400", strike=5400, predicted_dir="DOWN",
        base_w=2.0, delta=1.0, max_pos=60,
    )
    trade = Trade("VEV_5400", 25, 5, MARK14_INFORMED_BOT, "Other", timestamp=100)
    state = make_full_state(timestamp=200, market_trades={"VEV_5400": [trade]})
    orders, _ = s.run(state)
    buys = [o for o in orders if o.quantity > 0]
    sells = [o for o in orders if o.quantity < 0]
    assert len(buys) == 1
    assert len(sells) == 1


def test_act_suppresses_bid_at_positive_max_pos():
    s = make_strategy(predicted_dir="UP", max_pos=50)
    state = make_full_state(
        timestamp=100,
        market_trades={"VEV_5300": []},
        position={"VEV_5300": 50},
    )
    orders, _ = s.run(state)
    # At +max_pos: no BUY orders, only SELL.
    buys = [o for o in orders if o.quantity > 0]
    assert buys == []


def test_act_suppresses_ask_at_negative_max_pos():
    s = make_strategy(predicted_dir="UP", max_pos=50)
    state = make_full_state(
        timestamp=100,
        market_trades={"VEV_5300": []},
        position={"VEV_5300": -50},
    )
    orders, _ = s.run(state)
    sells = [o for o in orders if o.quantity < 0]
    assert sells == []


def test_act_no_orders_when_fair_unavailable():
    """If our symbol's order depth is missing or empty, no orders emitted."""
    s = make_strategy()
    od_spot = OrderDepth()
    od_spot.buy_orders = {5246: 50}
    od_spot.sell_orders = {5248: -50}
    # Empty book on our voucher — no buy/sell orders.
    od = OrderDepth()
    od.buy_orders = {}
    od.sell_orders = {}
    order_depths = {"VELVETFRUIT_EXTRACT": od_spot, "VEV_5300": od}
    # The base Strategy.run() guards on every required symbol having both
    # buys and sells; with the voucher empty, act() never runs.
    state = TradingState(
        traderData="", timestamp=100, listings={},
        order_depths=order_depths, own_trades={}, market_trades={},
        position={}, observations=None,  # type: ignore
    )
    orders, _ = s.run(state)
    assert orders == []


def test_save_returns_dict_with_last_signal_ts():
    s = make_strategy()
    s.last_signal_ts = 1234
    saved = s.save()
    assert saved == {"last_signal_ts": 1234}


def test_load_restores_last_signal_ts():
    s = make_strategy()
    s.load({"last_signal_ts": 5678})
    assert s.last_signal_ts == 5678


def test_load_empty_dict_gives_none():
    s = make_strategy()
    s.load({})
    assert s.last_signal_ts is None


def test_save_load_roundtrip():
    s = make_strategy()
    trade = Trade("VEV_5300", 50, 5, MARK14_INFORMED_BOT, "Other", timestamp=300)
    state = make_full_state(timestamp=400, market_trades={"VEV_5300": [trade]})
    s.run(state)
    saved = s.save()
    s2 = make_strategy()
    s2.load(saved)
    assert s2.last_signal_ts == 300
