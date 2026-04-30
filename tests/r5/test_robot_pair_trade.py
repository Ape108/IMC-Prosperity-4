import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r5" / "groups"))

import pytest
from datamodel import Order, OrderDepth, Observation, TradingState
from robot import R5BaseMMStrategy, R5PairTradeStrategy, Trader


def make_state(
    products: dict[str, tuple[int, int, int, int]],
    positions: dict[str, int] | None = None,
    timestamp: int = 0,
    trader_data: str = "",
) -> TradingState:
    """Build a TradingState. products maps symbol -> (bid, bid_vol, ask, ask_vol)."""
    order_depths: dict[str, OrderDepth] = {}
    for sym, (bid, bid_vol, ask, ask_vol) in products.items():
        od = OrderDepth()
        od.buy_orders = {bid: bid_vol}
        od.sell_orders = {ask: -ask_vol}
        order_depths[sym] = od
    return TradingState(
        traderData=trader_data,
        timestamp=timestamp,
        listings={},
        order_depths=order_depths,
        own_trades={},
        market_trades={},
        position=positions or {},
        observations=Observation({}, {}),
    )


def test_constructor_and_required_symbols():
    """Class instantiates with all parameters; exposes both legs as required symbols."""
    s = R5PairTradeStrategy(
        symbol_a="ROBOT_LAUNDRY",
        symbol_b="ROBOT_VACUUMING",
        limit=10,
        window=200,
        z_entry=2.0,
        z_exit=0.5,
        max_hold_ticks=500,
    )
    assert s.symbol == "ROBOT_LAUNDRY"
    assert s.symbol_a == "ROBOT_LAUNDRY"
    assert s.symbol_b == "ROBOT_VACUUMING"
    assert s.limit == 10
    assert s.window == 200
    assert s.z_entry == 2.0
    assert s.z_exit == 0.5
    assert s.max_hold_ticks == 500
    assert s.spread_history == []
    assert s.entry_tick is None
    assert sorted(s.get_required_symbols()) == ["ROBOT_LAUNDRY", "ROBOT_VACUUMING"]


def test_warmup_no_orders_until_window_full():
    """Window=3. After 2 ticks, no orders. After 3 ticks, history full but z=0 → no orders."""
    s = R5PairTradeStrategy(
        symbol_a="A", symbol_b="B", limit=10,
        window=3, z_entry=2.0, z_exit=0.5, max_hold_ticks=500,
    )
    # Tick 1: spread = 100 - 100 = 0
    state = make_state({"A": (99, 5, 101, 5), "B": (99, 5, 101, 5)})
    orders, _ = s.run(state)
    assert orders == []
    assert s.spread_history == [0.0]

    # Tick 2: same spread = 0
    orders, _ = s.run(state)
    assert orders == []
    assert s.spread_history == [0.0, 0.0]

    # Tick 3: same spread = 0 — history full but std=0 → no entry
    orders, _ = s.run(state)
    assert orders == []
    assert s.spread_history == [0.0, 0.0, 0.0]


def test_spread_history_rolls_at_window_capacity():
    """Window=3. After 4 ticks, history holds the most recent 3 spreads."""
    s = R5PairTradeStrategy(
        symbol_a="A", symbol_b="B", limit=10,
        window=3, z_entry=10.0, z_exit=0.5, max_hold_ticks=500,
    )
    # 4 different spreads
    for a_mid in (100, 102, 104, 106):
        bid, ask = a_mid - 1, a_mid + 1
        state = make_state({"A": (bid, 5, ask, 5), "B": (99, 5, 101, 5)})
        s.run(state)
    # History should be [102-100, 104-100, 106-100] = [2.0, 4.0, 6.0]
    assert s.spread_history == [2.0, 4.0, 6.0]


def test_entry_z_positive_shorts_a_longs_b():
    """Spread spikes high (A rich vs B). z > z_entry → short A at bid, long B at ask, full limit each."""
    s = R5PairTradeStrategy(
        symbol_a="A", symbol_b="B", limit=10,
        window=3, z_entry=1.0, z_exit=0.5, max_hold_ticks=500,
    )
    # Prime history with 3 spreads = 0
    flat = make_state({"A": (99, 5, 101, 5), "B": (99, 5, 101, 5)})
    s.run(flat)
    s.run(flat)
    s.run(flat)
    # Now spike A's mid up by 10 (spread = 10) — z very positive
    spike = make_state({"A": (109, 5, 111, 5), "B": (99, 5, 101, 5)}, timestamp=400)
    orders, _ = s.run(spike)

    # Expect 2 orders: sell A at bid=109 size 10, buy B at ask=101 size 10
    by_symbol = {(o.symbol, o.price): o.quantity for o in orders}
    assert ("A", 109) in by_symbol and by_symbol[("A", 109)] == -10
    assert ("B", 101) in by_symbol and by_symbol[("B", 101)] == +10
    assert s.entry_tick == 400


def test_entry_z_negative_longs_a_shorts_b():
    """Spread plunges low (A cheap vs B). z < -z_entry → long A at ask, short B at bid."""
    s = R5PairTradeStrategy(
        symbol_a="A", symbol_b="B", limit=10,
        window=3, z_entry=1.0, z_exit=0.5, max_hold_ticks=500,
    )
    flat = make_state({"A": (99, 5, 101, 5), "B": (99, 5, 101, 5)})
    s.run(flat); s.run(flat); s.run(flat)
    # Drop A's mid by 10 (spread = -10)
    drop = make_state({"A": (89, 5, 91, 5), "B": (99, 5, 101, 5)}, timestamp=400)
    orders, _ = s.run(drop)

    by_symbol = {(o.symbol, o.price): o.quantity for o in orders}
    assert ("A", 91) in by_symbol and by_symbol[("A", 91)] == +10
    assert ("B", 99) in by_symbol and by_symbol[("B", 99)] == -10
    assert s.entry_tick == 400


def test_no_entry_when_z_below_threshold():
    """Spread moves but |z| < z_entry → no orders, entry_tick stays None."""
    s = R5PairTradeStrategy(
        symbol_a="A", symbol_b="B", limit=10,
        window=3, z_entry=10.0, z_exit=0.5, max_hold_ticks=500,
    )
    flat = make_state({"A": (99, 5, 101, 5), "B": (99, 5, 101, 5)})
    s.run(flat); s.run(flat); s.run(flat)
    small = make_state({"A": (100, 5, 102, 5), "B": (99, 5, 101, 5)}, timestamp=400)
    orders, _ = s.run(small)
    assert orders == []
    assert s.entry_tick is None


def test_exit_on_z_revert_to_below_z_exit():
    """When holding short A / long B, spread reverts toward 0 (|z| < z_exit) → flatten both legs."""
    s = R5PairTradeStrategy(
        symbol_a="A", symbol_b="B", limit=10,
        window=3, z_entry=1.0, z_exit=0.5, max_hold_ticks=500,
    )
    # Prime + spike entry
    flat = make_state({"A": (99, 5, 101, 5), "B": (99, 5, 101, 5)})
    s.run(flat); s.run(flat); s.run(flat)
    spike = make_state({"A": (109, 5, 111, 5), "B": (99, 5, 101, 5)}, timestamp=400)
    s.run(spike)
    # Now we're "holding" — simulate position state and a reverted spread (z near 0)
    # mean of [0,0,0,10] = 2.5; std = sqrt(((0-2.5)^2 * 3 + (10-2.5)^2)/4) = sqrt((18.75+56.25)/4) = sqrt(18.75) = 4.33
    # New spread = 2.5 → z = 0 → exit
    revert = make_state(
        {"A": (101, 5, 103, 5), "B": (99, 5, 101, 5)},  # mid_A=102, mid_B=100, spread=2 ~= mean
        positions={"A": -10, "B": +10},
        timestamp=500,
    )
    orders, _ = s.run(revert)

    by_symbol = {(o.symbol, o.price): o.quantity for o in orders}
    # Flatten: buy A 10 at ask=103, sell B 10 at bid=99
    assert ("A", 103) in by_symbol and by_symbol[("A", 103)] == +10
    assert ("B", 99) in by_symbol and by_symbol[("B", 99)] == -10
    # entry_tick stays set during the exit attempt — fills haven't confirmed yet
    assert s.entry_tick == 400

    # Next tick with positions confirmed flat → entry_tick cleared
    fills_confirmed = make_state(
        {"A": (101, 5, 103, 5), "B": (99, 5, 101, 5)},
        positions={"A": 0, "B": 0},
        timestamp=600,
    )
    s.run(fills_confirmed)
    assert s.entry_tick is None


def test_no_exit_when_still_above_z_exit():
    """When holding and |z| > z_exit, do not flatten — wait."""
    s = R5PairTradeStrategy(
        symbol_a="A", symbol_b="B", limit=10,
        window=3, z_entry=1.0, z_exit=0.5, max_hold_ticks=500,
    )
    flat = make_state({"A": (99, 5, 101, 5), "B": (99, 5, 101, 5)})
    s.run(flat); s.run(flat); s.run(flat)
    spike = make_state({"A": (109, 5, 111, 5), "B": (99, 5, 101, 5)}, timestamp=400)
    s.run(spike)
    # Spread still elevated (z still > z_exit) — no exit
    still_wide = make_state(
        {"A": (108, 5, 110, 5), "B": (99, 5, 101, 5)},  # spread still ~= 9
        positions={"A": -10, "B": +10},
        timestamp=500,
    )
    orders, _ = s.run(still_wide)
    assert orders == []
    assert s.entry_tick == 400  # unchanged


def test_max_hold_ticks_forces_exit():
    """Holding past max_hold_ticks → flatten even if spread hasn't reverted."""
    s = R5PairTradeStrategy(
        symbol_a="A", symbol_b="B", limit=10,
        window=3, z_entry=1.0, z_exit=0.5, max_hold_ticks=3,  # 3 ticks * 100 = timestamp delta of 300
    )
    flat = make_state({"A": (99, 5, 101, 5), "B": (99, 5, 101, 5)})
    s.run(flat); s.run(flat); s.run(flat)
    # Entry at timestamp 400
    spike = make_state({"A": (109, 5, 111, 5), "B": (99, 5, 101, 5)}, timestamp=400)
    s.run(spike)
    assert s.entry_tick == 400

    # 3 ticks pass, spread still wide — should NOT exit yet (delta == max * 100)
    still_at_limit = make_state(
        {"A": (109, 5, 111, 5), "B": (99, 5, 101, 5)},
        positions={"A": -10, "B": +10},
        timestamp=700,  # delta = 300 = 3 ticks
    )
    orders, _ = s.run(still_at_limit)
    assert orders == []  # exactly at limit, no exit yet
    assert s.entry_tick == 400

    # One more tick → delta = 400 > 300 → force exit
    over_limit = make_state(
        {"A": (109, 5, 111, 5), "B": (99, 5, 101, 5)},
        positions={"A": -10, "B": +10},
        timestamp=800,
    )
    orders, _ = s.run(over_limit)
    by_symbol = {(o.symbol, o.price): o.quantity for o in orders}
    assert ("A", 111) in by_symbol and by_symbol[("A", 111)] == +10
    assert ("B", 99) in by_symbol and by_symbol[("B", 99)] == -10
    # entry_tick stays set during the exit attempt — fills haven't confirmed yet
    assert s.entry_tick == 400

    # Next tick with positions confirmed flat → entry_tick cleared
    fills_confirmed = make_state(
        {"A": (109, 5, 111, 5), "B": (99, 5, 101, 5)},
        positions={"A": 0, "B": 0},
        timestamp=900,
    )
    s.run(fills_confirmed)
    assert s.entry_tick is None


def test_partial_fill_exit_does_not_crash_on_residual_position():
    """
    Regression: under the old behavior, exit cleared entry_tick eagerly. If exit orders
    only partially filled, the next tick would have a residual position with entry_tick=None,
    triggering the invariant assertion. Under the fix, entry_tick stays set until positions
    actually reach zero — partial-fill exit retries cleanly.
    """
    s = R5PairTradeStrategy(
        symbol_a="A", symbol_b="B", limit=10,
        window=3, z_entry=1.0, z_exit=0.5, max_hold_ticks=500,
    )
    flat = make_state({"A": (99, 5, 101, 5), "B": (99, 5, 101, 5)})
    s.run(flat); s.run(flat); s.run(flat)

    # Entry at timestamp 400
    spike = make_state({"A": (109, 5, 111, 5), "B": (99, 5, 101, 5)}, timestamp=400)
    s.run(spike)
    assert s.entry_tick == 400

    # Z reverts; exit fires with positions still at full
    revert = make_state(
        {"A": (101, 5, 103, 5), "B": (99, 5, 101, 5)},
        positions={"A": -10, "B": +10},
        timestamp=500,
    )
    s.run(revert)
    assert s.entry_tick == 400  # not cleared during exit attempt

    # Simulate partial fill: A flattened to -2, B fully flat
    partial = make_state(
        {"A": (101, 5, 103, 5), "B": (99, 5, 101, 5)},
        positions={"A": -2, "B": 0},
        timestamp=600,
    )
    # Should NOT raise the invariant assertion — entry_tick is still 400
    orders, _ = s.run(partial)
    assert s.entry_tick == 400  # still tracking the original entry

    # Final fill confirms flat → entry_tick clears
    final = make_state(
        {"A": (101, 5, 103, 5), "B": (99, 5, 101, 5)},
        positions={"A": 0, "B": 0},
        timestamp=700,
    )
    s.run(final)
    assert s.entry_tick is None


def test_save_load_roundtrip_preserves_state():
    """After running 5 ticks and entering a position, save → fresh load → state matches."""
    s = R5PairTradeStrategy(
        symbol_a="A", symbol_b="B", limit=10,
        window=3, z_entry=1.0, z_exit=0.5, max_hold_ticks=500,
    )
    flat = make_state({"A": (99, 5, 101, 5), "B": (99, 5, 101, 5)})
    s.run(flat); s.run(flat); s.run(flat)
    spike = make_state({"A": (109, 5, 111, 5), "B": (99, 5, 101, 5)}, timestamp=400)
    s.run(spike)

    saved = s.save()
    assert saved == {"spread_history": [0.0, 0.0, 10.0], "entry_tick": 400}

    s2 = R5PairTradeStrategy(
        symbol_a="A", symbol_b="B", limit=10,
        window=3, z_entry=1.0, z_exit=0.5, max_hold_ticks=500,
    )
    s2.load(saved)
    assert s2.spread_history == [0.0, 0.0, 10.0]
    assert s2.entry_tick == 400


def test_load_handles_missing_or_none_entry_tick():
    """load() with missing keys or null entry_tick produces a clean flat state."""
    s = R5PairTradeStrategy(
        symbol_a="A", symbol_b="B", limit=10,
        window=3, z_entry=1.0, z_exit=0.5, max_hold_ticks=500,
    )
    s.load({})
    assert s.spread_history == []
    assert s.entry_tick is None

    s.load({"spread_history": [1.0, 2.0], "entry_tick": None})
    assert s.spread_history == [1.0, 2.0]
    assert s.entry_tick is None


def test_trader_run_dispatches_multi_symbol_orders_correctly():
    """
    Trader registered with R5PairTradeStrategy under 'A' must place orders for both 'A' and 'B'
    in the returned orders dict, even though the strategy is registered under one key.
    """
    t = Trader()
    t.strategies = {
        "A": R5PairTradeStrategy(
            symbol_a="A", symbol_b="B", limit=10,
            window=3, z_entry=1.0, z_exit=0.5, max_hold_ticks=500,
        ),
    }
    # Prime
    flat = make_state({"A": (99, 5, 101, 5), "B": (99, 5, 101, 5)})
    t.run(flat)
    t.run(flat)
    t.run(flat)
    # Spike → entry on both legs
    spike = make_state({"A": (109, 5, 111, 5), "B": (99, 5, 101, 5)}, timestamp=400)
    orders, conversions, trader_data = t.run(spike)

    assert "A" in orders and "B" in orders
    assert any(o.symbol == "A" and o.quantity == -10 for o in orders["A"])
    assert any(o.symbol == "B" and o.quantity == +10 for o in orders["B"])


def test_trader_run_persists_pair_trade_state():
    """trader_data round-trips spread_history and entry_tick across run() invocations."""
    t = Trader()
    t.strategies = {
        "A": R5PairTradeStrategy(
            symbol_a="A", symbol_b="B", limit=10,
            window=3, z_entry=1.0, z_exit=0.5, max_hold_ticks=500,
        ),
    }
    flat = make_state({"A": (99, 5, 101, 5), "B": (99, 5, 101, 5)})
    _, _, td1 = t.run(flat)

    parsed = json.loads(td1)
    assert "A" in parsed
    assert parsed["A"]["spread_history"] == [0.0]


def test_trader_run_does_not_overwrite_orders_from_separate_strategies():
    """
    Two strategies registered: pair trade on (A, B), base MM on C. All three symbols' orders
    must appear; pair trade's B orders must not overwrite base MM on C.
    """
    t = Trader()
    t.strategies = {
        "A": R5PairTradeStrategy(
            symbol_a="A", symbol_b="B", limit=10,
            window=3, z_entry=1.0, z_exit=0.5, max_hold_ticks=500,
        ),
        "C": R5BaseMMStrategy("C", limit=10, width=2),
    }
    flat = make_state({"A": (99, 5, 101, 5), "B": (99, 5, 101, 5), "C": (99, 5, 101, 5)})
    t.run(flat); t.run(flat); t.run(flat)
    spike = make_state(
        {"A": (109, 5, 111, 5), "B": (99, 5, 101, 5), "C": (99, 5, 101, 5)},
        timestamp=400,
    )
    orders, _, _ = t.run(spike)

    assert set(orders.keys()) >= {"A", "B", "C"}
    assert len(orders["C"]) > 0  # base MM produced quotes for C


def test_trader_run_load_path_restores_state_from_trader_data():
    """
    End-to-end: Trader.run serializes state into trader_data; on a subsequent run with that
    trader_data fed back as state.traderData, Trader.run must call strategy.load() so the
    strategy resumes from the saved state, not from __init__-initialized state.
    """
    t1 = Trader()
    t1.strategies = {
        "A": R5PairTradeStrategy(
            symbol_a="A", symbol_b="B", limit=10,
            window=3, z_entry=1.0, z_exit=0.5, max_hold_ticks=500,
        ),
    }
    flat = make_state({"A": (99, 5, 101, 5), "B": (99, 5, 101, 5)})
    _, _, td1 = t1.run(flat)
    _, _, td2 = t1.run(flat)
    _, _, td3 = t1.run(flat)
    # td3 captures spread_history = [0.0, 0.0, 0.0]

    # Fresh Trader, fresh strategy — must restore via load()
    t2 = Trader()
    t2.strategies = {
        "A": R5PairTradeStrategy(
            symbol_a="A", symbol_b="B", limit=10,
            window=3, z_entry=1.0, z_exit=0.5, max_hold_ticks=500,
        ),
    }
    # Pre-run, state should be __init__-empty
    assert t2.strategies["A"].spread_history == []
    assert t2.strategies["A"].entry_tick is None

    # Feed td3 as the platform would; Trader.run must call load() before strategy.run()
    state_with_td = make_state({"A": (99, 5, 101, 5), "B": (99, 5, 101, 5)}, trader_data=td3)
    t2.run(state_with_td)

    # Now spread_history should contain the loaded state PLUS this tick's spread
    # Loaded: [0.0, 0.0, 0.0]. After this tick, append 0 → [0.0, 0.0, 0.0, 0.0], then pop because window=3 → [0.0, 0.0, 0.0]
    assert t2.strategies["A"].spread_history == [0.0, 0.0, 0.0]
