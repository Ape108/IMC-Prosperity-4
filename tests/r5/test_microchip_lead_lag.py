import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r5" / "groups"))

from datamodel import Order, OrderDepth, Observation, TradingState
from microchip import R5LeadLagMMStrategy


def make_state(
    products: dict[str, tuple[int, int, int, int]],
    positions: dict[str, int] | None = None,
) -> TradingState:
    order_depths: dict[str, OrderDepth] = {}
    for sym, (bid, bid_vol, ask, ask_vol) in products.items():
        od = OrderDepth()
        od.buy_orders = {bid: bid_vol}
        od.sell_orders = {ask: -ask_vol}
        order_depths[sym] = od
    return TradingState(
        traderData="",
        timestamp=0,
        listings={},
        order_depths=order_depths,
        own_trades={},
        market_trades={},
        position=positions or {},
        observations=Observation({}, {}),
    )


def _make_strategy(lag_ticks: int = 50, k: float = 1.0) -> R5LeadLagMMStrategy:
    return R5LeadLagMMStrategy(
        symbol="MICROCHIP_OVAL",
        limit=10,
        width=1,
        leader_symbol="MICROCHIP_CIRCLE",
        lag_ticks=lag_ticks,
        k=k,
    )


def test_constructor():
    s = _make_strategy()
    assert s.symbol == "MICROCHIP_OVAL"
    assert s.limit == 10
    assert s.width == 1
    assert s.leader_symbol == "MICROCHIP_CIRCLE"
    assert s.lag_ticks == 50
    assert s.k == 1.0
    assert s.leader_history == []
    assert s.bias_fired == 0
    assert s.warmup_ticks == 0


def test_first_tick_no_bias():
    """Empty leader_history -> get_true_value returns microprice unchanged."""
    s = _make_strategy()
    state = make_state({
        "MICROCHIP_OVAL":   (100, 5, 110, 5),
        "MICROCHIP_CIRCLE": (1000, 5, 1010, 5),
    })
    # Microprice for OVAL with equal volumes = (100*5 + 110*5)/(5+5) = 105
    assert s.get_true_value(state) == 105.0


def test_save_load_round_trip():
    s = _make_strategy()
    s.leader_history = [1000.0, 1001.0, 1002.5]
    s.bias_fired = 7
    s.warmup_ticks = 53
    data = s.save()

    s2 = _make_strategy()
    s2.load(data)
    assert s2.leader_history == [1000.0, 1001.0, 1002.5]
    assert s2.bias_fired == 7
    assert s2.warmup_ticks == 53


def test_warmup_no_bias_until_full_window():
    """leader_history shorter than lag_ticks + 2 -> no bias applied."""
    s = _make_strategy(lag_ticks=3, k=1.0)  # need 5 mids before bias fires
    state = make_state({
        "MICROCHIP_OVAL":   (100, 5, 110, 5),
        "MICROCHIP_CIRCLE": (1010, 5, 1010, 5),
    })
    # To test warmup, start with 3 entries — after append, len=4 < max_len=5
    s.leader_history = [1000.0, 1001.0, 1002.0]
    fair = s.get_true_value(state)
    assert fair == 105.0, f"expected microprice 105 (no bias during warmup), got {fair}"
    assert len(s.leader_history) == 4, f"expected leader_history grown to 4, got {len(s.leader_history)}"


def test_positive_leader_return_biases_up():
    """Once warmed, a positive single-tick leader return at t-N shifts fair UP by k * ret * base."""
    s = _make_strategy(lag_ticks=3, k=1.0)
    # Pre-fill so that after this tick's append, leader_history = [1000.0, 1010.0, 1020.0, 1030.0, 1040.0]
    # The bias uses (history[1] - history[0]) / history[0] = (1010 - 1000) / 1000 = +0.01
    s.leader_history = [1000.0, 1010.0, 1020.0, 1030.0]  # after append of 1040 -> len=5
    state = make_state({
        "MICROCHIP_OVAL":   (100, 5, 110, 5),
        "MICROCHIP_CIRCLE": (1039, 5, 1041, 5),  # mid=1040
    })
    # base microprice for OVAL = 105
    # leader_return = (1010 - 1000) / 1000 = 0.01
    # bias = k * leader_return * base = 1.0 * 0.01 * 105 = 1.05
    # fair = 105 + 1.05 = 106.05
    fair = s.get_true_value(state)
    assert abs(fair - 106.05) < 1e-9, f"expected 106.05, got {fair}"


def test_negative_leader_return_biases_down():
    """A negative single-tick leader return at t-N shifts fair DOWN."""
    s = _make_strategy(lag_ticks=3, k=1.0)
    s.leader_history = [1000.0, 990.0, 1020.0, 1030.0]
    state = make_state({
        "MICROCHIP_OVAL":   (100, 5, 110, 5),
        "MICROCHIP_CIRCLE": (1039, 5, 1041, 5),
    })
    # leader_return = (990 - 1000) / 1000 = -0.01
    # bias = 1.0 * -0.01 * 105 = -1.05
    # fair = 105 - 1.05 = 103.95
    fair = s.get_true_value(state)
    assert abs(fair - 103.95) < 1e-9, f"expected 103.95, got {fair}"


def test_window_cap_holds_at_lag_ticks_plus_2():
    """After lag_ticks + 5 tick mids appended, leader_history length stays at lag_ticks + 2."""
    s = _make_strategy(lag_ticks=3, k=1.0)
    state = make_state({
        "MICROCHIP_OVAL":   (100, 5, 110, 5),
        "MICROCHIP_CIRCLE": (1000, 5, 1002, 5),  # mid = 1001
    })
    for _ in range(8):  # 8 calls > lag_ticks + 2 = 5
        s.get_true_value(state)
    assert len(s.leader_history) == 5, f"expected length cap at 5, got {len(s.leader_history)}"


def test_leader_book_missing_resets_history():
    """Missing leader_depth -> leader_history reset; returns base microprice."""
    s = _make_strategy(lag_ticks=3, k=1.0)
    s.leader_history = [1000.0, 1010.0, 1020.0, 1030.0, 1040.0]
    # State has OVAL but no CIRCLE entry at all
    state = make_state({
        "MICROCHIP_OVAL": (100, 5, 110, 5),
    })
    fair = s.get_true_value(state)
    assert fair == 105.0, f"expected base microprice 105, got {fair}"
    assert s.leader_history == [], "leader_history should reset on missing leader"


def test_leader_one_sided_book_resets_history():
    """One-sided leader book (e.g. empty sell_orders) -> same reset behavior."""
    s = _make_strategy(lag_ticks=3, k=1.0)
    s.leader_history = [1000.0, 1010.0, 1020.0, 1030.0, 1040.0]
    od_oval = OrderDepth()
    od_oval.buy_orders = {100: 5}
    od_oval.sell_orders = {110: -5}
    od_circle_one_sided = OrderDepth()
    od_circle_one_sided.buy_orders = {1000: 5}
    od_circle_one_sided.sell_orders = {}
    state = TradingState(
        traderData="", timestamp=0, listings={},
        order_depths={"MICROCHIP_OVAL": od_oval, "MICROCHIP_CIRCLE": od_circle_one_sided},
        own_trades={}, market_trades={}, position={},
        observations=Observation({}, {}),
    )
    fair = s.get_true_value(state)
    assert fair == 105.0
    assert s.leader_history == []


def test_bias_fired_counter():
    """bias_fired increments only on ticks where the bias is actually applied."""
    s = _make_strategy(lag_ticks=3, k=1.0)
    state_warmup = make_state({
        "MICROCHIP_OVAL":   (100, 5, 110, 5),
        "MICROCHIP_CIRCLE": (1000, 5, 1002, 5),
    })
    # First 4 calls fill the window (max_len=5 not yet reached) -> no bias fires
    for _ in range(4):
        s.get_true_value(state_warmup)
    assert s.bias_fired == 0
    assert s.warmup_ticks == 4
    # 5th call hits max_len -> bias fires
    s.get_true_value(state_warmup)
    assert s.bias_fired == 1
    # 6th call still has full window -> bias fires again
    s.get_true_value(state_warmup)
    assert s.bias_fired == 2


def test_warmup_ticks_counter():
    """warmup_ticks increments on warmup AND on leader-missing ticks."""
    s = _make_strategy(lag_ticks=3, k=1.0)
    state_full = make_state({
        "MICROCHIP_OVAL":   (100, 5, 110, 5),
        "MICROCHIP_CIRCLE": (1000, 5, 1002, 5),
    })
    state_no_leader = make_state({
        "MICROCHIP_OVAL": (100, 5, 110, 5),
    })
    # 2 warmup ticks (window not full)
    s.get_true_value(state_full)
    s.get_true_value(state_full)
    assert s.warmup_ticks == 2
    # Leader vanishes -> warmup increments and history resets
    s.get_true_value(state_no_leader)
    assert s.warmup_ticks == 3
    assert s.leader_history == []
    # Leader returns -> next 4 ticks are warmup again
    for _ in range(4):
        s.get_true_value(state_full)
    assert s.warmup_ticks == 7
    assert s.bias_fired == 0
    # 5th post-reset tick -> bias fires
    s.get_true_value(state_full)
    assert s.bias_fired == 1


def test_trader_baseline_registers_5_base_mm_strategies():
    """Smoke test: with mm_baseline active, Trader has 5 R5BaseMMStrategy entries."""
    import microchip
    from microchip import R5BaseMMStrategy
    t = microchip.Trader()
    assert len(t.strategies) == 5
    for sym in microchip.SYMBOLS:
        assert sym in t.strategies
        assert isinstance(t.strategies[sym], R5BaseMMStrategy)


def test_trader_lead_lag_variant_wires_oval_with_lead_lag():
    """If lead_lag_oval_k1 is activated, OVAL is R5LeadLagMMStrategy with correct config."""
    # Direct construction check (variant functions are private to Trader.__init__).
    # Confirms the class accepts the wired-in arguments.
    from microchip import R5LeadLagMMStrategy
    s = R5LeadLagMMStrategy(
        "MICROCHIP_OVAL", 10, 1,
        leader_symbol="MICROCHIP_CIRCLE", lag_ticks=50, k=1.0,
    )
    assert s.leader_symbol == "MICROCHIP_CIRCLE"
    assert s.lag_ticks == 50
    assert s.k == 1.0
