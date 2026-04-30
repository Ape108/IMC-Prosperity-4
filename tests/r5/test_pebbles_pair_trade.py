import sys
from pathlib import Path

# conftest already adds these, but be explicit for IDE / standalone runs
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r5" / "groups"))

from datamodel import Order, OrderDepth, Observation, TradingState
import pebbles


def make_state(
    products: dict[str, tuple[int, int, int, int]],
    positions: dict[str, int] | None = None,
    trader_data: str = "",
) -> TradingState:
    """products maps symbol -> (bid, bid_vol, ask, ask_vol)."""
    order_depths: dict[str, OrderDepth] = {}
    for sym, (bid, bid_vol, ask, ask_vol) in products.items():
        od = OrderDepth()
        od.buy_orders = {bid: bid_vol}
        od.sell_orders = {ask: -ask_vol}
        order_depths[sym] = od
    return TradingState(
        traderData=trader_data,
        timestamp=0,
        listings={},
        order_depths=order_depths,
        own_trades={},
        market_trades={},
        position=positions or {},
        observations=Observation({}, {}),
    )


class _MultiSymbolStrategy(pebbles.Strategy):
    """Stub strategy that returns orders on two different symbols."""

    def __init__(self, symbol: str, partner: str, limit: int) -> None:
        super().__init__(symbol, limit)
        self.partner = partner

    def get_required_symbols(self) -> list[str]:
        return [self.symbol, self.partner]

    def act(self, state: TradingState) -> None:
        self.orders.append(Order(self.symbol, 100, 1))
        self.orders.append(Order(self.partner, 200, -1))


def test_run_dispatch_accumulates_multi_symbol_orders():
    """Strategy returning orders on two symbols must populate orders[both_symbols]."""
    trader = pebbles.Trader()
    trader.strategies = {
        "PEBBLES_M": _MultiSymbolStrategy("PEBBLES_M", "PEBBLES_XL", limit=10),
    }
    state = make_state({
        "PEBBLES_M":  (100, 5, 110, 5),
        "PEBBLES_XL": (200, 5, 210, 5),
    })
    orders, _, _ = trader.run(state)
    m_orders = orders.get("PEBBLES_M", [])
    xl_orders = orders.get("PEBBLES_XL", [])
    assert any(o.price == 100 and o.quantity == 1 for o in m_orders), \
        f"PEBBLES_M order missing: {m_orders}"
    assert any(o.price == 200 and o.quantity == -1 for o in xl_orders), \
        f"PEBBLES_XL order missing: {xl_orders}"


from pebbles import R5PairTradeStrategy


def _make_pair() -> R5PairTradeStrategy:
    return R5PairTradeStrategy(
        symbol_a="PEBBLES_M",
        symbol_b="PEBBLES_XL",
        limit=10,
        window=5,
        z_entry=2.0,
        z_exit=0.5,
        max_hold_ticks=500,
    )


def _build_state(mid_a: float, mid_b: float, positions: dict[str, int] | None = None) -> TradingState:
    """Construct a state with M and XL bid/ask centered around mid values, spread=2."""
    return make_state(
        {
            "PEBBLES_M":  (int(mid_a) - 1, 5, int(mid_a) + 1, 5),
            "PEBBLES_XL": (int(mid_b) - 1, 5, int(mid_b) + 1, 5),
        },
        positions=positions,
    )


def test_pair_constructor():
    s = _make_pair()
    assert s.symbol_a == "PEBBLES_M"
    assert s.symbol_b == "PEBBLES_XL"
    assert s.symbol == "PEBBLES_M"  # dispatch key
    assert s.limit == 10
    assert s.window == 5
    assert s.z_entry == 2.0
    assert s.z_exit == 0.5
    assert s.max_hold_ticks == 500
    assert s.spread_history == []
    assert s.entry_tick is None


def test_no_orders_until_window_full():
    """Window=5; no orders for first 4 ticks regardless of spread."""
    s = _make_pair()
    for i in range(4):
        state = _build_state(100, 50)  # constant spread=50
        orders, _ = s.run(state)
        assert orders == [], f"tick {i}: expected no orders, got {orders}"
    assert len(s.spread_history) == 4


def test_entry_when_z_above_threshold():
    """After window fills with low-variance spread, a large spike triggers entry."""
    s = _make_pair()
    # Seed history: spreads of 49,50,51,50,49 (mean=49.8, std small)
    s.spread_history = [49.0, 50.0, 51.0, 50.0, 49.0]
    # Feed a 6th tick where spread is 200 -> z is huge positive
    state = _build_state(220, 20)  # mid_a=220, mid_b=20, spread=200
    orders, _ = s.run(state)
    # z is positive (spread rich) -> short A, long B at full limit
    a_orders = [o for o in orders if o.symbol == "PEBBLES_M"]
    b_orders = [o for o in orders if o.symbol == "PEBBLES_XL"]
    assert len(a_orders) == 1 and a_orders[0].quantity == -10, \
        f"expected short PEBBLES_M @ -10, got {a_orders}"
    assert len(b_orders) == 1 and b_orders[0].quantity == 10, \
        f"expected long PEBBLES_XL @ +10, got {b_orders}"


def test_entry_negative_z_long_a_short_b():
    """Negative z (spread cheap) -> long A, short B."""
    s = _make_pair()
    s.spread_history = [49.0, 50.0, 51.0, 50.0, 49.0]
    state = _build_state(20, 120)  # spread = -100, much below mean ~50
    orders, _ = s.run(state)
    a_orders = [o for o in orders if o.symbol == "PEBBLES_M"]
    b_orders = [o for o in orders if o.symbol == "PEBBLES_XL"]
    assert len(a_orders) == 1 and a_orders[0].quantity == 10, \
        f"expected long PEBBLES_M @ +10, got {a_orders}"
    assert len(b_orders) == 1 and b_orders[0].quantity == -10, \
        f"expected short PEBBLES_XL @ -10, got {b_orders}"


def test_exit_when_z_below_exit_threshold():
    """Holding position, |z| drops below z_exit -> flatten both legs."""
    s = _make_pair()
    s.spread_history = [49.0, 50.0, 51.0, 50.0, 49.0]
    s.entry_tick = 1
    s.tick = 5
    # Currently long A (10), short B (-10); spread back near mean -> exit.
    state = _build_state(100, 50, positions={"PEBBLES_M": 10, "PEBBLES_XL": -10})
    orders, _ = s.run(state)
    a_orders = [o for o in orders if o.symbol == "PEBBLES_M"]
    b_orders = [o for o in orders if o.symbol == "PEBBLES_XL"]
    assert len(a_orders) == 1 and a_orders[0].quantity == -10, "should sell A to flatten"
    assert len(b_orders) == 1 and b_orders[0].quantity == 10, "should buy B to flatten"


def test_time_stop_exits_held_pair():
    """Held past max_hold_ticks -> flatten regardless of z."""
    s = R5PairTradeStrategy(
        symbol_a="PEBBLES_M", symbol_b="PEBBLES_XL", limit=10,
        window=5, z_entry=2.0, z_exit=0.5, max_hold_ticks=2,
    )
    s.spread_history = [49.0, 50.0, 51.0, 50.0, 49.0]
    s.entry_tick = 0
    s.tick = 3  # tick - entry_tick = 3 > max_hold_ticks=2
    state = _build_state(100, 50, positions={"PEBBLES_M": 10, "PEBBLES_XL": -10})
    orders, _ = s.run(state)
    a_orders = [o for o in orders if o.symbol == "PEBBLES_M"]
    b_orders = [o for o in orders if o.symbol == "PEBBLES_XL"]
    assert any(o.quantity == -10 for o in a_orders), "should flatten A long"
    assert any(o.quantity == 10 for o in b_orders), "should flatten B short"


def test_pair_save_load_round_trip():
    s = _make_pair()
    s.spread_history = [10.0, 20.0, 30.0]
    s.tick = 7
    s.entry_tick = 3
    data = s.save()
    s2 = _make_pair()
    s2.load(data)
    assert s2.spread_history == [10.0, 20.0, 30.0]
    assert s2.tick == 7
    assert s2.entry_tick == 3
