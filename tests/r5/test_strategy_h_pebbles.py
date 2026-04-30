import sys
from pathlib import Path

# conftest.py already adds submissions/r5/ but be explicit for IDE / standalone runs
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r5"))

from datamodel import Order, OrderDepth, Observation, TradingState
import strategy_h
from strategy_h import (
    Trader,
    R5BaseMMStrategy,
    R5CorrMMStrategy,
    R5PairTradeStrategy,
)


def _make_state(
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


# ── Wiring assertions ───────────────────────────────────────────────


def test_pebbles_s_is_base_mm_width_2():
    t = Trader()
    s = t.strategies["PEBBLES_S"]
    assert isinstance(s, R5BaseMMStrategy)
    assert s.width == 2
    # Must NOT be the corr or pair-trade subclass
    assert not isinstance(s, R5CorrMMStrategy)
    assert not isinstance(s, R5PairTradeStrategy)


def test_pebbles_l_not_registered():
    t = Trader()
    assert "PEBBLES_L" not in t.strategies


def test_pebbles_xs_not_registered():
    t = Trader()
    assert "PEBBLES_XS" not in t.strategies


def test_pebbles_xl_not_separately_registered():
    """PEBBLES_XL is handled inside the pair trade entry; no standalone strategy."""
    t = Trader()
    assert "PEBBLES_XL" not in t.strategies


def test_pebbles_m_key_holds_pair_trade():
    t = Trader()
    s = t.strategies["PEBBLES_M"]
    assert isinstance(s, R5PairTradeStrategy)
    assert s.symbol_a == "PEBBLES_M"
    assert s.symbol_b == "PEBBLES_XL"
    assert s.limit == 10
    assert s.window == 200
    assert s.z_entry == 2.0
    assert s.z_exit == 0.5
    assert s.max_hold_ticks == 500


def test_no_corr_overlay_anywhere():
    """The lag-1 partner-lean overlay must be fully removed from the trader."""
    t = Trader()
    for sym, s in t.strategies.items():
        assert not isinstance(s, R5CorrMMStrategy), \
            f"{sym} still uses R5CorrMMStrategy — should be removed"


# ── Multi-symbol dispatch ───────────────────────────────────────────


class _MultiSymbolStub(strategy_h.Strategy):
    """Stub strategy that emits orders on two distinct symbols."""

    def __init__(self, symbol: str, partner: str, limit: int) -> None:
        super().__init__(symbol, limit)
        self.partner = partner

    def get_required_symbols(self) -> list[str]:
        return [self.symbol, self.partner]

    def act(self, state: TradingState) -> None:
        self.orders.append(Order(self.symbol, 100, 1))
        self.orders.append(Order(self.partner, 200, -1))


def test_dispatch_routes_orders_by_order_symbol():
    """Trader.run must group orders by order.symbol, not by iteration key."""
    t = Trader()
    t.strategies = {"PEBBLES_M": _MultiSymbolStub("PEBBLES_M", "PEBBLES_XL", limit=10)}
    state = _make_state({
        "PEBBLES_M":  (100, 5, 110, 5),
        "PEBBLES_XL": (200, 5, 210, 5),
    })
    orders, _, _ = t.run(state)
    m_orders = orders.get("PEBBLES_M", [])
    xl_orders = orders.get("PEBBLES_XL", [])
    assert any(o.price == 100 and o.quantity == 1 for o in m_orders), \
        f"PEBBLES_M order missing or misrouted: {m_orders}"
    assert any(o.price == 200 and o.quantity == -1 for o in xl_orders), \
        f"PEBBLES_XL order missing or misrouted: {xl_orders}"


def test_held_position_keeps_entry_tick_after_exit_attempt():
    """The robot.py-style fix: entry_tick must reconcile against state.position
    each tick, not be eagerly cleared when exit orders are submitted. Otherwise
    a partial fill would strand the time-stop on the residual position."""
    pair = R5PairTradeStrategy(
        symbol_a="PEBBLES_M",
        symbol_b="PEBBLES_XL",
        limit=10,
        window=5,
        z_entry=2.0,
        z_exit=0.5,
        max_hold_ticks=500,
    )
    # Pre-seed state: held a position with prior entry_tick
    pair.spread_history = [49.0, 50.0, 51.0, 50.0, 49.0]
    pair.entry_tick = 100  # earlier timestamp

    # Tick 1: spread back near mean -> exit fires; pretend partial fill leaves residual
    state = _make_state(
        {"PEBBLES_M": (99, 5, 101, 5), "PEBBLES_XL": (49, 5, 51, 5)},
        positions={"PEBBLES_M": 10, "PEBBLES_XL": -10},
    )
    state.timestamp = 200
    pair.orders = []
    pair.act(state)
    # After exit attempt, entry_tick must still be set — fills not yet confirmed
    assert pair.entry_tick == 100, \
        f"entry_tick was eagerly cleared during exit attempt: {pair.entry_tick}"

    # Tick 2: positions confirmed flat -> reconcile clears entry_tick
    state2 = _make_state(
        {"PEBBLES_M": (99, 5, 101, 5), "PEBBLES_XL": (49, 5, 51, 5)},
        positions={"PEBBLES_M": 0, "PEBBLES_XL": 0},
    )
    state2.timestamp = 300
    pair.orders = []  # reset accumulator (run() does this automatically; act() does not)
    pair.act(state2)
    assert pair.entry_tick is None, \
        "entry_tick should be cleared once positions are confirmed flat"


def test_cold_start_with_carried_position_sets_entry_tick():
    """If the strategy starts with a non-zero position (e.g. --carry without
    --persist) and no saved entry_tick, reconcile should seed entry_tick from
    state.timestamp so the time-stop has a reference point."""
    pair = R5PairTradeStrategy(
        symbol_a="PEBBLES_M",
        symbol_b="PEBBLES_XL",
        limit=10,
        window=5,
        z_entry=2.0,
        z_exit=0.5,
        max_hold_ticks=500,
    )
    pair.spread_history = [49.0, 50.0, 51.0, 50.0, 49.0]
    pair.entry_tick = None  # cold start

    state = _make_state(
        {"PEBBLES_M": (99, 5, 101, 5), "PEBBLES_XL": (49, 5, 51, 5)},
        positions={"PEBBLES_M": 10, "PEBBLES_XL": -10},
    )
    state.timestamp = 5000
    pair.orders = []
    pair.act(state)
    assert pair.entry_tick == 5000, \
        f"cold-start entry_tick should be set to state.timestamp, got {pair.entry_tick}"
