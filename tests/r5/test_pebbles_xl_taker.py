import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r5" / "groups"))

from datamodel import Order, OrderDepth, Observation, TradingState
from pebbles import R5XLTakerStrategy


def make_state(
    products: dict[str, tuple[int, int, int, int]],
    positions: dict[str, int] | None = None,
    timestamp: int = 0,
) -> TradingState:
    order_depths: dict[str, OrderDepth] = {}
    for sym, (bid, bid_vol, ask, ask_vol) in products.items():
        od = OrderDepth()
        od.buy_orders = {bid: bid_vol}
        od.sell_orders = {ask: -ask_vol}
        order_depths[sym] = od
    return TradingState(
        traderData="",
        timestamp=timestamp,
        listings={},
        order_depths=order_depths,
        own_trades={},
        market_trades={},
        position=positions or {},
        observations=Observation({}, {}),
    )


def _make_strategy() -> R5XLTakerStrategy:
    return R5XLTakerStrategy(
        symbol="PEBBLES_M",
        limit=10,
        partner_symbol="PEBBLES_XL",
        entry_threshold=0.0015,
        exit_threshold=0.0005,
        max_hold_ticks=50,
        k_clip=5,
    )


def test_constructor():
    s = _make_strategy()
    assert s.symbol == "PEBBLES_M"
    assert s.partner_symbol == "PEBBLES_XL"
    assert s.entry_threshold == 0.0015
    assert s.exit_threshold == 0.0005
    assert s.max_hold_ticks == 50
    assert s.k_clip == 5
    assert s.last_xl_mid is None
    assert s.entry_tick is None
    assert s.tick == 0


def test_first_tick_no_orders():
    """No prior XL mid -> no signal -> no orders."""
    s = _make_strategy()
    state = make_state({
        "PEBBLES_M":  (100, 5, 110, 5),
        "PEBBLES_XL": (1000, 5, 1010, 5),
    })
    orders, _ = s.run(state)
    assert orders == []


def test_below_threshold_no_orders():
    """Partner return below entry threshold -> no orders."""
    s = _make_strategy()
    s.last_xl_mid = 1005.0
    # XL mid 1005.5, return ~ 0.0005 < 0.0015
    state = make_state({
        "PEBBLES_M":  (100, 5, 110, 5),
        "PEBBLES_XL": (1005, 5, 1006, 5),
    })
    orders, _ = s.run(state)
    assert orders == []


def test_xl_drop_above_threshold_buys_at_ask():
    """XL drop > entry threshold and flat -> cross ask on PEBBLES_M."""
    s = _make_strategy()
    s.last_xl_mid = 1010.0
    # XL mid now 1000, return ~ -0.01
    state = make_state({
        "PEBBLES_M":  (100, 5, 110, 5),
        "PEBBLES_XL": (999, 5, 1001, 5),
    })
    orders, _ = s.run(state)
    assert len(orders) == 1
    assert orders[0].symbol == "PEBBLES_M"
    assert orders[0].price == 110, "should cross ask"
    assert orders[0].quantity == 5, "should buy k_clip=5"


def test_xl_rise_above_threshold_sells_at_bid():
    """XL rise > entry threshold and flat -> cross bid on PEBBLES_M."""
    s = _make_strategy()
    s.last_xl_mid = 1000.0
    state = make_state({
        "PEBBLES_M":  (100, 5, 110, 5),
        "PEBBLES_XL": (1009, 5, 1011, 5),  # XL mid 1010, return +0.01
    })
    orders, _ = s.run(state)
    assert len(orders) == 1
    assert orders[0].symbol == "PEBBLES_M"
    assert orders[0].price == 100, "should cross bid"
    assert orders[0].quantity == -5, "should sell k_clip=5"


def test_signal_flip_exits_long():
    """Holding long, partner return > exit_threshold -> sell to flatten."""
    s = _make_strategy()
    s.last_xl_mid = 1000.0
    # XL mid 1002, return +0.002 > exit_threshold(0.0005), opposite of long position
    state = make_state(
        {"PEBBLES_M": (100, 5, 110, 5), "PEBBLES_XL": (1001, 5, 1003, 5)},
        positions={"PEBBLES_M": 5},
    )
    orders, _ = s.run(state)
    assert len(orders) == 1
    assert orders[0].quantity < 0, "should sell to exit long"
    assert orders[0].price == 100, "should hit bid to exit"


def test_time_stop_exits_held_position():
    """Held position past max_hold_ticks -> flatten regardless of signal."""
    s = _make_strategy()
    s.last_xl_mid = 1000.0
    s.entry_tick = 0
    s.tick = 51  # exceeds max_hold_ticks=50
    state = make_state(
        {"PEBBLES_M": (100, 5, 110, 5), "PEBBLES_XL": (1000, 5, 1001, 5)},
        positions={"PEBBLES_M": 3},
    )
    orders, _ = s.run(state)
    assert len(orders) == 1
    assert orders[0].quantity == -3, "should flatten 3 long"


def test_save_load_round_trip():
    s = _make_strategy()
    s.last_xl_mid = 1234.5
    s.entry_tick = 7
    s.tick = 12
    data = s.save()
    s2 = _make_strategy()
    s2.load(data)
    assert s2.last_xl_mid == 1234.5
    assert s2.entry_tick == 7
    assert s2.tick == 12
