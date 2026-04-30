import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r5" / "groups"))

from datamodel import Order, OrderDepth, Observation, TradingState
from pebbles import R5XLSkewMMStrategy


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


def _bid_ask(orders: list[Order]) -> tuple[set[int], set[int]]:
    """Return (set of bid prices, set of ask prices)."""
    return (
        {o.price for o in orders if o.quantity > 0},
        {o.price for o in orders if o.quantity < 0},
    )


def _make_strategy() -> R5XLSkewMMStrategy:
    return R5XLSkewMMStrategy(
        symbol="PEBBLES_M",
        limit=10,
        width=2,
        partner_symbol="PEBBLES_XL",
        threshold=0.001,
        k_ticks=2,
    )


def test_constructor():
    s = _make_strategy()
    assert s.symbol == "PEBBLES_M"
    assert s.partner_symbol == "PEBBLES_XL"
    assert s.threshold == 0.001
    assert s.k_ticks == 2
    assert s.width == 2
    assert s.last_xl_mid is None


def test_first_tick_no_skew_quotes_symmetric():
    """No prior XL mid -> standard symmetric quoting at width=2."""
    s = _make_strategy()
    state = make_state({
        "PEBBLES_M":  (100, 5, 110, 5),
        "PEBBLES_XL": (1000, 5, 1010, 5),
    })
    orders, _ = s.run(state)
    bids, asks = _bid_ask(orders)
    # microprice for M = 105, max_buy_price=104, passive_buy = 104 - (2-1) = 103
    # min_sell_price = 106, passive_sell = 106 + (2-1) = 107
    assert bids == {103}, f"expected bid=103, got {bids}"
    assert asks == {107}, f"expected ask=107, got {asks}"


def test_below_threshold_no_skew():
    """XL return below threshold -> still symmetric."""
    s = _make_strategy()
    s.last_xl_mid = 1005.0
    # XL mid moves to 1005.5, return = 0.0005 < 0.001
    state = make_state({
        "PEBBLES_M":  (100, 5, 110, 5),
        "PEBBLES_XL": (1005, 5, 1006, 5),
    })
    orders, _ = s.run(state)
    bids, asks = _bid_ask(orders)
    assert bids == {103}
    assert asks == {107}


def test_xl_drop_above_threshold_raises_bid_keeps_ask():
    """XL drop > threshold -> bid raised by k_ticks, ask unchanged."""
    s = _make_strategy()
    s.last_xl_mid = 1010.0
    # XL mid now 1000, return = -0.0099 << -0.001
    state = make_state({
        "PEBBLES_M":  (100, 5, 110, 5),
        "PEBBLES_XL": (999, 5, 1001, 5),
    })
    orders, _ = s.run(state)
    bids, asks = _bid_ask(orders)
    # bid: standard 103 + k_ticks(2) = 105
    # ask: unchanged at 107
    assert bids == {105}, f"expected bid=105 (raised by k_ticks), got {bids}"
    assert asks == {107}, f"expected ask=107 (unchanged), got {asks}"


def test_xl_rise_above_threshold_lowers_ask_keeps_bid():
    """XL rise > threshold -> ask lowered by k_ticks, bid unchanged."""
    s = _make_strategy()
    s.last_xl_mid = 1000.0
    # XL mid now 1010, return = +0.01 >> +0.001
    state = make_state({
        "PEBBLES_M":  (100, 5, 110, 5),
        "PEBBLES_XL": (1009, 5, 1011, 5),
    })
    orders, _ = s.run(state)
    bids, asks = _bid_ask(orders)
    # bid: unchanged at 103
    # ask: standard 107 - k_ticks(2) = 105
    assert bids == {103}, f"expected bid=103 (unchanged), got {bids}"
    assert asks == {105}, f"expected ask=105 (lowered by k_ticks), got {asks}"


def test_save_load_round_trip():
    s = _make_strategy()
    s.last_xl_mid = 1234.5
    data = s.save()
    s2 = _make_strategy()
    s2.load(data)
    assert s2.last_xl_mid == 1234.5


def test_load_handles_none_last_mid():
    s = _make_strategy()
    s.load({"last_xl_mid": None})
    assert s.last_xl_mid is None


def test_partner_hiccup_resets_last_mid_no_spurious_skew():
    """Partner book missing for one tick -> last_xl_mid resets; subsequent
    return computation does not span the gap. No spurious skew when partner
    reappears after a hiccup."""
    s = _make_strategy()
    s.last_xl_mid = 1010.0

    # Tick A: partner book is present in state.order_depths but ONE-SIDED
    # (sell_orders empty). act() should still run because own book is fine,
    # _partner_return resets last_xl_mid to None and returns None.
    od_m = OrderDepth()
    od_m.buy_orders = {100: 5}
    od_m.sell_orders = {110: -5}
    od_xl_one_sided = OrderDepth()
    od_xl_one_sided.buy_orders = {1000: 5}
    od_xl_one_sided.sell_orders = {}  # missing
    state_a = TradingState(
        traderData="", timestamp=0, listings={},
        order_depths={"PEBBLES_M": od_m, "PEBBLES_XL": od_xl_one_sided},
        own_trades={}, market_trades={}, position={},
        observations=Observation({}, {}),
    )
    orders_a, _ = s.run(state_a)
    bids_a, asks_a = _bid_ask(orders_a)
    # Symmetric quoting because partner_return is None
    assert bids_a == {103}, f"hiccup tick: expected symmetric bid=103, got {bids_a}"
    assert asks_a == {107}, f"hiccup tick: expected symmetric ask=107, got {asks_a}"
    assert s.last_xl_mid is None, "last_xl_mid should reset on hiccup"

    # Tick B: partner reappears with a much-changed mid. _partner_return
    # should return None again (last_xl_mid was reset, so this tick re-seeds it),
    # NOT compute a spurious return spanning the gap.
    state_b = make_state({
        "PEBBLES_M":  (100, 5, 110, 5),
        "PEBBLES_XL": (999, 5, 1001, 5),  # mid 1000 — would have been -1% if compared to 1010
    })
    orders_b, _ = s.run(state_b)
    bids_b, asks_b = _bid_ask(orders_b)
    # No spurious skew — should still be symmetric on the reseed tick
    assert bids_b == {103}, f"reseed tick: expected symmetric bid=103, got {bids_b}"
    assert asks_b == {107}, f"reseed tick: expected symmetric ask=107, got {asks_b}"
    assert s.last_xl_mid == 1000.0, "last_xl_mid should be reseeded to current"


def test_xl_signal_skew_variant_wires_correct_strategies():
    """xl_signal_skew variant: M/L/S/XS use R5XLSkewMMStrategy, XL uses R5BaseMMStrategy."""
    import pebbles
    from pebbles import R5BaseMMStrategy

    # Sanity check that the classes are independently wireable in the expected pattern.
    # Full integration testing of the variant function happens via the backtester.
    skew = R5XLSkewMMStrategy(
        symbol="PEBBLES_M", limit=10, width=2,
        partner_symbol="PEBBLES_XL", threshold=0.001, k_ticks=2,
    )
    base = R5BaseMMStrategy("PEBBLES_XL", limit=10, width=2)
    assert skew.partner_symbol == "PEBBLES_XL"
    assert base.symbol == "PEBBLES_XL"
