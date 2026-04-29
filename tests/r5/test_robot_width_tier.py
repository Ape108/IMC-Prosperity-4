import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r5" / "groups"))

from datamodel import OrderDepth, Observation, TradingState
from robot import R5BaseMMStrategy, SYMBOLS


def make_state(products, positions=None):
    order_depths = {}
    for sym, (bid, bid_vol, ask, ask_vol) in products.items():
        od = OrderDepth()
        od.buy_orders = {bid: bid_vol}
        od.sell_orders = {ask: -ask_vol}
        order_depths[sym] = od
    return TradingState(
        traderData="", timestamp=0, listings={}, order_depths=order_depths,
        own_trades={}, market_trades={}, position=positions or {},
        observations=Observation({}, {}),
    )


def test_width_2_passive_quotes_are_2_ticks_inside_touch():
    """
    width=2 base MM with no inventory: passive bid is 2 ticks below best ask, passive ask
    is 2 ticks above best bid. (At width=1, passive sits 1 tick from each side.)
    """
    s = R5BaseMMStrategy("ROBOT_MOPPING", limit=10, width=2)
    state = make_state({"ROBOT_MOPPING": (95, 5, 105, 5)})  # microprice = 100
    orders, _ = s.run(state)
    # max_buy_price = floor(100) - 1 if integer = 99 (microprice is integer)
    # passive_buy = 99 - 2 + 1 = 98
    # min_sell_price = 101
    # passive_sell = 101 + 2 - 1 = 102
    buy_prices = sorted(o.price for o in orders if o.quantity > 0)
    sell_prices = sorted(o.price for o in orders if o.quantity < 0)
    assert 98 in buy_prices
    assert 102 in sell_prices


def test_width_1_vs_width_2_quotes_differ():
    """Regression: width=1 quotes 1 tick from anchor; width=2 quotes 2 ticks from anchor."""
    state = make_state({"ROBOT_MOPPING": (95, 5, 105, 5)})

    s1 = R5BaseMMStrategy("ROBOT_MOPPING", limit=10, width=1)
    o1, _ = s1.run(state)
    passive_buy_1 = max(o.price for o in o1 if o.quantity > 0)
    passive_sell_1 = min(o.price for o in o1 if o.quantity < 0)

    s2 = R5BaseMMStrategy("ROBOT_MOPPING", limit=10, width=2)
    o2, _ = s2.run(state)
    passive_buy_2 = max(o.price for o in o2 if o.quantity > 0)
    passive_sell_2 = min(o.price for o in o2 if o.quantity < 0)

    # Width 2 sits one tick further out than width 1 on each side
    assert passive_buy_1 - passive_buy_2 == 1
    assert passive_sell_2 - passive_sell_1 == 1


def test_width_tier_2_covers_all_five_robot_symbols():
    """SYMBOLS list contains exactly the five ROBOT products."""
    assert sorted(SYMBOLS) == sorted([
        "ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES",
        "ROBOT_LAUNDRY", "ROBOT_IRONING",
    ])
    assert len(SYMBOLS) == 5
