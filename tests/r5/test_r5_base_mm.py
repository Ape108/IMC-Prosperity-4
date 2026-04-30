import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r5"))

import pytest
from datamodel import Order, OrderDepth, Observation, TradingState
from strategy import R5BaseMMStrategy


def make_state(
    products: dict[str, tuple[int, int, int, int]],
    positions: dict[str, int] | None = None,
) -> TradingState:
    """Build a TradingState. products maps symbol -> (bid, bid_vol, ask, ask_vol)."""
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


# ── microprice ───────────────────────────────────────────────────────────────


def test_microprice_symmetric_volumes():
    """Equal volumes → microprice = simple mid."""
    s = R5BaseMMStrategy("TEST", 10, width=1)
    state = make_state({"TEST": (100, 5, 110, 5)})
    assert s.get_true_value(state) == 105.0


def test_microprice_asymmetric_volumes():
    """bid=100 vol=10, ask=110 vol=30 → microprice = (100*30 + 110*10)/(30+10) = 102.5."""
    s = R5BaseMMStrategy("TEST", 10, width=1)
    state = make_state({"TEST": (100, 10, 110, 30)})
    assert s.get_true_value(state) == 102.5


# ── act() with width ────────────────────────────────────────────────────────


def test_act_takes_sell_order_below_fair():
    """Sell order at 99, fair=99.375 (best_ask=99 shifts microprice down) → should buy at 99."""
    s = R5BaseMMStrategy("TEST", 10, width=1)
    od = OrderDepth()
    od.buy_orders = {100: 5}
    od.sell_orders = {99: -3, 110: -5}
    state = TradingState(
        traderData="", timestamp=0, listings={},
        order_depths={"TEST": od},
        own_trades={}, market_trades={}, position={},
        observations=Observation({}, {}),
    )
    orders, _ = s.run(state)
    buys = [o for o in orders if o.quantity > 0]
    assert any(o.price == 99 and o.quantity == 3 for o in buys)


def test_act_passive_buy_at_width_1_integer_fair():
    """fair=105.0 (integer) → max_buy=104, passive_buy = 104 - 1 + 1 = 104."""
    s = R5BaseMMStrategy("TEST", 10, width=1)
    state = make_state({"TEST": (100, 5, 110, 5)})
    orders, _ = s.run(state)
    buys = [o for o in orders if o.quantity > 0]
    passive_buys = [o for o in buys if o.price != 99]
    assert len(passive_buys) >= 1
    assert passive_buys[-1].price == 104


def test_act_passive_sell_at_width_1_integer_fair():
    """fair=105.0 (integer) → min_sell=106, passive_sell = 106 + 1 - 1 = 106."""
    s = R5BaseMMStrategy("TEST", 10, width=1)
    state = make_state({"TEST": (100, 5, 110, 5)})
    orders, _ = s.run(state)
    sells = [o for o in orders if o.quantity < 0]
    passive_sells = [o for o in sells if o.price != 111]
    assert len(passive_sells) >= 1
    assert passive_sells[-1].price == 106


def test_act_width_3_passive_spread_is_6_integer_fair():
    """fair=105.0 → passive buy=102, passive sell=108 → spread=6."""
    s = R5BaseMMStrategy("TEST", 10, width=3)
    state = make_state({"TEST": (100, 5, 110, 5)})
    orders, _ = s.run(state)
    buys = [o for o in orders if o.quantity > 0]
    sells = [o for o in orders if o.quantity < 0]
    passive_buy = min(o.price for o in buys)
    passive_sell = max(o.price for o in sells)
    assert passive_sell - passive_buy == 6


def test_act_width_2_fractional_fair_spread_is_3():
    """Fractional microprice → passive spread = 2*width - 1 = 3."""
    s = R5BaseMMStrategy("TEST", 10, width=2)
    state = make_state({"TEST": (100, 10, 110, 30)})  # microprice=102.5
    orders, _ = s.run(state)
    buys = [o for o in orders if o.quantity > 0]
    sells = [o for o in orders if o.quantity < 0]
    passive_buy = min(o.price for o in buys)
    passive_sell = max(o.price for o in sells)
    assert passive_sell - passive_buy == 3


def test_act_position_at_limit_no_buys():
    """At position limit (10), to_buy=0 → no buy orders."""
    s = R5BaseMMStrategy("TEST", 10, width=1)
    state = make_state({"TEST": (100, 5, 110, 5)}, positions={"TEST": 10})
    orders, _ = s.run(state)
    buys = [o for o in orders if o.quantity > 0]
    assert len(buys) == 0


def test_act_position_at_neg_limit_no_sells():
    """At position -10, to_sell=0 → no sell orders."""
    s = R5BaseMMStrategy("TEST", 10, width=1)
    state = make_state({"TEST": (100, 5, 110, 5)}, positions={"TEST": -10})
    orders, _ = s.run(state)
    sells = [o for o in orders if o.quantity < 0]
    assert len(sells) == 0
