import sys
from pathlib import Path

# conftest already adds these, but be explicit for IDE / standalone runs
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r5" / "groups"))

import pytest
from datamodel import Order, OrderDepth, Observation, TradingState
from snackpack import R5BasketCapMMStrategy, R5BaseMMStrategy


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


def test_constructor_and_required_symbols():
    """Class instantiates with leg_sign + partners, exposes both partners as required."""
    s = R5BasketCapMMStrategy(
        "CHOC", limit=10, width=3,
        leg_sign=+1, partners={"VAN": -1},
    )
    assert s.symbol == "CHOC"
    assert s.limit == 10
    assert s.width == 3
    assert s.leg_sign == +1
    assert s.partners == {"VAN": -1}
    assert s.get_required_symbols() == ["CHOC", "VAN"]


def _collect_orders_for_choc(positions):
    """Run R5BasketCapMMStrategy on a CHOC/VAN make_state, return orders."""
    s = R5BasketCapMMStrategy(
        "CHOC", limit=10, width=3,
        leg_sign=+1, partners={"VAN": -1},
    )
    state = make_state(
        {"CHOC": (100, 5, 110, 5), "VAN": (100, 5, 110, 5)},
        positions=positions,
    )
    orders, _ = s.run(state)
    return orders


def test_factor_system1_hedged():
    """CHOC=+5, VAN=+5 (hedged) -> factor=0, effective_C=0; to_buy=5 (per-leg cap), to_sell=10 (basket-aware)."""
    orders = _collect_orders_for_choc({"CHOC": 5, "VAN": 5})
    buy_total = sum(o.quantity for o in orders if o.quantity > 0)
    sell_total = sum(-o.quantity for o in orders if o.quantity < 0)
    assert buy_total == 5, f"expected to_buy=5, got {buy_total}"
    assert sell_total == 10, f"expected to_sell=10, got {sell_total}"


def test_factor_system1_directional():
    """CHOC=+5, VAN=-5 (directional) -> factor=+10, effective_C=+5; to_buy=5, to_sell=15 (matches base MM)."""
    orders = _collect_orders_for_choc({"CHOC": 5, "VAN": -5})
    buy_total = sum(o.quantity for o in orders if o.quantity > 0)
    sell_total = sum(-o.quantity for o in orders if o.quantity < 0)
    assert buy_total == 5
    assert sell_total == 15


def test_factor_system1_max_hedged():
    """CHOC=+10, VAN=+10 -> factor=0, effective_C=0; to_buy=0 (per-leg cap), to_sell=10 (basket-aware caps)."""
    orders = _collect_orders_for_choc({"CHOC": 10, "VAN": 10})
    buy_total = sum(o.quantity for o in orders if o.quantity > 0)
    sell_total = sum(-o.quantity for o in orders if o.quantity < 0)
    assert buy_total == 0
    assert sell_total == 10


def _collect_orders_for_rasp(positions):
    """Run R5BasketCapMMStrategy on RASP with full System-2 partners."""
    s = R5BasketCapMMStrategy(
        "RASP", limit=10, width=3,
        leg_sign=-1, partners={"STRAW": +1, "PIST": +1},
    )
    state = make_state(
        {
            "RASP":  (100, 5, 110, 5),
            "STRAW": (100, 5, 110, 5),
            "PIST":  (100, 5, 110, 5),
        },
        positions=positions,
    )
    orders, _ = s.run(state)
    return orders


def test_factor_system2_three_legs_partial():
    """
    RASP=+5, STRAW=+5, PIST=+5 -> factor_2 = -5+5+5 = +5; effective_R = -5/3 ~= -1.67.
    to_buy_R = floor(10 - (-1.67)) = 11, clamp to per-leg = 5.
    to_sell_R = floor(10 + (-1.67)) = 8, clamp to per-leg = 8.
    """
    orders = _collect_orders_for_rasp({"RASP": 5, "STRAW": 5, "PIST": 5})
    buy_total = sum(o.quantity for o in orders if o.quantity > 0)
    sell_total = sum(-o.quantity for o in orders if o.quantity < 0)
    assert buy_total == 5, f"expected to_buy=5 (per-leg cap), got {buy_total}"
    assert sell_total == 8, f"expected to_sell=8 (basket-aware), got {sell_total}"


def test_factor_system2_max_directional():
    """
    RASP=+10, STRAW=-10, PIST=-10 -> factor_2 = -10-10-10 = -30; effective_R = -(-30)/3 = +10.
    to_buy_R = 0, to_sell_R = 20 (matches base MM in max-directional state).
    """
    orders = _collect_orders_for_rasp({"RASP": 10, "STRAW": -10, "PIST": -10})
    buy_total = sum(o.quantity for o in orders if o.quantity > 0)
    sell_total = sum(-o.quantity for o in orders if o.quantity < 0)
    assert buy_total == 0
    assert sell_total == 20


def test_partner_book_missing_returns_no_orders():
    """When partner symbol's book is missing, run() gate skips act() and returns empty orders."""
    s = R5BasketCapMMStrategy(
        "CHOC", limit=10, width=3,
        leg_sign=+1, partners={"VAN": -1},
    )
    # Only CHOC has a book; VAN missing entirely
    state = make_state({"CHOC": (100, 5, 110, 5)}, positions={"CHOC": 0})
    orders, conversions = s.run(state)
    assert orders == []
    assert conversions == 0


def test_per_leg_cap_overrides_basket_capacity():
    """
    CHOC=+10, VAN=+10: B2-cap formula gives to_buy=10, but per-leg cap forces 0.
    Confirms the per-leg hard cap is the binding constraint when basket says otherwise.
    """
    orders = _collect_orders_for_choc({"CHOC": 10, "VAN": 10})
    buy_total = sum(o.quantity for o in orders if o.quantity > 0)
    assert buy_total == 0, f"expected to_buy=0 (per-leg cap), got {buy_total}"


def test_orders_match_base_mm_in_directional_state():
    """
    In a max-directional state where effective_pos == own_pos, B2-cap orders should
    match R5BaseMMStrategy orders exactly. Regression check on directional behavior.

    CHOC=+5, VAN=-5: factor=+10, effective_C = +5 = CHOC_pos.
    """
    state = make_state(
        {"CHOC": (100, 5, 110, 5), "VAN": (100, 5, 110, 5)},
        positions={"CHOC": 5, "VAN": -5},
    )

    basket = R5BasketCapMMStrategy(
        "CHOC", limit=10, width=3,
        leg_sign=+1, partners={"VAN": -1},
    )
    base = R5BaseMMStrategy("CHOC", limit=10, width=3)

    basket_orders, _ = basket.run(state)
    base_orders, _ = base.run(state)

    # Sort by (price, quantity) for comparison
    assert sorted((o.price, o.quantity) for o in basket_orders) == \
           sorted((o.price, o.quantity) for o in base_orders)
