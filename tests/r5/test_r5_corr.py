import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r5"))

import pytest
from datamodel import Order, OrderDepth, Observation, TradingState
from strategy import R5CorrMMStrategy


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
        traderData="", timestamp=0, listings={},
        order_depths=order_depths,
        own_trades={}, market_trades={},
        position=positions or {},
        observations=Observation({}, {}),
    )


def test_first_tick_no_lean():
    """First tick: last_partner_mid is None → returns base microprice."""
    s = R5CorrMMStrategy("A", 10, width=3, partner_symbol="B", beta=0.46)
    state = make_state({"A": (100, 5, 110, 5), "B": (200, 5, 210, 5)})
    tv = s.get_true_value(state)
    assert tv == 105.0


def test_positive_beta_partner_up_leans_down():
    """Partner goes up, beta=+0.46 (fade) → our fair value goes down."""
    s = R5CorrMMStrategy("A", 10, width=3, partner_symbol="B", beta=0.46)
    # Tick 1
    state1 = make_state({"A": (100, 5, 110, 5), "B": (200, 5, 210, 5)})
    s.get_true_value(state1)  # partner_mid = 205

    # Tick 2: partner went up
    state2 = make_state({"A": (100, 5, 110, 5), "B": (210, 5, 220, 5)})
    tv = s.get_true_value(state2)
    base_microprice = 105.0
    partner_return = (215 - 205) / 205
    expected = base_microprice - 0.46 * partner_return * base_microprice
    assert abs(tv - expected) < 0.01
    assert tv < base_microprice  # leaned down (fade)


def test_negative_beta_partner_up_leans_up():
    """Partner goes up, beta=-0.46 (follow) → our fair value goes up."""
    s = R5CorrMMStrategy("A", 10, width=3, partner_symbol="B", beta=-0.46)
    # Tick 1
    state1 = make_state({"A": (100, 5, 110, 5), "B": (200, 5, 210, 5)})
    s.get_true_value(state1)

    # Tick 2: partner went up
    state2 = make_state({"A": (100, 5, 110, 5), "B": (210, 5, 220, 5)})
    tv = s.get_true_value(state2)
    assert tv > 105.0  # leaned up (follow)


def test_partner_missing_returns_base():
    """If partner symbol not in order_depths → returns base microprice."""
    s = R5CorrMMStrategy("A", 10, width=3, partner_symbol="B", beta=0.46)
    state = make_state({"A": (100, 5, 110, 5)})  # B missing
    tv = s.get_true_value(state)
    assert tv == 105.0


def test_save_load_roundtrip():
    s = R5CorrMMStrategy("A", 10, width=3, partner_symbol="B", beta=0.46)
    state = make_state({"A": (100, 5, 110, 5), "B": (200, 5, 210, 5)})
    s.get_true_value(state)
    assert s.last_partner_mid == 205.0

    data = s.save()
    assert data == {"last_partner_mid": 205.0}

    s2 = R5CorrMMStrategy("A", 10, width=3, partner_symbol="B", beta=0.46)
    s2.load(data)
    assert s2.last_partner_mid == 205.0
