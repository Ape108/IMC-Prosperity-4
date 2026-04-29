import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r5"))

import pytest
from datamodel import Order, OrderDepth, Observation, TradingState
from strategy import R5AutocorrMMStrategy


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
    """First tick: last_mid is None → get_true_value returns base microprice."""
    s = R5AutocorrMMStrategy("TEST", 10, width=2, alpha=0.12)
    state = make_state({"TEST": (100, 5, 110, 5)})
    tv = s.get_true_value(state)
    assert tv == 105.0  # pure microprice, no lean


def test_positive_return_leans_down():
    """Price went up → last_return > 0 → base -= alpha * last_return * base → lean DOWN (fade)."""
    s = R5AutocorrMMStrategy("TEST", 10, width=2, alpha=0.1)
    # Tick 1: mid = 105
    state1 = make_state({"TEST": (100, 5, 110, 5)})
    s.get_true_value(state1)

    # Tick 2: mid = 115 (price went up); symmetric vols so microprice == mid
    state2 = make_state({"TEST": (110, 5, 120, 5)})
    tv = s.get_true_value(state2)
    base_microprice = 115.0
    last_return = (115 - 105) / 105
    expected = base_microprice - 0.1 * last_return * base_microprice
    assert abs(tv - expected) < 0.01
    assert tv < base_microprice  # leaned down


def test_negative_return_leans_up():
    """Price went down → last_return < 0 → base -= alpha * last_return * base → lean UP (fade)."""
    s = R5AutocorrMMStrategy("TEST", 10, width=2, alpha=0.1)
    # Tick 1: mid = 115
    state1 = make_state({"TEST": (110, 5, 120, 5)})
    s.get_true_value(state1)

    # Tick 2: mid = 105 (price went down)
    state2 = make_state({"TEST": (100, 5, 110, 5)})
    tv = s.get_true_value(state2)
    base_microprice = 105.0
    assert tv > base_microprice  # leaned up


def test_save_load_roundtrip():
    s = R5AutocorrMMStrategy("TEST", 10, width=2, alpha=0.12)
    state = make_state({"TEST": (100, 5, 110, 5)})
    s.get_true_value(state)
    assert s.last_mid == 105.0

    data = s.save()
    assert data == {"last_mid": 105.0}

    s2 = R5AutocorrMMStrategy("TEST", 10, width=2, alpha=0.12)
    s2.load(data)
    assert s2.last_mid == 105.0


def test_save_before_any_tick():
    """Before any tick, last_mid is None → save returns {"last_mid": None}."""
    s = R5AutocorrMMStrategy("TEST", 10, width=2, alpha=0.12)
    assert s.save() == {"last_mid": None}
