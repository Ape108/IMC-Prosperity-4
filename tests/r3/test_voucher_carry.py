import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'submissions', 'r3'))

import pytest
from strategy import VoucherStrategy
from datamodel import TradingState, OrderDepth, Listing, Observation


def make_strategy(**kwargs) -> VoucherStrategy:
    defaults = dict(symbol="VEV_5000", limit=300, strike=5000, carry_window=100, carry_threshold=0.020)
    defaults.update(kwargs)
    return VoucherStrategy(**defaults)


# ── _apply_carry unit tests ────────────────────────────────────────────────

def test_apply_carry_returns_scalper_when_window_not_full():
    s = make_strategy()
    s.residual_history = [0.05] * 50  # only 50 of 100 needed
    assert s._apply_carry(150) == 150
    assert s._apply_carry(-150) == -150


def test_apply_carry_clamps_buy_when_structural_short():
    s = make_strategy()
    s.residual_history = [0.05] * 100  # mean 0.05 > threshold 0.020
    assert s._apply_carry(150) == 0    # scalper says buy → clamped to 0
    assert s._apply_carry(-150) == -150  # scalper says sell → allowed


def test_apply_carry_clamps_sell_when_structural_long():
    s = make_strategy()
    s.residual_history = [-0.05] * 100  # mean -0.05 < -0.020
    assert s._apply_carry(-150) == 0   # scalper says sell → clamped to 0
    assert s._apply_carry(150) == 150  # scalper says buy → allowed


def test_apply_carry_passes_through_when_mean_within_threshold():
    s = make_strategy()
    s.residual_history = [0.01] * 100  # mean 0.01 < threshold 0.020
    assert s._apply_carry(150) == 150
    assert s._apply_carry(-150) == -150


def test_apply_carry_passes_through_at_exact_threshold():
    # Boundary: mean == threshold is NOT active (requires strictly greater)
    s = make_strategy()
    s.residual_history = [0.020] * 100
    assert s._apply_carry(150) == 150


# ── residual_history state tests ──────────────────────────────────────────

def test_residual_history_starts_empty():
    s = make_strategy()
    assert s.residual_history == []


def test_save_roundtrip_preserves_history():
    s = make_strategy()
    s.residual_history = [0.01, 0.02, -0.03]
    data = s.save()
    s2 = make_strategy()
    s2.load(data)
    assert s2.residual_history == [0.01, 0.02, -0.03]


def test_load_empty_dict_gives_empty_history():
    s = make_strategy()
    s.load({})
    assert s.residual_history == []


def test_save_returns_dict_with_residual_history_key():
    s = make_strategy()
    s.residual_history = [0.05]
    data = s.save()
    assert "residual_history" in data
    assert data["residual_history"] == [0.05]


# ── history overflow trim ─────────────────────────────────────────────────

def test_residual_history_capped_at_carry_window():
    s = make_strategy(carry_window=5)
    s.residual_history = [0.01, 0.02, 0.03, 0.04, 0.05]  # exactly at window
    # Simulate what act() does when appending one more entry
    residual = 0.06
    s.residual_history.append(residual)
    if len(s.residual_history) > s.carry_window:
        s.residual_history.pop(0)
    assert len(s.residual_history) == 5
    assert s.residual_history[0] == 0.02  # oldest was popped
    assert s.residual_history[-1] == 0.06  # newest is present


# ── act() integration: carry clamp suppresses buy orders ─────────────────

def _make_order_depth(bid: int, ask: int) -> OrderDepth:
    od = OrderDepth()
    od.buy_orders[bid] = 10
    od.sell_orders[ask] = -10
    return od


def test_carry_clamp_suppresses_buy_orders_in_act():
    """
    With residual_history full of large positive values (structural short signal),
    _apply_carry() must clamp any positive (buy) scalper target to 0.
    No buy orders should be emitted even if the instantaneous IV residual is negative.
    """
    # Build order depths for all required symbols
    spot_sym = "VELVETFRUIT_EXTRACT"
    voucher_prices = {
        "VEV_4000": (1246, 1248),
        "VEV_4500": (746, 748),
        "VEV_5000": (200, 210),   # low price relative to spot → appears underpriced → scalper says buy
        "VEV_5100": (167, 169),
        "VEV_5200": (97, 99),
        "VEV_5300": (48, 50),
        "VEV_5400": (18, 20),
        "VEV_5500": (7, 9),
        "VEV_6000": (0, 1),
        "VEV_6500": (0, 1),
    }

    order_depths: dict = {spot_sym: _make_order_depth(4998, 5002)}
    listings: dict = {
        spot_sym: Listing(spot_sym, spot_sym, "SEASHELLS"),
    }
    for sym, (bid, ask) in voucher_prices.items():
        order_depths[sym] = _make_order_depth(bid, ask)
        listings[sym] = Listing(sym, sym, "SEASHELLS")

    state = TradingState(
        traderData="",
        timestamp=500_000,   # mid-round, TTE ≈ 4.5 days
        listings=listings,
        order_depths=order_depths,
        own_trades={},
        market_trades={},
        position={},
        observations=Observation(plainValueObservations={}, conversionObservations={}),
    )

    s = make_strategy(symbol="VEV_5000", limit=300, strike=5000, carry_window=100, carry_threshold=0.020)
    s.residual_history = [0.05] * 100  # structural short signal active (mean 0.05 >> 0.020)

    strategy_orders, _ = s.run(state)

    buy_orders = [o for o in strategy_orders if o.quantity > 0]
    assert buy_orders == [], (
        f"Expected no buy orders under structural short carry signal, got: {buy_orders}"
    )
