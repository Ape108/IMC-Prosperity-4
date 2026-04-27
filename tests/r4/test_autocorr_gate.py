import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'submissions', 'r4'))

import pytest
import numpy as np
from strategy_h import autocorr_1lag, VoucherStrategy
from datamodel import TradingState, OrderDepth, Listing, Observation


# ── autocorr_1lag unit tests ──────────────────────────────────────────────────

def test_autocorr_1lag_empty_series_returns_zero():
    assert autocorr_1lag([]) == 0.0


def test_autocorr_1lag_single_element_returns_zero():
    assert autocorr_1lag([0.5]) == 0.0


def test_autocorr_1lag_constant_series_returns_zero():
    # Zero variance in both slices → NaN → safe fallback 0.0
    assert autocorr_1lag([0.05] * 10) == 0.0


def test_autocorr_1lag_alternating_series_is_negative():
    # Perfectly alternating = strong negative autocorrelation
    series = [-0.05, 0.05] * 15  # 30 elements
    r = autocorr_1lag(series)
    assert r < -0.9, f"Expected strong negative autocorr, got {r}"


def test_autocorr_1lag_trending_series_is_positive():
    # Monotonically increasing = strong positive autocorrelation
    series = [float(i) * 0.01 for i in range(30)]
    r = autocorr_1lag(series)
    assert r > 0.9, f"Expected strong positive autocorr, got {r}"


def test_autocorr_1lag_two_element_series():
    # Minimum valid series: [a, b] — should return a valid float, not crash
    r = autocorr_1lag([0.01, 0.02])
    assert isinstance(r, float)


# ── VoucherStrategy constructor parameter tests ───────────────────────────────

def make_strategy(**kwargs) -> VoucherStrategy:
    defaults = dict(
        symbol="VEV_5000", limit=300, strike=5000,
        k=150, min_residual=0.01, max_otm_moneyness=0.996,
        carry_window=100, carry_threshold=0.020,
        autocorr_window=30, autocorr_threshold=-0.05,
    )
    defaults.update(kwargs)
    return VoucherStrategy(**defaults)


def test_autocorr_window_stored_on_strategy():
    s = make_strategy(autocorr_window=20)
    assert s.autocorr_window == 20


def test_autocorr_threshold_stored_on_strategy():
    s = make_strategy(autocorr_threshold=-0.10)
    assert s.autocorr_threshold == -0.10


def test_autocorr_window_default_is_30():
    s = VoucherStrategy(symbol="VEV_5000", limit=300, strike=5000)
    assert s.autocorr_window == 30


def test_autocorr_threshold_default_is_minus_005():
    s = VoucherStrategy(symbol="VEV_5000", limit=300, strike=5000)
    assert s.autocorr_threshold == -0.05


# ── act() integration helpers ─────────────────────────────────────────────────

def _make_order_depth(bid: int, ask: int) -> OrderDepth:
    od = OrderDepth()
    od.buy_orders[bid] = 10
    od.sell_orders[ask] = -10
    return od


def _make_state(timestamp: int = 500_000) -> TradingState:
    spot_sym = "VELVETFRUIT_EXTRACT"
    voucher_prices = {
        "VEV_4000": (1246, 1248),
        "VEV_4500": (746, 748),
        "VEV_5000": (200, 210),
        "VEV_5100": (167, 169),
        "VEV_5200": (97, 99),
        "VEV_5300": (48, 50),
        "VEV_5400": (18, 20),
        "VEV_5500": (7, 9),
        "VEV_6000": (0, 1),
        "VEV_6500": (0, 1),
    }
    order_depths = {spot_sym: _make_order_depth(4998, 5002)}
    listings = {spot_sym: Listing(spot_sym, spot_sym, "SEASHELLS")}
    for sym, (bid, ask) in voucher_prices.items():
        order_depths[sym] = _make_order_depth(bid, ask)
        listings[sym] = Listing(sym, sym, "SEASHELLS")
    return TradingState(
        traderData="",
        timestamp=timestamp,
        listings=listings,
        order_depths=order_depths,
        own_trades={},
        market_trades={},
        position={},
        observations=Observation(plainValueObservations={}, conversionObservations={}),
    )


# ── warmup guard ──────────────────────────────────────────────────────────────

def test_warmup_guard_suppresses_orders_before_window_full():
    """No orders when residual_history has fewer than autocorr_window entries after append."""
    # Use max_otm_moneyness=1.020 so the OTM gate does not fire before the warmup guard.
    # With spot=5000 and strike=5000: moneyness = 1.0, which is <= 1.020.
    s = make_strategy(autocorr_window=30, max_otm_moneyness=1.020, min_residual=0.0)
    s.residual_history = []  # cold start — after one tick it will have 1 entry < 30
    state = _make_state()

    strategy_orders, _ = s.run(state)

    assert strategy_orders == [], (
        f"Expected no orders during warmup, got: {strategy_orders}"
    )


# ── autocorr gate ─────────────────────────────────────────────────────────────

def test_gate_suppresses_orders_when_trending():
    """
    With residual_history full of constant values (autocorr = 0.0 >= -0.05),
    the gate fires and no orders are emitted even if scalper target is non-zero.
    """
    # Use max_otm_moneyness=1.020 so the OTM gate does not fire before the autocorr gate.
    s = make_strategy(autocorr_window=5, autocorr_threshold=-0.05, max_otm_moneyness=1.020, min_residual=0.0)
    s.residual_history = [0.05] * 10
    state = _make_state()

    strategy_orders, _ = s.run(state)

    assert strategy_orders == [], (
        f"Expected no orders when autocorr gate fires (trending history), got: {strategy_orders}"
    )


# ── Trader instantiation ──────────────────────────────────────────────────────

def test_all_vouchers_use_autocorr_params():
    from strategy_h import Trader, STRIKES, VoucherStrategy
    trader = Trader()
    # VEV_5300/5400/5500 are now Mark14FollowerStrategy, not VoucherStrategy — skip them
    for strike in STRIKES:
        sym = f"VEV_{strike}"
        s = trader.strategies[sym]
        if not isinstance(s, VoucherStrategy):
            continue
        assert s.autocorr_window == 30, f"{sym}: expected autocorr_window=30, got {s.autocorr_window}"
        assert s.autocorr_threshold == -0.05, f"{sym}: expected autocorr_threshold=-0.05, got {s.autocorr_threshold}"
