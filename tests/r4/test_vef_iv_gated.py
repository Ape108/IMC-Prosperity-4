"""Unit + integration tests for VelvetfruitIVGatedStrategy."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r4"))

import pytest
from datamodel import OrderDepth, TradingState
from strategy_h import VelvetfruitIVGatedStrategy, VelvetfruitStrategy


def make_state(
    vef_bid: int = 5198,
    vef_ask: int = 5202,
    vev_bid: float | None = 40.0,
    vev_ask: float | None = 46.0,
    atm_strike: int = 5200,
    position: int = 0,
    ts: int = 0,
) -> TradingState:
    vef_od = OrderDepth()
    vef_od.buy_orders = {vef_bid: 10}
    vef_od.sell_orders = {vef_ask: -10}
    order_depths: dict = {"VELVETFRUIT_EXTRACT": vef_od}
    if vev_bid is not None and vev_ask is not None:
        vev_od = OrderDepth()
        vev_od.buy_orders = {int(vev_bid): 5}
        vev_od.sell_orders = {int(vev_ask): -5}
        order_depths[f"VEV_{atm_strike}"] = vev_od
    return TradingState(
        traderData="",
        timestamp=ts,
        listings={},
        order_depths=order_depths,
        own_trades={},
        market_trades={},
        position={"VELVETFRUIT_EXTRACT": position} if position else {},
        observations=None,  # type: ignore
    )


def make_strategy(**kwargs) -> VelvetfruitIVGatedStrategy:
    defaults = dict(symbol="VELVETFRUIT_EXTRACT", limit=200, iv_floor=0.20)
    defaults.update(kwargs)
    return VelvetfruitIVGatedStrategy(**defaults)


# -- _compute_atm_iv ----------------------------------------------------------

def test_compute_atm_iv_no_vev_book_returns_none() -> None:
    s = make_strategy()
    assert s._compute_atm_iv(make_state(vev_bid=None, vev_ask=None)) is None


def test_compute_atm_iv_mid_at_intrinsic_returns_none() -> None:
    # spot~=5200, VEV_5200 bid=0 ask=1 -> mid=0.5 <= intrinsic(0)+0.5 -> None
    s = make_strategy()
    state = make_state(vev_bid=0.0, vev_ask=1.0, atm_strike=5200)
    assert s._compute_atm_iv(state) is None


def test_compute_atm_iv_valid_returns_positive() -> None:
    s = make_strategy()
    iv = s._compute_atm_iv(make_state(vev_bid=40.0, vev_ask=46.0))
    assert iv is not None
    assert 0.0 < iv < 5.0


# -- _capped_limit ------------------------------------------------------------

def test_capped_limit_at_floor_returns_base_limit() -> None:
    s = make_strategy(limit=200, iv_floor=0.20, k=2.0, min_frac=0.5, iv_alpha=0.0)
    s._iv_ema = 0.20
    assert s._capped_limit(make_state(vev_bid=None, vev_ask=None)) == 200


def test_capped_limit_above_floor_reduces() -> None:
    # iv_ema=0.30, floor=0.20 -> excess=0.10 -> frac=1-2*0.10=0.80 -> 160
    s = make_strategy(limit=200, iv_floor=0.20, k=2.0, min_frac=0.5, iv_alpha=0.0)
    s._iv_ema = 0.30
    assert s._capped_limit(make_state(vev_bid=None, vev_ask=None)) == 160


def test_capped_limit_never_below_min_frac() -> None:
    # iv_ema=0.70, floor=0.20 -> raw frac=-0.0, clamped to min_frac=0.5 -> 100
    s = make_strategy(limit=200, iv_floor=0.20, k=2.0, min_frac=0.5, iv_alpha=0.0)
    s._iv_ema = 0.70
    assert s._capped_limit(make_state(vev_bid=None, vev_ask=None)) == 100


def test_capped_limit_below_floor_returns_full() -> None:
    # iv_ema=0.15 < floor=0.20 -> excess=0, frac=1.0 -> 200
    s = make_strategy(limit=200, iv_floor=0.20, k=2.0, min_frac=0.5, iv_alpha=0.0)
    s._iv_ema = 0.15
    assert s._capped_limit(make_state(vev_bid=None, vev_ask=None)) == 200


# -- EMA ---------------------------------------------------------------------

def test_ema_does_not_update_when_iv_none() -> None:
    s = make_strategy(iv_floor=0.20, iv_alpha=0.50)
    s._iv_ema = 0.25
    s._capped_limit(make_state(vev_bid=None, vev_ask=None))
    assert s._iv_ema == pytest.approx(0.25)


def test_ema_updates_when_iv_not_none(monkeypatch: pytest.MonkeyPatch) -> None:
    s = make_strategy(iv_floor=0.20, iv_alpha=1.0)
    s._iv_ema = 0.20
    monkeypatch.setattr(s, "_compute_atm_iv", lambda state: 0.30)
    s._capped_limit(make_state(vev_bid=None, vev_ask=None))
    assert s._iv_ema == pytest.approx(0.30)


# -- Day-boundary guard -------------------------------------------------------

def test_day_boundary_resets_ema_to_floor() -> None:
    s = make_strategy(iv_floor=0.20)
    s._iv_ema = 0.35
    s._last_ts = 999_000
    s._capped_limit(make_state(ts=0, vev_bid=None, vev_ask=None))
    assert s._iv_ema == pytest.approx(0.20)


# -- save / load --------------------------------------------------------------

def test_save_load_round_trip() -> None:
    s = make_strategy(iv_floor=0.20)
    s._iv_ema = 0.28
    s._last_ts = 42000
    data = json.loads(json.dumps(s.save()))
    s2 = make_strategy(iv_floor=0.20)
    s2.load(data)
    assert s2._iv_ema == pytest.approx(0.28)
    assert s2._last_ts == 42000


# -- Integration: act() -------------------------------------------------------

def test_act_at_floor_orders_identical_to_baseline() -> None:
    state = make_state(vev_bid=None, vev_ask=None, position=0)
    baseline = VelvetfruitStrategy("VELVETFRUIT_EXTRACT", 200)
    baseline_orders, _ = baseline.run(state)

    gated = make_strategy(limit=200, iv_floor=0.20, k=2.0, min_frac=0.5, iv_alpha=0.0)
    gated._iv_ema = 0.20
    gated_orders, _ = gated.run(state)

    assert sorted([(o.price, o.quantity) for o in baseline_orders]) == sorted(
        [(o.price, o.quantity) for o in gated_orders]
    )


def test_act_restores_limit_after_call() -> None:
    s = make_strategy(limit=200, iv_floor=0.20, k=2.0, min_frac=0.5, iv_alpha=0.0)
    s._iv_ema = 0.45
    s.run(make_state(vev_bid=None, vev_ask=None))
    assert s.limit == 200
