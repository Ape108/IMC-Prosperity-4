"""Unit tests for eda_vef_iv helpers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r4"))

import pandas as pd
import pytest
from eda_vef_iv import (
    _angle_b_verdict,
    _angle_a_verdict,
    _angle_c_verdict,
    build_iv_spike_events,
    compute_smile_skew,
    compute_iv_for_strike,
    find_atm_strike,
    fit_smile_from_mids,
    get_session_open_iv,
)


def test_find_atm_strike_nearest_below() -> None:
    assert find_atm_strike(5247.0) == 5200


def test_find_atm_strike_nearest_above() -> None:
    assert find_atm_strike(5280.0) == 5300


def test_find_atm_strike_exact_match() -> None:
    assert find_atm_strike(5000.0) == 5000


def test_compute_iv_for_strike_below_intrinsic_returns_none() -> None:
    # spot=5000, strike=4000: intrinsic=1000, mid=150 << intrinsic -> None
    result = compute_iv_for_strike(5000.0, 4000, 4.0 / 365, 100.0, 200.0)
    assert result is None


def test_compute_iv_for_strike_mid_at_boundary_returns_none() -> None:
    # spot=5200, strike=5200: intrinsic=0, mid=0.5 -> mid <= 0+0.5 -> None
    result = compute_iv_for_strike(5200.0, 5200, 4.0 / 365, 0.0, 1.0)
    assert result is None


def test_compute_iv_for_strike_valid_atm_returns_positive_float() -> None:
    # ATM call: spot=5200, strike=5200, T=4/365, bid=40, ask=46 -> mid=43
    result = compute_iv_for_strike(5200.0, 5200, 4.0 / 365, 40.0, 46.0)
    assert result is not None
    assert 0.0 < result < 5.0


def test_get_session_open_iv_empty_df_returns_none() -> None:
    df = pd.DataFrame(columns=["product", "timestamp", "bid_price_1", "ask_price_1"])
    assert get_session_open_iv(df) is None


def test_get_session_open_iv_no_vev_returns_none() -> None:
    rows = [
        {
            "product": "VELVETFRUIT_EXTRACT",
            "timestamp": 0,
            "bid_price_1": 5198.0,
            "ask_price_1": 5202.0,
        }
    ]
    assert get_session_open_iv(pd.DataFrame(rows)) is None


def test_get_session_open_iv_valid_returns_correct_keys() -> None:
    rows = [
        {
            "product": "VELVETFRUIT_EXTRACT",
            "timestamp": 0,
            "bid_price_1": 5198.0,
            "ask_price_1": 5202.0,
        },
        {
            "product": "VEV_5200",
            "timestamp": 0,
            "bid_price_1": 40.0,
            "ask_price_1": 46.0,
        },
    ]
    result = get_session_open_iv(pd.DataFrame(rows))
    assert result is not None
    assert result["atm_strike"] == 5200
    assert result["spot"] == pytest.approx(5200.0)
    assert 0.0 < result["iv"] < 5.0


def test_angle_a_verdict_confirmed() -> None:
    rows = [
        {"day": 1, "atm_iv_open": 0.185},
        {"day": 2, "atm_iv_open": 0.195},
        {"day": 3, "atm_iv_open": 0.240},
    ]
    # mean(0.185, 0.195) = 0.190; 0.240 > 0.190 + 0.02 -> CONFIRMED
    assert _angle_a_verdict(rows) == "SIGNAL CONFIRMED"


def test_angle_a_verdict_no_signal() -> None:
    rows = [
        {"day": 1, "atm_iv_open": 0.185},
        {"day": 2, "atm_iv_open": 0.195},
        {"day": 3, "atm_iv_open": 0.205},
    ]
    # mean = 0.190; 0.205 < 0.190 + 0.02 -> NO SIGNAL
    assert _angle_a_verdict(rows) == "NO SIGNAL"


def test_fit_smile_from_mids_returns_none_for_fewer_than_3() -> None:
    assert fit_smile_from_mids(5200.0, 4.0 / 365, {5200: 43.0, 5300: 15.0}) is None


def test_fit_smile_from_mids_returns_3_coeffs_for_valid_data() -> None:
    mids = {5100: 110.0, 5200: 43.0, 5300: 15.0}
    result = fit_smile_from_mids(5200.0, 4.0 / 365, mids)
    assert result is not None
    assert len(result) == 3


def test_compute_smile_skew_returns_none_for_insufficient_strikes() -> None:
    assert compute_smile_skew(5200.0, 4.0 / 365, {5200: 43.0}) is None


def test_compute_smile_skew_returns_float_for_valid_data() -> None:
    mids = {5000: 200.0, 5200: 43.0, 5400: 22.0}
    result = compute_smile_skew(5200.0, 4.0 / 365, mids)
    assert result is not None
    assert isinstance(result, float)


def test_angle_b_verdict_static() -> None:
    rows = [
        {"day": 1, "mean_skew": 0.017},
        {"day": 2, "mean_skew": 0.018},
        {"day": 3, "mean_skew": 0.021},
    ]
    # |0.021 - 0.0175| = 0.0035 < 0.005 -> STATIC
    assert _angle_b_verdict(rows) == "STATIC"


def test_angle_b_verdict_moving() -> None:
    rows = [
        {"day": 1, "mean_skew": 0.010},
        {"day": 2, "mean_skew": 0.012},
        {"day": 3, "mean_skew": 0.025},
    ]
    # |0.025 - 0.011| = 0.014 > 0.005 -> MOVING
    assert _angle_b_verdict(rows) == "MOVING"


def test_build_iv_spike_events_empty_when_no_spikes() -> None:
    residuals = pd.Series([0.10, 0.10, 0.10, 0.10, 0.10])
    vef_mid = pd.Series([5200.0] * 5)
    result = build_iv_spike_events(residuals, vef_mid, thresholds=(0.015,), horizons=(2,))
    assert result.empty


def test_build_iv_spike_events_detects_spike() -> None:
    # delta at index 2 = +0.020 > threshold 0.015
    residuals = pd.Series([0.10, 0.10, 0.12, 0.12, 0.12, 0.12])
    vef_mid = pd.Series([5200.0, 5200.0, 5200.0, 5203.0, 5207.0, 5210.0])
    result = build_iv_spike_events(residuals, vef_mid, thresholds=(0.015,), horizons=(2,))
    assert not result.empty
    # positive spike (dr=+0.02) -> signed_move = (vef_mid[4] - vef_mid[2]) * +1 = +7.0
    assert float(result["signed_move"].iloc[0]) == pytest.approx(7.0)


def test_build_iv_spike_events_no_event_beyond_bounds() -> None:
    residuals = pd.Series([0.10, 0.10, 0.12])
    vef_mid = pd.Series([5200.0] * 3)
    result = build_iv_spike_events(residuals, vef_mid, thresholds=(0.015,), horizons=(5,))
    assert result.empty


def test_angle_c_verdict_edge() -> None:
    summary = pd.DataFrame(
        [
            {
                "threshold": 0.010,
                "horizon_N": 10,
                "n": 8,
                "mean_move": 3.0,
                "hit_rate_pct": 55.0,
            }
        ]
    )
    assert _angle_c_verdict(summary) == "EDGE"


def test_angle_c_verdict_no_edge_low_n() -> None:
    summary = pd.DataFrame(
        [
            {
                "threshold": 0.010,
                "horizon_N": 10,
                "n": 3,
                "mean_move": 3.0,
                "hit_rate_pct": 55.0,
            }
        ]
    )
    assert _angle_c_verdict(summary) == "NO EDGE"
