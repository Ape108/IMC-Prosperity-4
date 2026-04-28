"""Unit tests for eda_vef_extreme_mr helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from eda_vef_extreme_mr import REVERSION_BREAKEVEN


def _make_mid(n_warmup: int = 50, spike: float = 0.0, tail: int = 25) -> pd.Series:
    """Alternating 99/101 warmup (mean=100, std≈1), optional spike, flat tail."""
    warmup = [99.0 if i % 2 == 0 else 101.0 for i in range(n_warmup)]
    return pd.Series(warmup + [100.0 + spike] + [100.0] * tail)


# ---------------------------------------------------------------------------
# build_events
# ---------------------------------------------------------------------------

def test_build_events_flat_series_produces_no_events():
    """Constant mid → rolling std = 0 → z-score NaN → no events."""
    from eda_vef_extreme_mr import build_events
    mid = pd.Series([100.0] * 100)
    events = build_events(mid, period=20, thresholds=(1.5,), horizons=(1,))
    assert events.empty


def test_build_events_warmup_ticks_excluded():
    """Ticks 0..period-1 must not produce events (no z-score yet)."""
    from eda_vef_extreme_mr import build_events
    # spike at tick 5 (within warmup=20) should not fire
    mid = pd.Series([100.0] * 5 + [200.0] + [100.0] * 74)
    events = build_events(mid, period=20, thresholds=(1.5,), horizons=(1,))
    assert events.empty


def test_build_events_spike_fires_for_correct_thresholds():
    """z ≈ 5 should fire for thresholds 1.5, 2.0, 2.5, 3.0 but a z ≈ 2.2
    should fire for 1.5, 2.0 only."""
    from eda_vef_extreme_mr import build_events
    # spike=5 → z ≈ 5 (std ≈ 1 from alternating 99/101)
    mid_big = _make_mid(n_warmup=50, spike=5.0, tail=5)
    events_big = build_events(mid_big, period=50, thresholds=(1.5, 2.0, 2.5, 3.0), horizons=(1,))
    fired_thresholds = set(events_big["threshold"].unique())
    assert fired_thresholds == {1.5, 2.0, 2.5, 3.0}

    # spike=2.2 → z ≈ 2.2
    mid_small = _make_mid(n_warmup=50, spike=2.2, tail=5)
    events_small = build_events(mid_small, period=50, thresholds=(1.5, 2.0, 2.5, 3.0), horizons=(1,))
    fired_small = set(events_small["threshold"].unique())
    assert 1.5 in fired_small
    assert 2.0 in fired_small
    assert 2.5 not in fired_small
    assert 3.0 not in fired_small


def test_build_events_positive_spike_signed_reversion_positive_on_return():
    """Price spikes up then returns → signed_reversion should be positive."""
    from eda_vef_extreme_mr import build_events
    mid = _make_mid(n_warmup=50, spike=5.0, tail=5)
    events = build_events(mid, period=50, thresholds=(1.5,), horizons=(1,))
    # At horizon=1, price drops back toward 100 → signed_reversion > 0
    h1 = events[events["horizon_N"] == 1]
    assert len(h1) == 1
    assert h1.iloc[0]["signed_reversion"] > 0


def test_build_events_negative_spike_signed_reversion_positive_on_return():
    """Price drops then returns → signed_reversion should also be positive."""
    from eda_vef_extreme_mr import build_events
    mid = _make_mid(n_warmup=50, spike=-5.0, tail=5)
    events = build_events(mid, period=50, thresholds=(1.5,), horizons=(1,))
    h1 = events[events["horizon_N"] == 1]
    assert len(h1) == 1
    assert h1.iloc[0]["signed_reversion"] > 0


def test_build_events_reverted_flag_true_when_reversion_exceeds_breakeven():
    """reverted_2_5 is True when signed_reversion > 2.5."""
    from eda_vef_extreme_mr import build_events
    # spike=5, tail returns to 100 → signed_reversion ≈ 5 at horizon=1
    mid = _make_mid(n_warmup=50, spike=5.0, tail=5)
    events = build_events(mid, period=50, thresholds=(1.5,), horizons=(1,))
    h1 = events[events["horizon_N"] == 1]
    assert h1.iloc[0]["reverted_2_5"] is True


def test_build_events_reverted_flag_false_when_reversion_below_breakeven():
    """reverted_2_5 is False when signed_reversion <= 2.5."""
    from eda_vef_extreme_mr import build_events
    # spike=5 but tail stays at spike level (no reversion) → signed_rev < 0
    warmup = [99.0 if i % 2 == 0 else 101.0 for i in range(50)]
    mid = pd.Series(warmup + [105.0] * 6)  # spike and stays there
    events = build_events(mid, period=50, thresholds=(1.5,), horizons=(1,))
    h1 = events[events["horizon_N"] == 1]
    assert len(h1) >= 1
    assert h1.iloc[0]["reverted_2_5"] is False


def test_build_events_out_of_bounds_horizon_dropped():
    """If i + N >= len(mid), that (event, horizon) pair must be dropped."""
    from eda_vef_extreme_mr import build_events
    # spike at last possible tick with tail=0 → horizon=5 is out of bounds
    warmup = [99.0 if i % 2 == 0 else 101.0 for i in range(50)]
    mid = pd.Series(warmup + [106.0] + [100.0] * 2)  # only 2 tail ticks
    events = build_events(mid, period=50, thresholds=(1.5,), horizons=(1, 2, 5, 10, 20))
    horizons_present = set(events["horizon_N"].unique())
    assert 5 not in horizons_present
    assert 10 not in horizons_present
    assert 20 not in horizons_present
    assert 1 in horizons_present
    assert 2 in horizons_present


def test_build_events_mae_zero_when_price_immediately_reverts():
    """If price moves toward reversion at every sub-tick, MAE = 0."""
    from eda_vef_extreme_mr import build_events
    warmup = [99.0 if i % 2 == 0 else 101.0 for i in range(50)]
    # spike up, then strictly down at every subsequent tick
    mid = pd.Series(warmup + [108.0, 106.0, 104.0, 102.0, 100.0, 98.0])
    events = build_events(mid, period=50, thresholds=(1.5,), horizons=(5,))
    h5 = events[events["horizon_N"] == 5]
    assert len(h5) == 1
    assert h5.iloc[0]["mae"] == pytest.approx(0.0, abs=0.01)


def test_build_events_mae_positive_when_price_worsens_before_reverting():
    """If price moves further against us before reverting, MAE > 0."""
    from eda_vef_extreme_mr import build_events
    warmup = [99.0 if i % 2 == 0 else 101.0 for i in range(50)]
    # spike up, then goes even higher before returning
    mid = pd.Series(warmup + [108.0, 110.0, 112.0, 110.0, 108.0, 100.0])
    events = build_events(mid, period=50, thresholds=(1.5,), horizons=(5,))
    h5 = events[events["horizon_N"] == 5]
    assert len(h5) == 1
    assert h5.iloc[0]["mae"] > 0


def test_build_events_output_columns():
    """Output DataFrame has exactly the expected columns."""
    from eda_vef_extreme_mr import build_events
    mid = _make_mid(n_warmup=50, spike=5.0, tail=5)
    events = build_events(mid, period=50, thresholds=(2.0,), horizons=(1,))
    expected_cols = {"period", "threshold", "horizon_N", "signed_reversion", "reverted_2_5", "mae"}
    assert set(events.columns) == expected_cols


# ---------------------------------------------------------------------------
# aggregate_events
# ---------------------------------------------------------------------------

def _make_events(
    n: int,
    mean_rev: float,
    hit_rate_frac: float,
    period: int = 20,
    threshold: float = 2.0,
    horizon_N: int = 1,
) -> pd.DataFrame:
    """Build a synthetic events DataFrame with controlled mean and hit rate."""
    n_reverted = int(round(n * hit_rate_frac))
    n_not = n - n_reverted
    reverted_rows = [
        {
            "period": period,
            "threshold": threshold,
            "horizon_N": horizon_N,
            "signed_reversion": mean_rev,
            "reverted_2_5": True,
            "mae": 0.5,
        }
    ] * n_reverted
    not_rev_rows = [
        {
            "period": period,
            "threshold": threshold,
            "horizon_N": horizon_N,
            "signed_reversion": -mean_rev,
            "reverted_2_5": False,
            "mae": 2.0,
        }
    ] * n_not
    return pd.DataFrame(reverted_rows + not_rev_rows)


def test_aggregate_events_empty_returns_empty():
    from eda_vef_extreme_mr import aggregate_events
    result = aggregate_events(pd.DataFrame())
    assert result.empty


def test_aggregate_events_verdict_no_edge_when_n_too_low():
    """n < 5 → NO_EDGE regardless of reversion quality."""
    from eda_vef_extreme_mr import aggregate_events
    events = pd.DataFrame([
        {"period": 20, "threshold": 2.0, "horizon_N": 1,
         "signed_reversion": 5.0, "reverted_2_5": True, "mae": 0.0}
    ] * 4)  # n=4 < MIN_EVENTS=5
    result = aggregate_events(events)
    assert (result["verdict"] == "NO_EDGE").all()


def test_aggregate_events_verdict_no_edge_when_mean_too_low():
    """mean_reversion <= 2.5 → NO_EDGE even with high hit rate."""
    from eda_vef_extreme_mr import aggregate_events
    events = pd.DataFrame([
        {"period": 20, "threshold": 2.0, "horizon_N": 1,
         "signed_reversion": 2.0, "reverted_2_5": True, "mae": 0.0}
    ] * 10)  # mean=2.0, hit_rate=100%
    result = aggregate_events(events)
    assert (result["verdict"] == "NO_EDGE").all()


def test_aggregate_events_verdict_no_edge_when_hit_rate_too_low():
    """hit_rate < 40% → NO_EDGE even with high mean."""
    from eda_vef_extreme_mr import aggregate_events
    # 3 reverted (mean_rev=5.0) + 7 not (mean_rev=-5.0) → hit_rate=30%
    events = _make_events(n=10, mean_rev=5.0, hit_rate_frac=0.3)
    result = aggregate_events(events)
    assert (result["verdict"] == "NO_EDGE").all()


def test_aggregate_events_verdict_edge_when_all_criteria_met():
    """n>=5, mean>2.5, hit_rate>=40% → EDGE."""
    from eda_vef_extreme_mr import aggregate_events
    events = pd.DataFrame([
        {"period": 20, "threshold": 2.0, "horizon_N": 1,
         "signed_reversion": 3.0, "reverted_2_5": True, "mae": 0.5}
    ] * 10)  # n=10, mean=3.0, hit_rate=100%
    result = aggregate_events(events)
    assert (result["verdict"] == "EDGE").all()


def test_aggregate_events_groups_by_period_threshold_horizon():
    """Two distinct (period, threshold, horizon_N) groups produce two rows."""
    from eda_vef_extreme_mr import aggregate_events
    g1 = pd.DataFrame([
        {"period": 20, "threshold": 2.0, "horizon_N": 1,
         "signed_reversion": 3.0, "reverted_2_5": True, "mae": 0.0}
    ] * 5)
    g2 = pd.DataFrame([
        {"period": 50, "threshold": 2.5, "horizon_N": 5,
         "signed_reversion": 1.0, "reverted_2_5": False, "mae": 1.0}
    ] * 5)
    result = aggregate_events(pd.concat([g1, g2], ignore_index=True))
    assert len(result) == 2


def test_aggregate_events_output_columns():
    """Output DataFrame has exactly the expected columns."""
    from eda_vef_extreme_mr import aggregate_events
    events = pd.DataFrame([
        {"period": 20, "threshold": 2.0, "horizon_N": 1,
         "signed_reversion": 3.0, "reverted_2_5": True, "mae": 0.5}
    ] * 5)
    result = aggregate_events(events)
    expected_cols = {"period", "threshold", "horizon_N", "n_events",
                     "mean_reversion", "hit_rate_pct", "mean_mae", "verdict"}
    assert set(result.columns) == expected_cols


def test_aggregate_events_mean_reversion_computed_correctly():
    """mean_reversion in output matches mean of signed_reversion column."""
    from eda_vef_extreme_mr import aggregate_events
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    events = pd.DataFrame([
        {"period": 20, "threshold": 2.0, "horizon_N": 1,
         "signed_reversion": v, "reverted_2_5": v > 2.5, "mae": 0.0}
        for v in values
    ])
    result = aggregate_events(events)
    assert result.iloc[0]["mean_reversion"] == pytest.approx(3.0, abs=1e-6)
    assert result.iloc[0]["hit_rate_pct"] == pytest.approx(60.0, abs=1e-6)
