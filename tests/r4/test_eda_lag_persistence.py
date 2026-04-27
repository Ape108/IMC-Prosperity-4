"""Unit tests for classify_persistence (the persist/decay/revert rule)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r4"))

from eda_lag_persistence import classify_persistence


def test_persist_when_same_sign_and_ratio_at_or_above_0_6():
    # lead_5 = +0.225, lead_20 = +0.16 -> ratio = 0.71 -> persist
    corrs = {1: 0.2, 2: 0.21, 5: 0.225, 10: 0.18, 20: 0.16}
    assert classify_persistence(corrs) == "persist"


def test_revert_when_opposite_sign():
    # lead_5 = +0.225, lead_20 = -0.10 -> revert
    corrs = {1: 0.2, 2: 0.21, 5: 0.225, 10: 0.05, 20: -0.10}
    assert classify_persistence(corrs) == "revert"


def test_decay_when_same_sign_and_ratio_below_0_4():
    # lead_5 = +0.225, lead_20 = +0.05 -> ratio = 0.22 -> decay
    corrs = {1: 0.2, 2: 0.21, 5: 0.225, 10: 0.10, 20: 0.05}
    assert classify_persistence(corrs) == "decay"


def test_borderline_between_thresholds():
    # ratio = 0.5 -> between 0.4 and 0.6 -> borderline
    corrs = {1: 0.2, 2: 0.21, 5: 0.20, 10: 0.15, 20: 0.10}
    assert classify_persistence(corrs) == "borderline"


def test_indeterminate_when_lead_5_is_zero():
    corrs = {1: 0.0, 2: 0.0, 5: 0.0, 10: 0.0, 20: 0.0}
    assert classify_persistence(corrs) == "indeterminate"


def test_indeterminate_when_lead_20_is_zero():
    corrs = {1: 0.2, 2: 0.21, 5: 0.225, 10: 0.10, 20: 0.0}
    assert classify_persistence(corrs) == "indeterminate"


def test_negative_signal_persists():
    # VEV_5400/5500 case: large negative corr persisting
    corrs = {1: -0.30, 2: -0.31, 5: -0.319, 10: -0.25, 20: -0.22}
    assert classify_persistence(corrs) == "persist"


def test_persist_at_inclusive_lower_boundary():
    # ratio = 0.6 exactly -> persist (PERSIST_RATIO is inclusive)
    corrs = {1: 0.2, 2: 0.21, 5: 0.20, 10: 0.15, 20: 0.12}
    assert classify_persistence(corrs) == "persist"


def test_borderline_at_decay_upper_boundary():
    # ratio = 0.4 exactly -> borderline (DECAY_RATIO is strict less-than)
    # 0.10 / 0.25 == 0.4 exactly in IEEE 754; 0.4 < 0.4 is False -> borderline
    corrs = {1: 0.2, 2: 0.21, 5: 0.25, 10: 0.10, 20: 0.10}
    assert classify_persistence(corrs) == "borderline"


def test_lag_persistence_table_assembles_columns_in_lag_order(monkeypatch):
    """lag_persistence_table preserves the input lag order in column names + values."""
    import eda_lag_persistence as mod

    def fake_lead_lag_corr(trades_df, prices_df, bot, product, lags):
        # Return a deterministic value per (product, lag) so we can verify ordering.
        base = {"VEV_X": 1.0, "VEV_Y": 2.0}[product]
        return [base + lag * 0.01 for lag in lags]

    monkeypatch.setattr(mod, "lead_lag_corr", fake_lead_lag_corr)

    table = mod.lag_persistence_table(
        trades_df=None, prices_df=None,
        bot="Mark 14", products=["VEV_X", "VEV_Y"], lags=[1, 5, 20],
    )

    assert list(table.columns) == ["bot", "product", "lead_1_corr", "lead_5_corr", "lead_20_corr"]
    assert table.loc[0, "product"] == "VEV_X"
    assert round(float(table.loc[0, "lead_1_corr"]), 3) == 1.01
    assert round(float(table.loc[0, "lead_5_corr"]), 3) == 1.05
    assert round(float(table.loc[0, "lead_20_corr"]), 3) == 1.20
    assert table.loc[1, "product"] == "VEV_Y"
    assert round(float(table.loc[1, "lead_20_corr"]), 3) == 2.20
