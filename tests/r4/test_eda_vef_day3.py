"""Unit tests for eda_vef_day3 helpers."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r4"))

from eda_vef_day3 import compute_daily_stats, apply_gate

PRODUCT = "VELVETFRUIT_EXTRACT"


def make_prices(day: int, mids: list[float]) -> pd.DataFrame:
    n = len(mids)
    step = 1_000_000 // n
    return pd.DataFrame({
        "day": [day] * n,
        "timestamp": [i * step for i in range(n)],
        "product": [PRODUCT] * n,
        "mid_price": mids,
    })


def test_compute_daily_stats_rising_trend():
    """Net positive trend: net_move > 0, late_same_dir True."""
    df = make_prices(1, list(range(100, 200)))  # monotone rising
    stats = compute_daily_stats(df, PRODUCT)
    assert len(stats) == 1
    s = stats[0]
    assert s["day"] == 1
    assert s["net_move"] == pytest.approx(99.0)
    assert s["abs_net_move"] == pytest.approx(99.0)
    assert s["late_same_dir"] is True
    assert s["max_excursion"] == pytest.approx(99.0)  # monotone rise, max excursion = net move


def test_compute_daily_stats_falling_with_late_reversal():
    """Falls for 80% of session then rises in last 20%: late_same_dir False."""
    # 80 ticks falling, then 20 ticks rising
    mids = list(range(200, 120, -1)) + list(range(120, 140))  # 80 + 20 = 100 ticks
    df = make_prices(2, mids)
    stats = compute_daily_stats(df, PRODUCT)
    s = stats[0]
    net_move = mids[-1] - mids[0]  # 139 - 200 = -61
    assert s["net_move"] == pytest.approx(net_move)
    assert s["late_same_dir"] is False  # overall fell, but late 20% rose


def test_compute_daily_stats_multi_day():
    """Stats returned for each day independently."""
    df1 = make_prices(1, [100, 110, 120, 130, 140])
    df2 = make_prices(2, [200, 190, 180, 170, 160])
    df = pd.concat([df1, df2], ignore_index=True)
    stats = compute_daily_stats(df, PRODUCT)
    assert len(stats) == 2
    days = {s["day"] for s in stats}
    assert days == {1, 2}


def test_compute_daily_stats_product_filter():
    """Only returns stats for the requested product."""
    df = make_prices(1, [100, 200, 300])
    df["product"] = "OTHER_PRODUCT"
    stats = compute_daily_stats(df, PRODUCT)
    assert stats == []


def test_apply_gate_confirms_large_trend():
    """Day 3 abs_net_move > 2x avg of days 1/2, and late_same_dir=True -> confirmed."""
    stats = [
        {"day": 1, "abs_net_move": 5.0, "net_move": 5.0, "late_same_dir": True},
        {"day": 2, "abs_net_move": 5.0, "net_move": -5.0, "late_same_dir": False},
        {"day": 3, "abs_net_move": 25.0, "net_move": -25.0, "late_same_dir": True},
    ]
    # avg other days = 5, day 3 = 25 > 2*5 = 10, late_same_dir=True
    assert apply_gate(stats) is True


def test_apply_gate_rejects_small_day3_move():
    """Day 3 abs_net_move not > 2x avg -> rejected."""
    stats = [
        {"day": 1, "abs_net_move": 5.0, "net_move": 5.0, "late_same_dir": True},
        {"day": 2, "abs_net_move": 5.0, "net_move": 5.0, "late_same_dir": True},
        {"day": 3, "abs_net_move": 8.0, "net_move": 8.0, "late_same_dir": True},
    ]
    # avg = 5, 8 < 10 -> rejected
    assert apply_gate(stats) is False


def test_apply_gate_rejects_reverting_late_session():
    """Day 3 is large but trend reverses in late session -> rejected."""
    stats = [
        {"day": 1, "abs_net_move": 5.0, "net_move": 5.0, "late_same_dir": True},
        {"day": 2, "abs_net_move": 5.0, "net_move": 5.0, "late_same_dir": True},
        {"day": 3, "abs_net_move": 25.0, "net_move": -25.0, "late_same_dir": False},
    ]
    # large move but late_same_dir=False -> rejected
    assert apply_gate(stats) is False


def test_apply_gate_missing_day3_returns_false():
    stats = [
        {"day": 1, "abs_net_move": 5.0, "net_move": 5.0, "late_same_dir": True},
        {"day": 2, "abs_net_move": 5.0, "net_move": 5.0, "late_same_dir": True},
    ]
    assert apply_gate(stats) is False
