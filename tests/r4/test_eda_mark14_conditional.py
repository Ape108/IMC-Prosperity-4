"""Unit tests for eda_mark14_conditional slicing helpers and decision rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eda_mark14_conditional import (
    MARK14_BOT,
    THRESHOLD_MEAN_TICKS,
    THRESHOLD_N,
    THRESHOLD_T_STAT,
)


def _make_trades(rows: list[dict]) -> pd.DataFrame:
    """Build a synthetic trades DataFrame with the canonical columns."""
    return pd.DataFrame(rows, columns=["ts", "bot", "product", "signed_qty"])


def test_burst_size_counts_prior_trades_within_window():
    from eda_mark14_conditional import burst_size_in_window
    trades = _make_trades([
        {"ts": 100, "bot": MARK14_BOT, "product": "VEV_5300", "signed_qty": 5},
        {"ts": 150, "bot": MARK14_BOT, "product": "VEV_5300", "signed_qty": 5},
        {"ts": 200, "bot": MARK14_BOT, "product": "VEV_5300", "signed_qty": 5},
    ])
    assert burst_size_in_window(trades, ts=300, strike="VEV_5300", window=200) == 3


def test_burst_size_excludes_trade_at_exact_ts():
    from eda_mark14_conditional import burst_size_in_window
    trades = _make_trades([
        {"ts": 100, "bot": MARK14_BOT, "product": "VEV_5300", "signed_qty": 5},
        {"ts": 200, "bot": MARK14_BOT, "product": "VEV_5300", "signed_qty": 5},
    ])
    assert burst_size_in_window(trades, ts=200, strike="VEV_5300", window=200) == 1


def test_burst_size_respects_window_lower_bound():
    from eda_mark14_conditional import burst_size_in_window
    trades = _make_trades([
        {"ts": 50, "bot": MARK14_BOT, "product": "VEV_5300", "signed_qty": 5},
        {"ts": 250, "bot": MARK14_BOT, "product": "VEV_5300", "signed_qty": 5},
    ])
    assert burst_size_in_window(trades, ts=300, strike="VEV_5300", window=100) == 1


def test_burst_size_filters_by_strike():
    from eda_mark14_conditional import burst_size_in_window
    trades = _make_trades([
        {"ts": 100, "bot": MARK14_BOT, "product": "VEV_5300", "signed_qty": 5},
        {"ts": 150, "bot": MARK14_BOT, "product": "VEV_5400", "signed_qty": 5},
    ])
    assert burst_size_in_window(trades, ts=200, strike="VEV_5300", window=200) == 1


def test_burst_size_filters_by_bot():
    from eda_mark14_conditional import burst_size_in_window
    trades = _make_trades([
        {"ts": 100, "bot": MARK14_BOT, "product": "VEV_5300", "signed_qty": 5},
        {"ts": 150, "bot": "Mark 22", "product": "VEV_5300", "signed_qty": 5},
    ])
    assert burst_size_in_window(trades, ts=200, strike="VEV_5300", window=200) == 1


def _make_prices(rows: list[dict]) -> pd.DataFrame:
    """Build a synthetic prices DataFrame with the canonical columns."""
    return pd.DataFrame(
        rows,
        columns=["ts", "product", "bid_price_1", "ask_price_1", "mid_price"],
    )


def test_spread_at_tick_returns_book_spread_at_or_before_ts():
    from eda_mark14_conditional import spread_at_tick
    prices = _make_prices([
        {"ts": 100, "product": "VEV_5300", "bid_price_1": 50, "ask_price_1": 52, "mid_price": 51.0},
        {"ts": 200, "product": "VEV_5300", "bid_price_1": 50, "ask_price_1": 51, "mid_price": 50.5},
    ])
    assert spread_at_tick(prices, "VEV_5300", ts=150) == 2
    assert spread_at_tick(prices, "VEV_5300", ts=200) == 1
    assert spread_at_tick(prices, "VEV_5300", ts=250) == 1


def test_spread_at_tick_returns_none_when_no_prior_tick():
    from eda_mark14_conditional import spread_at_tick
    prices = _make_prices([
        {"ts": 100, "product": "VEV_5300", "bid_price_1": 50, "ask_price_1": 52, "mid_price": 51.0},
    ])
    assert spread_at_tick(prices, "VEV_5300", ts=50) is None


def test_spread_at_tick_filters_by_strike():
    from eda_mark14_conditional import spread_at_tick
    prices = _make_prices([
        {"ts": 100, "product": "VEV_5300", "bid_price_1": 50, "ask_price_1": 51, "mid_price": 50.5},
        {"ts": 100, "product": "VEV_5400", "bid_price_1": 30, "ask_price_1": 33, "mid_price": 31.5},
    ])
    assert spread_at_tick(prices, "VEV_5400", ts=150) == 3


def test_joint_bot_active_true_when_other_bot_traded_recently():
    from eda_mark14_conditional import joint_bot_active
    trades = _make_trades([
        {"ts": 50, "bot": "Mark 22", "product": "HYDROGEL_PACK", "signed_qty": 3},
    ])
    assert joint_bot_active(trades, ts=120, other_bot="Mark 22", window=100) is True


def test_joint_bot_active_false_when_other_bot_outside_window():
    from eda_mark14_conditional import joint_bot_active
    trades = _make_trades([
        {"ts": 10, "bot": "Mark 22", "product": "HYDROGEL_PACK", "signed_qty": 3},
    ])
    assert joint_bot_active(trades, ts=200, other_bot="Mark 22", window=100) is False


def test_joint_bot_active_excludes_trade_at_exact_ts():
    from eda_mark14_conditional import joint_bot_active
    trades = _make_trades([
        {"ts": 100, "bot": "Mark 22", "product": "HYDROGEL_PACK", "signed_qty": 3},
    ])
    assert joint_bot_active(trades, ts=100, other_bot="Mark 22", window=100) is False


def test_joint_bot_active_ignores_other_bots():
    from eda_mark14_conditional import joint_bot_active
    trades = _make_trades([
        {"ts": 50, "bot": "Mark 38", "product": "HYDROGEL_PACK", "signed_qty": 3},
    ])
    assert joint_bot_active(trades, ts=120, other_bot="Mark 22", window=100) is False


def test_cooling_off_returns_ticks_since_prior_bot_trade():
    from eda_mark14_conditional import cooling_off_ticks
    trades = _make_trades([
        {"ts": 100, "bot": MARK14_BOT, "product": "VEV_5300", "signed_qty": 5},
        {"ts": 250, "bot": MARK14_BOT, "product": "VEV_5400", "signed_qty": 5},
    ])
    assert cooling_off_ticks(trades, ts=300, bot=MARK14_BOT) == 50


def test_cooling_off_uses_most_recent_prior_trade_across_strikes():
    from eda_mark14_conditional import cooling_off_ticks
    trades = _make_trades([
        {"ts": 100, "bot": MARK14_BOT, "product": "VEV_5300", "signed_qty": 5},
        {"ts": 200, "bot": MARK14_BOT, "product": "VEV_5500", "signed_qty": 5},
    ])
    assert cooling_off_ticks(trades, ts=350, bot=MARK14_BOT) == 150


def test_cooling_off_returns_none_when_no_prior_trade():
    from eda_mark14_conditional import cooling_off_ticks
    trades = _make_trades([
        {"ts": 100, "bot": "Mark 22", "product": "VEV_5300", "signed_qty": 3},
    ])
    assert cooling_off_ticks(trades, ts=200, bot=MARK14_BOT) is None


def test_cooling_off_excludes_trade_at_exact_ts():
    from eda_mark14_conditional import cooling_off_ticks
    trades = _make_trades([
        {"ts": 100, "bot": MARK14_BOT, "product": "VEV_5300", "signed_qty": 5},
        {"ts": 200, "bot": MARK14_BOT, "product": "VEV_5300", "signed_qty": 5},
    ])
    assert cooling_off_ticks(trades, ts=200, bot=MARK14_BOT) == 100


def test_realized_vol_quartile_top_in_high_vol_segment():
    from eda_mark14_conditional import realized_vol_quartile
    n = 40
    constant = [100.0] * (n // 2)
    noisy = [100.0 + ((-1) ** i) * 5 for i in range(n // 2)]
    prices = _make_prices([
        {"ts": i, "product": "VEV_5300", "bid_price_1": p - 1, "ask_price_1": p + 1, "mid_price": p}
        for i, p in enumerate(constant + noisy)
    ])
    assert realized_vol_quartile(prices, "VEV_5300", ts=35, window=5) == "top"


def test_realized_vol_quartile_rest_in_low_vol_segment():
    from eda_mark14_conditional import realized_vol_quartile
    n = 40
    constant = [100.0] * (n // 2)
    noisy = [100.0 + ((-1) ** i) * 5 for i in range(n // 2)]
    prices = _make_prices([
        {"ts": i, "product": "VEV_5300", "bid_price_1": p - 1, "ask_price_1": p + 1, "mid_price": p}
        for i, p in enumerate(constant + noisy)
    ])
    assert realized_vol_quartile(prices, "VEV_5300", ts=15, window=5) == "rest"


def test_realized_vol_quartile_rest_during_warmup():
    from eda_mark14_conditional import realized_vol_quartile
    prices = _make_prices([
        {"ts": i, "product": "VEV_5300", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100.0 + i}
        for i in range(3)
    ])
    # ts=2: rolling-5 std needs 5 prior diffs, only 2 ticks of data → vol is NaN → "rest"
    assert realized_vol_quartile(prices, "VEV_5300", ts=2, window=5) == "rest"


def test_decision_pass_true_when_all_thresholds_met():
    from eda_mark14_conditional import decision_pass
    # n=10, mean=1.6, std=0.5 → t = sqrt(10) * 1.6 / 0.5 = 10.12
    assert decision_pass(n=10, mean=1.6, std=0.5) is True


def test_decision_pass_false_when_n_below_threshold():
    from eda_mark14_conditional import decision_pass
    assert decision_pass(n=5, mean=2.0, std=0.5) is False


def test_decision_pass_false_when_mean_below_threshold():
    from eda_mark14_conditional import decision_pass
    assert decision_pass(n=10, mean=1.4, std=0.5) is False


def test_decision_pass_false_when_t_stat_below_threshold():
    from eda_mark14_conditional import decision_pass
    # n=10, mean=1.6, std=10.0 → t = sqrt(10)*1.6/10 = 0.506
    assert decision_pass(n=10, mean=1.6, std=10.0) is False


def test_decision_pass_false_when_std_zero():
    from eda_mark14_conditional import decision_pass
    assert decision_pass(n=10, mean=1.6, std=0.0) is False


def test_decision_pass_handles_negative_mean_via_abs():
    from eda_mark14_conditional import decision_pass
    # signed_move already accounts for predicted direction; large negative mean
    # means the predicted direction was wrong — still |mean|>=1.5 but the pass
    # condition should NOT fire: a "passes" cell must be in the predicted
    # direction. abs(mean) is the right gate ONLY if mean>0.
    # We encode this: only positive (predicted-direction-aligned) means pass.
    assert decision_pass(n=10, mean=-1.6, std=0.5) is False
