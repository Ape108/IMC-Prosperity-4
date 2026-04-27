"""Mark 14 conditional-edge analysis.

Falsifies or surfaces a tradeable Mark 14 sub-regime on VEV_5300/5400/5500.
For each Mark 14 trade, compute signed price moves at horizons {5, 10, 20, 50}
ticks. Bucket trades along 6 conditional slices. Per (strike × slice × bucket
× lag) cell, compute (n, mean, std, t-stat, win-rate) and check against the
decision rule. Emit a single-line PASS/NO PASS verdict.

Spec: docs/superpowers/specs/2026-04-27-r4-mark14-conditional-edge-design.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from eda_mark_bots import DATASET_DIR, R4_DAYS

MARK14_BOT = "Mark 14"
JOINT_BOT = "Mark 22"
MARK14_STRIKES = ["VEV_5300", "VEV_5400", "VEV_5500"]
PREDICTED_DIR = {"VEV_5300": +1, "VEV_5400": -1, "VEV_5500": -1}
LAGS = [5, 10, 20, 50]  # lags 1/2 excluded — bid-ask noise (spec §Metrics)
TICKS_PER_DAY = 1_000_000

BURST_WINDOW = 200
JOINT_WINDOW = 100
COOLING_THRESHOLD = 100
VOL_WINDOW = 500
VOL_QUANTILE = 0.75

THRESHOLD_MEAN_TICKS = 1.5
THRESHOLD_T_STAT = 2.0
THRESHOLD_N = 10


# ── Slicing helpers ──────────────────────────────────────────────────────────


def burst_size_in_window(
    trades: pd.DataFrame,
    ts: int,
    strike: str,
    window: int = BURST_WINDOW,
    bot: str = MARK14_BOT,
) -> int:
    """Count `bot` trades on `strike` strictly before `ts`, within `window` ticks.

    `trades` must have columns: ts, bot, product. The current trade at `ts` is
    excluded — caller measures *prior* clustering.
    """
    mask = (
        (trades["bot"] == bot)
        & (trades["product"] == strike)
        & (trades["ts"] >= ts - window)
        & (trades["ts"] < ts)
    )
    return int(mask.sum())


def spread_at_tick(prices: pd.DataFrame, strike: str, ts: int) -> int | None:
    """Spread (ask_1 − bid_1) at the price tick at-or-before `ts`. None if no
    such tick exists or the book has missing values.
    """
    sub = (
        prices[prices["product"] == strike]
        .sort_values("ts")
        .reset_index(drop=True)
    )
    if sub.empty:
        return None
    ts_arr = sub["ts"].to_numpy()
    idx = int(np.searchsorted(ts_arr, ts, side="right")) - 1
    if idx < 0:
        return None
    bid = sub.iloc[idx]["bid_price_1"]
    ask = sub.iloc[idx]["ask_price_1"]
    if pd.isna(bid) or pd.isna(ask):
        return None
    return int(ask - bid)


def joint_bot_active(
    trades: pd.DataFrame,
    ts: int,
    other_bot: str,
    window: int = JOINT_WINDOW,
) -> bool:
    """True iff `other_bot` has any trade (any product) strictly before `ts`,
    within `window` ticks."""
    mask = (
        (trades["bot"] == other_bot)
        & (trades["ts"] >= ts - window)
        & (trades["ts"] < ts)
    )
    return bool(mask.any())


def cooling_off_ticks(
    trades: pd.DataFrame,
    ts: int,
    bot: str = MARK14_BOT,
) -> int | None:
    """Ticks since the most recent prior `bot` trade on any product. None if
    `bot` has no trade strictly before `ts`."""
    sub = trades[(trades["bot"] == bot) & (trades["ts"] < ts)]
    if sub.empty:
        return None
    return int(ts - sub["ts"].max())
