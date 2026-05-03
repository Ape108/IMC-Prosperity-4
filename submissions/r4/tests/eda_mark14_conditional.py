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

from submissions.r4.tests.eda_mark_bots import DATASET_DIR, R4_DAYS

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


def realized_vol_quartile(
    prices: pd.DataFrame,
    strike: str,
    ts: int,
    window: int = VOL_WINDOW,
    quantile: float = VOL_QUANTILE,
) -> str:
    """Return 'top' iff the rolling-`window`-tick mid-return std at the price
    tick at-or-before `ts` is at-or-above the `quantile` of the pooled vol
    distribution for `strike`. 'rest' for warmup ticks (vol=NaN) or when no
    prior tick exists.
    """
    sub = (
        prices[prices["product"] == strike]
        .sort_values("ts")
        .reset_index(drop=True)
    )
    if sub.empty:
        return "rest"
    rets = sub["mid_price"].diff()
    vol = rets.rolling(window).std()
    threshold = vol.quantile(quantile)
    if pd.isna(threshold):
        return "rest"
    ts_arr = sub["ts"].to_numpy()
    idx = int(np.searchsorted(ts_arr, ts, side="right")) - 1
    if idx < 0:
        return "rest"
    v = vol.iloc[idx]
    if pd.isna(v):
        return "rest"
    return "top" if v >= threshold else "rest"


# ── Decision rule ────────────────────────────────────────────────────────────


def decision_pass(
    n: int,
    mean: float,
    std: float,
    threshold_mean: float = THRESHOLD_MEAN_TICKS,
    threshold_t: float = THRESHOLD_T_STAT,
    threshold_n: int = THRESHOLD_N,
) -> bool:
    """A cell passes iff:
        n >= threshold_n
        AND mean >= threshold_mean (signed in the predicted direction)
        AND t-stat = sqrt(n) * mean / std >= threshold_t

    Negative `mean` means the predicted direction was wrong on aggregate; never
    a pass even if |mean| is large. `std<=0` is a degenerate cell — never a pass.
    """
    if n < threshold_n:
        return False
    if std <= 0:
        return False
    if mean < threshold_mean:
        return False
    t_stat = (n ** 0.5) * mean / std
    return t_stat >= threshold_t


# ── Loaders ──────────────────────────────────────────────────────────────────


def load_trades_with_day(days: list[int] = R4_DAYS) -> pd.DataFrame:
    """Load Mark-bot trades across `days` and return a DataFrame with columns:
        ts, day, bot, product, signed_qty
    where ts = day * TICKS_PER_DAY + raw_timestamp (strictly monotonic).
    """
    dfs: list[pd.DataFrame] = []
    for day in days:
        path = f"{DATASET_DIR}/trades_round_4_day_{day}.csv"
        df = pd.read_csv(path, sep=";", dtype={"buyer": str, "seller": str})
        df["day"] = day
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)

    buys = raw[raw["buyer"].str.startswith("Mark", na=False)].copy()
    buys["bot"] = buys["buyer"]
    buys["signed_qty"] = buys["quantity"]

    sells = raw[raw["seller"].str.startswith("Mark", na=False)].copy()
    sells["bot"] = sells["seller"]
    sells["signed_qty"] = -sells["quantity"]

    combined = pd.concat([buys, sells], ignore_index=True)
    combined["ts"] = combined["day"] * TICKS_PER_DAY + combined["timestamp"]
    return combined[["ts", "day", "bot", "symbol", "signed_qty"]].rename(
        columns={"symbol": "product"}
    )


def load_prices_with_book(days: list[int] = R4_DAYS) -> pd.DataFrame:
    """Load price ticks across `days` with bid_price_1 / ask_price_1 / mid_price
    preserved. ts = day * TICKS_PER_DAY + raw_timestamp."""
    dfs: list[pd.DataFrame] = []
    for day in days:
        path = f"{DATASET_DIR}/prices_round_4_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)
    raw["ts"] = raw["day"] * TICKS_PER_DAY + raw["timestamp"]
    return raw[["ts", "day", "product", "bid_price_1", "ask_price_1", "mid_price"]].copy()


# ── Pipeline: per-trade signed moves ─────────────────────────────────────────


def compute_signed_moves(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    strike: str,
    lag: int,
) -> pd.DataFrame:
    """For each Mark 14 trade on `strike`, compute signed_move at horizon
    `lag` price ticks. signed_move = (mid[idx+lag] - mid[idx]) * predicted_dir.

    Trades whose price tick is missing (idx<0) or whose lag-shifted tick is
    out-of-range are dropped. ts column is preserved for downstream join.
    """
    direction = PREDICTED_DIR[strike]
    bot_trades = trades[
        (trades["bot"] == MARK14_BOT) & (trades["product"] == strike)
    ].copy()
    sub = (
        prices[prices["product"] == strike]
        .sort_values("ts")
        .reset_index(drop=True)
    )
    ts_arr = sub["ts"].to_numpy()
    mid_arr = sub["mid_price"].to_numpy()

    out_rows: list[dict] = []
    for _, row in bot_trades.iterrows():
        t = int(row["ts"])
        idx = int(np.searchsorted(ts_arr, t, side="right")) - 1
        if idx < 0 or idx + lag >= len(mid_arr):
            continue
        raw_move = float(mid_arr[idx + lag] - mid_arr[idx])
        out_rows.append({
            "ts": t,
            "lag": lag,
            "raw_move": raw_move,
            "signed_move": raw_move * direction,
        })
    return pd.DataFrame(out_rows, columns=["ts", "lag", "raw_move", "signed_move"])


# ── Bucket helpers ────────────────────────────────────────────────────────────


def _burst_bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 2:
        return "1-2"
    return ">=3"


def _spread_bucket(spread: int | None) -> str:
    if spread is None:
        return "unknown"
    if spread <= 1:
        return "1t"
    if spread == 2:
        return "2t"
    return ">=3t"


def _cooling_bucket(cool: int | None) -> str:
    if cool is None or cool >= COOLING_THRESHOLD:
        return ">=100"
    return "<100"


# ── Events table ──────────────────────────────────────────────────────────────


def build_events_df(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    strikes: list[str] = MARK14_STRIKES,
    lags: list[int] = LAGS,
) -> pd.DataFrame:
    """One row per (Mark 14 trade × lag), with signed_move and all 6 bucket
    assignments. Used as the single input to aggregate_metrics.
    """
    rows: list[dict] = []
    for strike in strikes:
        direction = PREDICTED_DIR[strike]
        m14 = trades[
            (trades["bot"] == MARK14_BOT) & (trades["product"] == strike)
        ].copy()
        sub = (
            prices[prices["product"] == strike]
            .sort_values("ts")
            .reset_index(drop=True)
        )
        ts_arr = sub["ts"].to_numpy()
        mid_arr = sub["mid_price"].to_numpy()

        for _, row in m14.iterrows():
            t = int(row["ts"])
            day = int(row["day"])
            idx = int(np.searchsorted(ts_arr, t, side="right")) - 1
            if idx < 0:
                continue

            burst = burst_size_in_window(trades, t, strike, BURST_WINDOW)
            spread = spread_at_tick(prices, strike, t)
            cool = cooling_off_ticks(trades, t, MARK14_BOT)
            joint = joint_bot_active(trades, t, JOINT_BOT, JOINT_WINDOW)
            vol = realized_vol_quartile(prices, strike, t, VOL_WINDOW)

            burst_b = _burst_bucket(burst)
            spread_b = _spread_bucket(spread)
            cool_b = _cooling_bucket(cool)
            joint_b = "yes" if joint else "no"
            day_b = str(day)

            for lag in lags:
                if idx + lag >= len(mid_arr):
                    continue
                raw_move = float(mid_arr[idx + lag] - mid_arr[idx])
                signed_move = raw_move * direction
                rows.append({
                    "ts": t,
                    "strike": strike,
                    "lag": lag,
                    "signed_move": signed_move,
                    "burst": burst_b,
                    "spread": spread_b,
                    "day": day_b,
                    "cooling": cool_b,
                    "joint22": joint_b,
                    "vol": vol,
                })
    return pd.DataFrame(rows)


# ── Pipeline: aggregation ────────────────────────────────────────────────────

SLICE_COLUMNS = ["burst", "spread", "day", "cooling", "joint22", "vol"]


def _agg_one(values: pd.Series) -> dict:
    """n, mean, std, t-stat, win-rate for a series of signed moves."""
    n = int(len(values))
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "t_stat": float("nan"), "win_rate": float("nan")}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n >= 2 else 0.0
    t_stat = (n ** 0.5) * mean / std if std > 0 else float("nan")
    win_rate = float((values > 0).mean())
    return {"n": n, "mean": mean, "std": std, "t_stat": t_stat, "win_rate": win_rate}


def aggregate_metrics(events: pd.DataFrame) -> pd.DataFrame:
    """Flatten events into (strike x slice x bucket x lag) cells. Includes
    an 'unconditional' slice (bucket='all') as the baseline per strike x lag.
    """
    cells: list[dict] = []

    for (strike, lag), grp in events.groupby(["strike", "lag"]):
        agg = _agg_one(grp["signed_move"])
        cells.append({
            "strike": strike, "slice": "unconditional", "bucket": "all",
            "lag": int(lag), **agg,
        })

    for slice_col in SLICE_COLUMNS:
        for (strike, bucket, lag), grp in events.groupby(["strike", slice_col, "lag"]):
            agg = _agg_one(grp["signed_move"])
            cells.append({
                "strike": strike, "slice": slice_col, "bucket": str(bucket),
                "lag": int(lag), **agg,
            })

    df = pd.DataFrame(cells)
    df["passes"] = df.apply(
        lambda r: decision_pass(r["n"], r["mean"], r["std"]) if r["n"] > 0 else False,
        axis=1,
    )
    return df.sort_values(["strike", "slice", "bucket", "lag"]).reset_index(drop=True)


# ── Output ───────────────────────────────────────────────────────────────────


def _format_row(row: pd.Series) -> str:
    return (
        f"  {row['slice']:14s} {str(row['bucket']):8s} "
        f"lag={int(row['lag']):>2d} "
        f"n={int(row['n']):>3d} mean={row['mean']:+.3f}t "
        f"t={row['t_stat']:+.2f} wr={row['win_rate']:.2f}"
        + (" *** PASS ***" if row['passes'] else "")
    )


def emit_report(cells: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=== Mark 14 Conditional Edge Analysis ===")
    for strike in MARK14_STRIKES:
        direction = PREDICTED_DIR[strike]
        lines.append(f"\n{strike} (predicted_dir={direction:+d}):")
        sub = cells[cells["strike"] == strike]
        if sub.empty:
            lines.append("  (no events)")
            continue
        for _, row in sub.iterrows():
            lines.append(_format_row(row))

    lines.append("")
    passing = cells[cells["passes"]]
    lines.append("=== VERDICT ===")
    if passing.empty:
        lines.append("NO PASS — shelve Mark 14")
    else:
        for _, row in passing.iterrows():
            lines.append(
                f"PASS: {row['strike']} {row['slice']}={row['bucket']} "
                f"lag={int(row['lag'])} "
                f"(mean={row['mean']:+.3f}t t={row['t_stat']:+.2f} "
                f"wr={row['win_rate']:.2f} n={int(row['n'])})"
            )
    return "\n".join(lines)


def main() -> None:
    print("Loading R4 trades and prices (days 1/2/3)...")
    trades = load_trades_with_day()
    prices = load_prices_with_book()
    m14 = trades[trades["bot"] == MARK14_BOT]
    print(f"  {len(m14)} Mark 14 trades on {sorted(m14['product'].unique())}.")

    events = build_events_df(trades, prices)
    cells = aggregate_metrics(events)
    print(emit_report(cells))


if __name__ == "__main__":
    main()
