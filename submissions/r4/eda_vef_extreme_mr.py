"""
VEF Extreme Mean-Reversion EDA.

Tests whether VELVETFRUIT_EXTRACT reverts >2.5 ticks at 2-3σ z-score extremes.
Decision rule: ≥1 EDGE cell → high-threshold signal strategy viable.

Run with:
  .venv/Scripts/python.exe submissions/r4/eda_vef_extreme_mr.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "datasets" / "round4"
DAYS = (1, 2, 3)
PERIODS = (20, 50, 75)
THRESHOLDS = (1.5, 2.0, 2.5, 3.0)
HORIZONS = (1, 2, 5, 10, 20)
REVERSION_BREAKEVEN = 2.5
MIN_EVENTS = 5
MIN_HIT_RATE = 40.0


def build_events(
    mid: pd.Series,
    period: int,
    thresholds: tuple[float, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    roll = mid.rolling(window=period, min_periods=period)
    roll_mean = roll.mean()
    roll_std = roll.std()
    zscore = (mid - roll_mean) / roll_std

    rows: list[dict] = []
    n = len(mid)
    for i in range(period, n):
        z = zscore.iloc[i]
        if pd.isna(z) or z == 0.0:
            continue
        sign_z = 1 if z > 0 else -1
        abs_z = abs(z)
        for threshold in thresholds:
            if abs_z <= threshold:
                continue
            for N in horizons:
                if i + N >= n:
                    continue
                fwd_move = mid.iloc[i + N] - mid.iloc[i]
                signed_rev = fwd_move * (-sign_z)
                reverted = bool(signed_rev > REVERSION_BREAKEVEN)
                window = mid.iloc[i + 1 : i + N + 1]
                adverse = ((window - mid.iloc[i]) * sign_z).clip(lower=0)
                mae = float(adverse.max()) if len(adverse) > 0 else 0.0
                rows.append({
                    "period": period,
                    "threshold": threshold,
                    "horizon_N": N,
                    "signed_reversion": float(signed_rev),
                    "reverted_2_5": reverted,
                    "mae": mae,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["reverted_2_5"] = df["reverted_2_5"].astype(object)
    return df


def aggregate_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    def _summarize(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        mean_rev = float(g["signed_reversion"].mean())
        hit_rate = 100.0 * float(g["reverted_2_5"].mean())
        mean_mae = float(g["mae"].mean())
        is_edge = (
            n >= MIN_EVENTS
            and mean_rev > REVERSION_BREAKEVEN
            and hit_rate >= MIN_HIT_RATE
        )
        return pd.Series({
            "n_events": n,
            "mean_reversion": mean_rev,
            "hit_rate_pct": hit_rate,
            "mean_mae": mean_mae,
            "verdict": "EDGE" if is_edge else "NO_EDGE",
        })

    return (
        events.groupby(["period", "threshold", "horizon_N"])
        .apply(_summarize, include_groups=False)
        .reset_index()
    )


def load_mid(day: int) -> pd.Series:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    vef = df[df["product"] == "VELVETFRUIT_EXTRACT"].sort_values("timestamp")
    return ((vef["bid_price_1"] + vef["ask_price_1"]) / 2.0).reset_index(drop=True)


def main() -> None:
    all_events: list[pd.DataFrame] = []
    for day in DAYS:
        mid = load_mid(day)
        for period in PERIODS:
            chunk = build_events(mid, period, THRESHOLDS, HORIZONS)
            if not chunk.empty:
                all_events.append(chunk)

    if not all_events:
        print("No events found.")
        return

    events_df = pd.concat(all_events, ignore_index=True)
    summary = aggregate_events(events_df)

    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    pd.set_option("display.width", 120)
    print("=" * 70)
    print("VEF Extreme Mean-Reversion EDA")
    print("Horizons pooled across days 1/2/3 | break-even: 2.5 ticks")
    print("=" * 70)
    print(summary.to_string(index=False))
    print()

    edge_count = int((summary["verdict"] == "EDGE").sum())
    if edge_count > 0:
        print(
            f"DECISION: {edge_count} EDGE cell(s) found -> "
            "high-threshold VEF signal strategy is viable."
        )
    else:
        print(
            "DECISION: No EDGE cells -> "
            "VEF cross-spread MR is uneconomic at all tested parameters."
        )


if __name__ == "__main__":
    main()
