"""VEF Day 3 Diagnosis -- price-path analysis across R4 days 1/2/3.

Loads price data and computes per-day stats for VELVETFRUIT_EXTRACT:
  - net daily move (mid[last] - mid[first])
  - max intraday excursion from open
  - late-session (last 20%) direction relative to full-day trend

Decision gate: if day 3 |net_move| > 2x avg of days 1/2 AND late-session
continues the trend (same sign) -> inventory accumulation hypothesis confirmed.

Run (WSL2):
    cd ~/prosperity_rust_backtester
    python "$PROSP4/submissions/r4/eda_vef_day3.py"
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

PRODUCT = "VELVETFRUIT_EXTRACT"
LATE_FRACTION = 0.80  # last 20% of session


# -- Helpers ------------------------------------------------------------------


def compute_daily_stats(prices: pd.DataFrame, product: str) -> list[dict]:
    """Compute per-day price-path statistics for `product`.

    Returns a list of dicts (one per day) with keys:
        day, open_mid, close_mid, net_move, abs_net_move,
        max_excursion, late_move, late_same_dir
    Sorted by day ascending.
    """
    sub = prices[prices["product"] == product].copy()
    if sub.empty:
        return []

    results: list[dict] = []
    for day in sorted(sub["day"].unique()):
        day_df = sub[sub["day"] == day].sort_values("timestamp").reset_index(drop=True)
        if len(day_df) < 2:
            continue

        ts = day_df["timestamp"].to_numpy()
        mids = day_df["mid_price"].to_numpy()

        open_mid = float(mids[0])
        close_mid = float(mids[-1])
        net_move = close_mid - open_mid
        max_excursion = float(np.max(np.abs(mids - open_mid)))

        t_start = float(ts[0])
        t_end = float(ts[-1])
        t_late = t_start + LATE_FRACTION * (t_end - t_start)
        late_mask = ts >= t_late
        late_prices = mids[late_mask]
        if len(late_prices) >= 2:
            late_open = float(late_prices[0])
            late_move = close_mid - late_open
        else:
            late_move = 0.0

        # late_same_dir: the late-session move continues the full-day trend
        late_same_dir = bool(net_move * late_move > 0)

        results.append({
            "day": int(day),
            "open_mid": open_mid,
            "close_mid": close_mid,
            "net_move": net_move,
            "abs_net_move": abs(net_move),
            "max_excursion": max_excursion,
            "late_move": late_move,
            "late_same_dir": late_same_dir,
        })

    return results


def apply_gate(stats: list[dict]) -> bool:
    """Return True if day 3 confirms the inventory-accumulation hypothesis.

    Gate: day 3 |net_move| > 2 x mean(|net_move| for days 1/2)
          AND day 3 late_same_dir is True.
    """
    day3 = next((s for s in stats if s["day"] == 3), None)
    others = [s for s in stats if s["day"] in (1, 2)]
    if day3 is None or not others:
        return False
    avg_abs = sum(s["abs_net_move"] for s in others) / len(others)
    return day3["abs_net_move"] > 2 * avg_abs and day3["late_same_dir"]


# -- Loader -------------------------------------------------------------------


def load_prices(days: list[int] | None = None) -> pd.DataFrame:
    if days is None:
        days = R4_DAYS
    dfs: list[pd.DataFrame] = []
    for day in days:
        path = f"{DATASET_DIR}/prices_round_4_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# -- Main ---------------------------------------------------------------------


def main() -> None:
    prices = load_prices()
    stats = compute_daily_stats(prices, PRODUCT)

    print(f"\n=== VEF Day 3 Diagnosis -- {PRODUCT} ===\n")
    header = f"{'day':>4}  {'open':>8}  {'close':>8}  {'net':>8}  {'|net|':>7}  {'max_exc':>8}  {'late_mv':>8}  {'late_dir':>10}"
    print(header)
    print("-" * len(header))
    for s in stats:
        d = s["late_same_dir"]
        print(
            f"{s['day']:>4}  "
            f"{s['open_mid']:>8.2f}  "
            f"{s['close_mid']:>8.2f}  "
            f"{s['net_move']:>+8.2f}  "
            f"{s['abs_net_move']:>7.2f}  "
            f"{s['max_excursion']:>8.2f}  "
            f"{s['late_move']:>+8.2f}  "
            f"{'CONTINUES' if d else 'REVERTS':>10}"
        )

    others = [s for s in stats if s["day"] != 3]
    if others:
        avg_abs = sum(s["abs_net_move"] for s in others) / len(others)
        print(f"\nDays 1/2 avg |net_move|: {avg_abs:.2f}")
        print(f"2x threshold:            {2 * avg_abs:.2f}")

    confirmed = apply_gate(stats)
    print(f"\n=== VERDICT ===")
    if confirmed:
        print("CONFIRMED -- day 3 trend is large and persistent. Implement EOD flatten.")
    else:
        print("NOT CONFIRMED -- day 3 variance is data-specific. No code change needed.")


if __name__ == "__main__":
    main()
