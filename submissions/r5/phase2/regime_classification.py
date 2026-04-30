"""Entry #3 — Per-day regime classification.

For each of days 2/3/4 and each product, computes:
  - trend_score = abs(cumulative log return)
  - vol_score = realized variance of returns
  - drawdown_score = max drawdown of cumulative log mid
  - acf_score = lag-1 autocorrelation of returns

Surfaces products whose per-day PnL pattern correlates strongly with one
of these regime axes (e.g., SNACKPACK_VANILLA / STRAWBERRY lose only on
trending days). For products with regime-explained per-day variance >= 30%,
flags as candidates for regime gating.

Per-day PnL is read from baseline_pnl.csv (Task 2 output).

Input:  datasets/round5/prices_round_5_day_{2,3,4}.csv
        submissions/r5/phase2/results/baseline_pnl.csv
Output: stdout report; submissions/r5/phase2/results/regime_classification.csv

NO modification of strategy_h.py. User decides which (if any) regime gates to add.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _data import DAYS, GROUPS, RESULTS_DIR, full_symbol, load_mids

INPUT_CSV = RESULTS_DIR / "baseline_pnl.csv"
OUTPUT_CSV = RESULTS_DIR / "regime_classification.csv"
R2_THRESHOLD = 0.30


def regime_metrics(symbol: str, day: int) -> dict[str, float]:
    mids = load_mids(day, symbol).dropna()
    if len(mids) < 100:
        return {"trend": float("nan"), "vol": float("nan"), "drawdown": float("nan"), "acf": float("nan")}
    rets = mids.pct_change().dropna()
    log_rets = np.log(mids / mids.iloc[0])
    cum_max = log_rets.cummax()
    drawdown = float((cum_max - log_rets).max())
    return {
        "trend": float(abs(log_rets.iloc[-1])),
        "vol": float(rets.std()),
        "drawdown": drawdown,
        "acf": float(rets.autocorr(lag=1)),
    }


def per_product_per_day_pnl(baseline: pd.DataFrame) -> dict[tuple[str, int], float]:
    out = {}
    for _, row in baseline.iterrows():
        for d in DAYS:
            out[(row["product"], d)] = row.get(f"default_d{d}", float("nan"))
    return out


def run_product(symbol: str, pnl_lookup: dict[tuple[str, int], float]) -> dict | None:
    pnls = []
    metrics_by_day = {}
    for d in DAYS:
        pnl = pnl_lookup.get((symbol, d), float("nan"))
        if pd.isna(pnl):
            return None
        pnls.append(pnl)
        metrics_by_day[d] = regime_metrics(symbol, d)

    row = {"symbol": symbol}
    for axis in ("trend", "vol", "drawdown", "acf"):
        xs = np.array([metrics_by_day[d][axis] for d in DAYS])
        ys = np.array(pnls)
        if np.any(np.isnan(xs)) or np.std(xs) == 0:
            row[f"r2_{axis}"] = float("nan")
            row[f"beta_{axis}"] = float("nan")
            continue
        beta = np.polyfit(xs, ys, 1)[0]
        pred = beta * xs + (ys.mean() - beta * xs.mean())
        ss_res = float(np.sum((ys - pred) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        row[f"r2_{axis}"] = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        row[f"beta_{axis}"] = beta
        row[f"x_{axis}_d2"] = xs[0]
        row[f"x_{axis}_d3"] = xs[1]
        row[f"x_{axis}_d4"] = xs[2]
    row["pnl_d2"] = pnls[0]
    row["pnl_d3"] = pnls[1]
    row["pnl_d4"] = pnls[2]
    valid_r2 = [row[f"r2_{a}"] for a in ("trend", "vol", "drawdown", "acf") if not pd.isna(row[f"r2_{a}"])]
    row["max_r2"] = max(valid_r2) if valid_r2 else float("nan")
    return row


def main() -> None:
    print("=" * 90)
    print("Entry #3 — Per-day regime classification")
    print("=" * 90)
    baseline = pd.read_csv(INPUT_CSV)
    pnl_lookup = per_product_per_day_pnl(baseline)
    print(f"  Products with baseline PnL: {len(baseline)}")
    print(f"  Threshold for regime-gate candidate: max R^2 across axes >= {R2_THRESHOLD}")
    print()

    rows = []
    for group, products in GROUPS.items():
        for product in products:
            symbol = full_symbol(group, product)
            row = run_product(symbol, pnl_lookup)
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows).sort_values("max_r2", ascending=False)
    df.to_csv(OUTPUT_CSV, index=False)

    candidates = df[df["max_r2"] >= R2_THRESHOLD]
    print(f"  {'symbol':<40} {'r2_trend':>9} {'r2_vol':>9} {'r2_dd':>9} {'r2_acf':>9} {'max_r2':>9}")
    for _, row in df.iterrows():
        flag = "  <-- CANDIDATE" if row["max_r2"] >= R2_THRESHOLD else ""
        r2_trend = row["r2_trend"] if not pd.isna(row["r2_trend"]) else float("nan")
        r2_vol = row["r2_vol"] if not pd.isna(row["r2_vol"]) else float("nan")
        r2_dd = row["r2_drawdown"] if not pd.isna(row["r2_drawdown"]) else float("nan")
        r2_acf = row["r2_acf"] if not pd.isna(row["r2_acf"]) else float("nan")
        print(f"  {row['symbol']:<40} {r2_trend:>9.3f} {r2_vol:>9.3f} "
              f"{r2_dd:>9.3f} {r2_acf:>9.3f} {row['max_r2']:>9.3f}{flag}")

    print()
    print("=" * 90)
    print(f"Regime-gate candidates (max R^2 >= {R2_THRESHOLD}): {len(candidates)}")
    print("=" * 90)
    print("  Note: with only 3 days, R^2 is noisy. A high R^2 with consistent")
    print("  per-day sign on a known-fragile product (e.g., VANILLA Day-3 loss)")
    print("  is more actionable than a high R^2 on a stable product.")

    print()
    print("=" * 90)
    print(f"NEXT QUESTION FOR USER: {len(candidates)} regime-gate candidate(s). "
          "Which (if any) to wire as a per-day filter in strategy_h.py?")
    print("=" * 90)


if __name__ == "__main__":
    main()
