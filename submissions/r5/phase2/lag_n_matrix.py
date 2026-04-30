"""Entry #1 — Within-group lag-N cross-correlation matrix.

For every within-group pair (10 pairs/group * 10 groups = 100 pairs),
computes Pearson correlation of mid-return series at lags k in [-5, +5].
Distinguishes lag-0 simultaneous mirror (don't overlay; microprice already
has the info — same lesson as SNACKPACK) from non-zero peak lag (overlay
candidate, analog of the shipped PEBBLES XL->XS skew).

Convention: corr(ret_a[t], ret_b[t+k]). Positive k means b leads a.

Input:  datasets/round5/prices_round_5_day_{2,3,4}.csv
Output: stdout report; submissions/r5/phase2/results/lag_n_matrix.csv

NO modification of strategy_h.py. User decides which (if any) overlay
candidates to test.
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _data import DAYS, GROUPS, RESULTS_DIR, full_symbol, load_mids

OUTPUT_CSV = RESULTS_DIR / "lag_n_matrix.csv"
LAGS = tuple(range(-5, 6))
PEAK_THRESHOLD = 0.10


def lag_corrs(sym_a: str, sym_b: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {k: [] for k in LAGS}
    for day in DAYS:
        r_a = load_mids(day, sym_a).pct_change().dropna()
        r_b = load_mids(day, sym_b).pct_change().dropna()
        idx = r_a.index.intersection(r_b.index)
        for k in LAGS:
            out[k].append(r_a.loc[idx].corr(r_b.loc[idx].shift(k)))
    return out


def peak_lag(corrs: dict[int, list[float]]) -> tuple[int, float]:
    means = {k: float(np.mean([abs(c) for c in v if not np.isnan(c)])) for k, v in corrs.items()}
    means = {k: v for k, v in means.items() if not np.isnan(v)}
    if not means:
        return 0, 0.0
    k_peak = max(means, key=means.get)
    return k_peak, means[k_peak]


def run_group(group: str) -> list[dict]:
    products = GROUPS[group]
    rows = []
    print(f"\n  Group: {group}")
    print(f"  {'Pair':<30} {'peak_k':>7} {'peak|c|':>10} {'lag0|c|':>10} {'flag':>20}")
    for a, b in combinations(products, 2):
        sym_a = full_symbol(group, a)
        sym_b = full_symbol(group, b)
        corrs = lag_corrs(sym_a, sym_b)
        k_peak, peak_abs = peak_lag(corrs)
        lag0_vals = [abs(c) for c in corrs[0] if not np.isnan(c)]
        lag0_abs = float(np.mean(lag0_vals)) if lag0_vals else 0.0
        if k_peak == 0:
            flag = "lag-0 mirror"
        elif peak_abs >= PEAK_THRESHOLD:
            flag = "OVERLAY CANDIDATE"
        else:
            flag = "noise"
        label = f"{a} <-> {b}"
        print(f"  {label:<30} {k_peak:>7} {peak_abs:>10.4f} {lag0_abs:>10.4f} {flag:>20}")
        rows.append({
            "group": group,
            "pair": label,
            "sym_a": sym_a,
            "sym_b": sym_b,
            "peak_lag": k_peak,
            "peak_abs_corr": peak_abs,
            "lag0_abs_corr": lag0_abs,
            "flag": flag,
            **{f"k_{k}_d{d}": corrs[k][i] for k in LAGS for i, d in enumerate(DAYS)},
        })
    return rows


def main() -> None:
    print("=" * 90)
    print("Entry #1 — Within-group lag-N cross-correlation matrix")
    print("=" * 90)
    print(f"  Lags: {LAGS}")
    print(f"  Threshold: peak |corr| >= {PEAK_THRESHOLD} at lag != 0 = OVERLAY CANDIDATE.")

    all_rows = []
    for group in GROUPS:
        all_rows.extend(run_group(group))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    candidates = df[df["flag"] == "OVERLAY CANDIDATE"]
    mirrors = df[df["flag"] == "lag-0 mirror"]
    print()
    print("=" * 90)
    print(f"Mirror pairs (lag-0 peak): {len(mirrors)}")
    print(f"Overlay candidates (non-zero peak, |corr| >= {PEAK_THRESHOLD}): {len(candidates)}")
    print("=" * 90)
    if len(candidates) > 0:
        print(f"  {'pair':<35} {'group':<15} {'peak_k':>7} {'peak|c|':>10}")
        for _, row in candidates.iterrows():
            print(f"  {row['pair']:<35} {row['group']:<15} {row['peak_lag']:>7} {row['peak_abs_corr']:>10.4f}")
    else:
        print("  No overlay candidates surfaced. Confirms the Phase 1 priors.")

    print()
    print("=" * 90)
    print(f"NEXT QUESTION FOR USER: {len(candidates)} overlay candidate(s). "
          "Which (if any) to test with an XL-skew-style overlay?")
    print("=" * 90)


if __name__ == "__main__":
    main()
