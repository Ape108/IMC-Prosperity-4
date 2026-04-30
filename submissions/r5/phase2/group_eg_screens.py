"""Entries #5 + #8 + #15 — Within-group EG screens for unrun groups.

Runs Engle-Granger cointegration on every within-group pair for the four
groups that did not run a formal EG screen during Phase 1 dives:
  - MICROCHIP   (10 pairs)
  - GALAXY_SOUNDS (10 pairs)
  - OXYGEN_SHAKE (10 pairs; replacement bar applies for any survivor)
  - ROBOT       (10 pairs; spread is 7.2 ticks, cost gate is binding)

For each pair, EG p-value per day and pass-all-3-days flag. For survivors,
half-life of the spread (OU regression) and spread sigma in ticks.

Input:  datasets/round5/prices_round_5_day_{2,3,4}.csv
Output: stdout report; submissions/r5/phase2/results/group_eg_screens.csv

NO modification of strategy_h.py. User decides which (if any) survivors to ship.
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from _data import DAYS, GROUPS, RESULTS_DIR, full_symbol, load_mids

OUTPUT_CSV = RESULTS_DIR / "group_eg_screens.csv"
UNRUN_GROUPS = ("MICROCHIP", "GALAXY_SOUNDS", "OXYGEN_SHAKE", "ROBOT")
P_THRESHOLD = 0.05


def eg_pvals(sym_a: str, sym_b: str) -> list[float]:
    pvals = []
    for day in DAYS:
        s_a = load_mids(day, sym_a).dropna()
        s_b = load_mids(day, sym_b).dropna()
        idx = s_a.index.intersection(s_b.index)
        _, p, _ = coint(s_a.loc[idx], s_b.loc[idx])
        pvals.append(p)
    return pvals


def half_life(spread: pd.Series) -> float:
    """OU half-life = -ln(2) / beta."""
    s = spread.dropna()
    ds = s.diff().dropna()
    s_lag = s.shift(1).dropna()
    idx = ds.index.intersection(s_lag.index)
    if len(idx) < 50:
        return float("inf")
    beta = np.polyfit(s_lag.loc[idx].values, ds.loc[idx].values, 1)[0]
    if beta >= 0:
        return float("inf")
    return -np.log(2) / beta


def spread_stats(sym_a: str, sym_b: str) -> tuple[float, float]:
    hls, sigmas = [], []
    for day in DAYS:
        s_a = load_mids(day, sym_a).dropna()
        s_b = load_mids(day, sym_b).dropna()
        idx = s_a.index.intersection(s_b.index)
        spread = s_a.loc[idx] - s_b.loc[idx]
        hls.append(half_life(spread))
        sigmas.append(spread.std())
    return float(np.median(hls)), float(np.median(sigmas))


def run_group(group: str) -> list[dict]:
    products = GROUPS[group]
    rows = []
    print(f"\n  Group: {group}")
    print(f"  {'Pair':<35} {'D2 p':>8} {'D3 p':>8} {'D4 p':>8} {'pass_all':>10}")
    for a, b in combinations(products, 2):
        sym_a = full_symbol(group, a)
        sym_b = full_symbol(group, b)
        pvals = eg_pvals(sym_a, sym_b)
        pass_all = all(p < P_THRESHOLD for p in pvals)
        label = f"{a} <-> {b}"
        flag = "  <-- PASS" if pass_all else ""
        print(
            f"  {label:<35} {pvals[0]:>8.4f} {pvals[1]:>8.4f} {pvals[2]:>8.4f} "
            f"{'YES' if pass_all else 'no':>10}{flag}"
        )
        row = {
            "group": group,
            "pair": label,
            "sym_a": sym_a,
            "sym_b": sym_b,
            "p_d2": pvals[0],
            "p_d3": pvals[1],
            "p_d4": pvals[2],
            "pass_all_3_days": pass_all,
            "half_life_ticks": None,
            "spread_sigma_ticks": None,
        }
        if pass_all:
            hl, sigma = spread_stats(sym_a, sym_b)
            row["half_life_ticks"] = hl
            row["spread_sigma_ticks"] = sigma
        rows.append(row)
    return rows


def main() -> None:
    print("=" * 90)
    print("Entries #5 + #8 + #15 — Within-group EG screens for unrun groups")
    print("=" * 90)
    print(f"  Threshold: pass = EG p < {P_THRESHOLD} on ALL 3 days.")

    all_rows = []
    for group in UNRUN_GROUPS:
        all_rows.extend(run_group(group))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    survivors = df[df["pass_all_3_days"]]
    print()
    print("=" * 90)
    print("Survivors (pass all 3 days)")
    print("=" * 90)
    if len(survivors) == 0:
        print("  None. Kill criterion fired for all 4 groups (consistent with Phase 1 priors).")
    else:
        print(f"  {len(survivors)} pair(s):")
        for _, row in survivors.iterrows():
            print(
                f"    {row['pair']:<35}  half_life={row['half_life_ticks']:.0f}  "
                f"sigma={row['spread_sigma_ticks']:.2f} ticks"
            )
        print()
        print("  Replacement-bar reminders:")
        print("    OXYGEN_SHAKE: any pair survivor must beat its two legs' existing wiring.")
        print("      MORNING<->EVENING bar = +67,582 default (per eda_gaps.md #15).")
        print("    ROBOT: spread ~7.2 ticks; spread sigma must exceed round-trip cost.")
        print("    MICROCHIP: tight tier, modest position-limit budget.")

    print()
    print("=" * 90)
    print(f"NEXT QUESTION FOR USER: {len(survivors)} pair(s) survived. "
          "Which (if any) to wire as a pair trade?")
    print("=" * 90)


if __name__ == "__main__":
    main()
