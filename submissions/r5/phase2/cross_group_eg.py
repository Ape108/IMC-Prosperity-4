"""Entry #14 — Cross-group Engle-Granger cointegration screen.

Runs EG on every cross-group pair: 1,225 - 100 = 1,125 pairs total, * 3 days
= 3,375 coint() calls. Multiple-comparisons risk is real; survivors must
additionally pass the spread_sigma > 2 * round_trip_cost gate plus an
economic-plausibility check before they ship.

Includes a sanity-check pair (SNACKPACK_CHOCOLATE <-> OXYGEN_SHAKE_CHOCOLATE):
literal name collision with no actual structural relationship — should NOT
cointegrate. If it does, the test is broken.

Input:  datasets/round5/prices_round_5_day_{2,3,4}.csv
Output: stdout report; submissions/r5/phase2/results/cross_group_eg.csv

NO modification of strategy_h.py. User decides which (if any) to ship.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from _data import DAYS, GROUPS, RESULTS_DIR, full_symbol, load_mids

OUTPUT_CSV = RESULTS_DIR / "cross_group_eg.csv"
P_THRESHOLD = 0.05
SIGMA_COST_RATIO = 2.0
ROUND_TRIP_COST_TICKS = 5.0


def cross_group_pairs() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    groups = list(GROUPS.keys())
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            for a in GROUPS[groups[i]]:
                for b in GROUPS[groups[j]]:
                    out.append((full_symbol(groups[i], a), full_symbol(groups[j], b)))
    return out


def eg_pvals_and_spread(sym_a: str, sym_b: str) -> tuple[list[float], float]:
    pvals = []
    sigmas = []
    for day in DAYS:
        s_a = load_mids(day, sym_a).dropna()
        s_b = load_mids(day, sym_b).dropna()
        idx = s_a.index.intersection(s_b.index)
        if len(idx) < 100:
            pvals.append(1.0)
            sigmas.append(float("nan"))
            continue
        try:
            _, p, _ = coint(s_a.loc[idx], s_b.loc[idx])
        except Exception:
            p = 1.0
        pvals.append(p)
        sigmas.append((s_a.loc[idx] - s_b.loc[idx]).std())
    return pvals, float(np.nanmedian(sigmas))


def main() -> None:
    print("=" * 90)
    print("Entry #14 — Cross-group EG screen")
    print("=" * 90)
    pairs = cross_group_pairs()
    sanity_pair = ("SNACKPACK_CHOCOLATE", "OXYGEN_SHAKE_CHOCOLATE")
    print(f"  Total cross-group pairs: {len(pairs)} (3 days = {len(pairs)*3} coint calls).")
    print(f"  Sanity-check pair: {sanity_pair} (should NOT cointegrate).")
    print(f"  Threshold: pass_all_3_days AND spread_sigma > {SIGMA_COST_RATIO}x round_trip_cost ({ROUND_TRIP_COST_TICKS} ticks)")
    print()

    rows = []
    for n, (sym_a, sym_b) in enumerate(pairs, 1):
        if n % 100 == 0:
            print(f"  ... {n}/{len(pairs)} pairs ...")
        pvals, sigma = eg_pvals_and_spread(sym_a, sym_b)
        pass_all = all(p < P_THRESHOLD for p in pvals)
        sigma_pass = sigma > SIGMA_COST_RATIO * ROUND_TRIP_COST_TICKS
        rows.append({
            "sym_a": sym_a,
            "sym_b": sym_b,
            "p_d2": pvals[0],
            "p_d3": pvals[1],
            "p_d4": pvals[2],
            "pass_all_3_days": pass_all,
            "spread_sigma_ticks": sigma,
            "sigma_passes_cost_gate": sigma_pass,
            "ship_candidate": pass_all and sigma_pass,
        })

    df = pd.DataFrame(rows).sort_values("pass_all_3_days", ascending=False)
    df.to_csv(OUTPUT_CSV, index=False)

    sanity = df[(df["sym_a"] == sanity_pair[0]) & (df["sym_b"] == sanity_pair[1])]
    if len(sanity) == 0:
        sanity = df[(df["sym_a"] == sanity_pair[1]) & (df["sym_b"] == sanity_pair[0])]
    print()
    print("Sanity-check row:")
    if len(sanity) > 0:
        s = sanity.iloc[0]
        print(f"  {s['sym_a']} <-> {s['sym_b']}  p={s['p_d2']:.3f}/{s['p_d3']:.3f}/{s['p_d4']:.3f}  pass_all={s['pass_all_3_days']}")
        if s["pass_all_3_days"]:
            print("  WARNING: sanity pair passed cointegration. Test may be broken; investigate before trusting other survivors.")
        else:
            print("  OK: sanity pair did not cointegrate as expected.")

    survivors = df[df["pass_all_3_days"]]
    candidates = df[df["ship_candidate"]]
    print()
    print("=" * 90)
    print(f"Survivors (EG pass all 3 days): {len(survivors)} of {len(pairs)}")
    print(f"Ship candidates (also sigma > {SIGMA_COST_RATIO}x cost): {len(candidates)}")
    print("=" * 90)
    if len(candidates) > 0:
        print(f"  {'pair':<60} {'sigma':>8} {'p_d2':>7} {'p_d3':>7} {'p_d4':>7}")
        # Show top 30 by sigma
        for _, row in candidates.sort_values("spread_sigma_ticks", ascending=False).head(30).iterrows():
            label = f"{row['sym_a']} <-> {row['sym_b']}"
            print(f"  {label:<60} {row['spread_sigma_ticks']:>8.2f} "
                  f"{row['p_d2']:>7.4f} {row['p_d3']:>7.4f} {row['p_d4']:>7.4f}")
        if len(candidates) > 30:
            print(f"  ... and {len(candidates) - 30} more (full list in {OUTPUT_CSV.name}).")
    else:
        print("  No cross-group candidates survived both gates.")

    print()
    print("=" * 90)
    print(f"NEXT QUESTION FOR USER: {len(candidates)} cross-group ship candidate(s). "
          "Which (if any) to wire as a pair trade?")
    print("  Reminder: cross-group survivors carry multiple-comparisons risk;")
    print("  apply economic-plausibility judgment before shipping.")
    print("=" * 90)


if __name__ == "__main__":
    main()
