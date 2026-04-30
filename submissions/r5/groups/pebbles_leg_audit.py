"""PEBBLES_M <-> XL leg decomposition (eda_gaps #11).

Classifies the M leg as σ-scaled (Hypothesis A — both legs do equal information work)
or structural (Hypothesis B — M is mostly cointegration anchor; XL drives all alpha).

Streamlined from the original plan: instead of parsing per-fill round trips from
submission.log (which would require running --artifact-mode submission, locating
the log path, and writing a parser), this uses (a) per-day PnL already in the
backtest output and (b) per-day sigma from the dataset price CSVs. The classification
is the same in either case -- the sigma-normalized dollar-PnL ratio per leg per day.

Decision rule:
  sigma-normalized ratio (XL/M) ~ 1   -> A: equal work, keep M at 10/10
  sigma-normalized ratio (XL/M) >> 1  -> B: XL drives, M can be sized down

Inputs:
  datasets/round5/prices_round_5_day_{2,3,4}.csv (semicolon-delimited)
  Per-day pair-trade PnL (hardcoded from backtest output below).

Outputs (printed):
  Per-day sigma for M and XL.
  Per-day dollar PnL ratio XL/M.
  Per-day sigma-normalized ratio.
  Classification + recommendation.
"""

from pathlib import Path

import pandas as pd

# Per-day per-product pair-trade PnL from the post-Phase-A backtest output (2026-04-30).
# Source: rust_backtester --trader strategy_h.py --dataset round5 (default and conservative runs).
PNL_DEFAULT = {
    "PEBBLES_M":  {2: 5088.0,  3: 2778.0,   4: 10381.0},
    "PEBBLES_XL": {2: 5390.0,  3: 41505.0,  4: 54160.0},
}
PNL_CONSERVATIVE = {
    "PEBBLES_M":  {2: -1522.0, 3: -4032.0,  4: 2275.0},
    "PEBBLES_XL": {2: -2340.0, 3: 32405.0,  4: 43010.0},
}

DATASET_DIR = Path(__file__).resolve().parents[3] / "datasets" / "round5"


def load_mids(day: int, product: str) -> pd.Series:
    """Return mid_price series for a given product/day, indexed by timestamp."""
    path = DATASET_DIR / f"prices_round_5_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    return df[df["product"] == product].set_index("timestamp")["mid_price"].sort_index()


def per_day_sigma(product: str) -> dict[int, float]:
    """sigma of mid_price returns per day (pct_change().std())."""
    out = {}
    for day in (2, 3, 4):
        mids = load_mids(day, product)
        out[day] = mids.pct_change().std()
    return out


def main() -> None:
    sigma_m = per_day_sigma("PEBBLES_M")
    sigma_xl = per_day_sigma("PEBBLES_XL")

    print("=" * 80)
    print("PEBBLES_M <-> XL leg decomposition (eda_gaps #11)")
    print("=" * 80)

    print("\nPer-day sigma (return std of mid_price):")
    print(f"  {'Day':>5} {'sigma_M':>12} {'sigma_XL':>12} {'XL/M':>12}")
    sigma_ratios = {}
    for d in (2, 3, 4):
        ratio = sigma_xl[d] / sigma_m[d]
        sigma_ratios[d] = ratio
        print(f"  {d:>5} {sigma_m[d]:>12.6f} {sigma_xl[d]:>12.6f} {ratio:>12.3f}")

    for label, pnl in (("DEFAULT", PNL_DEFAULT), ("CONSERVATIVE", PNL_CONSERVATIVE)):
        print(f"\n{label} backtest:")
        print(f"  {'Day':>5} {'M PnL':>10} {'XL PnL':>10} {'$XL/$M':>10} {'sig ratio':>10} {'adj':>10}")
        # sigma-normalized ratio: divides the dollar ratio by the sigma ratio.
        # If equal information work, $ratio ~ sigma ratio (because both legs at size 10
        # with returns ~ N standard deviations realize PnL proportional to size * sigma * price).
        # adj > 1 means XL outperforms even after volatility adjustment -> structural.
        ratios = []
        for d in (2, 3, 4):
            m = pnl["PEBBLES_M"][d]
            xl = pnl["PEBBLES_XL"][d]
            # Use abs to make magnitudes comparable when one leg has small/sign-flipped PnL
            dollar_ratio = abs(xl) / abs(m) if m != 0 else float("inf")
            sigma_adj = dollar_ratio / sigma_ratios[d]
            ratios.append(sigma_adj)
            print(f"  {d:>5} {m:>10.0f} {xl:>10.0f} {dollar_ratio:>10.2f} {sigma_ratios[d]:>10.3f} {sigma_adj:>10.2f}")
        median_adj = sorted(ratios)[1]
        print(f"  Median sigma-adjusted ratio: {median_adj:.2f}")
        if 0.5 <= median_adj <= 2.0:
            classification = "A (sigma-scaled -- both legs do equal information work)"
        elif median_adj > 3.0:
            classification = "B (structural -- XL drives price discovery; M is anchor)"
        else:
            classification = "ambiguous"
        print(f"  Classification: {classification}")

    print("\n" + "=" * 80)
    print("Recommendation:")
    print("=" * 80)
    print("""
If both DEFAULT and CONSERVATIVE classify as A:
    Keep M at 10/10. Both legs do equal information work; sizing down M
    would lose spread coverage 1:1 with the size reduction. Freed position
    units for new pair trades = 0.

If both classify as B:
    Resize M to 5/10 (or smaller). XL drives all alpha; M is mostly a
    cointegration anchor. Halving M frees 5 position units for ONE
    additional pair trade in C.1 (PANEL) or C.2 (UV_VISOR).

If they disagree (one A, one B):
    The conservative classification is the truth-test (alpha-only). If
    DEFAULT is A but CONSERVATIVE is B, the M leg's queue value masks
    its lack of structural alpha — recommend resize anyway.

If either is ambiguous:
    Keep M at 10/10 conservatively. Revisit after C.1 / C.2 reveal
    whether the freed-headroom would matter.
""")


if __name__ == "__main__":
    main()
