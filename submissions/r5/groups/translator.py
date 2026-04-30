"""
TRANSLATOR group dive (2026-04-30).

Hypothesis: Color names (VOID_BLUE, ASTRO_BLACK, ECLIPSE_CHARCOAL, etc.) with no obvious
price-determining variable. Group is mostly broken — primary expectation is exhaustion.

Kill criterion: no pair has Engle-Granger p < 0.05 across all 3 days ->
ship VOID_BLUE-only base MM (already in TIGHT_TIER at width=1); drop ASTRO_BLACK + ECLIPSE_CHARCOAL.
Trigger kill criterion immediately and proceed to Variant D — do NOT iterate.

Time-box: 1 hour HARD CAP. Per spec Risk #4 — TRANSLATOR is the explicitly-named
kill-fast group. 4 of 5 products are alpha-negative already.

Phase A drops: GRAPHITE_MIST (-13,946 conservative), SPACE_GRAY (-16,719 conservative).
Current alpha state (3-day totals, default / conservative):
  VOID_BLUE      +13,251 / +4,998  <- only alpha-positive product
  ASTRO_BLACK     +5,487 / -1,497  marginal
  ECLIPSE_CHARCOAL +3,691 / -3,771  marginal
"""

import pandas as pd
from statsmodels.tsa.stattools import coint

SYMBOLS = [
    "TRANSLATOR_VOID_BLUE",
    "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL",
]

PAIRS = [
    ("TRANSLATOR_VOID_BLUE", "TRANSLATOR_ASTRO_BLACK"),
    ("TRANSLATOR_VOID_BLUE", "TRANSLATOR_ECLIPSE_CHARCOAL"),
    ("TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_ECLIPSE_CHARCOAL"),
]

DAYS = [2, 3, 4]
DATA_PATH = "datasets/round5/prices_round_5_day_{day}.csv"


def load_mid_prices(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH.format(day=day), sep=";")
    result = {}
    for sym in SYMBOLS:
        sub = df[df["product"] == sym][["timestamp", "bid_price_1", "ask_price_1"]].copy()
        sub = sub.dropna(subset=["bid_price_1", "ask_price_1"])
        sub["mid"] = (sub["bid_price_1"] + sub["ask_price_1"]) / 2
        result[sym] = sub.set_index("timestamp")["mid"]
    return pd.DataFrame(result).dropna()


def run_eg_screen() -> list[tuple[str, str]]:
    """Run Engle-Granger cointegration screen for all 3 pairs across 3 days."""
    pvals: dict[tuple[str, str], dict[int, float]] = {}

    for day in DAYS:
        mids = load_mid_prices(day)
        for sym_a, sym_b in PAIRS:
            _, pval, _ = coint(mids[sym_a], mids[sym_b])
            pvals.setdefault((sym_a, sym_b), {})[day] = pval

    print(f"{'Pair':<40} {'Day2':>8} {'Day3':>8} {'Day4':>8} {'All<0.05':>10}")
    print("-" * 80)
    survivors = []
    for (sym_a, sym_b), day_pvals in pvals.items():
        d2 = day_pvals[2]
        d3 = day_pvals[3]
        d4 = day_pvals[4]
        passes = d2 < 0.05 and d3 < 0.05 and d4 < 0.05
        label = "PASS" if passes else "FAIL"
        a_short = sym_a.replace("TRANSLATOR_", "")
        b_short = sym_b.replace("TRANSLATOR_", "")
        pair_name = f"{a_short}<>{b_short}"
        print(f"{pair_name:<40} {d2:>8.4f} {d3:>8.4f} {d4:>8.4f} {label:>10}")
        if passes:
            survivors.append((sym_a, sym_b))

    print()
    if not survivors:
        print("KILL CRITERION FIRED: no pair passes EG screen.")
        print("=> Ship VOID_BLUE only. Drop ASTRO_BLACK + ECLIPSE_CHARCOAL.")
    else:
        print(f"EG survivors ({len(survivors)}): {survivors}")
        print("=> Unexpected pass. Run lag_xcorr and Variant B pair trade before shipping.")

    return survivors


def lag_xcorr(sym_a: str, sym_b: str, lags: range = range(-3, 4)) -> None:
    """Compute lag-N return cross-correlation (run only if EG screen passes)."""
    print(f"\nLag cross-correlation: {sym_a.replace('TRANSLATOR_','')} <> {sym_b.replace('TRANSLATOR_','')}")
    print(f"{'Day':<6} " + " ".join(f"k={k:+d}" for k in lags))
    for day in DAYS:
        mids = load_mid_prices(day)
        ret_a = mids[sym_a].pct_change().dropna()
        ret_b = mids[sym_b].pct_change().dropna()
        row = []
        for k in lags:
            c = ret_a.corr(ret_b.shift(k))
            row.append(f"{c:>7.4f}")
        print(f"{day:<6} " + " ".join(row))


if __name__ == "__main__":
    survivors = run_eg_screen()
    if survivors:
        for sym_a, sym_b in survivors:
            lag_xcorr(sym_a, sym_b)
    else:
        print("\nTask 3 (lag-N matrix) and Task 4 (adverse-selection) skipped per kill criterion.")
        print("Proceed to Variant D backtest: drop ASTRO_BLACK + ECLIPSE_CHARCOAL, ship VOID_BLUE only.")
