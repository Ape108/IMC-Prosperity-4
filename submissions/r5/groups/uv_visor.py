"""
UV_VISOR group dive (2026-04-30).

Hypothesis: spectrum ordering YELLOW→AMBER→ORANGE→RED→MAGENTA mirrors PEBBLES
size ordering XS→S→M→L→XL. PEBBLES analog paid +101k from one pair trade leg.

Kill criterion: no pair has Engle-Granger p < 0.05 across all 3 days →
ship AMBER-only base MM (already in MEDIUM_TIER at width=2); drop ORANGE/RED/YELLOW.

Phase A decision: MAGENTA already dropped (conservative PnL -22,796).
Current alpha state (3-day totals, default / conservative):
  AMBER  +13,754 / +7,878  ← only alpha-positive UV_VISOR product
  ORANGE  +7,103 / -526    marginal
  RED     +4,163 / -3,893  marginal
  YELLOW  +2,696 / -5,625  marginal
"""

import pandas as pd
from statsmodels.tsa.stattools import coint

SYMBOLS = [
    "UV_VISOR_YELLOW",
    "UV_VISOR_AMBER",
    "UV_VISOR_ORANGE",
    "UV_VISOR_RED",
]

PAIRS = [
    ("UV_VISOR_AMBER", "UV_VISOR_ORANGE"),
    ("UV_VISOR_AMBER", "UV_VISOR_RED"),
    ("UV_VISOR_AMBER", "UV_VISOR_YELLOW"),
    ("UV_VISOR_ORANGE", "UV_VISOR_RED"),
    ("UV_VISOR_ORANGE", "UV_VISOR_YELLOW"),
    ("UV_VISOR_RED", "UV_VISOR_YELLOW"),
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


def run_eg_screen() -> None:
    """Run Engle-Granger cointegration screen for all 6 pairs across 3 days."""
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
        pair_name = f"{sym_a.split('_')[-1]}<>{sym_b.split('_')[-1]}"
        print(f"{pair_name:<40} {d2:>8.4f} {d3:>8.4f} {d4:>8.4f} {label:>10}")
        if passes:
            survivors.append((sym_a, sym_b))

    print()
    if not survivors:
        print("KILL CRITERION FIRED: no pair passes EG screen. Ship AMBER-only.")
    else:
        print(f"EG survivors ({len(survivors)}): {survivors}")

    return survivors


def lag_xcorr(sym_a: str, sym_b: str, lags: range = range(-3, 4)) -> None:
    """Compute lag-N return cross-correlation for a cointegrated pair."""
    print(f"\nLag cross-correlation: {sym_a.split('_')[-1]} <> {sym_b.split('_')[-1]}")
    print(f"{'Day':<6} " + " ".join(f"k={k:+d}" for k in lags))
    for day in DAYS:
        mids = load_mid_prices(day)
        ret_a = mids[sym_a].pct_change().dropna()
        ret_b = mids[sym_b].pct_change().dropna()
        row = []
        for k in lags:
            if k == 0:
                c = ret_a.corr(ret_b)
            elif k > 0:
                c = ret_a.corr(ret_b.shift(k))
            else:
                c = ret_a.corr(ret_b.shift(k))
            row.append(f"{c:>7.4f}")
        print(f"{day:<6} " + " ".join(row))


if __name__ == "__main__":
    survivors = run_eg_screen()
    if survivors:
        for sym_a, sym_b in survivors:
            lag_xcorr(sym_a, sym_b)
