"""
GALAXY_SOUNDS group dive (2026-04-30).

Hypothesis: No structural cointegration prior visible from product names
(BLACK_HOLES, PLANETARY_RINGS, DARK_MATTER, SOLAR_WINDS). Per-day PnL
stability separates ships from drops.

Kill criterion: EG screen empty → ship BLACK_HOLES (+ PLANETARY_RINGS
if width-tune positive); drop DARK_MATTER + SOLAR_WINDS from MEDIUM_TIER.

Phase A decision: SOLAR_FLAMES already dropped (conservative -23,644).
Current alpha state (3-day totals, default / conservative):
  BLACK_HOLES     +13,319 / +4,896   D2: -7,812  D3: +8,041  D4: +13,090
  PLANETARY_RINGS  +9,059 / +935     D2: +21,189 D3: -3,464  D4: -8,667
  DARK_MATTER      +1,825 / -5,625   (alpha-negative → drop candidate)
  SOLAR_WINDS      +1,664 / -5,974   (alpha-negative → drop candidate)
"""

import pandas as pd
from statsmodels.tsa.stattools import coint

SYMBOLS = [
    "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_SOLAR_WINDS",
]

PAIRS = [
    ("GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_PLANETARY_RINGS"),
    ("GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_DARK_MATTER"),
    ("GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_SOLAR_WINDS"),
    ("GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_DARK_MATTER"),
    ("GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS"),
    ("GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_SOLAR_WINDS"),
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
    """Run Engle-Granger cointegration screen for all 6 pairs across 3 days."""
    pvals: dict[tuple[str, str], dict[int, float]] = {}

    for day in DAYS:
        mids = load_mid_prices(day)
        for sym_a, sym_b in PAIRS:
            _, pval, _ = coint(mids[sym_a], mids[sym_b])
            pvals.setdefault((sym_a, sym_b), {})[day] = pval

    print(f"{'Pair':<36} {'Day2':>8} {'Day3':>8} {'Day4':>8} {'All<0.05':>10}")
    print("-" * 76)
    survivors = []
    for (sym_a, sym_b), day_pvals in pvals.items():
        d2 = day_pvals[2]
        d3 = day_pvals[3]
        d4 = day_pvals[4]
        passes = d2 < 0.05 and d3 < 0.05 and d4 < 0.05
        label = "PASS" if passes else "FAIL"
        a_short = sym_a.replace("GALAXY_SOUNDS_", "")
        b_short = sym_b.replace("GALAXY_SOUNDS_", "")
        pair_name = f"{a_short}<>{b_short}"
        print(f"{pair_name:<36} {d2:>8.4f} {d3:>8.4f} {d4:>8.4f} {label:>10}")
        if passes:
            survivors.append((sym_a, sym_b))

    print()
    if not survivors:
        print("KILL CRITERION FIRED: no pair passes EG screen.")
        print("Recommendation: ship BLACK_HOLES; width-tune PLANETARY_RINGS before decision.")
        print("Drop DARK_MATTER + SOLAR_WINDS (both alpha-negative under conservative test).")
    else:
        print(f"EG survivors ({len(survivors)}): {survivors}")
        print("→ proceed to lag_xcorr analysis to characterise lag-0 vs lead-lag.")

    return survivors


def lag_xcorr(sym_a: str, sym_b: str, lags: range = range(-3, 4)) -> None:
    """Compute lag-N return cross-correlation for a cointegrated pair."""
    a_short = sym_a.replace("GALAXY_SOUNDS_", "")
    b_short = sym_b.replace("GALAXY_SOUNDS_", "")
    print(f"\nLag cross-correlation: {a_short} <> {b_short}")
    print(f"{'Day':<6} " + " ".join(f"k={k:+d}" for k in lags))
    for day in DAYS:
        mids = load_mid_prices(day)
        ret_a = mids[sym_a].pct_change().dropna()
        ret_b = mids[sym_b].pct_change().dropna()
        row = []
        for k in lags:
            if k >= 0:
                c = ret_a.corr(ret_b.shift(k))
            else:
                c = ret_a.shift(-k).corr(ret_b)
            row.append(f"{c:>7.4f}")
        print(f"{day:<6} " + " ".join(row))


def planetary_rings_width_tune() -> None:
    """
    Per-day spread and adverse-selection diagnostics for PLANETARY_RINGS.

    Day 2 jackpot (+21,189) then bleed (-3,464 / -8,667) suggests:
    - Day 2: passive fills captured a one-time mispricing at width=2.
    - Days 3/4: microprice lags trending tape → adverse selection bleeds.

    Wider passive quote (width=3) demands more edge before committing
    inventory — fewer fills but less exposure on trending days.
    """
    print("\n=== PLANETARY_RINGS width-tune diagnostics ===")
    sym = "GALAXY_SOUNDS_PLANETARY_RINGS"
    per_day_pnl = {2: +21_189, 3: -3_464, 4: -8_667}
    per_day_default = {2: +21_189, 3: -3_464, 4: -8_667}

    for day in DAYS:
        mids = load_mid_prices(day)
        series = mids[sym]
        returns = series.pct_change().dropna()
        spread_est = (series.diff().dropna().abs()).mean()
        trend_strength = abs(series.iloc[-1] - series.iloc[0]) / series.std()
        acf1 = returns.autocorr(lag=1)
        print(
            f"  Day {day}: PnL={per_day_pnl[day]:>+8,}  "
            f"mean_tick_move={spread_est:.2f}  "
            f"trend_s={trend_strength:.2f}  "
            f"ACF(1)={acf1:+.4f}"
        )

    print()
    print("  Per-day pattern: D2 jackpot -> D3/D4 bleed.")
    print("  ACF(1) near 0 on D3/D4 -> trending, not mean-reverting.")
    print("  Width=3 backtest needed: run variant C below.")
    print()
    print("  Conservative alpha state (width=2 baseline): +935 (total D2+D3+D4)")
    print("  Width=3 hypothesis: D2 fills fewer but D3/D4 adverse selection reduced.")
    print("  Accept if: conservative total >= +2,000 OR width=3 per-day more consistent.")


def print_backtest_commands() -> None:
    print("\n=== Backtest commands to run in WSL2 ===")
    print("""
cd ~/prosperity_rust_backtester

# Variant A — baseline confirmation (current strategy_h.py, no changes)
rust_backtester --trader "$PROSP4/submissions/r5/strategy_h.py" --dataset round5 --persist --carry
rust_backtester --trader "$PROSP4/submissions/r5/strategy_h.py" --dataset round5 --queue-penetration 0 --price-slippage-bps 5 --persist --carry
for day in 2 3 4; do
  rust_backtester --trader "$PROSP4/submissions/r5/strategy_h.py" --dataset round5 --day $day --persist --carry
done

# Variant C — PLANETARY_RINGS width=3
# Edit strategy_h.py to add per-product block for PLANETARY_RINGS width=3
# (override medium-tier loop default width=2), then:
rust_backtester --trader "$PROSP4/submissions/r5/strategy_h.py" --dataset round5 --persist --carry
rust_backtester --trader "$PROSP4/submissions/r5/strategy_h.py" --dataset round5 --queue-penetration 0 --price-slippage-bps 5 --persist --carry
for day in 2 3 4; do
  rust_backtester --trader "$PROSP4/submissions/r5/strategy_h.py" --dataset round5 --day $day --persist --carry
done

# Variant D — drop DARK_MATTER + SOLAR_WINDS (remove from MEDIUM_TIER)
rust_backtester --trader "$PROSP4/submissions/r5/strategy_h.py" --dataset round5 --persist --carry
rust_backtester --trader "$PROSP4/submissions/r5/strategy_h.py" --dataset round5 --queue-penetration 0 --price-slippage-bps 5 --persist --carry
""")


if __name__ == "__main__":
    survivors = run_eg_screen()
    if survivors:
        for sym_a, sym_b in survivors:
            lag_xcorr(sym_a, sym_b)
    planetary_rings_width_tune()
    print_backtest_commands()
