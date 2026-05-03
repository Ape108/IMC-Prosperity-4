"""
R4 Mean-Reversion Signal EDA — produces a 1-page summary covering:

1. Returns autocorr on mid (lags 1, 5, 10) for VELVETFRUIT_EXTRACT and every
   traded VEV strike, per R4 day. Decision rule: negative autocorr at any lag
   with |corr| > 0.05 across ALL 3 days = MR candidate.
2. VEV_4000 spread/volume/depth distribution. Decision rule: median spread > 1
   tick OR trades/day < 50 = skip. (We already know VEV_4000 has 128-164/day.)
3. Velvetfruit late-tape behaviour — last-20% mid trajectory per day. If there
   is a consistent directional crash, MR is the WRONG answer for fruit (would
   average down into it). The fix would be inventory-flatten-near-close.

Run with:
  .venv/Scripts/python.exe submissions/r4/eda_mr_signal.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "datasets" / "round4"
DAYS = (1, 2, 3)
PRODUCTS = (
    "VELVETFRUIT_EXTRACT",
    "VEV_4000",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
)
LAGS = (1, 5, 10)


def load_prices(day: int) -> pd.DataFrame:
    return pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")


def returns_autocorr_table() -> pd.DataFrame:
    rows = []
    for day in DAYS:
        df = load_prices(day)
        for product in PRODUCTS:
            sub = df[df["product"] == product].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("timestamp")
            mid = (sub["bid_price_1"] + sub["ask_price_1"]) / 2.0
            mid = mid.dropna()
            if len(mid) < max(LAGS) + 5:
                continue
            ret = mid.diff().dropna()
            row = {"day": day, "product": product, "n": len(ret)}
            for lag in LAGS:
                if len(ret) > lag + 5:
                    row[f"ac_{lag}"] = ret.autocorr(lag=lag)
                else:
                    row[f"ac_{lag}"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def classify_mr(autocorr_df: pd.DataFrame) -> pd.DataFrame:
    """A product qualifies as MR-candidate if at least one lag shows negative
    autocorr with |corr| > 0.05 in EVERY day."""
    out = []
    for product in PRODUCTS:
        sub = autocorr_df[autocorr_df["product"] == product]
        if len(sub) < 3:
            out.append({"product": product, "verdict": "INSUFFICIENT_DATA", "detail": ""})
            continue
        details = []
        qualifying_lag = None
        for lag in LAGS:
            col = f"ac_{lag}"
            vals = sub[col].dropna()
            if len(vals) < 3:
                continue
            negs = (vals < -0.05).all()
            if negs:
                qualifying_lag = lag
            details.append(f"lag{lag}=" + ",".join(f"{v:+.3f}" for v in vals))
        if qualifying_lag is not None:
            verdict = f"MR_CANDIDATE (lag {qualifying_lag})"
        else:
            # Check for positive consistent autocorr (trend following / persistent)
            for lag in LAGS:
                vals = sub[f"ac_{lag}"].dropna()
                if len(vals) >= 3 and (vals > 0.05).all():
                    verdict = f"PERSISTENT (lag {lag})"
                    break
            else:
                verdict = "NEUTRAL/MIXED"
        out.append({"product": product, "verdict": verdict, "detail": " | ".join(details)})
    return pd.DataFrame(out)


def vev_4000_microstructure() -> pd.DataFrame:
    rows = []
    for day in DAYS:
        df = load_prices(day)
        sub = df[df["product"] == "VEV_4000"].copy()
        if sub.empty:
            continue
        spread = (sub["ask_price_1"] - sub["bid_price_1"]).dropna()
        bid_depth = sub["bid_volume_1"].fillna(0).abs()
        ask_depth = sub["ask_volume_1"].fillna(0).abs()
        rows.append({
            "day": day,
            "ticks": len(sub),
            "spread_mean": float(spread.mean()),
            "spread_median": float(spread.median()),
            "spread_p90": float(spread.quantile(0.9)),
            "bid_depth_median": float(bid_depth.median()),
            "ask_depth_median": float(ask_depth.median()),
            "mid_min": float(((sub["bid_price_1"] + sub["ask_price_1"]) / 2.0).min()),
            "mid_max": float(((sub["bid_price_1"] + sub["ask_price_1"]) / 2.0).max()),
        })
    return pd.DataFrame(rows)


def velvetfruit_late_tape() -> pd.DataFrame:
    """Last-20% mid trajectory per day, plus full-day net move for context."""
    rows = []
    for day in DAYS:
        df = load_prices(day)
        sub = df[df["product"] == "VELVETFRUIT_EXTRACT"].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        mid = (sub["bid_price_1"] + sub["ask_price_1"]) / 2.0
        n = len(mid)
        cut = int(n * 0.8)
        first_mid = mid.iloc[0]
        cut_mid = mid.iloc[cut]
        last_mid = mid.iloc[-1]
        full_move = last_mid - first_mid
        late_move = last_mid - cut_mid
        late_max = mid.iloc[cut:].max()
        late_min = mid.iloc[cut:].min()
        rows.append({
            "day": day,
            "ticks": n,
            "mid_start": float(first_mid),
            "mid_at_80pct": float(cut_mid),
            "mid_end": float(last_mid),
            "full_move": float(full_move),
            "late_move": float(late_move),
            "late_max": float(late_max),
            "late_min": float(late_min),
            "late_range": float(late_max - late_min),
        })
    return pd.DataFrame(rows)


def main() -> None:
    np.set_printoptions(suppress=True)
    pd.set_option("display.float_format", lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v))

    print("=" * 70)
    print("TEST 1 — Returns autocorr on mid (lags 1, 5, 10) per product per day")
    print("=" * 70)
    ac_df = returns_autocorr_table()
    print(ac_df.to_string(index=False))
    print()
    print("--- Verdict per product (MR-candidate = negative autocorr |>0.05| on every day) ---")
    print(classify_mr(ac_df).to_string(index=False))
    print()

    print("=" * 70)
    print("TEST 2 — VEV_4000 microstructure (spread / depth / range)")
    print("=" * 70)
    print(vev_4000_microstructure().to_string(index=False))
    print()

    print("=" * 70)
    print("TEST 3 — VELVETFRUIT_EXTRACT late-tape behaviour (last 20% of day)")
    print("=" * 70)
    print(velvetfruit_late_tape().to_string(index=False))
    print()
    print("If late_move is consistently negative across days while full_move is")
    print("smaller or positive, neutral MM averaged down into a closing crash —")
    print("this is the partial-vs-final PnL gap mechanism, and MR is the WRONG fix.")


if __name__ == "__main__":
    main()
