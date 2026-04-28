"""
VEF x IV Gated EDA.

Tests three angles:
  A: ATM IV at session open vs VEF daily PnL (regime gate hypothesis).
  B: Smile skew (right-wing minus left-wing) per day - expect static.
  C: IV residual spike event study.

Run with:
  .venv/Scripts/python.exe submissions/r4/eda_vef_iv.py

Spec: docs/superpowers/specs/2026-04-28-vef-iv-gated-design.md
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "datasets" / "round4"
DAYS = (1, 2, 3)

# Replicated from strategy_h.py - no strategy import needed in EDA
ROUND_START_TTE_DAYS = 4.0
TICKS_PER_DAY = 1_000_000
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEF_SYMBOL = "VELVETFRUIT_EXTRACT"

# Backtester results (neutral MM baseline; not recomputed here)
VEF_PNL: dict[int, int] = {1: 7482, 2: 9554, 3: -4386}

# Decision thresholds
ANGLE_A_IV_DELTA = 0.02
ANGLE_B_SKEW_DELTA = 0.005
ANGLE_C_MIN_EVENTS = 5
ANGLE_C_MOVE_THRESHOLD = 2.5
ANGLE_C_HIT_RATE_PCT = 50.0
ANGLE_C_THRESHOLDS = (0.010, 0.015, 0.020)
ANGLE_C_HORIZONS = (5, 10, 20, 50)


# -- Black-Scholes math (replicated from strategy_h.py) -----------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def _vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi)
    return S * sqrt_T * pdf_d1


def _implied_vol(
    S: float, K: float, T: float, market_price: float, max_iter: int = 100
) -> float | None:
    intrinsic = max(0.0, S - K)
    if market_price <= intrinsic + 1e-6:
        return None
    sigma = 0.5
    for _ in range(max_iter):
        price = _bs_call_price(S, K, T, sigma)
        diff = price - market_price
        if abs(diff) < 1e-6:
            return sigma
        v = _vega(S, K, T, sigma)
        if v < 1e-10:
            return None
        sigma -= diff / v
        if sigma <= 0:
            sigma = 1e-6
    return sigma


# -- Data loading -------------------------------------------------------------

def load_prices(day: int) -> pd.DataFrame:
    return pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")


# -- Core IV helpers ---------------------------------------------------------

def find_atm_strike(spot: float) -> int:
    return min(STRIKES, key=lambda s: abs(s - spot))


def compute_iv_for_strike(
    spot: float, strike: int, tte_years: float, bid: float, ask: float
) -> float | None:
    mid = (bid + ask) / 2.0
    intrinsic = max(0.0, spot - strike)
    if mid <= intrinsic + 0.5:
        return None
    return _implied_vol(spot, float(strike), tte_years, mid)


def get_strike_mids_at_ts(df: pd.DataFrame, ts: int, spot: float) -> dict[int, float]:
    result: dict[int, float] = {}
    for strike in STRIKES:
        rows = df[(df["product"] == f"VEV_{strike}") & (df["timestamp"] == ts)]
        if rows.empty:
            continue
        r = rows.iloc[0]
        bid, ask = r["bid_price_1"], r["ask_price_1"]
        if pd.isna(bid) or pd.isna(ask):
            continue
        mid = (float(bid) + float(ask)) / 2.0
        intrinsic = max(0.0, spot - strike)
        if mid <= intrinsic + 0.5:
            continue
        result[strike] = mid
    return result


def get_session_open_iv(df: pd.DataFrame) -> dict | None:
    vef = df[df["product"] == VEF_SYMBOL].sort_values("timestamp")
    for _, vrow in vef.iterrows():
        ts = int(vrow["timestamp"])
        bid_v, ask_v = vrow["bid_price_1"], vrow["ask_price_1"]
        if pd.isna(bid_v) or pd.isna(ask_v):
            continue
        spot = (float(bid_v) + float(ask_v)) / 2.0
        tte_years = max(ROUND_START_TTE_DAYS - ts / TICKS_PER_DAY, 0.001) / 365.0
        atm = find_atm_strike(spot)
        vev_rows = df[(df["product"] == f"VEV_{atm}") & (df["timestamp"] == ts)]
        if vev_rows.empty:
            continue
        vr = vev_rows.iloc[0]
        if pd.isna(vr["bid_price_1"]) or pd.isna(vr["ask_price_1"]):
            continue
        iv = compute_iv_for_strike(
            spot,
            atm,
            tte_years,
            float(vr["bid_price_1"]),
            float(vr["ask_price_1"]),
        )
        if iv is not None:
            return {"ts": ts, "spot": spot, "atm_strike": atm, "iv": iv}
    return None


def _angle_a_verdict(rows: list[dict]) -> str:
    ivs = {r["day"]: r.get("atm_iv_open") for r in rows}
    if ivs.get(1) is None or ivs.get(2) is None or ivs.get(3) is None:
        return "NO SIGNAL (missing IV data)"
    baseline = (ivs[1] + ivs[2]) / 2.0
    return "SIGNAL CONFIRMED" if ivs[3] > baseline + ANGLE_A_IV_DELTA else "NO SIGNAL"


def run_angle_a(days: tuple[int, ...] = DAYS) -> list[dict]:
    rows: list[dict] = []
    for day in days:
        df = load_prices(day)
        result = get_session_open_iv(df)
        rows.append(
            {
                "day": day,
                "spot_open": result["spot"] if result else None,
                "atm_strike": result["atm_strike"] if result else None,
                "atm_iv_open": result["iv"] if result else None,
                "vef_pnl": VEF_PNL[day],
            }
        )
    return rows


# -- Angle B: smile skew -----------------------------------------------------

def fit_smile_from_mids(
    spot: float, tte_years: float, strike_mids: dict[int, float]
) -> np.ndarray | None:
    moneynesses: list[float] = []
    ivs: list[float] = []
    for strike, mid in strike_mids.items():
        iv = _implied_vol(spot, float(strike), tte_years, mid)
        if iv is None:
            intrinsic = max(0.0, spot - strike)
            if mid <= intrinsic + 1e-6:
                iv = 0.0
        if iv is None:
            continue
        moneynesses.append(strike / spot)
        ivs.append(iv)
    if len(moneynesses) < 3:
        return None
    return np.polyfit(moneynesses, ivs, 2)


def compute_smile_skew(
    spot: float, tte_years: float, strike_mids: dict[int, float]
) -> float | None:
    coeffs = fit_smile_from_mids(spot, tte_years, strike_mids)
    if coeffs is None:
        return None
    return float(np.polyval(coeffs, 1.02)) - float(np.polyval(coeffs, 0.98))


def _angle_b_verdict(rows: list[dict]) -> str:
    skews = {r["day"]: r.get("mean_skew") for r in rows}
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in skews.values()):
        return "STATIC (missing data)"
    baseline = (skews[1] + skews[2]) / 2.0
    return "STATIC" if abs(skews[3] - baseline) < ANGLE_B_SKEW_DELTA else "MOVING"


def run_angle_b(days: tuple[int, ...] = DAYS) -> list[dict]:
    rows: list[dict] = []
    for day in days:
        df = load_prices(day)
        vef = df[df["product"] == VEF_SYMBOL].sort_values("timestamp")
        skews: list[float] = []
        for _, vrow in vef.iterrows():
            ts = int(vrow["timestamp"])
            bid_v, ask_v = vrow["bid_price_1"], vrow["ask_price_1"]
            if pd.isna(bid_v) or pd.isna(ask_v):
                continue
            spot = (float(bid_v) + float(ask_v)) / 2.0
            tte_years = max(ROUND_START_TTE_DAYS - ts / TICKS_PER_DAY, 0.001) / 365.0
            mids = get_strike_mids_at_ts(df, ts, spot)
            skew = compute_smile_skew(spot, tte_years, mids)
            if skew is not None:
                skews.append(skew)
        rows.append(
            {
                "day": day,
                "mean_skew": float(np.mean(skews)) if skews else float("nan"),
                "n_ticks": len(skews),
            }
        )
    return rows


# -- Angle C: IV residual spike event study ---------------------------------

def compute_iv_residual_series(df: pd.DataFrame, day: int) -> pd.DataFrame:
    vef = df[df["product"] == VEF_SYMBOL].sort_values("timestamp")
    rows: list[dict] = []
    for _, vrow in vef.iterrows():
        ts = int(vrow["timestamp"])
        bid_v, ask_v = vrow["bid_price_1"], vrow["ask_price_1"]
        if pd.isna(bid_v) or pd.isna(ask_v):
            continue
        spot = (float(bid_v) + float(ask_v)) / 2.0
        tte_years = max(ROUND_START_TTE_DAYS - ts / TICKS_PER_DAY, 0.001) / 365.0
        mids = get_strike_mids_at_ts(df, ts, spot)
        atm = find_atm_strike(spot)
        if atm not in mids:
            continue
        coeffs = fit_smile_from_mids(spot, tte_years, mids)
        if coeffs is None:
            continue
        atm_iv = _implied_vol(spot, float(atm), tte_years, mids[atm])
        if atm_iv is None:
            continue
        fitted = float(np.polyval(coeffs, atm / spot))
        rows.append(
            {
                "ts": ts,
                "atm_iv": atm_iv,
                "fitted_atm_iv": fitted,
                "residual": atm_iv - fitted,
            }
        )
    return pd.DataFrame(rows)


def build_iv_spike_events(
    residuals: pd.Series,
    vef_mid: pd.Series,
    thresholds: tuple[float, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict] = []
    n = len(residuals)
    delta = residuals.diff()
    for i in range(1, n):
        dr = delta.iloc[i]
        if pd.isna(dr) or dr == 0.0:
            continue
        abs_dr, sign_dr = abs(dr), (1 if dr > 0 else -1)
        for threshold in thresholds:
            if abs_dr <= threshold:
                continue
            for N in horizons:
                if i + N >= n:
                    continue
                signed_move = (vef_mid.iloc[i + N] - vef_mid.iloc[i]) * sign_dr
                rows.append(
                    {
                        "threshold": threshold,
                        "horizon_N": N,
                        "signed_move": float(signed_move),
                        "hit_2_5": bool(signed_move > ANGLE_C_MOVE_THRESHOLD),
                    }
                )
    return pd.DataFrame(rows)


def _angle_c_verdict(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "NO EDGE"
    edge = summary[
        (summary["n"] >= ANGLE_C_MIN_EVENTS)
        & (summary["mean_move"] > ANGLE_C_MOVE_THRESHOLD)
        & (summary["hit_rate_pct"] >= ANGLE_C_HIT_RATE_PCT)
    ]
    return "EDGE" if not edge.empty else "NO EDGE"


def run_angle_c(days: tuple[int, ...] = DAYS) -> pd.DataFrame:
    all_events: list[pd.DataFrame] = []
    for day in days:
        df = load_prices(day)
        residual_df = compute_iv_residual_series(df, day)
        if residual_df.empty:
            continue
        vef = df[df["product"] == VEF_SYMBOL].sort_values("timestamp")
        vef_mid_map: dict[int, float] = {
            int(r["timestamp"]): (float(r["bid_price_1"]) + float(r["ask_price_1"])) / 2.0
            for _, r in vef.iterrows()
            if not (pd.isna(r["bid_price_1"]) or pd.isna(r["ask_price_1"]))
        }
        vef_mid = pd.Series(
            [vef_mid_map.get(ts, float("nan")) for ts in residual_df["ts"].values]
        )
        valid = vef_mid.notna()
        events = build_iv_spike_events(
            residual_df["residual"].reset_index(drop=True)[valid].reset_index(drop=True),
            vef_mid[valid].reset_index(drop=True),
            ANGLE_C_THRESHOLDS,
            ANGLE_C_HORIZONS,
        )
        if not events.empty:
            all_events.append(events)
    if not all_events:
        return pd.DataFrame()
    combined = pd.concat(all_events, ignore_index=True)

    def _summarize(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        return pd.Series(
            {
                "n": n,
                "mean_move": float(g["signed_move"].mean()),
                "hit_rate_pct": 100.0 * float(g["hit_2_5"].mean()),
            }
        )

    return (
        combined.groupby(["threshold", "horizon_N"])
        .apply(_summarize, include_groups=False)
        .reset_index()
    )


# -- Verdict + main ----------------------------------------------------------

def write_verdict(
    angle_a_rows: list[dict],
    angle_b_rows: list[dict],
    angle_c_df: pd.DataFrame,
    path: Path,
) -> None:
    lines: list[str] = ["=" * 70, "VEF x IV EDA VERDICT", "=" * 70, ""]

    lines += ["-- ANGLE A: ATM IV at session open vs VEF PnL --", ""]
    lines.append(
        f"{'day':>4} {'spot_open':>12} {'atm_strike':>12} {'atm_iv_open':>14} {'vef_pnl':>10}"
    )
    for r in angle_a_rows:
        iv_str = f"{r['atm_iv_open']:.4f}" if r["atm_iv_open"] is not None else "    None"
        lines.append(
            f"{r['day']:>4} {(r['spot_open'] or 0):>12.1f} {(r['atm_strike'] or 0):>12d}"
            f" {iv_str:>14} {r['vef_pnl']:>10,}"
        )
    lines.append(f"\nVERDICT: {_angle_a_verdict(angle_a_rows)}\n")

    lines += ["-- ANGLE B: IV smile skew per day --", ""]
    lines.append(f"{'day':>4} {'mean_skew':>12} {'n_ticks':>10}")
    for r in angle_b_rows:
        lines.append(f"{r['day']:>4} {r['mean_skew']:>12.5f} {r['n_ticks']:>10d}")
    lines.append(f"\nVERDICT: {_angle_b_verdict(angle_b_rows)}\n")

    lines += ["-- ANGLE C: IV residual spikes -> VEF direction --", ""]
    if angle_c_df.empty:
        lines.append("No events.")
    else:
        lines.append(angle_c_df.to_string(index=False))
    lines.append(f"\nVERDICT: {_angle_c_verdict(angle_c_df)}\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Verdict written to {path}")


def main() -> None:
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 120)

    print("Running angle A (session-open ATM IV)...")
    angle_a = run_angle_a()
    a_verdict = _angle_a_verdict(angle_a)
    print(f"  {a_verdict}")

    print("Running angle B (per-tick smile fits - may be slow)...")
    angle_b = run_angle_b()
    b_verdict = _angle_b_verdict(angle_b)
    print(f"  {b_verdict}")

    if a_verdict == "SIGNAL CONFIRMED":
        print("Running angle C (gated on A)...")
        angle_c = run_angle_c()
        c_verdict = _angle_c_verdict(angle_c)
        print(f"  {c_verdict}")
    else:
        angle_c = pd.DataFrame()
        print("Skipping angle C (angle A returned NO SIGNAL).")

    verdict_path = Path(__file__).parent / "vef_iv_verdict.txt"
    write_verdict(angle_a, angle_b, angle_c, verdict_path)

    print("\n-- Angle A table --")
    print(pd.DataFrame(angle_a).to_string(index=False))


if __name__ == "__main__":
    main()
