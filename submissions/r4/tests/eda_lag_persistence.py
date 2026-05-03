"""Multi-lag corr analysis on Mark 14 × {VEV_5300, VEV_5400, VEV_5500}.

Outputs a per-strike table with one column per lag, then classifies each
strike using the spec's persist/decay/revert rule.

Reuses load_trades / load_prices / lead_lag_corr from eda_mark_bots.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running as a standalone script next to eda_mark_bots.py.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from eda_mark_bots import LAGS, R4_DAYS, lead_lag_corr, load_prices, load_trades

MARK14_BOT = "Mark 14"
MARK14_STRIKES = ["VEV_5300", "VEV_5400", "VEV_5500"]

PERSIST_RATIO = 0.6
DECAY_RATIO = 0.4


def classify_persistence(
    corrs: dict[int, float],
    lead_lag: int = 5,
    ref_lag: int = 20,
) -> str:
    """Apply the spec's persist/decay/revert rule.

    persist:       same sign as lead_lag AND |ref| / |lead| >= PERSIST_RATIO
    decay:         same sign AND |ref| / |lead| < DECAY_RATIO
    revert:        opposite sign
    borderline:    same sign AND DECAY_RATIO <= |ref| / |lead| < PERSIST_RATIO
    indeterminate: either corr is exactly zero
    """
    c_lead = corrs.get(lead_lag, 0.0)
    c_ref = corrs.get(ref_lag, 0.0)
    if c_lead == 0.0 or c_ref == 0.0:
        return "indeterminate"
    same_sign = (c_lead * c_ref) > 0
    if not same_sign:
        return "revert"
    ratio = abs(c_ref) / abs(c_lead)
    if ratio >= PERSIST_RATIO:
        return "persist"
    if ratio < DECAY_RATIO:
        return "decay"
    return "borderline"


def lag_persistence_table(
    trades_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    bot: str,
    products: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Per-(bot, product) row with one column per lag (lead_<lag>_corr)."""
    rows: list[dict] = []
    for product in products:
        corrs = lead_lag_corr(trades_df, prices_df, bot, product, lags)
        row: dict = {"bot": bot, "product": product}
        for lag, c in zip(lags, corrs):
            row[f"lead_{lag}_corr"] = round(c, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    print(f"Loading R4 trades and prices (days {R4_DAYS})...")
    trades = load_trades(days=R4_DAYS)
    prices = load_prices(days=R4_DAYS)

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.3f}".format)

    # Compute raw corrs once per strike; reuse for both display and classification.
    raw_per_product: dict[str, dict[int, float]] = {}
    rows: list[dict] = []
    for product in MARK14_STRIKES:
        corrs = lead_lag_corr(trades, prices, MARK14_BOT, product, LAGS)
        raw_per_product[product] = {lag: c for lag, c in zip(LAGS, corrs)}
        row: dict = {"bot": MARK14_BOT, "product": product}
        for lag, c in zip(LAGS, corrs):
            row[f"lead_{lag}_corr"] = round(c, 3)
        rows.append(row)
    table = pd.DataFrame(rows)

    print(f"\n{MARK14_BOT} lag-persistence table:")
    print(table.to_string(index=False))

    print("\nClassification per spec decision rule:")
    for product in MARK14_STRIKES:
        label = classify_persistence(raw_per_product[product])
        print(f"  {product:10s} -> {label}")


if __name__ == "__main__":
    main()
