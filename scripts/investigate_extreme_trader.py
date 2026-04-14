"""
Investigate whether an Olivia-like bot trades at daily price extremes.

In Prosp3, one anonymous bot consistently bought exactly at the daily low
and sold exactly at the daily high, with a fixed quantity (15 lots).
Buyer/seller names are not visible — we detect the pattern by:
  1. Finding the daily min/max mid-price (from prices data)
  2. Finding trades that occurred at timestamps where mid-price was at its daily extreme
  3. Checking whether those trades have a consistent, distinct quantity

Usage:
    python scripts/investigate_extreme_trader.py [--dataset round1]
"""

import argparse
import glob
from pathlib import Path

import pandas as pd


def load_data(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(__file__).parent.parent / "datasets" / dataset
    price_files = sorted(glob.glob(str(data_dir / "prices_*.csv")))
    trade_files = sorted(glob.glob(str(data_dir / "trades_*.csv")))

    prices = pd.concat([pd.read_csv(f, sep=";") for f in price_files], ignore_index=True)
    trades = pd.concat([pd.read_csv(f, sep=";") for f in trade_files], ignore_index=True)

    # Infer day column for trades from the filename
    trade_dfs = []
    for f in trade_files:
        df = pd.read_csv(f, sep=";")
        # Extract day from filename e.g. trades_round_1_day_-1.csv -> -1
        stem = Path(f).stem  # e.g. trades_round_1_day_-1
        day = int(stem.split("day_")[-1])
        df["day"] = day
        trade_dfs.append(df)
    trades = pd.concat(trade_dfs, ignore_index=True)

    return prices, trades


def investigate(dataset: str, tolerance: int = 2) -> None:
    """
    tolerance: how many ticks away from the daily extreme counts as 'at the extreme'
    """
    prices, trades = load_data(dataset)

    print(f"Dataset : {dataset}")
    print(f"Tolerance: ±{tolerance} ticks from daily extreme\n")

    for product in sorted(prices["product"].unique()):
        print(f"{'='*60}")
        print(f"Product: {product}")

        p = (
            prices[prices["product"] == product]
            .dropna(subset=["mid_price"])
            .query("mid_price > 0 and bid_price_1.notna() and ask_price_1.notna()")
            .copy()
        )
        t = trades[trades["symbol"] == product].copy()

        for day in sorted(p["day"].unique()):
            p_day = p[p["day"] == day]
            t_day = t[t["day"] == day]

            daily_min = p_day["mid_price"].min()
            daily_max = p_day["mid_price"].max()

            # Timestamps where mid-price is at daily extreme
            ts_at_min = set(p_day[p_day["mid_price"] <= daily_min + tolerance]["timestamp"])
            ts_at_max = set(p_day[p_day["mid_price"] >= daily_max - tolerance]["timestamp"])

            trades_at_min = t_day[t_day["timestamp"].isin(ts_at_min)]
            trades_at_max = t_day[t_day["timestamp"].isin(ts_at_max)]

            print(f"\n  Day {day:+d}  |  daily_min={daily_min:.1f}  daily_max={daily_max:.1f}  range={daily_max-daily_min:.1f}")
            print(f"  Trades at daily LOW  ({len(trades_at_min)} total): qty distribution = {dict(trades_at_min['quantity'].value_counts().head(5))}")
            print(f"  Trades at daily HIGH ({len(trades_at_max)} total): qty distribution = {dict(trades_at_max['quantity'].value_counts().head(5))}")

            # Cross-check: does that quantity appear disproportionately at extremes vs. overall?
            all_qty = t_day["quantity"].value_counts(normalize=True)
            for label, extreme_trades in [("LOW", trades_at_min), ("HIGH", trades_at_max)]:
                if extreme_trades.empty:
                    continue
                extreme_qty = extreme_trades["quantity"].value_counts(normalize=True)
                overrepresented = {
                    q: f"{extreme_qty[q]:.0%} vs {all_qty.get(q, 0):.0%} overall"
                    for q in extreme_qty.index
                    if extreme_qty[q] > all_qty.get(q, 0) * 2  # 2x overrepresented
                }
                if overrepresented:
                    print(f"  *** Overrepresented qty at {label}: {overrepresented}")

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="round1")
    parser.add_argument("--tolerance", type=int, default=2, help="Ticks from extreme to count as 'at extreme'")
    args = parser.parse_args()

    investigate(args.dataset, args.tolerance)
