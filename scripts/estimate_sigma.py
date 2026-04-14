"""
Estimate Avellaneda-Stoikov sigma (mid-price volatility per tick) from historical price data.

Usage:
    python scripts/estimate_sigma.py [--dataset round1]

Sigma is the std dev of tick-to-tick mid-price changes (first differences).
This is the value to plug into PepperRootStrategy and similar Stoikov-based strategies.
"""

import argparse
import glob
from pathlib import Path

import pandas as pd


def estimate_sigma(dataset: str) -> None:
    data_dir = Path(__file__).parent.parent / "datasets" / dataset
    files = sorted(glob.glob(str(data_dir / "prices_*.csv")))

    if not files:
        print(f"No price files found in {data_dir}")
        return

    dfs = [pd.read_csv(f, sep=";") for f in files]
    prices = pd.concat(dfs, ignore_index=True)

    print(f"Dataset : {dataset}")
    print(f"Days    : {sorted(prices['day'].unique())}")
    print(f"Files   : {[Path(f).name for f in files]}")
    print()

    for product in sorted(prices["product"].unique()):
        df = (
            prices[prices["product"] == product]
            .sort_values(["day", "timestamp"])
            .dropna(subset=["mid_price"])
            .copy()
        )

        # Filter to two-sided book ticks only — one-sided ticks (bid or ask missing)
        # cause artificial mid-price jumps that inflate sigma by ~300x.
        df_filtered = df[df["bid_price_1"].notna() & df["ask_price_1"].notna()]

        # First differences of mid-price (per 100-unit timestamp step)
        diffs = df_filtered["mid_price"].diff().dropna()

        sigma = diffs.std()
        mean_change = diffs.mean()
        n_ticks_raw = len(df)
        n_ticks_filtered = len(df_filtered)

        print(f"Product : {product}")
        print(f"  Ticks (raw)     : {n_ticks_raw}")
        print(f"  Ticks (filtered): {n_ticks_filtered}  (two-sided book only)")
        print(f"  Mean mid-price  : {df_filtered['mid_price'].mean():.2f}")
        print(f"  Mean tick change: {mean_change:.4f}")
        print(f"  Sigma (std dev) : {sigma:.4f}  <-- use this in Stoikov")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate Stoikov sigma from price data")
    parser.add_argument("--dataset", default="round1", help="Dataset name (default: round1)")
    args = parser.parse_args()

    estimate_sigma(args.dataset)
