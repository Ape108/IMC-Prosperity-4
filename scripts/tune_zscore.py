"""
Tune RollingZScoreStrategy parameters for a product by sweeping zscore_period,
smoothing_period, and threshold over historical price data.

Simulates the signal logic from RollingZScoreStrategy and reports how often
each parameter set fires, signal accuracy (did price revert after signal?),
and estimated edge per signal.

Usage:
    python scripts/tune_zscore.py [--dataset round1] [--product ASH_COATED_OSMIUM]
"""

import argparse
import glob
from itertools import product as iterproduct
from pathlib import Path

import pandas as pd


def load_prices(dataset: str, symbol: str) -> pd.Series:
    data_dir = Path(__file__).parent.parent / "datasets" / dataset
    files = sorted(glob.glob(str(data_dir / "prices_*.csv")))
    dfs = []
    for f in files:
        df = pd.read_csv(f, sep=";")
        stem = Path(f).stem
        day = int(stem.split("day_")[-1])
        df["day"] = day
        dfs.append(df)

    prices = pd.concat(dfs, ignore_index=True)
    df = (
        prices[prices["product"] == symbol]
        .query("bid_price_1.notna() and ask_price_1.notna()")
        .sort_values(["day", "timestamp"])
        .reset_index(drop=True)
    )
    return df["mid_price"]


def simulate_zscore(
    mid: pd.Series,
    zscore_period: int,
    smoothing_period: int,
    threshold: float,
    forward_ticks: int = 5,
) -> dict:
    """
    Simulate the RollingZScoreStrategy signal on a price series.

    Returns stats on signal quality:
    - n_signals: how many times the signal fired
    - accuracy: fraction of LONG signals where price rose in next forward_ticks
    - avg_edge: average price move in the correct direction after signal
    """
    hist = mid.rolling(zscore_period)
    zscore = ((mid - hist.mean()) / hist.std()).rolling(smoothing_period).mean()

    long_signals = zscore < -threshold
    short_signals = zscore > threshold

    forward_return = mid.shift(-forward_ticks) - mid

    long_idx = long_signals[long_signals].index
    short_idx = short_signals[short_signals].index

    # Accuracy: did price move in the right direction?
    long_correct = (forward_return[long_idx] > 0).sum() if len(long_idx) else 0
    short_correct = (forward_return[short_idx] < 0).sum() if len(short_idx) else 0

    n_long = len(long_idx)
    n_short = len(short_idx)
    n_signals = n_long + n_short

    accuracy = (long_correct + short_correct) / n_signals if n_signals > 0 else 0

    avg_long_edge = forward_return[long_idx].mean() if n_long > 0 else 0
    avg_short_edge = -forward_return[short_idx].mean() if n_short > 0 else 0
    avg_edge = (avg_long_edge + avg_short_edge) / 2

    return {
        "zscore_period": zscore_period,
        "smoothing_period": smoothing_period,
        "threshold": threshold,
        "n_signals": n_signals,
        "n_long": n_long,
        "n_short": n_short,
        "accuracy": accuracy,
        "avg_edge": avg_edge,
        "score": accuracy * avg_edge * n_signals,  # combined quality metric
    }


def tune(dataset: str, symbol: str) -> None:
    mid = load_prices(dataset, symbol)
    print(f"Loaded {len(mid)} ticks for {symbol}\n")

    zscore_periods   = [5, 10, 20, 50, 100]
    smoothing_periods = [1, 3, 5, 10]
    thresholds       = [0.3, 0.5, 0.75, 1.0, 1.5]

    results = []
    for zp, sp, th in iterproduct(zscore_periods, smoothing_periods, thresholds):
        if sp >= zp:
            continue
        r = simulate_zscore(mid, zp, sp, th)
        results.append(r)

    df = pd.DataFrame(results).sort_values("score", ascending=False)

    print(f"Top 15 parameter sets by score (accuracy × avg_edge × n_signals):")
    print(df.head(15).to_string(index=False, float_format="{:.4f}".format))

    print(f"\nAutocorrelation structure (context for choosing zscore_period):")
    changes = mid.diff().dropna()
    for lag in range(1, 6):
        print(f"  lag {lag}: {changes.autocorr(lag=lag):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="round1")
    parser.add_argument("--product", default="ASH_COATED_OSMIUM")
    args = parser.parse_args()

    tune(args.dataset, args.product)
