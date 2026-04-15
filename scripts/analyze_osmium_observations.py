r"""
Osmium observation variable analysis — R1

Investigates whether sunlight/humidity (ConversionObservation fields new in Prosperity 4)
correlate with ASH_COATED_OSMIUM price movement.

Run from PowerShell (project root):
    .venv\Scripts\python scripts\analyze_osmium_observations.py

Step 1 (discovery): prints column names + sample rows, lists all dataset files.
Step 2 (correlation): runs if observation columns are found.
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SYMBOL = "ASH_COATED_OSMIUM"
DAYS = [-2, -1, 0]
DEFAULT_DATASET = Path(r"\\wsl$\Ubuntu\home\heagen\prosperity_rust_backtester\datasets\round1")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover(dataset_dir: Path) -> None:
    print("=" * 60)
    print("DATASET DIRECTORY CONTENTS")
    print("=" * 60)
    files = sorted(dataset_dir.iterdir())
    for f in files:
        print(f"  {f.name}")
    print()

    print("=" * 60)
    print("PRICES CSV COLUMNS + SAMPLE (day 0)")
    print("=" * 60)
    prices_path = dataset_dir / "prices_round_1_day_0.csv"
    df = pd.read_csv(prices_path, sep=";")
    print(f"  Columns: {df.columns.tolist()}")
    print()

    # Show all unique products
    if "product" in df.columns:
        print(f"  Products: {sorted(df['product'].unique().tolist())}")
        print()

    # Show OSMIUM rows
    if "product" in df.columns:
        osm = df[df["product"] == SYMBOL]
    else:
        osm = df
    print(f"  OSMIUM sample (first 5 rows):")
    print(osm.head().to_string())
    print()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_prices(dataset_dir: Path, day: int) -> pd.DataFrame:
    path = dataset_dir / f"prices_round_1_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df["day"] = day
    return df


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def detect_observation_cols(df: pd.DataFrame) -> list[str]:
    """Return column names that look like observation variables."""
    keywords = ["sunlight", "humidity", "transport", "tariff", "export", "import",
                "bid_price", "ask_price", "observation", "conv"]
    # Exclude standard OHLCV columns
    standard = {"timestamp", "product", "day", "profit_and_loss",
                 "mid_price", "bid_price_1", "bid_volume_1",
                 "ask_price_1", "ask_volume_1"}
    candidates = []
    for col in df.columns:
        col_l = col.lower()
        if col in standard:
            continue
        if any(k in col_l for k in keywords):
            candidates.append(col)
    return candidates


def run_correlation(dataset_dir: Path) -> None:
    all_prices = pd.concat([load_prices(dataset_dir, d) for d in DAYS], ignore_index=True)

    if "product" not in all_prices.columns:
        print("ERROR: no 'product' column found — cannot filter for OSMIUM.")
        print(f"Columns: {all_prices.columns.tolist()}")
        return

    osm = all_prices[all_prices["product"] == SYMBOL].copy().sort_values(["day", "timestamp"]).reset_index(drop=True)

    if osm.empty:
        print(f"No rows found for {SYMBOL} in prices CSVs.")
        return

    # Identify mid price column
    mid_col = next((c for c in osm.columns if "mid" in c.lower()), None)
    if mid_col is None:
        print("No mid-price column found. Columns:", osm.columns.tolist())
        return

    # Compute forward price change (next tick mid − current mid, within same day)
    osm["mid_change_1"] = osm.groupby("day")[mid_col].diff(1).shift(-1)  # 1-tick forward
    osm["mid_change_5"] = osm.groupby("day")[mid_col].diff(5).shift(-5)  # 5-tick forward

    obs_cols = detect_observation_cols(osm)

    print("=" * 60)
    print("OBSERVATION COLUMNS DETECTED")
    print("=" * 60)
    if obs_cols:
        print(f"  Found: {obs_cols}")
        print()
        for col in obs_cols:
            n_nonnull = osm[col].notna().sum()
            print(f"  {col}: {n_nonnull} non-null values | "
                  f"range [{osm[col].min():.2f}, {osm[col].max():.2f}]")
    else:
        print("  None found — observation data may not be embedded in prices CSV.")
        print("  This is expected; Prosperity 4 observation data typically lives in")
        print("  state.observations, which is only accessible in the live/log environment.")
        print()
        print("  Falling back to price-only pattern analysis (autocorrelation, periodicity).")
    print()

    if obs_cols:
        print("=" * 60)
        print("PEARSON CORRELATION vs FORWARD PRICE CHANGE")
        print("=" * 60)
        print(f"  {'Column':<30} {'corr(Δmid+1)':>14} {'corr(Δmid+5)':>14}")
        for col in obs_cols:
            subset = osm[[col, "mid_change_1", "mid_change_5"]].dropna()
            if len(subset) < 20:
                continue
            c1 = subset[col].corr(subset["mid_change_1"])
            c5 = subset[col].corr(subset["mid_change_5"])
            flag = " *** " if abs(c1) > 0.1 or abs(c5) > 0.1 else ""
            print(f"  {col:<30} {c1:>+14.4f} {c5:>+14.4f}{flag}")
        print()

    # ---------------------------------------------------------------------------
    # Price-only pattern analysis (always runs)
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("PRICE AUTOCORRELATION (does price direction persist?)")
    print("=" * 60)
    osm["ret"] = osm.groupby("day")[mid_col].diff(1)
    for lag in [1, 2, 3, 5, 10]:
        series = osm["ret"].dropna()
        autocorr = series.autocorr(lag=lag)
        bar = "#" * int(abs(autocorr) * 40)
        direction = "+" if autocorr > 0 else "-"
        print(f"  lag={lag:>2}: {autocorr:>+.4f}  [{direction}{bar}]")
    print()
    print("  Interpretation:")
    print("    Strong POSITIVE autocorr → momentum (follow recent direction)")
    print("    Strong NEGATIVE autocorr → mean-reversion (fade recent direction)")
    print("    Near zero → no exploitable serial pattern")
    print()

    print("=" * 60)
    print("PRICE PERIODICITY CHECK (FFT — top 5 frequencies)")
    print("=" * 60)
    for day in DAYS:
        d = osm[osm["day"] == day][mid_col].dropna().values
        if len(d) < 50:
            continue
        # Detrend
        d_detrended = d - np.linspace(d[0], d[-1], len(d))
        fft = np.abs(np.fft.rfft(d_detrended))
        freqs = np.fft.rfftfreq(len(d))
        # Exclude DC (freq=0)
        fft[0] = 0
        top5_idx = np.argsort(fft)[-5:][::-1]
        periods = [f"{1/freqs[i]:.0f} ticks" if freqs[i] > 0 else "inf" for i in top5_idx]
        powers = [fft[i] for i in top5_idx]
        print(f"  Day {day:+d}: dominant periods = {', '.join(periods)}")
        print(f"          powers           = {[f'{p:.1f}' for p in powers]}")
    print()

    print("=" * 60)
    print("PRICE RANGE BY TIME-OF-DAY (10 equal buckets)")
    print("=" * 60)
    osm["bucket"] = pd.cut(osm["timestamp"], bins=10, labels=False)
    summary = osm.groupby("bucket")[mid_col].agg(["mean", "std", "min", "max"])
    summary.columns = ["mean", "std", "min", "max"]
    ts_bounds = osm.groupby("bucket")["timestamp"].agg(["min", "max"])
    summary["ts_range"] = ts_bounds.apply(lambda r: f"{r['min']:.0f}–{r['max']:.0f}", axis=1)
    print(summary[["ts_range", "mean", "std", "min", "max"]].to_string())
    print()

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    colors = {-2: "#2196F3", -1: "#FF9800", 0: "#4CAF50"}
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # 1) Mid price time series per day
    ax1 = fig.add_subplot(gs[0, :])
    for day in DAYS:
        d = osm[osm["day"] == day]
        ax1.plot(d["timestamp"], d[mid_col], label=f"Day {day:+d}",
                 color=colors[day], linewidth=0.8, alpha=0.85)
    ax1.axhline(10000, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="FV 10,000")
    ax1.set_title(f"{SYMBOL} — Mid-Price by Timestamp")
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Mid Price")
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    # 2) Return distribution
    ax2 = fig.add_subplot(gs[1, 0])
    for day in DAYS:
        d = osm[osm["day"] == day]["ret"].dropna()
        ax2.hist(d, bins=40, alpha=0.5, label=f"Day {day:+d}", color=colors[day])
    ax2.set_title("Per-tick Return Distribution")
    ax2.set_xlabel("Δ mid-price")
    ax2.set_ylabel("Count")
    ax2.legend()

    # 3) Autocorrelation bar chart
    ax3 = fig.add_subplot(gs[1, 1])
    lags = range(1, 21)
    series = osm["ret"].dropna()
    acorrs = [series.autocorr(lag=lag) for lag in lags]
    colors_bar = ["green" if a > 0 else "red" for a in acorrs]
    ax3.bar(list(lags), acorrs, color=colors_bar, alpha=0.75)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.axhline(0.05, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax3.axhline(-0.05, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax3.set_title("Price Return Autocorrelation (lags 1–20)")
    ax3.set_xlabel("Lag (ticks)")
    ax3.set_ylabel("Autocorrelation")
    ax3.grid(True, alpha=0.25, axis="y")

    # Observation subplot (if found)
    if obs_cols:
        col = obs_cols[0]
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_title(f"Observation: {col}")
        for day in DAYS:
            d = osm[osm["day"] == day]
            ax4.plot(d["timestamp"], d[col], label=f"Day {day:+d}", color=colors[day])
        ax4.legend()

    out_path = Path(__file__).parent.parent / "osmium_observations_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------

def main(dataset_dir: Path) -> None:
    discover(dataset_dir)
    run_correlation(dataset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Osmium observation variable analyser — R1")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to round1 dataset directory",
    )
    args = parser.parse_args()
    main(args.dataset)
