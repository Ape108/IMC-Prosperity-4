#!/usr/bin/env python3
r"""
Osmium bot trade analysis — R1

Run from PowerShell (project root):
    .venv\Scripts\python scripts\analyze_osmium_bot.py

Override the dataset path if needed:
    .venv\Scripts\python scripts\analyze_osmium_bot.py --dataset "\\wsl$\Ubuntu\home\heagen\prosperity_rust_backtester\datasets\round1"
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SYMBOL = "ASH_COATED_OSMIUM"
DAYS = [-2, -1, 0]
FAIR_VALUE = 10_000
DEFAULT_DATASET = Path(r"\\wsl$\Ubuntu\home\heagen\prosperity_rust_backtester\datasets\round1")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_trades(dataset_dir: Path, day: int) -> pd.DataFrame:
    path = dataset_dir / f"trades_round_1_day_{day}.csv"
    df = pd.read_csv(path, sep=";", dtype={"buyer": str, "seller": str})
    df["day"] = day
    # Normalise empty strings to NaN so isna() works cleanly
    df["buyer"] = df["buyer"].str.strip().replace("", pd.NA)
    df["seller"] = df["seller"].str.strip().replace("", pd.NA)
    return df


def load_prices(dataset_dir: Path, day: int) -> pd.DataFrame:
    path = dataset_dir / f"prices_round_1_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df["day"] = day
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_mid_col(df: pd.DataFrame) -> str | None:
    """Return the first column name that looks like a mid-price."""
    candidates = [c for c in df.columns if "mid" in c.lower()]
    return candidates[0] if candidates else None


def get_osm_mid_at(prices_by_day: dict, day: int, ts: float, mid_col: str) -> float | None:
    dp = prices_by_day.get(day)
    if dp is None:
        return None
    prev = dp[dp["timestamp"] <= ts]
    if prev.empty:
        return None
    return prev.iloc[-1][mid_col]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main(dataset_dir: Path) -> None:
    # ---- Load trades ----
    all_trades = pd.concat([load_trades(dataset_dir, d) for d in DAYS], ignore_index=True)
    osm = all_trades[all_trades["symbol"] == SYMBOL].copy()

    # Bot trades = both buyer and seller are anonymous (NA)
    osm["is_bot"] = osm["buyer"].isna() & osm["seller"].isna()
    bot = osm[osm["is_bot"]].copy()
    named = osm[~osm["is_bot"]].copy()

    # ---- Summary ----
    print("=" * 60)
    print("OSMIUM TRADE OVERVIEW")
    print("=" * 60)
    print(f"  Total trades across all days : {len(osm)}")
    print(f"  Anonymous bot trades         : {len(bot)}")
    print(f"  Named / player trades        : {len(named)}")
    print()

    print("BOT TRADES BY DAY")
    print("-" * 60)
    for day in DAYS:
        d = bot[bot["day"] == day]
        if d.empty:
            print(f"  Day {day:+d}: 0 bot trades")
        else:
            print(
                f"  Day {day:+d}: {len(d):3d} trades | "
                f"price [{d['price'].min():.0f}, {d['price'].max():.0f}] | "
                f"qty   [{d['quantity'].min():.0f}, {d['quantity'].max():.0f}] | "
                f"Δ FV  [{d['price'].min()-FAIR_VALUE:+.0f}, {d['price'].max()-FAIR_VALUE:+.0f}]"
            )
    print()

    # ---- Per-trade detail with mid-price context ----
    try:
        all_prices = pd.concat([load_prices(dataset_dir, d) for d in DAYS], ignore_index=True)

        # Detect product column name
        prod_col = "product" if "product" in all_prices.columns else None
        if prod_col:
            osm_prices = all_prices[all_prices[prod_col] == SYMBOL].copy()
        else:
            # Some versions use separate files per product — fall through
            osm_prices = pd.DataFrame()

        mid_col = get_mid_col(osm_prices) if not osm_prices.empty else None
        prices_by_day = (
            {d: osm_prices[osm_prices["day"] == d].sort_values("timestamp") for d in DAYS}
            if not osm_prices.empty
            else {}
        )

        if mid_col:
            print("BOT TRADE DETAIL  (price vs mid-price at that tick)")
            print("-" * 60)
            print(f"  {'Day':>4}  {'ts':>8}  {'price':>7}  {'qty':>4}  {'mid':>7}  {'Δmid':>6}")
            for _, row in bot.sort_values(["day", "timestamp"]).iterrows():
                mid = get_osm_mid_at(prices_by_day, row["day"], row["timestamp"], mid_col)
                mid_str = f"{mid:.1f}" if mid is not None else "   N/A"
                delta = f"{row['price']-mid:+.1f}" if mid is not None else "   N/A"
                print(
                    f"  {row['day']:>+4}  {row['timestamp']:>8.0f}  "
                    f"{row['price']:>7.0f}  {row['quantity']:>4.0f}  "
                    f"{mid_str:>7}  {delta:>6}"
                )
            print()

            # ---- Post-trade price movement ----
            print("POST-TRADE PRICE MOVEMENT  (next 10 price ticks after each bot trade)")
            print("-" * 60)
            print(f"  {'Day':>4}  {'ts':>8}  {'Δ10':>6}  direction")
            for _, row in bot.sort_values(["day", "timestamp"]).iterrows():
                dp = prices_by_day.get(row["day"], pd.DataFrame())
                if dp.empty:
                    continue
                after = dp[dp["timestamp"] > row["timestamp"]].head(10)
                if after.empty:
                    continue
                start = get_osm_mid_at(prices_by_day, row["day"], row["timestamp"], mid_col)
                end = after.iloc[-1][mid_col]
                if start is None:
                    continue
                delta = end - start
                direction = "UP  ▲" if delta > 0 else ("DOWN ▼" if delta < 0 else "FLAT")
                print(f"  {row['day']:>+4}  {row['timestamp']:>8.0f}  {delta:>+6.1f}  {direction}")
            print()

        else:
            print("(prices CSV has no mid-price column — skipping mid/post-trade analysis)")
            print(f"  Columns found: {list(all_prices.columns)}")
            print()

    except FileNotFoundError as e:
        print(f"(Could not load prices CSV: {e} — skipping mid-price analysis)\n")
        osm_prices = pd.DataFrame()
        mid_col = None
        prices_by_day = {}

    # ---- Price bucketing: where does the bot trade? ----
    print("PRICE BUCKET BREAKDOWN  (distance from fair value 10,000)")
    print("-" * 60)
    buckets = pd.cut(
        bot["price"] - FAIR_VALUE,
        bins=[-50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50],
        include_lowest=True,
    )
    counts = bot.groupby([buckets, "day"])["quantity"].agg(["count", "sum"])
    print(counts.to_string())
    print()

    # ---- Named trader check (in case any Osmium trades ARE named) ----
    if not named.empty:
        print("NAMED OSMIUM TRADES  (buyer or seller identified)")
        print("-" * 60)
        print(named[["day", "timestamp", "buyer", "seller", "price", "quantity"]]
              .sort_values(["day", "timestamp"])
              .to_string(index=False))
        print()
    else:
        print("No named traders on OSMIUM (confirms buyer/seller fields are empty for all trades)\n")

    # ---- Plot ----
    colors = {-2: "#2196F3", -1: "#FF9800", 0: "#4CAF50"}
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1) Price vs timestamp scatter (bubble = qty)
    ax1 = fig.add_subplot(gs[0, :2])
    for day in DAYS:
        d = bot[bot["day"] == day]
        if d.empty:
            continue
        ax1.scatter(
            d["timestamp"], d["price"],
            s=d["quantity"] * 8, alpha=0.75, label=f"Day {day:+d}",
            color=colors[day], zorder=3, edgecolors="white", linewidths=0.5,
        )
    ax1.axhline(FAIR_VALUE, color="red", linestyle="--", linewidth=1, alpha=0.6, label="FV 10,000")
    ax1.set_title(f"{SYMBOL} — Bot Trade Prices\n(bubble size ∝ quantity)")
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Trade Price")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # 2) Price histogram
    ax2 = fig.add_subplot(gs[0, 2])
    for day in DAYS:
        d = bot[bot["day"] == day]
        if d.empty:
            continue
        ax2.hist(d["price"], bins=20, alpha=0.55, label=f"Day {day:+d}", color=colors[day])
    ax2.axvline(FAIR_VALUE, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_title("Bot Trade Price Distribution")
    ax2.set_xlabel("Price")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=9)

    # 3) Quantity vs timestamp
    ax3 = fig.add_subplot(gs[1, :2])
    for day in DAYS:
        d = bot[bot["day"] == day]
        if d.empty:
            continue
        ax3.scatter(d["timestamp"], d["quantity"], alpha=0.7, label=f"Day {day:+d}", color=colors[day])
    ax3.set_title("Bot Trade Quantity by Timestamp")
    ax3.set_xlabel("Timestamp")
    ax3.set_ylabel("Quantity")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25)

    # 4) Cumulative bot quantity by day
    ax4 = fig.add_subplot(gs[1, 2])
    for day in DAYS:
        d = bot[bot["day"] == day].sort_values("timestamp")
        if d.empty:
            continue
        ax4.plot(d["timestamp"], d["quantity"].cumsum(), label=f"Day {day:+d}", color=colors[day])
    ax4.set_title("Cumulative Bot Quantity")
    ax4.set_xlabel("Timestamp")
    ax4.set_ylabel("Cumulative Qty")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)

    out_path = Path(__file__).parent.parent / "osmium_bot_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Osmium bot trade analyser — R1")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the round1 dataset directory (default: ~/prosperity_rust_backtester/datasets/round1)",
    )
    args = parser.parse_args()
    main(args.dataset)
