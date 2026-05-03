from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR = str(Path(__file__).parent.parent.parent / "datasets" / "round4")
R4_DAYS = [1, 2, 3]


def load_trades(days: list[int], dataset_dir: str = DATASET_DIR) -> pd.DataFrame:
    """Return Mark-bot trades with columns: timestamp, bot, product, signed_qty.

    Each trade is recorded twice: once as a buy (from the buyer's perspective)
    and once as a sell (from the seller's perspective).
    """
    dfs: list[pd.DataFrame] = []
    for day in days:
        path = f"{dataset_dir}/trades_round_4_day_{day}.csv"
        df = pd.read_csv(path, sep=";", dtype={"buyer": str, "seller": str})
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)

    # In R4 all trades are Mark-vs-Mark; record both perspectives so each bot's
    # behaviour can be analysed independently.
    buys = raw[raw["buyer"].str.startswith("Mark", na=False)].copy()
    buys["bot"] = buys["buyer"]
    buys["signed_qty"] = buys["quantity"]

    sells = raw[raw["seller"].str.startswith("Mark", na=False)].copy()
    sells["bot"] = sells["seller"]
    sells["signed_qty"] = -sells["quantity"]

    combined = pd.concat([buys, sells], ignore_index=True)
    return combined[["timestamp", "bot", "symbol", "signed_qty"]].rename(
        columns={"symbol": "product"}
    )


def load_prices(days: list[int], dataset_dir: str = DATASET_DIR) -> pd.DataFrame:
    """Return mid-price series with columns: timestamp, product, mid_price."""
    dfs: list[pd.DataFrame] = []
    for day in days:
        path = f"{dataset_dir}/prices_round_4_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)
    return raw[["timestamp", "product", "mid_price"]].copy()


def lead_lag_corr(
    trades_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    bot: str,
    product: str,
    lags: list[int],
) -> list[float]:
    """Pearson correlation between bot signed_qty at tick T and price change T → T+lag ticks."""
    bot_trades = trades_df[
        (trades_df["bot"] == bot) & (trades_df["product"] == product)
    ].copy()
    prod_prices = (
        prices_df[prices_df["product"] == product][["timestamp", "mid_price"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if bot_trades.empty or len(prod_prices) < 2:
        return [0.0] * len(lags)

    price_vals = prod_prices["mid_price"].to_numpy()
    price_ts = prod_prices["timestamp"].to_numpy()

    results: list[float] = []
    for lag in lags:
        signed_qtys: list[float] = []
        price_changes: list[float] = []

        for _, row in bot_trades.iterrows():
            t = row["timestamp"]
            sq = float(row["signed_qty"])
            # Find index of the price tick at or just before this trade
            idx = int(np.searchsorted(price_ts, t, side="right")) - 1
            if idx < 0 or idx + lag >= len(price_vals):
                continue
            signed_qtys.append(sq)
            price_changes.append(float(price_vals[idx + lag] - price_vals[idx]))

        if len(signed_qtys) < 3:
            results.append(0.0)
            continue

        corr_matrix = np.corrcoef(signed_qtys, price_changes)
        corr = float(corr_matrix[0, 1])
        results.append(corr if not np.isnan(corr) else 0.0)

    return results


def classify_bot(stats: dict) -> str:
    """Classify a (bot, product) pair based on trading stats."""
    lead_corr = abs(stats["lead_5_corr"])
    net_dir = abs(stats["net_direction"])

    if lead_corr > 0.15 and net_dir > 0.3:
        return "INFORMED"
    if net_dir < 0.1 and stats.get("trade_count", 0) > 50:
        return "MARKET_MAKER"
    return "NOISE"


LAGS = [1, 2, 5, 10, 20]


def compute_stats(
    trades_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-(bot, product) stats and classifications."""
    rows: list[dict] = []
    for (bot, product), group in trades_df.groupby(["bot", "product"]):
        trade_count = len(group)
        total_qty = group["signed_qty"].abs().sum()
        net_direction = float(group["signed_qty"].sum() / total_qty) if total_qty > 0 else 0.0
        avg_size = float(group["signed_qty"].abs().mean())

        intervals = group["timestamp"].sort_values().diff().dropna()
        median_interval = float(intervals.median()) if len(intervals) > 0 else 0.0

        corrs = lead_lag_corr(trades_df, prices_df, bot, product, LAGS)
        lag_labels = {f"lead_{lag}_corr": c for lag, c in zip(LAGS, corrs)}

        stats = {
            "bot": bot,
            "product": product,
            "trade_count": trade_count,
            "net_direction": round(net_direction, 3),
            "avg_size": round(avg_size, 1),
            "median_interval": round(median_interval, 0),
            **lag_labels,
        }
        stats["label"] = classify_bot({"lead_5_corr": corrs[LAGS.index(5)], **stats})
        rows.append(stats)

    return pd.DataFrame(rows).sort_values("lead_5_corr", ascending=False)


def main() -> None:
    print("Loading R4 trades and prices...")
    trades = load_trades(days=R4_DAYS)
    prices = load_prices(days=R4_DAYS)
    print(f"  {len(trades)} Mark-bot trade rows across {trades['bot'].nunique()} bots, {trades['product'].nunique()} products\n")

    stats_df = compute_stats(trades, prices)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(stats_df.to_string(index=False))

    print("\n--- INFORMED bots (lead-5 corr > 0.15, net dir > 0.3) ---")
    informed = stats_df[stats_df["label"] == "INFORMED"]
    if informed.empty:
        print("  None found — no Mark bot is directionally predictive at threshold.")
        print("  DECISION: shift focus to fixing VelvetfruitStrategy (-4k R3 result).")
    else:
        print(informed[["bot", "product", "trade_count", "net_direction", "lead_5_corr"]].to_string(index=False))
        print("\n  DECISION: build follower strategy for INFORMED bot×product pairs above.")


if __name__ == "__main__":
    main()
