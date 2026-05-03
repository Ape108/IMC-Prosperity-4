"""Bot Imitation-PnL Sweep (R4 Phase 1, single-bot).

For every (bot × product × horizon) cell, simulate imitating that bot's trades
on a 1-tick lag and exiting `horizon` ticks later. Aggregate per-cell signed
mean / std / t-stat / hit-rate. A cell passes iff its sign-aware mean PnL
(in mid-to-mid ticks) clears the per-product median round-trip spread cost
with n>=10 and |t|>=2. Emit one PASS line per passing cell, or a single
NO PASS verdict.

Spec: docs/superpowers/specs/2026-04-27-r4-bot-imitation-pnl-sweep-design.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from submissions.r4.tests.eda_mark_bots import DATASET_DIR, R4_DAYS

TICKS_PER_DAY = 1_000_000

BOTS: list[str] = [
    "Mark 01", "Mark 14", "Mark 22", "Mark 38",
    "Mark 49", "Mark 55", "Mark 67",
]

PRODUCTS: list[str] = [
    "HYDROGEL_PACK",
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
]

HORIZONS: list[int] = [5, 10, 20, 50]

THRESHOLD_N = 10
THRESHOLD_T_STAT = 2.0

# ── Decision rule ────────────────────────────────────────────────────────────


def cell_passes(
    n: int,
    mean_pnl: float,
    std_pnl: float,
    cost: float,
    threshold_n: int = THRESHOLD_N,
    threshold_t: float = THRESHOLD_T_STAT,
) -> bool:
    """A (bot × product × horizon) cell passes iff:
        n >= threshold_n
        AND mean_pnl >= cost (sign-aware mean clears round-trip cost)
        AND t-stat = sqrt(n) * mean_pnl / std_pnl >= threshold_t

    `std_pnl <= 0` is degenerate (can't compute t-stat) — never a pass.
    `mean_pnl < cost` (including negative) — never a pass.
    """
    if n < threshold_n:
        return False
    if std_pnl <= 0:
        return False
    if mean_pnl < cost:
        return False
    t_stat = (n ** 0.5) * mean_pnl / std_pnl
    return t_stat >= threshold_t


# ── Cost model ───────────────────────────────────────────────────────────────


def median_round_trip_cost(prices: pd.DataFrame, product: str) -> float:
    """Median (best_ask - best_bid) for `product` across all ticks. Returns NaN
    if the product has no rows or all rows have missing book.

    Imitation crosses the spread on entry (pays half-spread above mid) and on
    exit (pays half-spread below mid). Round-trip cost in mid-to-mid terms =
    one full spread. The pooled median is a strict-enough proxy across
    days 1/2/3 for products in scope.
    """
    sub = prices[prices["product"] == product]
    if sub.empty:
        return float("nan")
    spreads = sub["ask_price_1"] - sub["bid_price_1"]
    spreads = spreads.dropna()
    if spreads.empty:
        return float("nan")
    return float(spreads.median())


# ── Imitation-PnL helper ─────────────────────────────────────────────────────


def compute_imitation_pnl(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    bot: str,
    product: str,
    horizon: int,
) -> list[float]:
    """Sign-aware PnL (in mid-to-mid ticks) of imitating each (bot, product)
    trade with a 1-tick entry lag and `horizon`-tick hold.

    For each trade by `bot` in `product`:
        entry_tick = first price tick with ts strictly > trade.ts (within day)
        exit_tick  = entry_idx + horizon  (within day)
        pnl_ticks  = sign(signed_qty) * (mid[exit_tick] - mid[entry_tick])
    Trades whose entry is missing or whose exit is past the day's last tick
    are skipped. Per-day slicing prevents day N's last trade from picking up
    day N+1's opening prices as its exit.
    """
    if trades.empty or prices.empty:
        return []

    bot_trades = trades[
        (trades["bot"] == bot) & (trades["product"] == product)
    ]
    if bot_trades.empty:
        return []

    # Multi-level lifts in a single tick produce multiple rows with the same
    # (bot, product, ts). They represent ONE decision point; dedup so they
    # don't inflate n/t-stat with duplicate PnL values.
    bot_trades = bot_trades.drop_duplicates(subset=["ts"])

    pnls: list[float] = []
    for day, day_trades in bot_trades.groupby("day"):
        day_prices = (
            prices[(prices["product"] == product) & (prices["day"] == day)]
            .sort_values("ts")
            .reset_index(drop=True)
        )
        if day_prices.empty:
            continue
        ts_arr = day_prices["ts"].to_numpy()
        mid_arr = day_prices["mid_price"].to_numpy()

        for _, row in day_trades.iterrows():
            t = int(row["ts"])
            sq = float(row["signed_qty"])
            if sq == 0:
                continue
            entry_idx = int(np.searchsorted(ts_arr, t, side="right"))
            if entry_idx >= len(ts_arr):
                continue
            exit_idx = entry_idx + horizon
            if exit_idx >= len(ts_arr):
                continue
            entry_mid = float(mid_arr[entry_idx])
            exit_mid = float(mid_arr[exit_idx])
            sign = 1.0 if sq > 0 else -1.0
            pnls.append(sign * (exit_mid - entry_mid))
    return pnls


# ── Loaders ──────────────────────────────────────────────────────────────────


def load_trades_with_day(days: list[int] = R4_DAYS) -> pd.DataFrame:
    """Load Mark-bot trades across `days`. Returns columns:
        ts, day, bot, product, signed_qty
    where ts = day * TICKS_PER_DAY + raw_timestamp (strictly monotonic).

    Each trade is recorded twice — once from the buyer's perspective
    (signed_qty > 0) and once from the seller's perspective (signed_qty < 0).
    """
    dfs: list[pd.DataFrame] = []
    for day in days:
        path = f"{DATASET_DIR}/trades_round_4_day_{day}.csv"
        df = pd.read_csv(path, sep=";", dtype={"buyer": str, "seller": str})
        df["day"] = day
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)

    buys = raw[raw["buyer"].str.startswith("Mark", na=False)].copy()
    buys["bot"] = buys["buyer"]
    buys["signed_qty"] = buys["quantity"]

    sells = raw[raw["seller"].str.startswith("Mark", na=False)].copy()
    sells["bot"] = sells["seller"]
    sells["signed_qty"] = -sells["quantity"]

    combined = pd.concat([buys, sells], ignore_index=True)
    combined["ts"] = combined["day"] * TICKS_PER_DAY + combined["timestamp"]
    return combined[["ts", "day", "bot", "symbol", "signed_qty"]].rename(
        columns={"symbol": "product"}
    )


def load_prices_with_book(days: list[int] = R4_DAYS) -> pd.DataFrame:
    """Load price ticks across `days` with bid_price_1 / ask_price_1 / mid_price
    preserved. Returns columns: ts, day, product, bid_price_1, ask_price_1,
    mid_price, where ts = day * TICKS_PER_DAY + raw_timestamp.
    """
    dfs: list[pd.DataFrame] = []
    for day in days:
        path = f"{DATASET_DIR}/prices_round_4_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)
    raw["ts"] = raw["day"] * TICKS_PER_DAY + raw["timestamp"]
    return raw[["ts", "day", "product", "bid_price_1", "ask_price_1", "mid_price"]].copy()


# ── Sweep ────────────────────────────────────────────────────────────────────


def compute_cells(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    bots: list[str] = BOTS,
    products: list[str] = PRODUCTS,
    horizons: list[int] = HORIZONS,
) -> pd.DataFrame:
    """Per (bot × product × horizon), simulate imitation and aggregate stats.
    Returns a DataFrame with columns:
        bot, product, horizon, n, mean_pnl_ticks, std_pnl_ticks,
        cost, net_pnl_ticks, t_stat, hit_rate, passes
    sorted by net_pnl_ticks descending (NaNs last).
    """
    cost_by_product = {p: median_round_trip_cost(prices, p) for p in products}
    rows: list[dict] = []
    for bot in bots:
        for product in products:
            cost = cost_by_product[product]
            for horizon in horizons:
                pnls = compute_imitation_pnl(trades, prices, bot, product, horizon)
                n = len(pnls)
                if n == 0:
                    rows.append({
                        "bot": bot, "product": product, "horizon": horizon,
                        "n": 0,
                        "mean_pnl_ticks": float("nan"),
                        "std_pnl_ticks": float("nan"),
                        "cost": cost,
                        "net_pnl_ticks": float("nan"),
                        "t_stat": float("nan"),
                        "hit_rate": float("nan"),
                        "passes": False,
                    })
                    continue
                arr = np.asarray(pnls, dtype=float)
                mean_pnl = float(arr.mean())
                # n==1 sentinel: 0.0 forces t_stat to NaN and cell_passes to
                # reject via std_pnl <= 0; np.std(ddof=1) on n=1 returns NaN.
                std_pnl = float(arr.std(ddof=1)) if n >= 2 else 0.0
                t_stat = (n ** 0.5) * mean_pnl / std_pnl if std_pnl > 0 else float("nan")
                hit_rate = float((arr > 0).mean())
                cost_ok = not np.isnan(cost)
                net = (mean_pnl - cost) if cost_ok else float("nan")
                passes = cell_passes(n, mean_pnl, std_pnl, cost) if cost_ok else False
                rows.append({
                    "bot": bot, "product": product, "horizon": horizon,
                    "n": n,
                    "mean_pnl_ticks": mean_pnl,
                    "std_pnl_ticks": std_pnl,
                    "cost": cost,
                    "net_pnl_ticks": net,
                    "t_stat": t_stat,
                    "hit_rate": hit_rate,
                    "passes": passes,
                })
    df = pd.DataFrame(rows)
    # Multi-key sort with deterministic tie-break (bot/product/horizon) so the
    # verdict-file output is byte-stable across reruns when data is unchanged.
    return df.sort_values(
        ["net_pnl_ticks", "bot", "product", "horizon"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


# ── Output ───────────────────────────────────────────────────────────────────


def _format_row(row: pd.Series) -> str:
    n = int(row["n"])
    mean = row["mean_pnl_ticks"]
    cost = row["cost"]
    net = row["net_pnl_ticks"]
    t = row["t_stat"]
    hit = row["hit_rate"]
    pass_str = "YES" if row["passes"] else "no"

    def _f(x: float, fmt: str = "+.3f") -> str:
        return f"{x:{fmt}}" if not (isinstance(x, float) and np.isnan(x)) else "  nan "

    return (
        f"{row['bot']:8s}  {row['product']:14s}  "
        f"horizon={int(row['horizon']):>2d}  "
        f"n={n:>4d}  "
        f"mean_t={_f(mean):>8s}  "
        f"cost={_f(cost, '.3f'):>6s}  "
        f"net_t={_f(net):>8s}  "
        f"t={_f(t, '+.2f'):>7s}  "
        f"hit={_f(hit, '.2f'):>5s}  "
        f"PASS={pass_str}"
    )


def emit_report(cells: pd.DataFrame) -> str:
    """Render the full table and append a verdict footer."""
    lines: list[str] = []
    lines.append("=== Bot Imitation-PnL Sweep ===")
    lines.append(
        f"Cells: {len(cells)}  "
        f"with n>=10: {(cells['n'] >= 10).sum()}  "
        f"passing: {cells['passes'].sum()}"
    )
    lines.append("")
    for _, row in cells.iterrows():
        lines.append(_format_row(row))

    lines.append("")
    lines.append("=== VERDICT ===")
    passing = cells[cells["passes"]]
    if passing.empty:
        lines.append("NO PASS — close bot-following")
    else:
        for _, row in passing.iterrows():
            lines.append(
                f"PASS: {row['bot']} {row['product']} "
                f"horizon={int(row['horizon'])} "
                f"(mean={row['mean_pnl_ticks']:+.3f}t "
                f"cost={row['cost']:.3f} "
                f"t={row['t_stat']:+.2f} "
                f"n={int(row['n'])})"
            )
    return "\n".join(lines)


def main() -> None:
    print("Loading R4 trades and prices (days 1/2/3)...")
    trades = load_trades_with_day()
    prices = load_prices_with_book()
    n_trade_rows = len(trades)
    print(
        f"  {n_trade_rows} Mark-bot trade rows × {len(BOTS)} bots × "
        f"{len(PRODUCTS)} products."
    )

    cells = compute_cells(trades, prices)
    print(emit_report(cells))


if __name__ == "__main__":
    main()
