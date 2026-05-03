"""R4 Inside-MM Bias EDA.

Re-tests the top-K cells from submissions/r4/imitation_pnl_verdict.txt with a
passive-at-touch fill model. Dual-regime: optimistic (any same-side trade in
window = fill) vs conservative (trade through our level = fill). A cell passes
iff the conservative regime clears the four-condition gate from the spec.

Spec: docs/superpowers/specs/2026-04-28-r4-inside-mm-bias-eda-design.md
"""
from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from eda_imitation_pnl import (
    DATASET_DIR,
    R4_DAYS,
    TICKS_PER_DAY,
    load_prices_with_book,
    load_trades_with_day,
)


# ── Loaders ──────────────────────────────────────────────────────────────────


def load_market_trades(
    days: list[int] = R4_DAYS,
    dataset_dir: str = DATASET_DIR,
) -> pd.DataFrame:
    """Raw market trade prints across `days`. Returns columns: ts, day, product, price.

    Unlike load_trades_with_day, this does NOT split by buyer/seller perspective.
    Each row represents one market print — used for fill simulation against
    hypothetical resting orders.
    """
    dfs: list[pd.DataFrame] = []
    for day in days:
        path = f"{dataset_dir}/trades_round_4_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        df["day"] = day
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)
    raw["ts"] = raw["day"] * TICKS_PER_DAY + raw["timestamp"]
    return raw[["ts", "day", "symbol", "price"]].rename(columns={"symbol": "product"})


# ── Top-K cell parser ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TopKCell:
    bot: str
    product: str
    horizon: int
    mean_t: float
    t_stat: float
    direction: str  # "follow" or "fade"


_VERDICT_LINE_RE = re.compile(
    r"(Mark \d+)\s+(\S+)\s+horizon=\s*(\d+)\s+"
    r"n=\s*(\d+)\s+mean_t=\s*([+-]?\d+\.\d+)\s+"
    r"cost=\s*\d+\.\d+\s+net_t=\s*[+-]?\d+\.\d+\s+"
    r"t=\s*([+-]?\d+\.\d+|nan)"
)


def parse_top_k_cells(
    verdict_path: str | Path,
    n_min: int = 30,
    t_min: float = 1.5,
) -> list[TopKCell]:
    """Parse imitation_pnl_verdict.txt and return cells passing n>=n_min AND |t|>=t_min.

    Direction derived: "follow" if mean_t >= 0, "fade" otherwise. Mark 38 x
    HYDROGEL_PACK is excluded as a mirror of Mark 14 x HYDROGEL_PACK (same dyad
    signal, opposite side).
    """
    cells: list[TopKCell] = []
    with open(verdict_path, encoding="cp1252") as f:
        for line in f:
            m = _VERDICT_LINE_RE.search(line)
            if not m:
                continue
            bot, product, h_str, n_str, mean_str, t_str = m.groups()
            t_val = float(t_str)  # regex allowed "nan", float("nan") returns math.nan
            if math.isnan(t_val):
                continue
            n_int = int(n_str)
            if n_int < n_min or abs(t_val) < t_min:
                continue
            mean_val = float(mean_str)
            direction = "follow" if mean_val >= 0 else "fade"
            cells.append(
                TopKCell(
                    bot=bot,
                    product=product,
                    horizon=int(h_str),
                    mean_t=mean_val,
                    t_stat=t_val,
                    direction=direction,
                )
            )

    # Mirror exclusion: drop Mark 38 x HYDROGEL_PACK if Mark 14 x HYDROGEL_PACK present.
    has_m14_hydrogel = any(
        c.bot == "Mark 14" and c.product == "HYDROGEL_PACK" for c in cells
    )
    if has_m14_hydrogel:
        cells = [
            c for c in cells
            if not (c.bot == "Mark 38" and c.product == "HYDROGEL_PACK")
        ]
    return cells


# ── Fill simulation ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Fill:
    event_idx: int   # original positional index in `events`
    fill_tick: int   # ts at which fill occurred
    fill_price: float


def _bias_sign(direction: str, signed_qty: float) -> int:
    """Long bias = +1 (post bid), short bias = -1 (post ask)."""
    base = 1 if signed_qty > 0 else -1
    return base if direction == "follow" else -base


def simulate_passive_fills(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    market_trades: pd.DataFrame,
    product: str,
    horizon: int,
    direction: str,
    regime: str,
) -> list[Fill]:
    """For each event, find the first market trade in the window that fills our resting quote.

    Window = the next `horizon` price ticks after event.ts on the same day.
    Long bias (signed_qty > 0 follow, or signed_qty < 0 fade): post bid at best_bid(t').
        optimistic: any market trade with price ≤ best_bid(t') → fill at best_bid(t')
        conservative: market trade with price < best_bid(t') → fill at best_bid(t')
    Short bias mirrors with best_ask and price ≥ / >.

    Day-boundary: events whose window extends past last price tick of their day
    are skipped (no fill).
    """
    if events.empty or prices.empty:
        return []

    prod_prices = prices[prices["product"] == product].sort_values("ts").reset_index(drop=True)
    if prod_prices.empty:
        return []
    prod_trades = (
        market_trades[market_trades["product"] == product]
        .sort_values("ts").reset_index(drop=True)
    )

    fills: list[Fill] = []
    events_indexed = events.reset_index(drop=True)

    for day, day_prices in prod_prices.groupby("day", sort=True):
        day_prices = day_prices.sort_values("ts").reset_index(drop=True)
        day_ts = day_prices["ts"].to_numpy()
        day_bid = day_prices["bid_price_1"].to_numpy()
        day_ask = day_prices["ask_price_1"].to_numpy()

        day_trades = prod_trades[prod_trades["day"] == day]
        trade_ts = day_trades["ts"].to_numpy()
        trade_price = day_trades["price"].to_numpy()

        day_events = events_indexed[events_indexed["day"] == day]

        for event_idx, event in day_events.iterrows():
            t = int(event["ts"])
            sq = float(event["signed_qty"])
            if sq == 0:
                continue
            sign = _bias_sign(direction, sq)

            # First price-tick index strictly after event.ts.
            entry_idx = int(np.searchsorted(day_ts, t, side="right"))
            if entry_idx >= len(day_ts):
                continue
            exit_idx = entry_idx + horizon
            if exit_idx >= len(day_ts):
                # Day-boundary: not enough ticks remaining for the holding window.
                continue

            window_max_ts = int(day_ts[exit_idx])
            # Market trades with t < trade.ts <= window_max_ts.
            lo = int(np.searchsorted(trade_ts, t, side="right"))
            hi = int(np.searchsorted(trade_ts, window_max_ts, side="right"))
            if lo >= hi:
                continue

            for ti in range(lo, hi):
                tt = int(trade_ts[ti])
                tp = float(trade_price[ti])
                # Find the price tick at or just before this trade.
                pidx = int(np.searchsorted(day_ts, tt, side="right")) - 1
                if pidx < entry_idx or pidx > exit_idx:
                    continue
                bid = float(day_bid[pidx])
                ask = float(day_ask[pidx])

                if sign > 0:
                    qualifies = (regime == "optimistic" and tp <= bid) or (
                        regime == "conservative" and tp < bid
                    )
                    if qualifies:
                        fills.append(Fill(int(event_idx), tt, bid))
                        break
                else:
                    qualifies = (regime == "optimistic" and tp >= ask) or (
                        regime == "conservative" and tp > ask
                    )
                    if qualifies:
                        fills.append(Fill(int(event_idx), tt, ask))
                        break

    return fills


# ── PnL computation ──────────────────────────────────────────────────────────


def compute_pnls(
    fills: list[Fill],
    events: pd.DataFrame,
    prices: pd.DataFrame,
    product: str,
    horizon: int,
    direction: str,
) -> list[tuple[float, float]]:
    """For each fill, return (pnl_mid, pnl_flat).

    Hold from fill_tick until t + horizon (signal horizon, not fill+h):
        long bias: pnl_mid  = mid(t+h) - fill_price
                   pnl_flat = best_bid(t+h) - fill_price
        short bias: pnl_mid  = fill_price - mid(t+h)
                    pnl_flat = fill_price - best_ask(t+h)

    Day-boundary: if t + horizon exceeds the day's last tick, the fill is dropped
    (returned list is shorter than `fills`). In practice simulate_passive_fills
    already enforces this — this guard is defensive.
    """
    if not fills:
        return []
    events_indexed = events.reset_index(drop=True)
    prod_prices = prices[prices["product"] == product].sort_values("ts").reset_index(drop=True)
    if prod_prices.empty:
        return []

    out: list[tuple[float, float]] = []
    for fill in fills:
        ev = events_indexed.iloc[fill.event_idx]
        t = int(ev["ts"])
        day = int(ev["day"])
        sq = float(ev["signed_qty"])
        sign = _bias_sign(direction, sq)

        day_prices = prod_prices[prod_prices["day"] == day].sort_values("ts").reset_index(drop=True)
        day_ts = day_prices["ts"].to_numpy()
        day_bid = day_prices["bid_price_1"].to_numpy()
        day_ask = day_prices["ask_price_1"].to_numpy()
        day_mid = day_prices["mid_price"].to_numpy()

        entry_idx = int(np.searchsorted(day_ts, t, side="right"))
        exit_idx = entry_idx + horizon
        if entry_idx >= len(day_ts) or exit_idx >= len(day_ts):
            continue

        mid_exit = float(day_mid[exit_idx])
        bid_exit = float(day_bid[exit_idx])
        ask_exit = float(day_ask[exit_idx])
        fp = fill.fill_price
        if sign > 0:
            out.append((mid_exit - fp, bid_exit - fp))
        else:
            out.append((fp - mid_exit, fp - ask_exit))
    return out


# ── Decision rule ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CellMetrics:
    n_events: int
    n_fills: int
    fill_rate: float
    mean_pnl_mid: float
    std_pnl_mid: float
    t_stat_mid: float
    mean_pnl_flat: float
    std_pnl_flat: float
    t_stat_flat: float
    hit_rate_flat: float


N_FILLS_MIN = 30
FILL_RATE_MIN = 0.10
T_STAT_MIN = 2.0


def cell_passes(metrics: CellMetrics) -> bool:
    """Conservative-regime gate per spec:
        n_fills >= 30
        fill_rate >= 0.10
        mean_pnl_mid > 0 AND |t_stat_mid| >= 2.0
        mean_pnl_flat > 0
    """
    if metrics.n_fills < N_FILLS_MIN:
        return False
    if metrics.fill_rate < FILL_RATE_MIN:
        return False
    if metrics.mean_pnl_mid <= 0:
        return False
    if abs(metrics.t_stat_mid) < T_STAT_MIN:
        return False
    if metrics.mean_pnl_flat <= 0:
        return False
    return True


# ── Aggregator ───────────────────────────────────────────────────────────────


def _aggregate(
    pnls: list[tuple[float, float]],
    n_events: int,
) -> CellMetrics:
    n_fills = len(pnls)
    fill_rate = (n_fills / n_events) if n_events > 0 else 0.0
    if n_fills == 0:
        return CellMetrics(
            n_events=n_events, n_fills=0, fill_rate=fill_rate,
            mean_pnl_mid=0.0, std_pnl_mid=0.0, t_stat_mid=0.0,
            mean_pnl_flat=0.0, std_pnl_flat=0.0, t_stat_flat=0.0,
            hit_rate_flat=0.0,
        )
    mids = np.asarray([p[0] for p in pnls], dtype=float)
    flats = np.asarray([p[1] for p in pnls], dtype=float)
    mean_mid = float(mids.mean())
    mean_flat = float(flats.mean())
    std_mid = float(mids.std(ddof=1)) if n_fills >= 2 else 0.0
    std_flat = float(flats.std(ddof=1)) if n_fills >= 2 else 0.0
    t_mid = (n_fills ** 0.5) * mean_mid / std_mid if std_mid > 0 else 0.0
    t_flat = (n_fills ** 0.5) * mean_flat / std_flat if std_flat > 0 else 0.0
    hit_flat = float((flats > 0).mean())
    return CellMetrics(
        n_events=n_events, n_fills=n_fills, fill_rate=fill_rate,
        mean_pnl_mid=mean_mid, std_pnl_mid=std_mid, t_stat_mid=t_mid,
        mean_pnl_flat=mean_flat, std_pnl_flat=std_flat, t_stat_flat=t_flat,
        hit_rate_flat=hit_flat,
    )


def compute_cell_metrics(
    cell: TopKCell,
    events: pd.DataFrame,
    prices: pd.DataFrame,
    market_trades: pd.DataFrame,
) -> tuple[CellMetrics, CellMetrics]:
    """Run the simulator + PnL for one cell in both regimes. Returns (optimistic, conservative)."""
    bot_events = events[
        (events["bot"] == cell.bot) & (events["product"] == cell.product)
    ]
    n_events = int(len(bot_events))

    fills_opt = simulate_passive_fills(
        bot_events, prices, market_trades,
        cell.product, cell.horizon, cell.direction, "optimistic",
    )
    pnls_opt = compute_pnls(fills_opt, bot_events, prices, cell.product, cell.horizon, cell.direction)

    fills_cons = simulate_passive_fills(
        bot_events, prices, market_trades,
        cell.product, cell.horizon, cell.direction, "conservative",
    )
    pnls_cons = compute_pnls(fills_cons, bot_events, prices, cell.product, cell.horizon, cell.direction)

    return _aggregate(pnls_opt, n_events), _aggregate(pnls_cons, n_events)


# ── Report ───────────────────────────────────────────────────────────────────


def _fmt_metrics(m: CellMetrics) -> str:
    return (
        f"n_fill={m.n_fills:>4d} fill%={m.fill_rate:>5.2f} "
        f"mean_mid={m.mean_pnl_mid:+.3f} t_mid={m.t_stat_mid:+.2f} "
        f"mean_flat={m.mean_pnl_flat:+.3f} t_flat={m.t_stat_flat:+.2f}"
    )


def emit_report(
    results: list[tuple[TopKCell, CellMetrics, CellMetrics]],
) -> str:
    """Render per-cell metrics and the PASS / NO PASS verdict."""
    lines: list[str] = []
    lines.append("=== Inside-MM Bias EDA ===")
    lines.append(f"{len(results)} top-K cells. R4 days 1/2/3.")
    lines.append("")
    lines.append(
        f"{'bot':10s} {'product':14s} {'h':>3s} {'dir':>6s} {'n_ev':>5s}  "
        f"OPT[{'n_fill fill% mean_mid t_mid mean_flat t_flat'}]   "
        f"CONS[{'n_fill fill% mean_mid t_mid mean_flat t_flat'}]   PASS"
    )
    for cell, opt, cons in results:
        passes = cell_passes(cons)
        lines.append(
            f"{cell.bot:10s} {cell.product:14s} {cell.horizon:>3d} "
            f"{cell.direction:>6s} {opt.n_events:>5d}  "
            f"OPT[{_fmt_metrics(opt)}]   "
            f"CONS[{_fmt_metrics(cons)}]   "
            f"PASS={'YES' if passes else 'no'}"
        )

    lines.append("")
    lines.append("=== VERDICT ===")
    passing = [(c, o, k) for (c, o, k) in results if cell_passes(k)]
    if not passing:
        lines.append("NO PASS — close imitation-PnL bot-following branch")
    else:
        passing.sort(key=lambda r: -r[2].mean_pnl_flat)
        for cell, _opt, cons in passing:
            lines.append(
                f"PASS: {cell.bot} {cell.product} horizon={cell.horizon} "
                f"direction={cell.direction} "
                f"(cons: n={cons.n_fills} mean_flat={cons.mean_pnl_flat:+.3f}t "
                f"t_flat={cons.t_stat_flat:+.2f})"
            )
    return "\n".join(lines)


# ── Entry point ──────────────────────────────────────────────────────────────


VERDICT_FILE = str(_HERE / "imitation_pnl_verdict.txt")


def main() -> None:
    print("Loading R4 trades, prices, and market prints (days 1/2/3)...")
    bot_trades = load_trades_with_day()
    prices = load_prices_with_book()
    market_trades = load_market_trades()
    print(
        f"  {len(bot_trades)} bot-trade rows, {len(prices)} price ticks, "
        f"{len(market_trades)} market prints."
    )

    cells = parse_top_k_cells(VERDICT_FILE, n_min=30, t_min=1.5)
    print(f"  {len(cells)} top-K cells from {VERDICT_FILE}.")
    if not cells:
        print("No top-K cells. Exiting.")
        return

    results: list[tuple[TopKCell, CellMetrics, CellMetrics]] = []
    for cell in cells:
        opt, cons = compute_cell_metrics(cell, bot_trades, prices, market_trades)
        results.append((cell, opt, cons))

    print(emit_report(results))


if __name__ == "__main__":
    main()
