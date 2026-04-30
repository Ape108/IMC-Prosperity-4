"""Entry #12 — PEBBLES_M <-> XL parameter sensitivity sweep.

Grid backtest over (window, z_entry, z_exit, max_hold_ticks) on the
PEBBLES_M <-> XL pair trade. Output: PnL surface per-day and combined,
plus stability classification (robust plateau vs single-day cliff).

The pair is the single largest line item in strategy_h.py
(+71,956 conservative). Cliff vs plateau matters because regime shift
in eval would blow up a cliffed pair's PnL.

Mechanism: writes a temp trader file with each parameter combo, invokes
rust_backtester via subprocess, parses the per-product PnL line for
PEBBLES_M and PEBBLES_XL.

REQUIRES: WSL2 (rust_backtester is a Linux binary). Run from WSL2 shell.
$PROSP4 must be set in WSL2 ~/.bashrc to the project root.

Input:  None (writes temp trader files in /tmp).
Output: stdout report; submissions/r5/phase2/results/pebbles_param_sweep.csv

NO modification of strategy_h.py. User decides param adjustments.
"""

import itertools
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from _data import RESULTS_DIR

OUTPUT_CSV = RESULTS_DIR / "pebbles_param_sweep.csv"

# Grid (4*4*4*4 = 256 combos)
WINDOWS = (100, 200, 400, 800)
Z_ENTRIES = (1.5, 2.0, 2.5, 3.0)
Z_EXITS = (0.3, 0.5, 0.8, 1.2)
MAX_HOLDS = (200, 500, 1000, 2000)

CURRENT_PARAMS = (200, 2.0, 0.5, 500)

PROSP4 = Path(os.environ.get("PROSP4", str(Path(__file__).resolve().parents[3])))
RUST_BACKTESTER_DIR = Path(os.environ.get("HOME", ".")) / "prosperity_rust_backtester"
if not RUST_BACKTESTER_DIR.exists():
    raise RuntimeError(
        f"rust_backtester dir not found at {RUST_BACKTESTER_DIR}. "
        "Run this script from WSL2 with $HOME pointing to your Linux home dir."
    )


def make_trader_src(window: int, z_entry: float, z_exit: float, max_hold: int) -> str:
    """Build a self-contained trader file for the given pair-trade params.

    Reads the boilerplate (Logger / Strategy / StatefulStrategy / R5PairTradeStrategy)
    from submissions/r5/groups/panel.py.
    """
    panel_path = PROSP4 / "submissions" / "r5" / "groups" / "panel.py"
    panel_text = panel_path.read_text()
    start = panel_text.index("class Logger:")
    end = panel_text.index("# ── Strategy variants")
    boilerplate = panel_text[start:end]

    return f'''"""Temp pebbles M<->XL pair trade for param sweep (window={window}, z_entry={z_entry}, z_exit={z_exit}, max_hold={max_hold})."""
import json
from abc import abstractmethod
from pathlib import Path
from typing import Any
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

{boilerplate}

LIMIT = 10

class Trader:
    def __init__(self) -> None:
        self.strategies = {{
            "PEBBLES_M": R5PairTradeStrategy(
                symbol_a="PEBBLES_M",
                symbol_b="PEBBLES_XL",
                limit=5,
                limit_b=LIMIT,
                window={window},
                z_entry={z_entry},
                z_exit={z_exit},
                max_hold_ticks={max_hold},
            )
        }}

    def run(self, state):
        orders = {{}}
        conversions = 0
        raw = json.loads(state.traderData) if state.traderData not in ("", None) else {{}}
        old = raw if isinstance(raw, dict) else {{}}
        new = {{}}
        for sym, strat in self.strategies.items():
            if isinstance(strat, StatefulStrategy) and sym in old:
                strat.load(old[sym])
            o, c = strat.run(state)
            for order in o:
                orders.setdefault(order.symbol, []).append(order)
            conversions += c
            if isinstance(strat, StatefulStrategy):
                new[sym] = strat.save()
        td = json.dumps(new, separators=(",", ":"))
        logger.flush(state, orders, conversions, td)
        return orders, conversions, td

logger = Logger()
'''


def parse_pnl(stdout: str) -> tuple[float, float]:
    """Extract PEBBLES_M and PEBBLES_XL totals from rust_backtester stdout."""
    m_pnl = xl_pnl = 0.0
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("PEBBLES_M "):
            nums = re.findall(r"-?\d+\.?\d*", s)
            if nums:
                m_pnl = float(nums[-1])
        elif s.startswith("PEBBLES_XL "):
            nums = re.findall(r"-?\d+\.?\d*", s)
            if nums:
                xl_pnl = float(nums[-1])
    return m_pnl, xl_pnl


def run_backtest(window: int, z_entry: float, z_exit: float, max_hold: int, conservative: bool) -> tuple[float, float]:
    trader_src = make_trader_src(window, z_entry, z_exit, max_hold)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(trader_src)
        trader_path = f.name

    cmd = [
        "rust_backtester",
        "--trader", trader_path,
        "--dataset", "round5",
        "--persist", "--carry",
    ]
    if conservative:
        cmd += ["--queue-penetration", "0", "--price-slippage-bps", "5"]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=RUST_BACKTESTER_DIR, timeout=120)
        return parse_pnl(proc.stdout)
    finally:
        os.unlink(trader_path)


def main() -> None:
    print("=" * 90)
    print("Entry #12 — PEBBLES_M <-> XL parameter sensitivity sweep")
    print("=" * 90)
    grid = list(itertools.product(WINDOWS, Z_ENTRIES, Z_EXITS, MAX_HOLDS))
    print(f"  Grid size: {len(grid)} combos.")
    print(f"  Current params: window={CURRENT_PARAMS[0]} z_entry={CURRENT_PARAMS[1]} z_exit={CURRENT_PARAMS[2]} max_hold={CURRENT_PARAMS[3]}")

    rows = []
    for n, (w, ze, zx, mh) in enumerate(grid, 1):
        if n % 25 == 0:
            print(f"  ... {n}/{len(grid)} ...")
        m_def, xl_def = run_backtest(w, ze, zx, mh, conservative=False)
        m_cons, xl_cons = run_backtest(w, ze, zx, mh, conservative=True)
        rows.append({
            "window": w, "z_entry": ze, "z_exit": zx, "max_hold": mh,
            "default_m": m_def, "default_xl": xl_def, "default_total": m_def + xl_def,
            "conservative_m": m_cons, "conservative_xl": xl_cons,
            "conservative_total": m_cons + xl_cons,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    cur = df[
        (df["window"] == CURRENT_PARAMS[0])
        & (df["z_entry"] == CURRENT_PARAMS[1])
        & (df["z_exit"] == CURRENT_PARAMS[2])
        & (df["max_hold"] == CURRENT_PARAMS[3])
    ].iloc[0]
    cur_default = cur["default_total"]
    cur_cons = cur["conservative_total"]

    best_default = df.loc[df["default_total"].idxmax()]
    best_cons = df.loc[df["conservative_total"].idxmax()]

    print()
    print("=" * 90)
    print(f"Current point default: {cur_default:.0f} ; conservative: {cur_cons:.0f}")
    print(f"Best default: {best_default['default_total']:.0f} at "
          f"window={best_default['window']} z_entry={best_default['z_entry']} "
          f"z_exit={best_default['z_exit']} max_hold={best_default['max_hold']}")
    print(f"Best conservative: {best_cons['conservative_total']:.0f} at "
          f"window={best_cons['window']} z_entry={best_cons['z_entry']} "
          f"z_exit={best_cons['z_exit']} max_hold={best_cons['max_hold']}")

    threshold = 0.9 * cur_default
    plateau = df[df["default_total"] >= threshold]
    print(f"Default plateau (>= 90% of current): {len(plateau)} of {len(df)} grid points.")
    if len(plateau) >= len(df) * 0.5:
        cls = "ROBUST PLATEAU (>50% of grid within 10% of current)"
    elif len(plateau) <= len(df) * 0.1:
        cls = "FRAGILE CLIFF (<10% of grid within 10% of current)"
    else:
        cls = "MIXED"
    print(f"Classification: {cls}")

    print()
    print("=" * 90)
    print("NEXT QUESTION FOR USER: keep current params, switch to best-conservative point, "
          "or further investigate plateau structure?")
    print("=" * 90)


if __name__ == "__main__":
    main()
