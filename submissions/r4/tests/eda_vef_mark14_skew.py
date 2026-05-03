"""Gate A: Mark 14 × VELVETFRUIT_EXTRACT buyer/seller directional skew check.

Decision rule: |net_direction| >= 0.15 → proceed to Gate B.
               |net_direction| <  0.15 → close Mark 14 × VEF branch.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from submissions.r4.tests.eda_imitation_pnl import load_trades_with_day

PRODUCT = "VELVETFRUIT_EXTRACT"
BOT = "Mark 14"
THRESHOLD = 0.15


def compute_skew(trades):
    """Return (n_buys, n_sells, net_direction) for BOT on PRODUCT."""
    m14_vef = trades[(trades["bot"] == BOT) & (trades["product"] == PRODUCT)]
    n_buys = int((m14_vef["signed_qty"] > 0).sum())
    n_sells = int((m14_vef["signed_qty"] < 0).sum())
    total = n_buys + n_sells
    net = (n_buys - n_sells) / total if total > 0 else 0.0
    return n_buys, n_sells, net


def main() -> None:
    trades = load_trades_with_day()
    m14_vef = trades[(trades["bot"] == BOT) & (trades["product"] == PRODUCT)]

    print(f"{BOT} × {PRODUCT} activity (days 1/2/3):")
    for day in [1, 2, 3]:
        d = m14_vef[m14_vef["day"] == day]
        n_b = int((d["signed_qty"] > 0).sum())
        n_s = int((d["signed_qty"] < 0).sum())
        print(f"  Day {day}: buys={n_b:>3d}  sells={n_s:>3d}")

    n_buys, n_sells, net = compute_skew(trades)
    total = n_buys + n_sells
    print(f"\n  TOTAL  buys={n_buys:>3d}  sells={n_sells:>3d}  "
          f"net_direction={net:+.3f}  (n={total})")

    if total == 0:
        print("\nVERDICT: FAIL — no Mark 14 VEF trades found")
        return

    if abs(net) >= THRESHOLD:
        print(f"\nVERDICT: PASS — directional skew |{net:.3f}| >= {THRESHOLD}")
        print("         Proceed to Gate B: eda_vef_imitation_pnl.py")
    else:
        print(f"\nVERDICT: FAIL — symmetric |{net:.3f}| < {THRESHOLD}")
        print("         Mark 14 is a market-maker on VEF. Close branch.")


if __name__ == "__main__":
    main()
