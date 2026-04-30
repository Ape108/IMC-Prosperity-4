"""Entry #10 — Queue-vs-alpha decomposition across all shipped products.

Decomposes per-product total PnL into:
  queue_value = default_total - conservative_total   (queue-priority benefit)
  alpha       = conservative_total                    (signal-driven PnL)

Surfaces "queue-priority illusions" — products with alpha <= 0 whose default
PnL appears positive only because of optimistic queue-fill assumptions.

Input:  submissions/r5/phase2/results/baseline_pnl.csv (built in Task 2)
Output: stdout report; submissions/r5/phase2/results/queue_alpha_decomp.csv

NO modification of strategy_h.py. User decides all drop decisions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from _data import RESULTS_DIR

INPUT_CSV = RESULTS_DIR / "baseline_pnl.csv"
OUTPUT_CSV = RESULTS_DIR / "queue_alpha_decomp.csv"


def decompose(df: pd.DataFrame) -> pd.DataFrame:
    """Add queue_value and alpha columns; rank by alpha ascending."""
    out = df[["product", "default_total", "conservative_total"]].copy()
    out["queue_value"] = out["default_total"] - out["conservative_total"]
    out["alpha"] = out["conservative_total"]
    out["queue_pct"] = out["queue_value"] / out["default_total"].replace(0, pd.NA)
    return out.sort_values("alpha", ascending=True).reset_index(drop=True)


def print_report(decomp: pd.DataFrame) -> None:
    print("=" * 90)
    print("Entry #10 — Queue-vs-alpha decomposition")
    print("=" * 90)
    try:
        rel = INPUT_CSV.relative_to(Path.cwd())
    except ValueError:
        rel = INPUT_CSV
    print(f"  Input: {rel}")
    print(f"  {len(decomp)} shipped products in scope.")
    print()
    print(f"  {'product':<40} {'default':>10} {'cons':>10} {'queue_v':>10} {'alpha':>10} {'queue%':>8}")
    print("  " + "-" * 88)
    for _, row in decomp.iterrows():
        flag = "  <-- ILLUSION" if row["alpha"] <= 0 else ""
        qpct = f"{row['queue_pct']:.0%}" if pd.notna(row["queue_pct"]) else "n/a"
        print(
            f"  {row['product']:<40} {row['default_total']:>10.0f} {row['conservative_total']:>10.0f} "
            f"{row['queue_value']:>10.0f} {row['alpha']:>10.0f} {qpct:>8}{flag}"
        )

    print()
    print("-" * 90)
    print("Threshold: alpha <= 0 = queue-priority illusion (drop or downgrade candidate)")
    print("-" * 90)
    illusions = decomp[decomp["alpha"] <= 0]
    print(f"  {len(illusions)} products flagged as illusions:")
    for _, row in illusions.iterrows():
        print(f"    - {row['product']}: alpha={row['alpha']:.0f}, queue_value={row['queue_value']:.0f}")
    print()
    print(f"  Aggregate alpha across shipped products: {decomp['alpha'].sum():.0f}")
    print(f"  Aggregate default-total: {decomp['default_total'].sum():.0f}")
    print()
    print("=" * 90)
    print(f"NEXT QUESTION FOR USER: {len(illusions)} products have alpha <= 0 — which to drop?")
    print("=" * 90)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    decomp = decompose(df)
    decomp.to_csv(OUTPUT_CSV, index=False)
    print_report(decomp)


if __name__ == "__main__":
    main()
