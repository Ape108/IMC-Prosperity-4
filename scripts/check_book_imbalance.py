r"""
Order book imbalance signal check for ASH_COATED_OSMIUM.

Does total bid volume vs ask volume predict the next-tick price direction?

Run from PowerShell (project root):
    .venv\Scripts\python scripts\check_book_imbalance.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

SYMBOL = "ASH_COATED_OSMIUM"
DAYS = [-2, -1, 0]
DATASET = Path(r"\\wsl$\Ubuntu\home\heagen\prosperity_rust_backtester\datasets\round1")


def main() -> None:
    dfs = []
    for d in DAYS:
        df = pd.read_csv(DATASET / f"prices_round_1_day_{d}.csv", sep=";")
        df["day"] = d
        dfs.append(df)

    all_df = pd.concat(dfs)
    osm = all_df[all_df["product"] == SYMBOL].copy().sort_values(["day", "timestamp"]).reset_index(drop=True)

    # Total book volume each side
    osm["bid_vol"] = osm[["bid_volume_1", "bid_volume_2", "bid_volume_3"]].sum(axis=1)
    osm["ask_vol"] = osm[["ask_volume_1", "ask_volume_2", "ask_volume_3"]].sum(axis=1)
    total = osm["bid_vol"] + osm["ask_vol"]

    # Normalised imbalance: +1 = all bids, -1 = all asks, 0 = balanced
    osm["imbalance"] = (osm["bid_vol"] - osm["ask_vol"]) / total.replace(0, np.nan)

    # Forward returns at 1, 3, 5 ticks (within same day only)
    for lag in [1, 3, 5]:
        osm[f"fwd_{lag}"] = osm.groupby("day")["mid_price"].diff(lag).shift(-lag)

    clean = osm[["imbalance", "fwd_1", "fwd_3", "fwd_5"]].dropna()

    print("=" * 55)
    print("ORDER BOOK IMBALANCE → FORWARD RETURN CORRELATION")
    print("=" * 55)
    print(f"  Samples: {len(clean)}")
    print(f"  Imbalance mean : {clean['imbalance'].mean():+.4f}  (0 = balanced)")
    print(f"  Imbalance std  : {clean['imbalance'].std():.4f}")
    print()

    print(f"  {'Horizon':<12} {'Pearson r':>10}  signal?")
    print(f"  {'-'*40}")
    for col, label in [("fwd_1", "1-tick"), ("fwd_3", "3-tick"), ("fwd_5", "5-tick")]:
        r = clean["imbalance"].corr(clean[col])
        flag = " *** exploitable" if abs(r) > 0.05 else ""
        print(f"  {label:<12} {r:>+10.4f}{flag}")
    print()

    # Directional accuracy: when imbalance > 0, does price go up?
    print("=" * 55)
    print("DIRECTIONAL ACCURACY  (imbalance > 0 → price up?)")
    print("=" * 55)
    for col, label in [("fwd_1", "1-tick"), ("fwd_3", "3-tick"), ("fwd_5", "5-tick")]:
        pos_imb = clean[clean["imbalance"] > 0.1]   # meaningfully bid-heavy
        neg_imb = clean[clean["imbalance"] < -0.1]  # meaningfully ask-heavy

        if len(pos_imb) > 0 and len(neg_imb) > 0:
            pos_acc = (pos_imb[col] > 0).mean()
            neg_acc = (neg_imb[col] < 0).mean()
            print(f"  {label}: bid-heavy→up {pos_acc:.1%} ({len(pos_imb)} obs) | "
                  f"ask-heavy→down {neg_acc:.1%} ({len(neg_imb)} obs)")
        else:
            print(f"  {label}: not enough samples with |imbalance| > 0.1")
    print()

    print("  Interpretation:")
    print("    ~50% accuracy = no signal")
    print("    >55% consistently = exploitable, worth adding to MM fair-value skew")


if __name__ == "__main__":
    main()
