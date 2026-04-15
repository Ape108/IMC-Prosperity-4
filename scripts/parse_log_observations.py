r"""
Parse competition log for ConversionObservation data (sunlight, humidity).

Run from PowerShell (project root):
    .venv\Scripts\python scripts\parse_log_observations.py

Each log line is JSON: [state, orders, conversions, trader_data, logs]
state[7] = [plainValueObservations, {product: [bid, ask, transport, exportTariff, importTariff, sunlight, humidity]}]
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

LOG_PATH = Path(r"logs\r1\sub5\128297.log")

# Indices within each conversion observation list
OBS_FIELDS = ["bidPrice", "askPrice", "transportFees", "exportTariff", "importTariff", "sunlight", "humidity"]


def parse_log(log_path: Path) -> pd.DataFrame:
    records = []
    errors = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            # entry = [state, orders, conversions, trader_data, logs]
            try:
                state = entry[0]
                timestamp = state[0]
                obs = state[7]             # [plainValueObs, conversionObs]
                conv_obs = obs[1]          # dict: product -> [bid, ask, ...]
            except (IndexError, TypeError, KeyError):
                errors += 1
                continue

            for product, values in conv_obs.items():
                record = {"timestamp": timestamp, "product": product}
                for i, field in enumerate(OBS_FIELDS):
                    record[field] = values[i] if i < len(values) else None
                records.append(record)

    if errors:
        print(f"  ({errors} lines skipped — parse errors or non-data lines)")

    return pd.DataFrame(records)


def main(log_path: Path) -> None:
    print("=" * 60)
    print(f"PARSING LOG: {log_path}")
    print("=" * 60)

    df = parse_log(log_path)

    if df.empty:
        print("No conversion observation data found in log.")
        print("This means state.observations.conversionObservations was empty")
        print("for every tick — Osmium has no external observation signal in R1.")
        return

    products = df["product"].unique().tolist()
    print(f"Products with conversion observations: {products}")
    print(f"Total records: {len(df)}")
    print()

    for product in products:
        pdf = df[df["product"] == product].sort_values("timestamp")
        print(f"--- {product} ---")
        for field in OBS_FIELDS:
            col = pdf[field].dropna()
            if col.empty:
                print(f"  {field:<20}: all null")
            else:
                print(f"  {field:<20}: [{col.min():.4f}, {col.max():.4f}]  mean={col.mean():.4f}  n={len(col)}")
        print()

        # Print first 5 rows
        print("  First 5 rows:")
        print(pdf.head().to_string(index=False))
        print()

    # If sunlight or humidity present, check correlation with Osmium mid-price
    obs_products = df["product"].unique().tolist()
    has_sunlight = df["sunlight"].notna().any()
    has_humidity = df["humidity"].notna().any()

    if not (has_sunlight or has_humidity):
        print("sunlight and humidity are both null across all products.")
        print("The hidden Osmium pattern is NOT accessible via observation variables in R1.")
        return

    print("=" * 60)
    print("SUNLIGHT / HUMIDITY OVER TIME")
    print("=" * 60)

    fig, axes = plt.subplots(len(obs_products), 2, figsize=(14, 4 * len(obs_products)), squeeze=False)

    for row_i, product in enumerate(obs_products):
        pdf = df[df["product"] == product].sort_values("timestamp")

        ax_s = axes[row_i][0]
        if has_sunlight and pdf["sunlight"].notna().any():
            ax_s.plot(pdf["timestamp"], pdf["sunlight"], color="#FF9800", linewidth=0.8)
            ax_s.set_title(f"{product} — sunlight")
            ax_s.set_xlabel("Timestamp")
            ax_s.grid(True, alpha=0.25)
        else:
            ax_s.set_visible(False)

        ax_h = axes[row_i][1]
        if has_humidity and pdf["humidity"].notna().any():
            ax_h.plot(pdf["timestamp"], pdf["humidity"], color="#2196F3", linewidth=0.8)
            ax_h.set_title(f"{product} — humidity")
            ax_h.set_xlabel("Timestamp")
            ax_h.grid(True, alpha=0.25)
        else:
            ax_h.set_visible(False)

    plt.tight_layout()
    out_path = Path("osmium_log_observations.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse competition log for ConversionObservation data")
    parser.add_argument("--log", type=Path, default=LOG_PATH)
    args = parser.parse_args()
    main(args.log)
