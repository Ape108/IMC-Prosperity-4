"""
Per-day lag-1 ACF stability check for MORNING_BREATH. Decides base MM vs autocorr overlay.

Kill criterion for autocorr overlay:
    max(|α_d − α_mean|) / |α_mean| < 0.5  →  stable → overlay candidate
    Else: base MM only.
"""

import pandas as pd
from pathlib import Path

PRODUCT = "OXYGEN_SHAKE_MORNING_BREATH"
DAYS = [2, 3, 4]
DATA_DIR = Path(__file__).parents[3] / "datasets" / "round5"

alphas: dict[int, float] = {}

for day in DAYS:
    path = DATA_DIR / f"prices_round_5_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    mid = df[df["product"] == PRODUCT]["mid_price"].reset_index(drop=True)
    returns = mid.pct_change().dropna()
    alpha_d = returns.autocorr(lag=1)
    alphas[day] = alpha_d
    print(f"Day {day}: alpha = {alpha_d:.4f}  (n={len(returns)} return obs)")

alpha_mean = sum(alphas.values()) / len(alphas)
max_dev = max(abs(alphas[d] - alpha_mean) for d in DAYS)
stability_ratio = max_dev / abs(alpha_mean) if alpha_mean != 0 else float("inf")

print()
print(f"alpha per day : {', '.join(f'{alphas[d]:.4f}' for d in DAYS)}")
print(f"alpha_mean    : {alpha_mean:.4f}")
print(f"max_dev       : {max_dev:.4f}")
print(f"ratio         : {stability_ratio:.3f}  (gate: < 0.5)")
print()
if stability_ratio < 0.5:
    print(f"STABILITY GATE: PASS  ->  autocorr overlay candidate  (alpha = {alpha_mean:.4f})")
else:
    print("STABILITY GATE: FAIL  ->  ship base MM only")
