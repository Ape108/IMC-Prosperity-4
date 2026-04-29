# R5 EDA Triage Summary

> Extracted from `eda_r5_triage.ipynb` — all 10 groups, 3 days (2/3/4).
> Source notebook has full charts, per-day breakdowns, and styled tables.

## Key Finding: No Textbook Stat Arb Exists

**No group has the classic setup** (cointegrated pair + short half-life + lead-lag). Every pair across all 10 groups has half-life > 700 ticks and spreads that are I(1) on most days. The signals that DO exist are:

1. **Simultaneous negative return correlations** (mirror pairs) — SNACKPACK, PEBBLES
2. **Single-product mean reversion** via negative lag-1 autocorrelation — DISHES, IRONING, EVENING_BREATH, CHOCOLATE
3. **One stable lead-lag** — CIRCLE→OVAL in MICROCHIP (lag=-50, perfectly stable, but weak at |xcorr|=0.05)
4. **Sporadic spread stationarity** — LAUNDRY/VACUUMING I(0) on 2/3 days, CIRCLE/OVAL I(0) on day 2

---

## Tier 1: Strong Structural Relationships

### SNACKPACK
| Metric | Value |
|--------|-------|
| PC1 | **59.7%** (highest across all groups) |
| Avg spread | 16-18 (widest) |
| Cointegration | None (all pairs p > 0.05 all days) |
| Best half-life | 767 (RASPBERRY/VANILLA) |

**Mirror pairs (lag 0, stable across all 3 days):**

| Pair | Returns Corr | Stability (std) |
|------|-------------|-----------------|
| RASPBERRY / STRAWBERRY | **-0.923** | 0.008 |
| CHOCOLATE / VANILLA | **-0.915** | 0.004 |
| PISTACHIO / STRAWBERRY | **+0.913** | 0.001 |
| PISTACHIO / RASPBERRY | **-0.831** | 0.003 |

All remaining 6 pairs have correlations near 0 (< 0.05). The group splits into two anti-correlated clusters: {CHOCOLATE, PISTACHIO} vs {RASPBERRY, STRAWBERRY, VANILLA}. No lead-lag — all simultaneous.

**Spread stationarity:** All pairs I(1) on all days (diff form). No ratio spreads tested as stationary either.

**Verdict:** Strongest structural signal in the dataset. The anti-correlation is extremely strong and stable. But no mean-reversion in levels — these are return-space mirrors. Alpha angle is basket/hedge construction, not spread trading.

---

### PEBBLES
| Metric | Value |
|--------|-------|
| PC1 | **49.4%** |
| Avg spread | 10-17 (XL widest at 16.6) |
| Cointegration | None (all pairs p > 0.05 all days) |
| Best half-life | 1867 (M/XL) |

**XL is the anti-correlated outlier:**

| Pair | Returns Corr | Stability (std) | Lead-lag |
|------|-------------|-----------------|----------|
| M / XL | **-0.506** | 0.008 | lag=0, std=0.0 |
| L / XL | **-0.493** | 0.001 | lag=0, std=0.0 |
| S / XL | **-0.483** | 0.003 | lag=0, std=0.0 |
| XL / XS | **-0.475** | 0.005 | lag=0, std=0.0 |

All non-XL pairs have near-zero correlation. XL has 2x the volatility (range 8052 vs 3000-5400 for others) and the widest spread (16.6).

**Spread stationarity:** All I(1) except S/XS on day 4 only.

**Verdict:** XL is structurally different — moves opposite to the group. Stable and consistent but simultaneous (no lead-lag). XL's wide spread (16.6) makes capturing the anti-correlation expensive.

---

## Tier 2: Moderate Signals

### MICROCHIP
| Metric | Value |
|--------|-------|
| PC1 | 22.9% |
| Avg spread | 7-12 (SQUARE widest at 11.7) |
| Best half-life | 2857 (OVAL/TRIANGLE) |

**CIRCLE→OVAL lead-lag (most stable in the entire dataset):**

| Day | Peak Lag | Peak |xcorr| |
|-----|----------|------|
| 2 | -50 | ~ |
| 3 | -50 | ~ |
| 4 | -50 | ~ |
| **Combined** | **-50** | **0.051** |
| **Lag std** | **0.0** | |

Negative lag means CIRCLE leads (CIRCLE's past predicts OVAL's present). But |xcorr|=0.051 is weak.

**Cointegration (most green cells of any group):**
- CIRCLE/OVAL: p=0.008 on day 2 (only)
- CIRCLE/RECTANGLE: p<0.05 on day 2 (only)
- CIRCLE/SQUARE: p<0.05 on days 2 and 4

**Other:** SQUARE lag-1 autocorr = -0.022 (mild MR signal).

**Verdict:** Best lead-lag candidate but the signal is very weak. CIRCLE/OVAL has perfect lag stability but 0.05 correlation — after spread costs (~8 ticks), this probably doesn't survive. Worth deeper investigation of whether the 50-tick lag contains exploitable info at shorter horizons.

---

### ROBOT
| Metric | Value |
|--------|-------|
| PC1 | **39.8%** |
| Avg spread | **6-8 (tightest across all groups)** |
| Best half-life | 1049 (LAUNDRY/VACUUMING) |

**Single-product autocorrelation (strong):**

| Product | Lag-1 ACF (combined) | Day 2 | Day 3 | Day 4 |
|---------|---------------------|-------|-------|-------|
| DISHES | **-0.222** | 0.000 | -0.004 | **-0.289** |
| IRONING | **-0.121** | -0.156 | -0.080 | -0.115 |

DISHES autocorrelation is extremely strong on day 4 but near-zero on day 2 — unstable.
IRONING is more consistent across days.

**Spread stationarity:**
- LAUNDRY/VACUUMING: I(0) on days 2 AND 3 (best pair in dataset for multi-day stationarity)
- DISHES/MOPPING: I(0) on day 4 only
- LAUNDRY/MOPPING: I(0) on day 4 only

**Lead-lag:** All pairs weak (|xcorr| < 0.02).

**Verdict:** DISHES and IRONING have the strongest single-product mean-reversion in the dataset. LAUNDRY/VACUUMING has the best spread stationarity. But spreads are the tightest (6-8), meaning even small alpha gets eaten by transaction costs. The DISHES day-4-only autocorrelation is suspicious — could be a regime shift or artifact.

---

### OXYGEN_SHAKE
| Metric | Value |
|--------|-------|
| PC1 | 24.3% |
| Avg spread | 12-15 |
| Best half-life | 1594 (CHOCOLATE/GARLIC) |

**Single-product autocorrelation:**

| Product | Lag-1 ACF (combined) | Day 2 | Day 3 | Day 4 |
|---------|---------------------|-------|-------|-------|
| EVENING_BREATH | **-0.118** | -0.163 | -0.095 | -0.078 |
| CHOCOLATE | **-0.082** | -0.119 | -0.008 | -0.101 |

EVENING_BREATH is the most consistent negative autocorrelation across days.

**Lead-lag stability (best pair):**
- EVENING_BREATH/MINT: lag_std=5.8 (good stability), peak_lag=18, |xcorr|=0.023 (weak)
- EVENING_BREATH/MORNING_BREATH: lag_std=5.9, peak_lag=-5, |xcorr|=0.019

**Cointegration:** None (all pairs p > 0.05 all days).

**Verdict:** EVENING_BREATH autocorrelation is real and consistent. Moderate spreads (12) give room to capture it. Best single-product MR target after IRONING (which has tighter spreads). Lead-lag signals exist but are very weak.

---

## Tier 3: No Compelling Signal

### UV_VISOR
- PC1 = 20.5%. All return correlations near 0. No cointegration except sporadic single-day hits (AMBER/MAGENTA d3, AMBER/RED d2). Half-lives > 4700. Lead-lag all weak (< 0.02) and unstable (lag_std > 13 for all). Autocorrelation near 0 for all products.

### PANEL
- PC1 = 20.5%. Return correlations near 0. No cointegration. Half-lives > 2300. Lead-lag weak and unstable. Some pairs show I(0) on single days (1X2/2X4 d4, 1X4/2X2 d4) but no consistency.

### SLEEP_POD
- PC1 = 20.4%. Return correlations near 0. One cointegration hit: COTTON/POLYESTER d4 only. Half-lives > 1100. COTTON/POLYESTER has the best half-life (1108) but no multi-day support. Lead-lag all weak and mostly unstable.

### TRANSLATOR
- PC1 = 20.4%. Return correlations near 0. Sporadic cointegration (ASTRO_BLACK/GRAPHITE_MIST d2, ECLIPSE_CHARCOAL/GRAPHITE_MIST d4). Half-lives > 1400. Best lead-lag: ECLIPSE_CHARCOAL/SPACE_GRAY at lag=-1, |xcorr|=0.022 — close to simultaneous.

### GALAXY_SOUNDS
- PC1 = 20.4%. Return correlations near 0. No cointegration (all red). Half-lives > 1700. Lead-lag weak and unstable. Widest lead-lag stability ranges (lag_std > 15 for all pairs).

---

## Cross-Group Signal Ranking

| Rank | Signal | Group | Strength | Stability | Spread Cost |
|------|--------|-------|----------|-----------|-------------|
| 1 | Mirror pairs (|ret_corr| > 0.83) | SNACKPACK | Very strong | Very stable | Wide (17) — favorable |
| 2 | XL anti-correlation (-0.49) | PEBBLES | Strong | Perfect (std=0) | XL wide (17) |
| 3 | DISHES lag-1 autocorr (-0.22) | ROBOT | Strong | **Unstable** (day-dependent) | Tight (7) — unfavorable |
| 4 | IRONING lag-1 autocorr (-0.12) | ROBOT | Moderate | Stable | Tight (6) — unfavorable |
| 5 | EVENING_BREATH autocorr (-0.12) | OXYGEN_SHAKE | Moderate | Stable | Moderate (12) |
| 6 | CIRCLE→OVAL lead-lag (0.05) | MICROCHIP | Weak | Perfect (std=0) | Moderate (8) |
| 7 | LAUNDRY/VACUUMING spread I(0) | ROBOT | Moderate | 2/3 days | Tight (7) |
| 8 | CHOCOLATE autocorr (-0.08) | OXYGEN_SHAKE | Weak | Variable | Moderate (12) |

## Microstructure Overview

| Group | Avg Spread | Zero-Move % | Typical Non-Zero Move | Price Mean Range |
|-------|-----------|-------------|----------------------|-----------------|
| ROBOT | 6.4–8.0 | varies | varies | 8.7k–11.1k |
| MICROCHIP | 7.4–11.7 | varies | varies | 8.2k–13.6k |
| TRANSLATOR | 8.4–9.5 | varies | varies | 9.4k–10.9k |
| SLEEP_POD | 8.6–10.3 | varies | varies | 9.6k–11.8k |
| PANEL | 8.4–11.5 | varies | varies | 8.9k–11.3k |
| OXYGEN_SHAKE | 11.9–15.1 | varies | varies | 9.3k–11.9k |
| PEBBLES | 9.7–16.6 | varies | varies | 7.4k–13.2k |
| UV_VISOR | 10.3–14.1 | varies | varies | 7.9k–11.1k |
| GALAXY_SOUNDS | 13.1–14.5 | varies | varies | 10.2k–11.5k |
| SNACKPACK | 15.9–17.8 | varies | varies | 9.5k–10.7k |

---

## What the EDA Does NOT Tell Us

1. **Styled tables (Level 3 spread search, Level 4a cointegration)** rendered as Styler objects — the per-pair p-values are in the notebook but not fully extracted here. Cointegration summary above is inferred from the CSS styling (green = p<0.05, red = p>=0.05).
2. **Charts** — normalized price overlays, lead-lag xcorr plots, spread time series — all in the notebook. Worth reviewing visually for regime changes or structural breaks.
3. **Cross-group relationships** — not tested. All analysis is within-group only.
4. **Higher-order relationships** — PCA beyond PC1, multi-product baskets, non-linear spreads.
