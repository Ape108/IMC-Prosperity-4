# R5 EDA Notebook — What We Built and Why

## Context

50 products, 10 groups of 5, position limit 10. No counterparty data means no Mark bot analysis. The primary alpha thesis is **within-group statistical arbitrage** — find pairs that are cointegrated or have lead-lag structure, trade the spread mean-reversion. The notebook runs a standardized 5-level template on every group so we can compare groups apples-to-apples and rank them by exploitability.

---

## Microstructure Overview (runs once at the top)

**Goal:** Get simulator-level properties for all 10 groups before doing any stat arb analysis.

**What it measures:** Mean bid-ask spread, fraction of ticks with zero price movement, typical non-zero move size, and price level — one representative product per group, all three days.

**Why it matters:** Spread size is the round-trip cost floor. A stat arb signal only generates alpha if the edge exceeds the spread. This is how we established the spread tiers: SNACKPACK (17-tick spread) is the most naturally exploitable; ROBOT (7-tick) needs a much stronger signal to pay its way. Also calibrates move size — tells us whether prices tick in whole integers or fractions, which affects how we define the spread.

---

## Level 1 — Individual Series Characterization

**Goal:** Determine whether each product's price series is stationary (I(0)) or has a unit root (I(1)), and whether returns have autocorrelation.

**What it runs:** Augmented Dickey-Fuller (ADF) test per product per day, plus combined across all days. Also lag-1 autocorrelation of percentage returns.

**Why it matters:** Engle-Granger cointegration is only valid if both series are I(1) — testing a stationary series for cointegration produces a meaningless result. If a product is already I(0), we need a different strategy (direct mean reversion, not pair trading). The lag-1 autocorr tells us whether short-horizon momentum or mean-reversion exists in individual returns — relevant for signal decay assumptions.

---

## Level 2 — Cross-Sectional Structure

**Goal:** Understand how the 5 products in a group move relative to each other — both in price levels and in returns.

**What it shows:**

- **Correlation heatmaps** on levels and returns, per day and combined. High level correlation is expected (shared macro exposure); high *returns* correlation means they actually co-move tick-by-tick.
- **Normalized price overlay** — plots all 5 products' price paths on the same scale (% change from start) so you can visually see which products diverge, converge, or track each other.
- **PCA on returns** — decomposes the group's return covariance into principal components. A dominant PC1 (>70% of variance) means the group is essentially one factor — all 5 products move together, limiting pair opportunities. Low PC1 concentration means the group has genuine idiosyncratic spread between members.

**Why it matters:** Tells us at a glance whether a group is "one thing" or has exploitable structure between members. If PC1 explains 95%, pairs will be highly correlated but the residual spread will be tiny. If PC1 explains 40%, there's real relative movement to capture.

---

## Level 3 — Spread Definition Search

**Goal:** For every pair (10 pairs per group), determine whether the spread is better expressed as a **difference** (A − B) or a **ratio** (A / B).

**What it tests:** ADF on `A − B` and `A / B` for each pair, per day and combined. The form with the lower ADF p-value (more stationary) is flagged as `better_form`.

**Why it matters:** For PEBBLES and PANEL, where prices scale with size (XS is cheap, XL is expensive), the ratio spread `A/B` is far more likely to be stationary than the raw difference — the levels relationship is multiplicative, not additive. Getting the spread form wrong means the ADF and half-life tests downstream will fail even if the pair is truly cointegrated. This was identified as a critical distinction before the notebook was built.

---

## Level 4a — Engle-Granger Cointegration

**Goal:** Test whether each pair is cointegrated — i.e., whether a linear combination of their levels is stationary even though each individual series is non-stationary.

**What it runs:** `statsmodels.tsa.stattools.coint` on every pair, per day. Results are color-coded green (<0.05) or red (≥0.05).

**Why it's "informational, not a gate":** EG requires both series to be I(1) — Level 1 is the precondition check. Also EG is a low-power test on 2500-tick series; a failing EG doesn't mean the pair isn't exploitable, it just means the stationarity is weak. We look at it alongside Level 3 ADF (which directly tests spread stationarity) rather than gatekeeping on it. Strong EG signal on all 3 days is a green flag, not a strict requirement.

---

## Level 4b — Half-Life

**Goal:** Measure how quickly the spread mean-reverts, using the Ornstein-Uhlenbeck half-life formula.

**What it calculates:** For each pair using the best spread form from Level 3, fits the regression `ΔS_t = β·S_{t-1} + ε`. Half-life = `−ln(2)/β`. Also plots the spread time series per day to visually confirm mean-reversion behavior.

**Why it matters:** Half-life is the key operational parameter. A spread with 50-tick half-life can be traded within a single day (2500 ticks). A 500-tick half-life means the strategy needs multiple days to close, increasing exposure to regime change. Our criteria for exploitability is half-life < 200 ticks. Also reveals if half-life is consistent across days (stable regime) vs. wildly variable (fragile).

---

## Level 4c — Lead-Lag

**Goal:** Find pairs where one product reliably moves before the other, creating a predictive signal.

**What it runs:** Cross-correlation of percentage returns across all lags from −50 to +50 ticks, per day and combined. Sorts all pairs by peak absolute cross-correlation. Direction convention: `peak_lag > 0` means **B leads A** (B's past correlates with A's present).

**Why it matters:** Even if a pair isn't cointegrated, a stable lead-lag relationship lets you trade the lagger after the leader moves. The plot shows the cross-correlation profile for all three days overlaid — if the peak is at the same lag on all 3 days, the signal is real structure, not noise. A flat profile (no clear peak) means no lead-lag exists.

---

## Level 5 — Multi-Day Stability

**Goal:** Validate that everything found in Levels 1–4 is stable across the 3 dataset days, not just a single-day artifact.

**What it checks:**

1. **Returns correlation stability** — pairwise returns correlation per day, sorted by standard deviation across days. Low std = stable relationship.
2. **Spread stationarity stability** — ADF on difference spread per day. A pair that's I(0) on days 2 and 3 but I(1) on day 4 is a regime-change warning.
3. **Lead-lag stability** — peak lag per day for every pair, sorted by standard deviation of the lag. If the peak lag moves from +3 to −8 between days, the signal is noise.

**Why it matters:** A signal that only exists on one day almost certainly won't survive the hidden eval days. Stability is the final filter before committing to implementation.

---

## Summary Table (bottom of notebook)

An empty template to fill in as each group's analysis completes, recording: cointegrated pairs, best half-life, lead-lag existence, and strategy class (stat arb, MM, or skip). This is the triage ranking output — the direct input to implementation prioritization.

| Group | Cointegrated Pairs | Best Half-Life | Lead-Lag? | Strategy Class |
|-------|-------------------|----------------|-----------|---------------|
| SNACKPACK | | | | |
| UV_VISOR | | | | |
| PEBBLES | | | | |
| PANEL | | | | |
| OXYGEN_SHAKE | | | | |
| SLEEP_POD | | | | |
| MICROCHIP | | | | |
| ROBOT | | | | |
| TRANSLATOR | | | | |
| GALAXY_SOUNDS | | | | |
