# R5 EDA — Second-Pass Backlog

> EDA techniques noticed during per-group debugging that
> weren't in the first-pass notebook (`eda_r5_triage.ipynb`).
> **Append entries only AFTER backtests complete and a hypothesis
> is formed about what variants did/didn't work.** This doc is forward-looking
> ("what EDA to try next if Phase 1 didn't work"), but its entries must be
> evidence-grounded, not design-time guesses.
>
> Scope: cross-group or per group analytical methods only
> Goal: After testing all groups, utilize this file to create a second-pass notebook to find new signal
>
> Each entry: name / what it would test / why we didn't try first /
> expected difficulty / inspired by.

---

## 1. Lag-N cross-correlation matrix per within-group pair

- **What it would test:** Pearson correlation of mid-return series at lags k ∈ {−3, −2, −1, 0, +1, +2, +3} for every within-group pair (10 pairs/group). Identify whether the dominant peak sits at lag 0 (information already in microprice → don't overlay) or at non-zero lag (predictive content available → overlay candidate).
- **Why we didn't try first:** First-pass EDA used contemporaneous correlation only. The Phase 1 corr / tick-residual overlays presupposed lag-1 predictive content; we needed empirical lag-N evidence to distinguish "simultaneous mirror" from "lead-lag."
- **Expected difficulty:** Easy. ~20 lines of pandas; one heatmap per group.
- **Inspired by:** SNACKPACK lag-1 corr overlay net −95k across 5 products (`snackpack.py::corr`); tick-residual lean −568k (`snackpack.py::tick_lean`) — both fail in the way a lag-0 pair predicts they would. PEBBLES XL-skew selectivity provides the converse evidence: asymmetric one-sided XL-driven lean (threshold=0.001, k_ticks=2) rescued PEBBLES_XS (−8,261 → −1,524; +6,737 delta, almost entirely Day 3) but had near-zero impact on S (−1,500), M (−1,973), L (−259). Non-monotonic in size — only XS responds — suggests the XL → {XS,S,M,L} cross-correlation has a peaked lag/magnitude structure that the matrix would expose. The matrix would tell us whether XL-driven overlays should fire on every small-size PEBBLES or only XS, instead of the current shipped configuration which applies XL-skew to PEBBLES_S without evidence of selectivity.

## 2. Position-utilization distribution per product

- **What it would test:** Histogram of `|position|` over each trading day per product; percentile of ticks where position sits within 1 / 2 / 3 of the ±10 limit. Confirms or rejects the hypothesis that capacity-aware skews don't engage at limit=10.
- **Why we didn't try first:** Phase 1 wired `R5BasketCapMMStrategy` assuming the system-level skew would differentiate from base MM. Identical PnL numbers in `snackpack.py::basket_cap` vs `snackpack.py::mm` (matching to the digit on all 5 products × 3 days) demanded an explanation; we need the position trace to confirm "factor never bites."
- **Expected difficulty:** Easy. Position trace is already in the backtester's per-tick CSV.
- **Inspired by:** `snackpack.py` showing `basket_cap()` PnL ≡ `mm()` PnL — CHOC +11,382, RASP +8,044, PIST +4,553, STRAW −1,097, VAN −3,818, identical totals across both variants.

## 3. Per-day regime score regressed on per-product PnL

- **What it would test:** For each of days 2 / 3 / 4 compute (a) trend metrics (absolute cumulative log return, realized variance, max drawdown of cumulative mid) and (b) serial-correlation metrics (realized lag-1 ACF of mid-returns over the same window an autocorr overlay would use). Regress per-product per-day PnL against each metric. Trend metrics classify trend-fragile vs trend-neutral products; serial-correlation metrics classify whether overlay-based strategies (autocorr lean, partner lean) had usable signal that day.
- **Why we didn't try first:** First-pass EDA looked at marginal product properties (price level, volatility, spread). It did not classify days, and treated autocorrelation as a stationary per-product property. Phase 1 surfaced both regime axes empirically — VAN / STRAW lose Day 3 only (price regime); DISHES day-4 jackpot, IRONING day-2 jackpot (serial-correlation regime) — but we have no quantitative day label on either axis to gate strategies on.
- **Expected difficulty:** Medium. Trend score and rolling ACF are straightforward; the regression is data-thin (3 days × N products) and needs care.
- **Inspired by:** SNACKPACK base-MM Phase 1: VAN −2,964 / STRAW −1,800 on Day 3, recovery Day 4 (price-regime motivation). ROBOT autocorr "lottery" pattern: DISHES (−30k / −34k / +383k); IRONING (+30k / −7k / −4k) (serial-correlation regime motivation — combined-day lag-1 ACF=−0.222 fit to a regime that was day-4-only).

## 4. Adverse-selection metric per product

- **What it would test:** For each fill in the `--queue-penetration 0` log, compute `signed_loss = side · (fill_price − mid_{t+k})` for k ∈ {1, 5, 10} ticks. Aggregate mean and tail; estimate per-fill expected pickoff. Decompose Phase 1 PnL into edge × volume × adverse-selection per product.
- **Why we didn't try first:** Phase 1 used PnL as the only fitness metric. PnL is a noisy aggregate; it cannot tell us whether PEBBLES failed from low edge, low volume, or pickoff — and the fix differs.
- **Expected difficulty:** Medium. Need post-hoc join of fill log with mid-price series.
- **Inspired by:** PEBBLES_S base MM (overlay removed) +40,098 vs PEBBLES_M base MM −25,618 — same MM logic, same width=2, comparable price tier within the PEBBLES family. The "high price → pickoff dominates" framing from the Phase-1-with-overlays −490k figures cannot differentiate S (works) from M (fails). Per-fill signed-loss decomposition at k ∈ {1, 5, 10} ticks across S, M, L (−2,985), XS (−8,261) under base MM would reveal whether the differentiator is edge (per-fill margin), volume (insufficient flow), or pickoff tail. Without this, "PEBBLES MM is broken on high-priced products" persists as a folk explanation contradicted by the within-PEBBLES variance.

## 5. Within-group pair cointegration screen

- **What it would test:** Engle-Granger test on mid-prices for every within-group pair (10/group). Filter for p < 0.05 on all 3 days. For survivors compute (a) OU half-life of the spread, (b) spread σ in ticks, (c) round-trip cost (2 × half-spread + queue penalty); rank by edge / cost ratio.
- **Why we didn't try first:** First-pass EDA picked pair candidates by inspection — PEBBLES_M ↔ XL; ROBOT_LAUNDRY ↔ VACUUMING. One win, one loss, no systematic basis. We need a screen to replace inspection before nominating the next pair candidate.
- **Expected difficulty:** Medium. `statsmodels.tsa.stattools.coint` is one-line; the cost / edge gate is the real engineering.
- **Inspired by:** PEBBLES_M ↔ XL pair trade: +120k default / +70k conservative (validated; nominated from explicit size-ordering structure XS→XL). ROBOT_LAUNDRY ↔ VACUUMING: −9,658 (nominated without any cointegration evidence — arbitrary first stab on a group with no obvious pairing axis). The screen is needed because pair selection so far has either followed visible structure (succeeded) or guessed (failed).

## 6. Rolling lag-1 ACF per ROBOT product, intra-day

- **What it would test:** Estimate lag-1 autocorrelation of mid-returns on a rolling N-tick window (try N ∈ {200, 500, 1000}) for each ROBOT product, segmented by day. Identify when ACF goes meaningfully negative (e.g. < −0.1) and how long the regime persists. Output: per-product per-day ACF time series, plus a gate criterion ("activate autocorr overlay only when rolling ACF has been negative for K consecutive ticks").
- **Why we didn't try first:** Phase 1 fit a single static α per product (DISHES α=0.222, IRONING α=0.121) using combined-day ACF estimates — treated autocorrelation as a stationary property. The backtest pattern revealed it isn't.
- **Expected difficulty:** Easy. `pd.Series.rolling(N).apply(autocorr_lag1)` plus a heatmap.
- **Inspired by:** ROBOT autocorr deep dive. DISHES static α=0.222: −30k / −34k / +383k (jackpot day-4 only). IRONING α=0.121: +30k / −7k / −4k (jackpot day-2 only). Static α is fit to a transient regime; concentrated wins / distributed losses is the fingerprint of an intermittent signal applied as if stationary.

## 7. MOPPING structural diagnostic — why every MM variant loses

- **What it would test:** ROBOT-specific specialization of entry 4 (adverse selection). Compute (a) per-fill signed loss at k ∈ {1, 5, 10} ticks for MOPPING vs DISHES (similar price level, profitable at MM); (b) trade-size distribution MOPPING vs DISHES; (c) book-replenishment latency after a top-of-book fill; (d) order-flow imbalance (depth-weighted bid vs ask). If standard adverse-selection metric is comparable to DISHES, escalate through (b)–(d). Goal: identify the structural property of MOPPING that makes every quoting width unprofitable.
- **Why we didn't try first:** Phase 1 used PnL as the only metric. PnL says MOPPING loses; it doesn't say why. Without a diagnostic the only response is "drop" — and we can't rule out that the same structural property exists on other products we ship.
- **Expected difficulty:** Medium. Per-fill join with mid trajectory is entry 4's machinery; (b)–(d) need additional book-snapshot processing.
- **Inspired by:** ROBOT deep dive — MOPPING is the unique structural loser: w=1 −21,336, w=2 −18,403, every overlay net-negative. DISHES at the same widths and a comparable price level is profitable. Whatever distinguishes MOPPING from DISHES is the diagnostic target.

## 8. ROBOT 10-pair cointegration screen

- **What it would test:** ROBOT-specific instance of entry 5 with cost threshold set from the group's ~7.2-tick average spread (per first-pass triage). Run Engle-Granger on all 10 within-group pairs from {DISHES, IRONING, LAUNDRY, MOPPING, VACUUMING}. Filter for p<0.05 on all 3 days. For survivors compute (a) OU half-life, (b) spread σ in ticks, (c) round-trip cost ≈ 7.2 ticks. Survivors with σ > round-trip cost are candidate replacements for the failed LAUNDRY↔VACUUMING.
- **Why we didn't try first:** Only 1 of 10 ROBOT pairs was tested (LAUNDRY↔VACUUMING) and it was selected without evidence — not an inspection screen, just a guess. Nine untested combinations remain. Entry 5 covers this in principle, but ROBOT's narrow spread makes the cost gate the binding constraint and deserves a targeted run before any second ROBOT pair-trade is shipped.
- **Expected difficulty:** Easy. 10 `coint()` calls; ranking is one sort.
- **Inspired by:** ROBOT_LAUNDRY ↔ VACUUMING −9,658 — only ROBOT pair tested, chosen without quantitative basis. PEBBLES_M ↔ XL succeeded after evidence-backed nomination — shows the screen is the right gate, and the ROBOT instance is where the gate is most likely to bite.

## 9. Group-residual MM for MOPPING / VACUUMING

- **What it would test:** Estimate a ROBOT group factor as a basket of {DISHES, IRONING, LAUNDRY} (the three profitable members) — weights from PCA or simple equal-weight. For each of MOPPING and VACUUMING, compute residual = own_mid − basket-implied price. Quote only when |residual| exceeds a threshold (proxy for "the market is mispricing the residual relative to the group factor"). Test whether filtering MM activity to high-|residual| ticks improves PnL on the loss-leaders.
- **Why we didn't try first:** Phase 1 ran independent per-product MM. No cross-product context entered the quoting decision for any ROBOT product. If MOPPING/VACUUMING losses come from being on the wrong side of group-wide flow, gating on residual reduces fills exactly when adverse selection is highest.
- **Expected difficulty:** Medium. Factor estimation is one regression; gating logic is straightforward.
- **Inspired by:** ROBOT deep dive — 2 of 5 group products consistently lose at every tested MM width (MOPPING −21k, VACUUMING −4.5k); the other 3 are profitable. Asymmetric within-group PnL pattern that pure per-product MM cannot exploit.

## 10. Queue-priority vs alpha PnL decomposition (per-product)

- **What it would test:** For each product, compute `queue_value = PnL_default − PnL_qp0` and `alpha = PnL_qp0`. Apply per-product to all 5 PEBBLES under base MM and to each pair-trade leg separately. Output: (queue, alpha, total) triple per product. Decision rule for ship-or-drop: alpha > 0. Flag products where total > 0 but alpha ≤ 0 as "queue-priority illusions" — fragile under any realistic fill model.
- **Why we didn't try first:** Phase 1 ran both default and `--queue-penetration 0` modes but compared totals side-by-side rather than decomposing into queue vs alpha contributions. Without the split, "PEBBLES_XL +12,464" looks shippable; with it, alpha is slightly negative.
- **Expected difficulty:** Easy. Both backtest runs already exist per product; arithmetic is column subtraction.
- **Inspired by:** PEBBLES_XL base MM 12,464 → −612 under qp=0; queue value (13,076) exceeds total PnL — the product *needs* queue priority to be break-even, doesn't merely benefit from it. PEBBLES_S 40,098 → 31,127: 22% queue / 78% alpha (clearly shippable). Pair-trade XL leg 100,327 → 71,963 (28% queue / 72% alpha); pair-trade M leg 19,777 → −2,059 (entire M-leg PnL is queue priority). The decomposition would have flagged PEBBLES_XL as drop-candidate before pair-trade selection and identified the M leg as carrying no alpha.

## 11. Pair-trade leg outperformance — volatility-scaled or structural?

- **What it would test:** Decompose PEBBLES_M↔XL pair PnL by completed round-trip (entry → exit). For each round-trip, record (a) per-leg dollar PnL, (b) per-leg price move in σ-units of own product's daily volatility, (c) entry direction (which leg was shorted). Hypothesis A: XL-leg dollar dominance scales 1:1 with XL/M σ ratio — both legs contribute equally to spread closure but XL's higher volatility produces larger dollar swings (mechanical, uninteresting). Hypothesis B: XL leg outperforms even after σ-normalization — XL drives price discovery and M is a hedge that could be sized down (structural, actionable).
- **Why we didn't try first:** Phase 1 logged only aggregate per-product PnL. The day-conditional asymmetry — Day 2 legs ≈ equal; Days 3–4 XL is 5–9× M — only became visible in the per-day breakdown after the pair trade had already been shipped at full-limit on both legs.
- **Expected difficulty:** Medium. Need round-trip identification from order log (entry tick → exit tick) and per-leg fill-price aggregation; σ-normalization is one division.
- **Inspired by:** PEBBLES_M↔XL pair trade per-day (default): Day 2 M +4,198 / XL +5,222 (≈equal); Day 3 M +4,618 / XL +41,125 (8.9×); Day 4 M +10,961 / XL +53,980 (4.9×). Under qp=0 the M leg is essentially flat (−2,059) while XL retains 71,963. Either the M leg's role is purely cointegration anchor (could trade 5 lots instead of 10, freeing position-limit headroom for a second pair) or the dollar asymmetry is purely σ-scaled (M is doing equal information work) — the diagnostic distinguishes.

## 12. Pair-trade parameter sensitivity — fit-risk audit

- **What it would test:** Grid backtest over (window ∈ {100, 200, 400, 800}, z_entry ∈ {1.5, 2.0, 2.5, 3.0}, z_exit ∈ {0.3, 0.5, 0.8, 1.2}, max_hold_ticks ∈ {200, 500, 1000, 2000}) on PEBBLES_M↔XL. Output: PnL surface in (z_entry, z_exit) for each (window, max_hold). Validate per-day (2/3/4 separately). Robust edge = wide profitable plateau centered near chosen point. Fit risk = sharp peak with cliff drops nearby. Cross-day stability of the optimum is the real signal — single-day plateaus that move between days indicate regime-fitted parameters.
- **Why we didn't try first:** Pair trade was nominated from explicit cointegration evidence in the size-ordered structure; parameters (window=200, z_entry=2.0, z_exit=0.5, max_hold=500) were chosen by reasonable defaults rather than swept. Combined +120k default / +70k conservative validated the *existence* of the edge but did not characterize whether the chosen point is robust or fitted.
- **Expected difficulty:** Easy. ~256 backtest runs; pcolormesh visualization. Reading the surface is the only judgment call.
- **Inspired by:** PEBBLES_M↔XL +120k default / +70k conservative validated at one parameter point. ROBOT_LAUNDRY↔VACUUMING failure (−9,658) and SNACKPACK lag-1 corr overlay catastrophe both demonstrate that one-shot parameter choices on cointegration-style strategies can sit on cliffs. Before doubling positions, adding a second pair, or wiring the strategy to round 5 production, confirm this one isn't a parameter-fit ridge.

## 13. Conditional lead-lag — gate the bias on regime

- **What it would test:** For the CIRCLE→OVAL lag-50 relationship, segment ticks by an a-priori regime variable (large |ret_circle[t-50]|, recent realized vol, or time-of-day window) and compute the conditional `corr(ret_circle[t-50], ret_oval[t] | regime)`. If the conditional correlation in some regime exceeds ~0.15, build a gated overlay that fires the bias only in that regime; otherwise quote at base. Validates whether the unconditional |xcorr|=0.05 is a noisy average of "0.20 in 25% of ticks, 0.0 in 75%" (recoverable) vs "0.05 everywhere" (genuinely flat noise floor — unrecoverable).
- **Why we didn't try first:** Phase 1 used the unconditional correlation directly as the calibration weight (k=0.05) and as the magnitude scale for over-bet variants (k=0.5, 1.0). Conditional structure was not tested. Without it we cannot distinguish "signal exists in a subset and is washed out by averaging" from "no exploitable signal."
- **Expected difficulty:** Medium. Notebook work — segment the joint return series, compute conditional correlations, visualize. Gating logic in code is straightforward (the `R5LeadLagMMStrategy` already has the `bias_fired` counter; add a regime predicate before the `bias_fired += 1` block).
- **Inspired by:** CIRCLE→OVAL lead-lag deep dive — three k values tested (1.0, 0.5, 0.05). PnL scales monotonically with k: −614k, −294k, −1.5k respectively. The k=0.05 calibrated variant lands within rounding of baseline, NOT the small positive expected if the unconditional |xcorr|=0.05 had any tradable content. Either the signal is genuinely uniformly weak (no rescue possible) or it is regime-conditional with the hot regime diluted by noise in the unconditional average. Entry 13 distinguishes the two; if regime-conditional, gating recovers; if uniform, drop the angle entirely.

## 15. OXYGEN_SHAKE within-group pair screen (EG + lag-N matrix)

- **What it would test:** Engle-Granger cointegration test + lag-N cross-correlation matrix (lags k ∈ {−5, …, +5}) for all 10 within-group OXYGEN_SHAKE pairs. Primary target: MORNING_BREATH ↔ EVENING_BREATH — the naming implies a cyclical relationship (opposite-phase substitutes), but whether this manifests as cointegration or lead-lag in the price series is empirical. Secondary targets: CHOCOLATE ↔ EVENING_BREATH (both already autocorr-overlaid — do they share a spread?) and any pair involving GARLIC or MINT (dropped at Phase A but still price-generating; their mid series could cointegrate with a survivor). For any pair that passes EG on all 3 days: compute OU half-life, spread σ in ticks vs round-trip cost (2 × width=2 + queue penalty ≈ 5 ticks), z-score entry/exit distribution.
- **Why we didn't try first:** The C.3 brief was explicitly a **finishing pass, not a pair search** — it skipped Tasks 2/3 (EG screen + lag-N matrix) because MORNING_BREATH was the only remaining ship candidate and the group was otherwise settled (CHOCOLATE + EVENING_BREATH already shipped on autocorr, MINT + GARLIC dropped Phase A). The within-group OXYGEN_SHAKE EG screen has never been run.
- **Expected difficulty:** Easy. 10 pairs × 3 days = 30 `coint()` calls; lag-N matrix is ~20 lines of pandas. The binding constraint is the **replacement bar**: any MORNING↔EVENING pair trade would displace the existing independent wiring on both legs (MORNING base MM +11,043 default / EVENING autocorr +56,539 default = **+67,582 default combined**). A pair trade must clear that combined bar on default *and* conservative before it warrants replacing the current wiring.
- **Inspired by:** Per-day PnL inverse pattern in the Phase 1 backtest: EVENING_BREATH (+35,326 / +16,780 / +4,433, declining Day 2→4) vs MORNING_BREATH (+2,406 / +2,454 / +6,183, ascending Day 2→4). The inverse trend could reflect EVENING's autocorr overlay fading as the momentum regime weakened — OR it could reflect the two products genuinely anti-cycling in price. The lag-N matrix and EG test distinguish the two hypotheses. Note: if the relationship is lag-0 anti-correlation (instantaneous mirror), it adds no overlay alpha (same lesson as SNACKPACK — microprice already incorporates it). Only a non-zero peak lag or EG cointegration with a tradable spread σ justifies further engineering.

## 14. Cross-group cointegration screen

- **What it would test:** Engle-Granger test on all *cross-group* product pairs — every pair where the two products belong to different groups. With 50 products in 10 groups of 5, that's `C(50,2) − 10·C(5,2) = 1,225 − 100 = 1,125 cross-group pairs`. Filter for p < 0.05 on all 3 days. For survivors compute (a) OU half-life, (b) spread σ in ticks, (c) round-trip cost vs spread σ, (d) economic plausibility check (do the two products share a price-determining variable that wasn't obvious from naming?). Rank by edge / cost ratio. Concrete candidates worth surfacing if they survive the screen: PEBBLES_XL ↔ MICROCHIP_RECTANGLE (both highest-priced in their group, ~10–14k tier); UV_VISOR colors ↔ panel sizes ↔ pebble sizes (a hidden ordinal axis spanning groups); SNACKPACK_CHOCOLATE ↔ OXYGEN_SHAKE_CHOCOLATE (literal name overlap — flag the trivial one to verify the test isn't broken).
- **Why we didn't try first:** All EG screens to date are *within-group* — Phase B (#11 PEBBLES leg audit) and the C.1–C.6 group dives (#5 instantiated per group). Within-group has strong priors (size for PEBBLES, area for PANEL, spectrum for UV_VISOR, materials for SLEEP_POD). Cross-group has no a-priori structure, so it's a brute-force screen — only worth running after within-group screens have exhausted their candidates. After the dives complete, any remaining cointegration alpha must lie cross-group.
- **Expected difficulty:** Easy-Medium. 1,125 × 3 = 3,375 `coint()` calls runs in a few minutes (parallelizable). Easy part is the screen itself; medium part is sizing pair trades within position-limit constraints, since each cross-group pair's leg-A and leg-B already have their own MM/pair-trade wiring that may be competing for limit=10 capacity. Position-budget bookkeeping is the binding engineering constraint.
- **Inspired by:** All shipped pair trades and pair-leg overlays so far (PEBBLES_M↔XL pair, PEBBLES_S XL-skew, hypothetical PANEL 1X4↔2X2 from C.1) follow visible within-group structure. The C.1–C.6 dive set systematically exhausts within-group cointegration screens. By construction, after the dives any remaining EG alpha is cross-group — a category zero existing screens have probed. Multiplicative-coverage argument: within-group covers 100 pairs (10 groups × 10 within-group pairs) out of 1,225 total possible product pairs (8.2% coverage); cross-group screens the other 91.8%.
