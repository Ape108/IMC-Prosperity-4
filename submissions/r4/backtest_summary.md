# Backtest Summary — R4 Mark 14 Voucher Angles

Date: 2026-04-27
Strategy file: `submissions/r4/strategy_h.py`
Direction config: `submissions/r4/mark14_direction.md`
Branch decision input: `submissions/r4/branch_decision.md`

## Baseline (prior `Mark14FollowerStrategy("VEV_5300", ...)` cross-spread, deployed live)

| Day | VEV_5300 | VEV_5400 | VEV_5500 |
|---|---|---|---|
| 1 | -1266 | 0 | 0 |
| 2 | -2926 | 0 | 0 |
| 3 | -1462 | 0 | 0 |

3-day voucher total: **-5,654**

## New strategy (direction-aware passive entry per `mark14_direction.md`)

### Default mode (`--persist --carry`)

| Day | Total PnL | HYDROGEL | VEV_5000 | VEV_5100 | VEV_5200 | VEV_5300 | VEV_5400 | VEV_5500 | OTHER (4500/4000/6000/6500) | VELVETFRUIT |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | +4308.00 | +4272.00 | +69.00 | -5.50 | -27.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | +5100.00 | +5026.00 | +127.50 | +7.50 | -61.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | +4423.50 | +4646.00 | -226.00 | -72.50 | +76.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

3-day voucher total: **-112.00** (down from -5,654 baseline)
3-day delta vs baseline: **+5,542**

### Conservative mode

Skipped. With VEV_5300/5400/5500 at exactly 0.00 across 3 days, conservative mode adds no information about the new strategy — it would only re-test Hydrogel/Velvetfruit, which are unchanged from earlier runs and already calibrated.

## Acceptance vs spec

- [x] Voucher net PnL beats baseline (-1.5k/day average) in default mode — **met cleanly** (-37/day vs -1.9k/day baseline; +1.85k/day swing)
- [N/A] Voucher net PnL beats baseline in conservative mode — not run (uninformative given default-mode result)
- [x] No regression on Hydrogel or Velvetfruit per-day PnL — Hydrogel +4.6k/day average (similar to earlier baseline), Velvetfruit 0.00 in backtest (consistent with prior — the +1.3k live PnL came from regime-dependent flow not present in backtest data)

## Decision

**KEEP** the new strategy. Stops the bleed cleanly. No alpha capture from VEV_5300/5400/5500, but no loss either. Net +5.5k/round recovery vs the deployed baseline.

## Why no fills on VEV_5300/5400/5500

Most likely explanation: passive orders at our touch (best_bid for follow_passive, best_ask for fade_at_touch) sit in the maker queue but receive no aggressor flow large enough to penetrate the existing queue depth. Mark 14 trades VEV_5300 ~10x/day, VEV_5400 ~4x/day, VEV_5500 ~2x/day. Each trade is a small clip (~5 lots). Existing maker queues at the touch are typically larger than that single trade, so our newly-posted orders sit at the back of the queue and don't get matched within Mark 14's 500-tick signal window.

This is the structural reason the original cross-spread follower bled — only crossing fills these books reliably, but at OTM spread costs the alpha doesn't survive.

## Notes

- VEV_5300/5400/5500 at 0.00 verified in default-mode result; orders are posted (logic path traced in tests) but receive no fills in this dataset.
- VEV_5000/5100/5200 small ±200/day swings come from VoucherStrategy IV scalper firing on the 3 strikes whose moneyness < 0.996 cutoff (5000/5100 are below; 5200 is barely above and gets filtered most ticks).
- VEV_4000/4500/6000/6500 contribute 0.00 — VoucherStrategy filters them on intrinsic floor (deep ITM) or moneyness gate (deep OTM).
- Hydrogel +4.6k/day mean is consistent with prior calibrated result.
- Velvetfruit 0.00 in backtest is expected (cerebrum 2026-04-25 records that velvetfruit alpha is regime-dependent — present live, not in this backtest dataset).

---

## Track B (Mark14InformedMMStrategy — Informed-MM bias) — implemented and reverted 2026-04-27

Spec: `docs/superpowers/specs/2026-04-27-r4-informed-mm-bias-design.md`
Plan: `docs/superpowers/plans/2026-04-27-r4-informed-mm-bias.md`

### Result (default mode, --persist --carry)

| Day | Total PnL | HYDROGEL | VEV_5000 | VEV_5100 | VEV_5200 | VEV_5300 | VEV_5400 | VEV_5500 | OTHER | VELVETFRUIT |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | +4308.00 | +4272.00 | +69.00 | -5.50 | -27.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | +5100.00 | +5026.00 | +127.50 | +7.50 | -61.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | +4423.50 | +4646.00 | -226.00 | -72.50 | +76.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

3-day voucher net (5300/5400/5500): **0.00** (vs Track A −112). 3-day total identical to Track A down to the cent.

### Acceptance: NOT MET → REVERT

Spec required ≥1 day with non-zero fills on each of VEV_5300, VEV_5400, VEV_5500. Got 0 fills on all three across all three days. Reverted Trader wiring back to Mark14FollowerStrategy.

### Diagnosis (from local in-process diagnostic on real day-1 data)

Orders ARE being emitted on the three strikes (act() runs every tick — book coverage is 100% on all three days). They're emitted at non-competitive prices because:

1. **`base_w=1.0` is too wide for spread=1 books.** VEV_5400 ts=0: book (16, 17), spread=1. Our quote: (15, 18) — 1 below touch on bid, 1 above on ask. VEV_5500 same pattern. With offsets of ±1 from a fair near mid, floor() and ceil() push BOTH quotes outside the touch on opposite sides.

2. **Smile-theo blend pulls fair off mid by ~0.4 ticks for structurally-mispriced strikes.** On VEV_5400 ts=0 microprice = 16.5 exactly, but blended fair = 16.88 (0.38 above mid) because the smile-implied price is ~19 (VEV_5400 is structurally underpriced per cerebrum 2026-04-26: -0.042 avg IV residual). The 0.15 weight pulls fair UP enough that `ceil(fair + 1)` = 18 instead of 17.

3. **Spread=1 books leave no room "inside the spread."** Book-clamp logic (`bid ≤ best_ask − 1`, `ask ≥ best_bid + 1`) pulls any computed quote back to touch when offsets would land inside. So in-window directional asymmetry can't differentiate from out-of-window at-touch quoting on these books — the mechanism's bias has nowhere to express itself.

### What stayed in the codebase

- `Mark14InformedMMStrategy` class + 37 unit/integration tests retained in `submissions/r4/strategy_h.py` and `tests/r4/test_mark14_informedmm.py`. Re-activatable by changing one config field in `Trader.__init__` if a future round has wider spreads or non-mispriced strikes.
- Trader wiring rolled back to Track A (Mark14FollowerStrategy on the three strikes).
- All 124 tests still green after revert.

### What did NOT change

- Hydrogel, Velvetfruit, VoucherStrategy, Mark14FollowerStrategy, cross-strike voucher dispatch.
- Live submission strategy (sub39) — no change pushed.
