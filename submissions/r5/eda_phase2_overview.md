# R5 Phase 2 EDA — Scope and Approach

## Context

Phase 1 ran a uniform 5-level template across all 10 groups (`eda_overview.md`,
`eda_r5_triage.ipynb`). Per-group dives extended that with deep work in
`submissions/r5/groups/<group>.py` and surfaced a 14-entry second-pass
backlog (`eda_gaps.md`).

Phase 2 condenses that backlog into a focused 6-entry bundle picked by
expected value, with explicit sequencing and user-owned cut decisions.
Phase 2 outputs feed back into `strategy_h.py` before R5 close.

Conservative baseline at start of Phase 2: ~+187,372 XIRECS
(post-Phase-A drops + Phase-B PEBBLES_M leg resize).

## In-scope: 6 entries

| # | Entry | Axis | Decision driven |
|---|---|---|---|
| #10 | Queue-vs-alpha decomposition (all shipped) | Defensive | Drop products with alpha ≤ 0 |
| #1 | Lag-N within-group cross-correlation matrix | Offensive | Surface lead-lag overlay candidates |
| #3 | Per-day regime classification | Refinement | Surface regime-gate candidates |
| #5 + #8 + #15 | Within-group EG for unrun groups (MICROCHIP, GALAXY_SOUNDS, OXYGEN_SHAKE, ROBOT) | Offensive | Surface new pair candidates |
| #14 | Cross-group EG screen (1,125 pairs) | Offensive | Surface cross-group pair candidates |
| #12 | PEBBLES_M↔XL parameter sensitivity | Defensive | Confirm pair params robust vs cliffed |

## Out of scope (deferred)

#2 position util, #4 adverse selection, #6 rolling intra-day ACF,
#7 MOPPING diagnostic, #9 group-residual MM, #11 (already absorbed in
`pebbles_leg_audit.py`), #13 conditional lead-lag.

See full justifications in
`docs/superpowers/specs/2026-04-30-r5-phase2-eda-design.md`.

## Sequencing — three gates

```
Gate 1 (defensive, blocks all)
└── #10 queue_alpha_decomp.py
    └── user-directed strategy_h.py updates → re-baseline conservative backtest

Gate 2 (offensive, parallel)
├── #5+#8+#15 group_eg_screens.py
├── #1 lag_n_matrix.py
└── #14 cross_group_eg.py
    └── user-directed wirings (new pair trades, overlay tests)

Gate 3 (refinement)
├── #12 pebbles_param_sweep.py
└── #3 regime_classification.py
    └── user-directed final adjustments (param tweaks, regime gates)
```

## Reporting convention (every script)

Each script outputs to stdout with this structure:

1. Header — entry number, script purpose, runtime, input data range.
2. Full metrics table — every row, unfiltered.
3. Threshold annotation — flags rows that meet the entry's stated criterion.
4. One concrete next-question to user.
5. Optional intermediate artifacts written to `submissions/r5/phase2/results/`.

**No script auto-modifies `strategy_h.py`.** User decides every change.

## Acceptance

Phase 2 complete when:
- All 6 scripts run successfully.
- Each gate has a corresponding CLAUDE.md update.
- Final conservative net change vs ~+187,372 baseline is reported.
- **Target: ≥ +10k swing.** Benchmark only — user owns "done enough".
