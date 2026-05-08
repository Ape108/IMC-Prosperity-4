# Round 5 — "The Final Stretch"

R5 introduced 50 new products across 10 thematic groups (5 products each), all with position limit 10. R1–R4 products were retired. The primary alpha source is **statistical arbitrage** (cointegration, lead-lag relationships), not market making — position limits make per-product MM PnL marginal.

**`strategy.py`** — Final shipped submission. Includes a PEBBLES M↔XL pair trade, OXYGEN_SHAKE autocorrelation overlays, and base market making across the remaining profitable products.

**`piors/`** — Archived development iterations.

**`eda_r5_triage.ipynb`** — Main EDA notebook. Runs a triage screen across all 10 groups: return correlations, Engle-Granger cointegration tests, and lead-lag cross-correlation analysis. Groups are ranked by signal strength to prioritize investigation depth. Outputs are rendered.

**`groups/`** — Per-group deep-dive analysis scripts for the top groups investigated:
- `panel.py`, `sleep_pod.py`, `snackpack.py`, `uv_visor.py` — cointegration + pair trade candidate analysis
- `galaxy_sounds.py`, `translator.py`, `oxygen_shake_completion.py` — lead-lag and autocorrelation checks
- `pebbles.py`, `pebbles_leg_audit.py`, `robot.py`, `microchip.py` — strategy variant testing
