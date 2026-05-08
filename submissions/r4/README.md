# Round 4 — "The More The Merrier"

R4 carried forward the R3 strategy set and added counterparty data (`Trade.buyer` / `Trade.seller` now populated — all named "Mark XX"). Products are identical to R3.

**`strategy.py`** — Final shipped submission. Includes `HydrogelStrategy` (microprice + Avellaneda skew), `VelvetfruitStrategy` (neutral mid-price MM), `VoucherStrategy` (IV smile scalper), and `Vev4000MMStrategy` (inside-spread passive MM on VEV_4000's 21-tick spread).

**`piors/`** — Archived development iterations, including unshipped Mark-bot-following strategies.

**`tests/`** — EDA scripts written during development. These are not unit tests; they are one-off analysis scripts used to investigate specific hypotheses:
- `eda_mark_bots.py` / `eda_mark14_conditional.py` — Mark bot direction analysis
- `eda_imitation_pnl.py` — Imitation-PnL sweep: can we profit by following a bot?
- `eda_vef_iv.py` / `eda_vef_extreme_mr.py` / `eda_vef_day3.py` — VELVETFRUIT mean-reversion and IV analysis
- `eda_mr_signal.py` / `eda_lag_persistence.py` / `eda_inside_mm_bias.py` — Signal and market structure EDA
