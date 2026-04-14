# Evaluating Strategy Performance with rust_backtester

## Prerequisites

### Working Directory
All `rust_backtester` commands **must be run from `~/prosperity_rust_backtester`** in WSL2. Dataset paths resolve relative to cwd — running from the project directory (`$PROSP4`) will cause `Error: no runnable datasets found`.

```bash
cd ~/prosperity_rust_backtester
rust_backtester --trader "$PROSP4/submissions/r1/strategy.py" --dataset round1
```

### Adding Round Data
Download `prices_round_X_day_Y.csv` and `trades_round_X_day_Y.csv` from the IMC platform, then copy them into the backtester's dataset directory:

```bash
mkdir -p ~/prosperity_rust_backtester/datasets/round1
cp "$PROSP4/datasets/round1/"*.csv ~/prosperity_rust_backtester/datasets/round1/
```

The backtester picks up `prices_*.csv` + `trades_*.csv` pairs automatically.

### Day Numbering
Historical days use **negative integers** (`-2`, `-1`) and the live day is `0`. Passing a positive day (e.g. `--day 2`) will match nothing and produce `Error: no runnable datasets found`.

Round 1 valid days: `-2`, `-1`, `0`.

---

## Why Raw PnL Isn't Enough

Highest PnL on backtester data is necessary but not sufficient:

- **Overfitting risk** — The backtester runs on historical data. A strategy tuned to maximize PnL on that data may have learned noise and will underperform live.
- **Risk-adjusted returns matter** — A strategy making 10k with wild swings is worse than one making 8k consistently.
- **The backtester uses `d-1`/`d-2` data** — Live rounds use new data you've never seen. Consistency across days is a better signal than peak PnL.

## Better Metrics

| Metric | What it tells you |
|--------|-------------------|
| PnL per day | Consistency — does it profit every day or just one? |
| PnL under conservative settings | Whether the edge is real or an artifact of optimistic defaults |
| Per-product breakdown | Which symbols actually drive returns |
| PnL with `--queue-penetration 0` | Survives without queue priority assumptions |

## Key Options

### `--day`
Test each day independently to check consistency:
```bash
rust_backtester --trader "$PROSP4/submissions/r1/strategy.py" --dataset round1 --day -2
rust_backtester --trader "$PROSP4/submissions/r1/strategy.py" --dataset round1 --day -1
rust_backtester --trader "$PROSP4/submissions/r1/strategy.py" --dataset round1 --day 0
```
If PnL is only good on one day, the strategy is fragile.

### `--queue-penetration`
Controls fill assumptions:
- `1` (default) — you fill at the front of the queue (optimistic)
- `0` — you only fill if price moves through your level (conservative, more realistic)

```bash
rust_backtester --trader ... --dataset round1 --queue-penetration 0
```
If PnL survives `--queue-penetration 0`, the edge is real. If it collapses, you're relying on queue priority that won't exist on the platform.

### `--price-slippage-bps`
Adds execution friction. Default is 0 (optimistic). Use 5bps as a sanity check:
```bash
rust_backtester --trader ... --dataset round1 --price-slippage-bps 5
```

### `--persist` and `--carry`
- `--persist` — saves `traderData` state between days (required for stateful strategies with rolling windows)
- `--carry` — carries positions between days

Use both when testing `StatefulStrategy` subclasses:
```bash
rust_backtester --trader ... --dataset round1 --persist --carry
```

### `--artifact-mode`
Controls what output files are written:
- `none` — no files
- `diagnostic` — internal debug logs
- `submission` — log formatted for the IMC visualizer
- `full` — everything

```bash
rust_backtester --trader ... --dataset round1 --artifact-mode submission
```
Upload `runs/<id>/submission.log` to the visualizer at https://prosperity.equirag.com/ for order-level detail and per-tick PnL.

## Standard Evaluation Protocol

Run these in sequence for any strategy candidate:

```bash
# 1. Quick sanity check (default, fast)
rust_backtester --trader "$PROSP4/submissions/r1/strategy.py" --dataset round1

# 2. Conservative reality check
rust_backtester --trader "$PROSP4/submissions/r1/strategy.py" --dataset round1 \
  --queue-penetration 0 --price-slippage-bps 5

# 3. Per-day consistency check
for day in -2 -1 0; do
  rust_backtester --trader "$PROSP4/submissions/r1/strategy.py" --dataset round1 --day $day
done

# 4. Full artifact for visualizer
rust_backtester --trader "$PROSP4/submissions/r1/strategy.py" --dataset round1 \
  --artifact-mode submission --persist
```

## Decision Rule

**Prefer a strategy that holds up under `--queue-penetration 0` over one that only works with optimistic defaults.**

Rank strategies by: **consistency across all days > conservative PnL > default PnL**
