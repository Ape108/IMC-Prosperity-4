# GeyzsoN/prosperity_rust_backtester

Source: https://github.com/GeyzsoN/prosperity_rust_backtester

Rust backtester for IMC Prosperity 4. Local-only, no API.

## Setup

```bash
git clone https://github.com/GeyzsoN/prosperity_rust_backtester.git
cd prosperity_rust_backtester
```

### Windows: Use WSL2

Open Ubuntu shell in WSL2 and run the same commands as macOS.

### macOS

```bash
xcode-select --install
curl https://sh.rustup.rs -sSf | sh
source "$HOME/.cargo/env"
make install   # install CLI
# or
make backtest  # run directly
```

## Usage

```bash
rust_backtester                    # auto-picks trader + latest dataset
make backtest                      # same as above
make tutorial                      # run tutorial dataset

# With options
make tutorial DAY=-1               # specific day
make round3 TRADER=traders/latest_trader.py
make round2 PERSIST=1              # write full artifact set
make tutorial FLAT=1               # flat output layout
make tutorial CARRY=1              # carry state across days
```

### Explicit CLI

```bash
rust_backtester --trader /path/to/trader.py --dataset tutorial
rust_backtester --trader /path/to/trader.py --dataset datasets/round1
rust_backtester --trader /path/to/trader.py --dataset /path/to/submission.log
```

## Dataset Structure

- `datasets/tutorial/` — bundled tutorial day CSVs + sample submission.log
- `datasets/round1/` through `datasets/round8/` — placeholders for round data
- Supports: normalized JSON, `prices_*.csv` + `trades_*.csv` pairs, `submission.log`

## Dataset Aliases

`latest`, `tutorial`/`tut`, `round1`/`r1` through `round8`/`r8`

## Artifact Modes

- `--artifact-mode none` — only `metrics.json`
- `--artifact-mode submission` (default) — + `submission.log`
- `--artifact-mode diagnostic` — + `bundle.json` with PnL series
- `--artifact-mode full` — everything: metrics, bundle, submission.log, activity.csv, pnl_by_product.csv, combined.log, trades.csv
- `--persist` implies `--artifact-mode full`

## Product Display

- `--products summary` (default) — top PnL contributors + rollup
- `--products full` — every product
- `--products off` — per-day total only

## Carry Mode

`--carry` carries positions, own trades, market trades, and trader state across non-submission day datasets. Normalizes timestamps into continuous timeline.

## Output

```text
trader: latest_trader.py [auto]
dataset: tutorial [default]
mode: fast
SET             DAY    TICKS  OWN_TRADES    FINAL_PNL  RUN_DIR
D-2              -2    10000          39       118.10  runs/backtest-123-day-2
D-1              -1    10000          42       123.45  runs/backtest-123-day-1
```

## Troubleshooting (macOS)

If build hangs (syspolicyd CPU):
```bash
make doctor
sudo killall syspolicyd
make build-release
```

Docker fallback:
```bash
make docker-build
make docker-smoke
```

## Layout

- `src/` — Rust backtester implementation
- `traders/latest_trader.py` — bundled default trader
- `runs/` — persisted outputs
