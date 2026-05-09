# IMC Prosperity 4

Algorithmic trading competition by [IMC Trading](https://prosperity.imc.com/) — 5 rounds, ~18,000 teams globally. Each round adds new products; you implement a `Trader` class that the platform calls once per tick to return orders.

Round-by-round writeup (in progress): [Substack](https://substack.com/@heagenbell)

## Final Results

* **Global Ranking:** #42
* **National Ranking (United States):** #14

![Overall and State Ranking](finish.png)

**Contributors:** [Cameron Akhtar](https://github.com/Ape108) · [Heagen Bell](https://github.com/heagenb03)

## Results by Round

| Round | Products | Strategy | PnL | 
|--- | --- | --- | --- | 
| R1 | ASH_COATED_OSMIUM, INTARIAN_PEPPER_ROOT | Avellaneda-Stoikov MM with data-driven drift estimation | 160,290 |
| R2 | Same as R1 | Refined drift estimation; MAF blind auction bid | 475,034 |
| R3 | HYDROGEL_PACK, VELVETFRUIT_EXTRACT, VEV options (10 strikes) | Microprice + Avellaneda inventory skew MM; Black-Scholes IV smile scalper | 87,276 |
| R4 | Same as R3 | Carried R3 MM; inside-spread passive MM on VEV_4000 (21-tick spread); Mark bot EDA | 207,815 | 
| R5 | 50 products across 10 groups | PEBBLES M↔XL pair trade; OXYGEN_SHAKE autocorrelation overlays; base MM everywhere else | 796,401 |

## Directory Structure

```text
├── datamodel.py          # Official IMC platform data model — do not modify
├── requirements.txt      # Python dependencies
├── datasets/             # prices_*.csv + trades_*.csv pairs for each round
│   ├── round_0/
│   ├── round1/
│   ├── round2/
│   ├── round3/
│   ├── round4/
│   └── round5/
├── logs/                 # Backtest and live submission logs
│   └── r1/ … r5/
├── manual/               # Manual trading analysis notebooks
│   └── r2/ … r5/
├── reference/            # Backtester docs; jmerle and timodiehm reference strategies
└── submissions/          # Shipped code per round
    ├── tutorial/         
    ├── r1/               
    ├── r2/               
    ├── r3/               
    ├── r4/               
    │   ├── piors/        # Archived development iterations
    │   ├── tests/        # EDA scripts
    │   └── strategy.py
    └── r5/               
        ├── piors/        # Archived development iterations
        ├── groups/       # Per-group analysis scripts (cointegration, lead-lag)
        └── strategy.py
```

## Architecture

### Credit

| Author | Resource | Role in this repo |
| --- | --- | --- |
| [jmerle](https://github.com/jmerle/imc-prosperity-3) | 25th place Prosp3 strategies | Class hierarchy (`Strategy`, `StatefulStrategy`, `MarketMakingStrategy`) & code reference |
| [timodiehm](https://github.com/TimoDiehm/imc-prosperity-3) | 2nd place Prosp3 strategies | Advanced theory/technique reference |
| [GeyzsoN](https://github.com/GeyzsoN/prosperity_rust_backtester) | Rust backtester | Faster & configurable backtesting |
| [Equirag](https://prosperity.equirag.com/) | Online visualizer | Upload `submission.log` artifacts for order-level detail |

### Class Hierarchy

Base classes are copied from `reference/jmerle_hybrid.py` into each submission.

```python
Strategy[T]
├── StatefulStrategy[T]
│   └── SignalStrategy
└── MarketMakingStrategy
```

## Workflow

### Windows Setup

```bash
.venv/Scripts/activate
pip install -r requirements.txt
```

### Backtesting

The backtester runs in WSL2. `$PROSP4` in the WSL2 `~/.bashrc` points to this directory.

```bash
cd ~/prosperity_rust_backtester

# 1. Quick default run
rust_backtester --trader "$PROSP4/submissions/rN/strategy.py" --dataset roundN

# 2. Conservative PnL check
rust_backtester --trader "$PROSP4/submissions/rN/strategy.py" --dataset roundN \
  --queue-penetration 0 --price-slippage-bps 5
```

Upload `runs/<backtest-id>/submission.log` to [prosperity.equirag.com](https://prosperity.equirag.com/) for order-level detail.
