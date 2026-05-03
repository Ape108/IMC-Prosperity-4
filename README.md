# IMC Prosperity 4

Algorithmic trading competition by [IMC](https://prosperity.imc.com/) — 5 rounds, ~6,000 teams globally. Each round adds new products; you implement a `Trader` class that the platform calls once per tick to return orders.

**Final placement:** [TBD] / ~6,000

**Contributors:** [Cameron Akhtar](https://github.com/Ape108) · [Heagen Bell](https://github.com/heagenb03)

A round-by-round write-up series covering our decision process is in progress on Substack - ([Profile](https://substack.com/@heagenbell)). Currently [0 of 6 posts] drafted.

---

## Directory Structure

```
├── datamodel.py          # Official IMC platform data model — do not modify
├── requirements.txt
├── submissions/
│   ├── r1/strategy.py    # Shipped submission
│   ├── r2/strategy.py
│   ├── r3/strategy.py
│   ├── r4/strategy.py
│   └── r5/strategy.py
└── manual/
    ├── r2/               # Manual trading notebooks
    ├── r3/
    ├── r4/
    └── r5/
```

---

## Architecture

### Credit

| Author | Resource | Role in this repo |
|--------|----------|-------------------|
| [jmerle](https://github.com/jmerle/imc-prosperity-3) | 25th place Prosp3 strategies | Class hierarchy (`Strategy`, `StatefulStrategy`, `MarketMakingStrategy`) & code reference |
| [timodiehm](https://github.com/TimoDiehm/imc-prosperity-3) | 2nd place Prosp3 strategies | Advanced theory/technique reference |
| [GeyzsoN](https://github.com/GeyzsoN/prosperity_rust_backtester) | Rust backtester | Faster & configurable tests mentioned below |
| [Equirag](https://prosperity.equirag.com/) | Online visualizer | Upload `submission.log` artifacts for order-level backtest/submission detail |

### Class Hierarchy

Base classes are copied (not imported) from `reference/jmerle_hybrid.py` into each submission.

```
Strategy[T]                   # Base: symbol, limit, buy(), sell(), convert()
├── StatefulStrategy[T]       # Adds save()/load() for persisting state across ticks
│   └── SignalStrategy        # LONG/SHORT/NEUTRAL signal with position flattening
└── MarketMakingStrategy      # Quotes around a fair value; fills existing orders first
```

To add a strategy: subclass `MarketMakingStrategy` and implement `get_true_value()`, or subclass `SignalStrategy` and implement `get_signal()`.

**Key constraints:**
- `Order(symbol, price, quantity)` — positive qty = buy, negative = sell
- `state.traderData` is a JSON string, not a dict — parse explicitly
- Use `logger.print()` not `print()` — bare `print()` corrupts visualizer output
- `order_depth.sell_orders` values are negative integers — use `abs()` in microprice/VWAP

---

## Workflow

### Windows Setup

```bash
.venv/Scripts/activate
pip install -r requirements.txt
```

### Backtesting

Backtester runs in WSL2. `$PROSP4` in WSL2 `~/.bashrc` points to this directory.

```bash
cd ~/prosperity_rust_backtester

# 1. Quick default run
rust_backtester --trader "$PROSP4/submissions/rN/strategy.py" --dataset roundN

# 2. Conservative PnL check
rust_backtester --trader "$PROSP4/submissions/rN/strategy.py" --dataset roundN \
  --queue-penetration 0 --price-slippage-bps 5
```

**Decision rule:** consistency across all days > conservative PnL > default PnL.

Upload `runs/<backtest-id>/submission.log` to [prosperity.equirag.com](https://prosperity.equirag.com/) for order-level detail.

> Day numbering varies by round: R1 uses `-2,-1,0`; R3 uses `0,1,2`; R4 uses `1,2,3`; R5 uses `2,3,4`.

---

## Results

**Final placement: [TBD] / ~6,000**

| Round | Products added | Primary strategy | Backtest PnL/day | Live PnL |
|-------|---------------|-----------------|-----------------|---------|
| R1 | [TBD] | [TBD] | [TBD] | [TBD] |
| R2 | [TBD] | [TBD] | [TBD] | [TBD] |
| R3 | Hydrogel, Velvetfruit, VEV vouchers | Microprice MM + IV scalper | ~+8k | +3,039 |
| R4 | VEV_4000 | Avellaneda MM + inside-spread MM | ~+10k | [TBD] |
| R5 | 50 products (10 groups) | Stat arb + base MM | [TBD] | [TBD] |
