"""Tests for VelvetfruitSignalStrategy — z-score aggressive signal."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r4"))

import pytest
from datamodel import OrderDepth, TradingState
from strategy_h import VelvetfruitSignalStrategy, SignalStrategy, Signal


SYM = "VELVETFRUIT_EXTRACT"


def make_strategy(**kwargs) -> VelvetfruitSignalStrategy:
    defaults = dict(
        symbol=SYM,
        limit=200,
        target_position=60,
        zscore_period=4,
        smoothing_period=1,
        threshold=1.0,
    )
    defaults.update(kwargs)
    return VelvetfruitSignalStrategy(**defaults)


def make_state(bid: int, ask: int, position: int = 0) -> TradingState:
    od = OrderDepth()
    od.buy_orders = {bid: 10}
    od.sell_orders = {ask: -10}
    return TradingState(
        traderData="",
        timestamp=0,
        listings={},
        order_depths={SYM: od},
        own_trades={},
        market_trades={},
        position={SYM: position} if position else {},
        observations=None,  # type: ignore
    )


def pump(strategy: VelvetfruitSignalStrategy, ticks: list[tuple[int, int]], final_position: int = 0) -> list:
    """Run strategy for all ticks; apply final_position only on the last tick.
    Returns orders from the last tick."""
    orders: list = []
    for i, (bid, ask) in enumerate(ticks):
        pos = final_position if i == len(ticks) - 1 else 0
        state = make_state(bid, ask, pos)
        orders, _ = strategy.run(state)
    return orders


# Tick sequences designed for zscore_period=4, smoothing_period=1, threshold=1.0.
# Each sequence has 5 ticks (4+1 = warmup). Mid prices are (bid+ask)/2.
# LONG:  mids [100,102,98,100,80]  → z ≈ -1.48 < -1.0
# SHORT: mids [100,102,98,100,120] → z ≈ +1.48 > +1.0
# NEUTRAL: mids [100,102,98,100,101] → z ≈ +0.44 (between ±1.0)
LONG_TICKS    = [(99,101),(101,103),(97,99),(99,101),(79,81)]
SHORT_TICKS   = [(99,101),(101,103),(97,99),(99,101),(119,121)]
NEUTRAL_TICKS = [(99,101),(101,103),(97,99),(99,101),(100,102)]


# ── warmup ───────────────────────────────────────────────────────────────────

def test_warmup_get_signal_returns_none_below_required():
    """get_signal returns None (not NEUTRAL) while history < zscore_period + smoothing_period."""
    s = make_strategy()
    # required = 4 + 1 = 5; give only 4 ticks — every call must return None
    for bid, ask in LONG_TICKS[:4]:
        result = s.get_signal(make_state(bid, ask))
        assert result is None


# ── LONG signal ───────────────────────────────────────────────────────────────

def test_long_signal_crosses_ask():
    """z-score < -threshold → buy at best_ask, quantity = target_position - position."""
    s = make_strategy()
    orders = pump(s, LONG_TICKS, final_position=0)
    assert len(orders) == 1
    assert orders[0].quantity == 60    # target_position - 0
    assert orders[0].price == 81       # best_ask of last tick


# ── SHORT signal ──────────────────────────────────────────────────────────────

def test_short_signal_crosses_bid():
    """z-score > +threshold → sell at best_bid, quantity = position - (-target_position)."""
    s = make_strategy()
    orders = pump(s, SHORT_TICKS, final_position=0)
    assert len(orders) == 1
    assert orders[0].quantity == -60   # negative = sell
    assert orders[0].price == 119      # best_bid of last tick


# ── NEUTRAL / flatten ─────────────────────────────────────────────────────────

def test_neutral_flattens_long_position():
    """NEUTRAL signal + positive position → sell position lots at best_bid."""
    s = make_strategy()
    orders = pump(s, NEUTRAL_TICKS, final_position=30)
    # z ≈ +0.44 → NEUTRAL; position +30 → should sell 30 at best_bid
    assert len(orders) == 1
    assert orders[0].quantity == -30
    assert orders[0].price == 100      # best_bid of last tick


def test_neutral_flattens_short_position():
    """NEUTRAL signal + negative position → buy |position| lots at best_ask."""
    s = make_strategy()
    orders = pump(s, NEUTRAL_TICKS, final_position=-30)
    assert len(orders) == 1
    assert orders[0].quantity == 30
    assert orders[0].price == 102      # best_ask of last tick


def test_no_orders_on_flat_neutral():
    """NEUTRAL signal + zero position → no orders (not a sell-0 edge case)."""
    s = make_strategy()
    orders = pump(s, NEUTRAL_TICKS, final_position=0)
    assert orders == []


def test_neutral_after_long_resets_signal_and_flattens():
    """get_signal must return Signal.NEUTRAL (not None) between thresholds.

    If get_signal returned None, self.signal would stay LONG after the long
    trigger and the strategy would try to buy more instead of flattening.
    This test pins the NEUTRAL-not-None contract.
    """
    s = make_strategy()
    pump(s, LONG_TICKS)           # drive signal to LONG
    assert s.signal == Signal.LONG

    # One neutral-zone tick with an existing long position
    orders = pump(s, NEUTRAL_TICKS[-1:], final_position=30)

    assert s.signal == Signal.NEUTRAL   # signal must have updated (not None)
    assert len(orders) == 1
    assert orders[0].quantity == -30    # sell 30 (flatten)
    assert orders[0].price == 100       # best_bid of (100, 102) tick


# ── position cap ──────────────────────────────────────────────────────────────

def test_position_cap_respected_on_long():
    """Already at target_position → no additional buy on LONG signal."""
    s = make_strategy()
    orders = pump(s, LONG_TICKS, final_position=60)
    # target - position = 60 - 60 = 0 → no order
    assert orders == []


def test_hard_limit_never_exceeded():
    """target_position > limit is clamped to limit at __init__."""
    s = make_strategy(target_position=250, limit=200)
    assert s.target_position == 200


# ── save / load ───────────────────────────────────────────────────────────────

def test_save_load_roundtrip():
    """Signal + history survive JSON serialisation."""
    import json
    s1 = make_strategy()
    pump(s1, LONG_TICKS)  # puts strategy into LONG signal
    saved = s1.save()
    blob = json.dumps(saved)

    s2 = make_strategy()
    s2.load(json.loads(blob))
    assert s2.signal == s1.signal
    assert s2.history == pytest.approx(s1.history)


# ── Trader wiring ─────────────────────────────────────────────────────────────

def test_velvetfruitstrategy_fallback_exists():
    """Neutral MM fallback is wired; VelvetfruitSignalStrategy class still importable."""
    from strategy_h import Trader, VelvetfruitStrategy, MarketMakingStrategy
    trader = Trader()
    strat = trader.strategies[SYM]
    assert isinstance(strat, VelvetfruitStrategy)
    assert isinstance(strat, MarketMakingStrategy)
    # VelvetfruitSignalStrategy class must still be importable for next iteration
    assert VelvetfruitSignalStrategy is not None
