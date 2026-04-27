import json
import pytest

from datamodel import Listing, Observation, OrderDepth, Trade, TradingState
from strategy_h import (
    MARK14_INFORMED_BOT,
    MARK14_SIGNAL_SYMBOL,
    MARK14_TARGET_SIZE,
    MARK14_WINDOW_TICKS,
    Mark14FollowerStrategy,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_od(bids: dict[int, int], asks: dict[int, int]) -> OrderDepth:
    od = OrderDepth()
    od.buy_orders = dict(bids)
    od.sell_orders = {p: -abs(v) for p, v in asks.items()}  # asks are negative
    return od


def make_state(
    timestamp: int = 0,
    position: int = 0,
    market_trades: list[Trade] | None = None,
    bids: dict[int, int] | None = None,
    asks: dict[int, int] | None = None,
    trader_data: str = "",
) -> TradingState:
    bids = bids if bids is not None else {100: 10}
    asks = asks if asks is not None else {102: 10}
    od = make_od(bids, asks)
    return TradingState(
        traderData=trader_data,
        timestamp=timestamp,
        listings={MARK14_SIGNAL_SYMBOL: Listing(MARK14_SIGNAL_SYMBOL, MARK14_SIGNAL_SYMBOL, "SEASHELLS")},
        order_depths={MARK14_SIGNAL_SYMBOL: od},
        own_trades={},
        market_trades={MARK14_SIGNAL_SYMBOL: market_trades or []},
        position={MARK14_SIGNAL_SYMBOL: position},
        observations=Observation({}, {}),
    )


def make_strategy() -> Mark14FollowerStrategy:
    return Mark14FollowerStrategy(MARK14_SIGNAL_SYMBOL, 300)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_no_mark14_trades_produces_no_orders():
    strat = make_strategy()
    state = make_state(timestamp=1000, market_trades=[])
    orders, _ = strat.run(state)
    assert orders == []
    assert strat.last_signal_ts is None


def test_mark14_buy_updates_last_signal_ts():
    strat = make_strategy()
    trade = Trade(MARK14_SIGNAL_SYMBOL, price=101, quantity=5,
                  buyer=MARK14_INFORMED_BOT, seller="Mark 22", timestamp=900)
    state = make_state(timestamp=1000, market_trades=[trade])
    strat.run(state)
    assert strat.last_signal_ts == 900


def test_mark14_as_seller_is_ignored():
    strat = make_strategy()
    trade = Trade(MARK14_SIGNAL_SYMBOL, price=101, quantity=5,
                  buyer="Mark 22", seller="Mark 14", timestamp=900)
    state = make_state(timestamp=1000, market_trades=[trade])
    orders, _ = strat.run(state)
    assert strat.last_signal_ts is None
    assert orders == []


def test_within_window_buys_at_best_ask():
    strat = make_strategy()
    trade = Trade(MARK14_SIGNAL_SYMBOL, price=101, quantity=5,
                  buyer=MARK14_INFORMED_BOT, seller="Mark 22", timestamp=900)
    # timestamp - last_signal_ts = 1000 - 900 = 100 < WINDOW_TICKS (500) → in window
    state = make_state(
        timestamp=1000,
        position=0,
        market_trades=[trade],
        bids={100: 10},
        asks={102: 200},
    )
    orders, _ = strat.run(state)
    assert len(orders) == 1
    o = orders[0]
    assert o.symbol == MARK14_SIGNAL_SYMBOL
    assert o.price == 102
    assert o.quantity == MARK14_TARGET_SIZE  # 100


def test_past_window_flattens_long_position():
    strat = make_strategy()
    strat.last_signal_ts = 100
    # timestamp - last_signal_ts = 1000 - 100 = 900 > WINDOW_TICKS (500) → past window
    state = make_state(
        timestamp=1000,
        position=MARK14_TARGET_SIZE,
        market_trades=[],
        bids={100: 200},
        asks={102: 50},
    )
    orders, _ = strat.run(state)
    assert len(orders) == 1
    o = orders[0]
    assert o.symbol == MARK14_SIGNAL_SYMBOL
    assert o.price == 100
    assert o.quantity == -MARK14_TARGET_SIZE  # negative = sell


def test_restacking_refreshes_last_signal_ts():
    strat = make_strategy()
    strat.last_signal_ts = 100
    # New buy at ts=300 → last_signal_ts should become 300, not stay at 100.
    new_trade = Trade(MARK14_SIGNAL_SYMBOL, price=101, quantity=5,
                      buyer=MARK14_INFORMED_BOT, seller="Mark 22", timestamp=300)
    state = make_state(timestamp=400, position=MARK14_TARGET_SIZE,
                       market_trades=[new_trade])
    orders, _ = strat.run(state)
    assert strat.last_signal_ts == 300
    assert orders == []  # already at target — must not double-buy


def test_day_boundary_clears_stale_signal():
    strat = make_strategy()
    strat.last_signal_ts = 999_500
    # New day: timestamp resets to 0 → last_signal_ts > current → stale → clear.
    state = make_state(timestamp=0, position=0, market_trades=[])
    orders, _ = strat.run(state)
    assert strat.last_signal_ts is None
    assert orders == []


def test_save_load_roundtrip_preserves_last_signal_ts():
    strat = make_strategy()
    strat.last_signal_ts = 12345

    serialised = strat.save()
    encoded = json.dumps(serialised)  # mimics traderData round-trip
    decoded = json.loads(encoded)

    new_strat = make_strategy()
    assert new_strat.last_signal_ts is None
    new_strat.load(decoded)
    assert new_strat.last_signal_ts == 12345


def test_load_with_missing_key_defaults_to_none():
    new_strat = make_strategy()
    new_strat.load({})
    assert new_strat.last_signal_ts is None
