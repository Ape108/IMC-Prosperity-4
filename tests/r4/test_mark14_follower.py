"""Tests for Mark14FollowerStrategy with direction + size config."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r4"))

from datamodel import Order, OrderDepth, Trade, TradingState  # noqa
from strategy_h import Mark14FollowerStrategy, MARK14_INFORMED_BOT, MARK14_WINDOW_TICKS


def make_od(buys: dict[int, int], sells: dict[int, int]) -> OrderDepth:
    od = OrderDepth()
    od.buy_orders = dict(buys)
    od.sell_orders = dict(sells)
    return od


def make_state(symbol: str, timestamp: int, market_trades: dict[str, list[Trade]],
               position: int = 0) -> TradingState:
    od = make_od({100: 5}, {102: -5})
    state = TradingState(
        traderData="",
        timestamp=timestamp,
        listings={},
        order_depths={symbol: od},
        own_trades={},
        market_trades=market_trades,
        position={symbol: position},
        observations=None,  # type: ignore
    )
    return state


def test_follow_passive_posts_at_best_bid_when_in_window():
    """direction=follow_passive: post BUY at best_bid (our touch), no cross."""
    strat = Mark14FollowerStrategy(
        "VEV_5300", limit=300, direction="follow_passive", size=50,
    )
    trade = Trade("VEV_5300", 101, 5, MARK14_INFORMED_BOT, "Other", timestamp=100)
    state = make_state("VEV_5300", timestamp=200, market_trades={"VEV_5300": [trade]})

    orders, _ = strat.run(state)

    assert len(orders) == 1
    assert orders[0].price == 100   # best_bid, not best_ask
    assert orders[0].quantity == 50


def test_fade_at_touch_posts_sell_at_best_ask_when_in_window():
    """direction=fade_at_touch: post SELL at best_ask, no cross."""
    strat = Mark14FollowerStrategy(
        "VEV_5400", limit=300, direction="fade_at_touch", size=60,
    )
    trade = Trade("VEV_5400", 101, 5, MARK14_INFORMED_BOT, "Other", timestamp=100)
    state = make_state("VEV_5400", timestamp=200, market_trades={"VEV_5400": [trade]})

    orders, _ = strat.run(state)

    assert len(orders) == 1
    assert orders[0].price == 102   # best_ask
    assert orders[0].quantity == -60   # negative = sell


def test_no_orders_outside_window_for_follow():
    """500-tick window — at timestamp - last_signal_ts > 500, no order."""
    strat = Mark14FollowerStrategy(
        "VEV_5300", limit=300, direction="follow_passive", size=50,
    )
    trade = Trade("VEV_5300", 101, 5, MARK14_INFORMED_BOT, "Other", timestamp=100)
    state = make_state("VEV_5300",
        timestamp=100 + MARK14_WINDOW_TICKS + 1,
        market_trades={"VEV_5300": [trade]},
    )

    orders, _ = strat.run(state)
    assert orders == []


def test_no_orders_outside_window_for_fade():
    """500-tick window — at timestamp - last_signal_ts > 500, no order, even for fade."""
    strat = Mark14FollowerStrategy(
        "VEV_5400", limit=300, direction="fade_at_touch", size=60,
    )
    trade = Trade("VEV_5400", 101, 5, MARK14_INFORMED_BOT, "Other", timestamp=100)
    state = make_state("VEV_5400",
        timestamp=100 + MARK14_WINDOW_TICKS + 1,
        market_trades={"VEV_5400": [trade]},
    )

    orders, _ = strat.run(state)
    assert orders == []


def test_follow_passive_reconcile_to_zero_outside_window():
    """When window passes for follow_passive, target=0; if we hold +pos, post passive exit."""
    strat = Mark14FollowerStrategy(
        "VEV_5300", limit=300, direction="follow_passive", size=50,
    )
    state = make_state("VEV_5300",
        timestamp=2000, market_trades={"VEV_5300": []}, position=50,
    )
    # No Mark 14 signal → target = 0, but we have +50 position → post SELL at best_ask
    orders, _ = strat.run(state)

    assert len(orders) == 1
    assert orders[0].quantity == -50
    assert orders[0].price == 102   # best_ask, passive exit


def test_fade_at_touch_reconcile_to_zero_outside_window():
    """When window passes for fade_at_touch, target=0; if we hold -pos, post passive exit (BUY at best_bid)."""
    strat = Mark14FollowerStrategy(
        "VEV_5400", limit=300, direction="fade_at_touch", size=60,
    )
    state = make_state("VEV_5400",
        timestamp=2000, market_trades={"VEV_5400": []}, position=-60,
    )
    # No Mark 14 signal → target = 0, but we have -60 position → post BUY at best_bid
    orders, _ = strat.run(state)

    assert len(orders) == 1
    assert orders[0].quantity == 60
    assert orders[0].price == 100   # best_bid, passive exit


def test_invalid_direction_raises_value_error():
    """Constructor rejects unknown direction strings, including 'skip' (no longer valid)."""
    import pytest
    with pytest.raises(ValueError):
        Mark14FollowerStrategy("VEV_5300", limit=300, direction="bogus", size=50)
    with pytest.raises(ValueError):
        Mark14FollowerStrategy("VEV_5300", limit=300, direction="skip", size=50)


def test_save_load_roundtrip():
    """save() returns a dict with last_signal_ts; load() restores it."""
    strat = Mark14FollowerStrategy(
        "VEV_5300", limit=300, direction="follow_passive", size=50,
    )
    trade = Trade("VEV_5300", 101, 5, MARK14_INFORMED_BOT, "Other", timestamp=300)
    state = make_state("VEV_5300", timestamp=400, market_trades={"VEV_5300": [trade]})
    strat.run(state)

    saved = strat.save()
    assert saved == {"last_signal_ts": 300}

    new_strat = Mark14FollowerStrategy(
        "VEV_5300", limit=300, direction="follow_passive", size=50,
    )
    new_strat.load(saved)
    assert new_strat.last_signal_ts == 300


def test_multiple_mark14_buys_uses_latest_timestamp():
    """When multiple Mark 14 buys appear in one tick, last_signal_ts uses max."""
    strat = Mark14FollowerStrategy(
        "VEV_5300", limit=300, direction="follow_passive", size=50,
    )
    trades = [
        Trade("VEV_5300", 101, 5, MARK14_INFORMED_BOT, "Other", timestamp=100),
        Trade("VEV_5300", 101, 5, MARK14_INFORMED_BOT, "Other", timestamp=200),
        Trade("VEV_5300", 101, 5, MARK14_INFORMED_BOT, "Other", timestamp=150),
    ]
    state = make_state("VEV_5300", timestamp=300, market_trades={"VEV_5300": trades})
    strat.run(state)
    assert strat.last_signal_ts == 200


def test_no_order_when_already_at_follow_target():
    """When position == target in window, no order emitted (avoid duplicate posts)."""
    strat = Mark14FollowerStrategy(
        "VEV_5300", limit=300, direction="follow_passive", size=50,
    )
    trade = Trade("VEV_5300", 101, 5, MARK14_INFORMED_BOT, "Other", timestamp=100)
    state = make_state(
        "VEV_5300", timestamp=200, market_trades={"VEV_5300": [trade]}, position=50,
    )
    orders, _ = strat.run(state)
    assert orders == []


def test_no_order_when_already_at_fade_target():
    """When position == -size in window, no order emitted for fade_at_touch."""
    strat = Mark14FollowerStrategy(
        "VEV_5400", limit=300, direction="fade_at_touch", size=60,
    )
    trade = Trade("VEV_5400", 101, 5, MARK14_INFORMED_BOT, "Other", timestamp=100)
    state = make_state(
        "VEV_5400", timestamp=200, market_trades={"VEV_5400": [trade]}, position=-60,
    )
    orders, _ = strat.run(state)
    assert orders == []
