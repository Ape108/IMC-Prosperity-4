"""Tests for Vev4000MMStrategy — inside-spread MM with hard position bands."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r4"))

import pytest
from datamodel import Order, OrderDepth, TradingState  # noqa
from strategy_h import Vev4000MMStrategy


def make_strategy(**kwargs) -> Vev4000MMStrategy:
    defaults = dict(symbol="VEV_4000", limit=300)
    defaults.update(kwargs)
    return Vev4000MMStrategy(**defaults)


# ── _compute_sizes ───────────────────────────────────────────────────────────

def test_sizes_inner_band_quotes_both_sides_full_clip():
    s = make_strategy(band_inner=20, band_cap=30, max_per_tick=10)
    bid_qty, ask_qty = s._compute_sizes(position=0)
    assert bid_qty == 10
    assert ask_qty == 10


def test_sizes_inner_band_at_lower_bound_includes_minus_band_inner():
    """pos = -band_inner is INSIDE the inner band → both sides quoted."""
    s = make_strategy(band_inner=20, band_cap=30, max_per_tick=10)
    bid_qty, ask_qty = s._compute_sizes(position=-20)
    assert bid_qty == 10
    assert ask_qty == 10


def test_sizes_inner_band_at_upper_bound_includes_plus_band_inner():
    """pos = +band_inner is INSIDE the inner band → both sides quoted."""
    s = make_strategy(band_inner=20, band_cap=30, max_per_tick=10)
    bid_qty, ask_qty = s._compute_sizes(position=20)
    assert bid_qty == 10
    assert ask_qty == 10


def test_sizes_outer_long_band_suppresses_bid_quotes_ask_only():
    """pos in (band_inner, band_cap]: bid suppressed, ask quotes the reducing side."""
    s = make_strategy(band_inner=20, band_cap=30, max_per_tick=10)
    bid_qty, ask_qty = s._compute_sizes(position=25)
    assert bid_qty == 0
    assert ask_qty == 10


def test_sizes_outer_short_band_suppresses_ask_quotes_bid_only():
    s = make_strategy(band_inner=20, band_cap=30, max_per_tick=10)
    bid_qty, ask_qty = s._compute_sizes(position=-25)
    assert bid_qty == 10
    assert ask_qty == 0


def test_sizes_at_long_cap_yields_zero_buy_room():
    """pos = +band_cap: no buy room (band_cap - position = 0)."""
    s = make_strategy(band_inner=20, band_cap=30, max_per_tick=10)
    bid_qty, ask_qty = s._compute_sizes(position=30)
    assert bid_qty == 0
    # At pos=30, ask room is band_cap + 30 = 60 lots, capped by max_per_tick=10
    assert ask_qty == 10


def test_sizes_above_long_cap_no_quotes():
    """|pos| > band_cap: should never happen but defensively returns (0, 0)."""
    s = make_strategy(band_inner=20, band_cap=30, max_per_tick=10)
    bid_qty, ask_qty = s._compute_sizes(position=31)
    assert bid_qty == 0
    assert ask_qty == 0


def test_sizes_partial_room_clamps_below_max_per_tick():
    """pos=29: outer band → bid suppressed regardless; sell room capped at max_per_tick."""
    s = make_strategy(band_inner=20, band_cap=30, max_per_tick=10)
    bid_qty, ask_qty = s._compute_sizes(position=29)
    # pos=29 is OUTSIDE the inner band (29 > 20), so bid is suppressed regardless.
    assert bid_qty == 0
    # Sell room = 30 + 29 = 59, capped by max_per_tick = 10.
    assert ask_qty == 10


# ── _compute_prices ──────────────────────────────────────────────────────────

def test_prices_normal_spread_returns_inside_spread_quotes():
    """Spread = 21, offset = 5 → bid 5 above best_bid, ask 5 below best_ask."""
    s = make_strategy(offset=5)
    od = OrderDepth()
    od.buy_orders = {1198: 11}
    od.sell_orders = {1219: -11}
    bid_price, ask_price = s._compute_prices(od)
    assert bid_price == 1203
    assert ask_price == 1214


def test_prices_narrow_spread_falls_back_to_at_touch():
    """Spread = 2, offset = 5 → no inside room → quote at touch (best_bid, best_ask)."""
    s = make_strategy(offset=5)
    od = OrderDepth()
    od.buy_orders = {100: 10}
    od.sell_orders = {102: -10}
    bid_price, ask_price = s._compute_prices(od)
    assert bid_price == 100
    assert ask_price == 102


def test_prices_just_wide_enough_returns_inside_spread():
    """Spread = 11, offset = 5 → 11 - 10 = 1 (>= 1) → inside-spread quotes,
    bid_quote = best_bid + 5, ask_quote = best_ask - 5, gap = 1."""
    s = make_strategy(offset=5)
    od = OrderDepth()
    od.buy_orders = {1000: 10}
    od.sell_orders = {1011: -10}
    bid_price, ask_price = s._compute_prices(od)
    assert bid_price == 1005
    assert ask_price == 1006


def test_prices_one_below_threshold_falls_back_to_at_touch():
    """Spread = 10, offset = 5 → 10 - 10 = 0 (< 1) → at-touch fallback."""
    s = make_strategy(offset=5)
    od = OrderDepth()
    od.buy_orders = {1000: 10}
    od.sell_orders = {1010: -10}
    bid_price, ask_price = s._compute_prices(od)
    assert bid_price == 1000
    assert ask_price == 1010


# ── act() integration ────────────────────────────────────────────────────────

def make_state(
    bid_price: int, ask_price: int,
    bid_vol: int = 11, ask_vol: int = 11,
    position: int = 0,
    timestamp: int = 0,
) -> TradingState:
    """Build a one-symbol TradingState with a VEV_4000 book."""
    od = OrderDepth()
    od.buy_orders = {bid_price: bid_vol}
    od.sell_orders = {ask_price: -ask_vol}
    return TradingState(
        traderData="",
        timestamp=timestamp,
        listings={},
        order_depths={"VEV_4000": od},
        own_trades={},
        market_trades={},
        position={"VEV_4000": position} if position else {},
        observations=None,  # type: ignore
    )


def test_act_emits_two_sided_inside_spread_quotes_in_inner_band():
    s = make_strategy(offset=5, band_inner=20, band_cap=30, max_per_tick=10)
    state = make_state(bid_price=1198, ask_price=1219, position=0)
    orders, _ = s.run(state)
    buys = [o for o in orders if o.quantity > 0]
    sells = [o for o in orders if o.quantity < 0]
    assert len(buys) == 1
    assert buys[0].price == 1203
    assert buys[0].quantity == 10
    assert len(sells) == 1
    assert sells[0].price == 1214
    assert sells[0].quantity == -10


def test_act_only_quotes_ask_when_long_above_band_inner():
    s = make_strategy(offset=5, band_inner=20, band_cap=30, max_per_tick=10)
    state = make_state(bid_price=1198, ask_price=1219, position=25)
    orders, _ = s.run(state)
    buys = [o for o in orders if o.quantity > 0]
    sells = [o for o in orders if o.quantity < 0]
    assert buys == []
    assert len(sells) == 1
    assert sells[0].price == 1214
    assert sells[0].quantity == -10


def test_act_only_quotes_bid_when_short_below_negative_band_inner():
    s = make_strategy(offset=5, band_inner=20, band_cap=30, max_per_tick=10)
    state = make_state(bid_price=1198, ask_price=1219, position=-25)
    orders, _ = s.run(state)
    buys = [o for o in orders if o.quantity > 0]
    sells = [o for o in orders if o.quantity < 0]
    assert len(buys) == 1
    assert buys[0].price == 1203
    assert buys[0].quantity == 10
    assert sells == []


def test_act_falls_back_to_at_touch_on_narrow_spread():
    s = make_strategy(offset=5, band_inner=20, band_cap=30, max_per_tick=10)
    state = make_state(bid_price=100, ask_price=102, position=0)
    orders, _ = s.run(state)
    buys = [o for o in orders if o.quantity > 0]
    sells = [o for o in orders if o.quantity < 0]
    assert len(buys) == 1
    assert buys[0].price == 100
    assert len(sells) == 1
    assert sells[0].price == 102


def test_act_emits_no_orders_when_book_is_empty():
    """Base Strategy.run() guards on both sides being non-empty; act() never
    runs and orders is empty."""
    s = make_strategy()
    od = OrderDepth()
    od.buy_orders = {}
    od.sell_orders = {}
    state = TradingState(
        traderData="", timestamp=0, listings={},
        order_depths={"VEV_4000": od}, own_trades={}, market_trades={},
        position={}, observations=None,  # type: ignore
    )
    orders, _ = s.run(state)
    assert orders == []


def test_act_emits_no_orders_above_band_cap():
    """Defensive: if external fills push position past band_cap, both sides
    suppressed."""
    s = make_strategy(offset=5, band_inner=20, band_cap=30, max_per_tick=10)
    state = make_state(bid_price=1198, ask_price=1219, position=31)
    orders, _ = s.run(state)
    assert orders == []


# ── Trader wiring ────────────────────────────────────────────────────────────

def test_trader_registers_vev4000_mm_strategy_on_vev_4000():
    from strategy_h import Trader
    trader = Trader()
    strat = trader.strategies["VEV_4000"]
    assert isinstance(strat, Vev4000MMStrategy)
    assert strat.symbol == "VEV_4000"
    assert strat.limit == 300


def test_trader_registers_voucher_strategy_on_other_strikes_unchanged():
    """Sanity: only VEV_4000 changes; other vouchers still go through the
    pre-existing dispatch (VoucherStrategy or Mark14FollowerStrategy)."""
    from strategy_h import Trader, VoucherStrategy, Mark14FollowerStrategy
    trader = Trader()
    # VEV_5000 → no mark14_config entry → VoucherStrategy
    assert isinstance(trader.strategies["VEV_5000"], VoucherStrategy)
    # VEV_5300 → mark14_config entry → Mark14FollowerStrategy
    assert isinstance(trader.strategies["VEV_5300"], Mark14FollowerStrategy)
