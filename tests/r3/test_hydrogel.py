import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'submissions', 'r3'))

from datamodel import OrderDepth, TradingState
from strategy import HydrogelStrategy, MAX_CLIP

def make_hg_state(best_bid: int, bid_vol: int, best_ask: int, ask_vol: int, position: int = 0) -> TradingState:
    od = OrderDepth()
    od.buy_orders  = {best_bid: bid_vol}
    od.sell_orders = {best_ask: -ask_vol}
    return TradingState(
        traderData="", timestamp=0, listings={},
        order_depths={"HYDROGEL_PACK": od},
        own_trades={}, market_trades={},
        position={"HYDROGEL_PACK": position} if position != 0 else {},
        observations=None,
    )

def test_max_clip_is_40():
    assert MAX_CLIP == 40

def test_buys_at_or_below_max_buy_price_at_flat_inventory():
    """position=0 → skew=0, width=1.5. max_buy_price=9990, min_sell_price=9994."""
    strat = HydrogelStrategy("HYDROGEL_PACK", 200)
    state = make_hg_state(9990, 10, 9992, 10, position=0)
    orders, _ = strat.run(state)
    buy_orders = [o for o in orders if o.quantity > 0]
    assert len(buy_orders) > 0
    assert all(o.price <= 9990 for o in buy_orders)

def test_sells_at_or_above_min_sell_price_at_flat_inventory():
    """position=0 → min_sell_price = ceil(9992.35 + 1.5) = ceil(9993.85) = 9994."""
    strat = HydrogelStrategy("HYDROGEL_PACK", 200)
    state = make_hg_state(9990, 10, 9992, 10, position=0)
    orders, _ = strat.run(state)
    sell_orders = [o for o in orders if o.quantity < 0]
    assert len(sell_orders) > 0
    assert all(o.price >= 9994 for o in sell_orders)

def test_inventory_skew_lowers_quotes_when_long():
    """Long position → skewed_value drops → sell quotes become more aggressive (lower)."""
    strat = HydrogelStrategy("HYDROGEL_PACK", 200)
    state_flat = make_hg_state(9990, 10, 9992, 10, position=0)
    state_long = make_hg_state(9990, 10, 9992, 10, position=100)
    orders_flat, _ = strat.run(state_flat)
    orders_long, _ = strat.run(state_long)
    # position=100: inventory_ratio=0.5, skew=5.0 → skewed_value = 9992.35-5.0 = 9987.35
    # min_sell_price(long) < min_sell_price(flat) → willing to sell cheaper to unwind
    sell_price_flat = min(o.price for o in orders_flat if o.quantity < 0)
    sell_price_long = min(o.price for o in orders_long if o.quantity < 0)
    assert sell_price_long < sell_price_flat

def test_dynamic_width_widens_with_inventory():
    """position=100 → inventory_ratio=0.5 → width=3.25. max_buy_price=floor(9984.1)=9984."""
    strat = HydrogelStrategy("HYDROGEL_PACK", 200)
    state = make_hg_state(9990, 10, 9992, 10, position=100)
    orders, _ = strat.run(state)
    buy_orders = [o for o in orders if o.quantity > 0]
    # best_ask=9992 > max_buy_price=9984, so passive buy posted at 9984
    assert any(o.price == 9984 for o in buy_orders)

def test_order_quantity_capped_at_max_clip():
    """Even with full 200-unit headroom available, each side capped at MAX_CLIP=40."""
    strat = HydrogelStrategy("HYDROGEL_PACK", 200)
    od = OrderDepth()
    od.buy_orders  = {9985: 200}
    od.sell_orders = {9987: -200}
    state = TradingState(
        traderData="", timestamp=0, listings={},
        order_depths={"HYDROGEL_PACK": od},
        own_trades={}, market_trades={}, position={}, observations=None,
    )
    orders, _ = strat.run(state)
    assert len(orders) > 0
    for o in orders:
        assert abs(o.quantity) <= MAX_CLIP
