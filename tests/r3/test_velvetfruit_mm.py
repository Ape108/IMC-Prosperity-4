import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'submissions', 'r3'))

from datamodel import OrderDepth, TradingState
from strategy import VelvetfruitStrategy, MarketMakingStrategy, StatefulStrategy

def make_vf_state(
    mid: float,
    bid_vol: int = 10,
    ask_vol: int = 10,
    position: int = 0,
) -> TradingState:
    spread = 2
    od = OrderDepth()
    od.buy_orders  = {int(mid) - spread: bid_vol}
    od.sell_orders = {int(mid) + spread: -ask_vol}
    pos = {"VELVETFRUIT_EXTRACT": position} if position != 0 else {}
    return TradingState(
        traderData="", timestamp=0, listings={},
        order_depths={"VELVETFRUIT_EXTRACT": od},
        own_trades={}, market_trades={}, position=pos, observations=None,
    )

def test_velvetfruit_is_market_making_strategy():
    strat = VelvetfruitStrategy("VELVETFRUIT_EXTRACT", 200)
    assert isinstance(strat, MarketMakingStrategy)

def test_velvetfruit_is_not_stateful():
    """No state to persist — Trader.run will not call save()/load() on it."""
    strat = VelvetfruitStrategy("VELVETFRUIT_EXTRACT", 200)
    assert not isinstance(strat, StatefulStrategy)

def test_velvetfruit_true_value_equals_mid_price():
    strat = VelvetfruitStrategy("VELVETFRUIT_EXTRACT", 200)
    state = make_vf_state(5250.0)
    # buy_orders={5248:10}, sell_orders={5252:-10}
    # get_mid_price picks popular bid (max volume → 5248) and popular ask (min volume → 5252)
    # mid = (5248 + 5252) / 2 = 5250.0
    assert strat.get_true_value(state) == 5250.0


def test_flat_inventory_posts_tight_symmetric_quotes():
    """Zero inventory: quotes straddle microprice by dynamic_width=0.5."""
    strat = VelvetfruitStrategy("VELVETFRUIT_EXTRACT", 200)
    state = make_vf_state(5250.0)
    # book: bid=5248 vol=10, ask=5252 vol=10
    # microprice = (5248*10 + 5252*10)/20 = 5250.0
    # base = 0.9*5250 + 0.1*5250 = 5250.0
    # skewed = 5250.0, width = 0.5
    # max_buy  = floor(5249.5) = 5249
    # min_sell = ceil(5250.5)  = 5251
    strat.run(state)
    buy_prices  = [o.price for o in strat.orders if o.quantity > 0]
    sell_prices = [o.price for o in strat.orders if o.quantity < 0]
    assert 5249 in buy_prices,  f"Expected buy at 5249, got {buy_prices}"
    assert 5251 in sell_prices, f"Expected sell at 5251, got {sell_prices}"


def test_long_inventory_lowers_buy_threshold():
    """Long 100/200 units: buy threshold drops below flat-inventory case."""
    strat_flat = VelvetfruitStrategy("VELVETFRUIT_EXTRACT", 200)
    strat_long = VelvetfruitStrategy("VELVETFRUIT_EXTRACT", 200)

    strat_flat.run(make_vf_state(5250.0, position=0))
    strat_long.run(make_vf_state(5250.0, position=100))

    # flat: inventory_ratio=0, skewed=5250, width=0.5, max_buy=5249
    # long: inventory_ratio=0.5, skewed=5247.5, width=1.25, max_buy=floor(5246.25)=5246
    flat_buy = max(o.price for o in strat_flat.orders if o.quantity > 0)
    long_buy = max(o.price for o in strat_long.orders if o.quantity > 0)
    assert long_buy < flat_buy, (
        f"Long inventory should lower buy threshold: flat={flat_buy} long={long_buy}"
    )


def test_short_inventory_raises_sell_threshold():
    """Short 100/200 units: sell threshold rises above flat-inventory case."""
    strat_flat  = VelvetfruitStrategy("VELVETFRUIT_EXTRACT", 200)
    strat_short = VelvetfruitStrategy("VELVETFRUIT_EXTRACT", 200)

    strat_flat.run(make_vf_state(5250.0, position=0))
    strat_short.run(make_vf_state(5250.0, position=-100))

    # flat:  inventory_ratio=0,    skewed=5250,   width=0.5,  min_sell=5251
    # short: inventory_ratio=-0.5, skewed=5252.5, width=1.25, min_sell=ceil(5253.75)=5254
    flat_sell  = min(o.price for o in strat_flat.orders  if o.quantity < 0)
    short_sell = min(o.price for o in strat_short.orders if o.quantity < 0)
    assert short_sell > flat_sell, (
        f"Short inventory should raise sell threshold: flat={flat_sell} short={short_sell}"
    )


def test_max_clip_limits_posted_quantity():
    """Every posted order must have abs(quantity) <= MAX_CLIP (40)."""
    strat = VelvetfruitStrategy("VELVETFRUIT_EXTRACT", 200)
    strat.run(make_vf_state(5250.0, position=0))
    for o in strat.orders:
        assert abs(o.quantity) <= 40, (
            f"Order qty {abs(o.quantity)} exceeds MAX_CLIP=40"
        )
