import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'submissions', 'r3'))

from datamodel import OrderDepth, TradingState
from strategy import VelvetfruitStrategy, MarketMakingStrategy, StatefulStrategy

def make_vf_state(mid: float) -> TradingState:
    spread = 2
    od = OrderDepth()
    od.buy_orders  = {int(mid) - spread: 10}
    od.sell_orders = {int(mid) + spread: -10}
    return TradingState(
        traderData="", timestamp=0, listings={},
        order_depths={"VELVETFRUIT_EXTRACT": od},
        own_trades={}, market_trades={}, position={}, observations=None,
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
