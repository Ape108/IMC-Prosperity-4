# Simple Market-Making Strat


### STRATEGIES ROUND 0 ###

```python
class TomatoStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> float:
        return self.get_mid_price(state, self.symbol)

class EmeraldStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> float:
        return self.get_mid_price(state, self.symbol)
```

Score: ~2,451