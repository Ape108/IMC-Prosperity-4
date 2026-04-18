import json
from abc import abstractmethod
from enum import IntEnum
from math import ceil, floor
from typing import Any

import pandas as pd
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

type JSON = dict[str, Any] | list[Any] | str | int | float | bool | None


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            [],  # self.compress_trades(state.own_trades),
            [],  # self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


class Strategy[T: JSON]:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def get_required_symbols(self) -> list[Symbol]:
        return [self.symbol]

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = list[Order]()
        self.conversions = 0

        if all(
            (
                v in state.order_depths
                and len(state.order_depths[v].buy_orders) > 0
                and len(state.order_depths[v].sell_orders) > 0
            )
            for v in self.get_required_symbols()
        ):
            self.act(state)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2


class StatefulStrategy[T: JSON](Strategy):
    @abstractmethod
    def save(self) -> T:
        raise NotImplementedError()

    @abstractmethod
    def load(self, data: T) -> None:
        raise NotImplementedError()


class Signal(IntEnum):
    NEUTRAL = 0
    SHORT = 1
    LONG = 2


class SignalStrategy(StatefulStrategy[int]):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.signal = Signal.NEUTRAL

    @abstractmethod
    def get_signal(self, state: TradingState) -> Signal | None:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        new_signal = self.get_signal(state)
        if new_signal is not None:
            self.signal = new_signal

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        if self.signal == Signal.NEUTRAL:
            if position < 0:
                self.buy(self.get_buy_price(order_depth), -position)
            elif position > 0:
                self.sell(self.get_sell_price(order_depth), position)
        elif self.signal == Signal.SHORT:
            self.sell(self.get_sell_price(order_depth), self.limit + position)
        elif self.signal == Signal.LONG:
            self.buy(self.get_buy_price(order_depth), self.limit - position)

    def get_buy_price(self, order_depth: OrderDepth) -> int:
        return min(order_depth.sell_orders.keys())

    def get_sell_price(self, order_depth: OrderDepth) -> int:
        return max(order_depth.buy_orders.keys())

    def save(self) -> int:
        return self.signal.value

    def load(self, data: int) -> None:
        self.signal = Signal(data)


class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

    @abstractmethod
    def get_true_value(self, state: TradingState) -> float:
        raise NotImplementedError()

    def act(self, state: TradingState, quote: int = 1) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        
        max_buy_price = int(true_value) - quote if true_value % 1 == 0 else floor(true_value)
        min_sell_price = int(true_value) + quote if true_value % 1 == 0 else ceil(true_value)

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0:
            price = next((price + 1 for price, _ in buy_orders if price < max_buy_price), max_buy_price)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0:
            price = next((price - 1 for price, _ in sell_orders if price > min_sell_price), min_sell_price)
            self.sell(price, to_sell)


class BuyHoldStrategy(Strategy):
    """
    Base buy-hold strategy: accumulate to max long and never sell.

    Each tick:
      1. Aggressively hits all available asks (no price filter).
      2. Posts any remaining capacity as a passive limit at best_bid+1 to
         get filled in future ticks.

    Subclass and override act() to add product-specific entry filters,
    or override get_max_price() to cap the price you're willing to pay.
    """

    def get_max_price(self, state: TradingState) -> int | None:
        """Return None to buy at any price, or an int to cap entry price."""
        return None

    def act(self, state: TradingState) -> None:
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        if to_buy <= 0:
            return

        order_depth = state.order_depths[self.symbol]
        sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        max_price = self.get_max_price(state)

        # Aggressively take asks (optionally filtered by max_price)
        for price, volume in sell_orders:
            if to_buy <= 0:
                break
            if max_price is not None and price > max_price:
                break
            quantity = min(to_buy, -volume)
            self.buy(price, quantity)
            to_buy -= quantity

        # Post remainder as passive limit buy to accumulate on future ticks
        if to_buy > 0:
            best_bid = buy_orders[0][0] if buy_orders else sell_orders[0][0] - 2
            passive_price = best_bid + 1
            if max_price is None or passive_price <= max_price:
                self.buy(passive_price, to_buy)


### STRATEGIES ROUND 1 ###


class OsmiumStrategy(MarketMakingStrategy):
    """ 
        Notes:
        - IMC hinted at a possible hidden trader like Olivia, couldn't find anything though
        
        Results:
        - 
    """
    def get_true_value(self, state: TradingState) -> float:
        expected_true_value = 10_000
        max_delta = 5
        mid_price = self.get_mid_price(state, self.symbol)
        if (expected_true_value - max_delta) <= mid_price <= (expected_true_value + max_delta):
            return expected_true_value
        return mid_price
        

class PepperRootStrategy(StatefulStrategy[list[float]]):
    """
        Notes:
        -

        Results:
        - 
    """

    GAMMA = 0.005       # Risk aversion (A-S)
    SIGMA = 1.75        # Per-tick mid-price volatility
    DRIFT_WINDOW = 100  # Observations kept for slope estimation
    DRIFT_MIN_OBS = 10  # Minimum observations before drift is trusted
    DRIFT_CAP = 3.0     # Clamp to prevent outlier-driven extremes

    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.mid_history: list[float] = []

    def _compute_drift(self) -> float:
        n = len(self.mid_history)
        if n < self.DRIFT_MIN_OBS:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(self.mid_history) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(self.mid_history))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den == 0:
            return 0.0
        slope = num / den
        return max(-self.DRIFT_CAP, min(self.DRIFT_CAP, slope))

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_mid = self.get_mid_price(state, self.symbol)
        self.mid_history.append(popular_mid)
        if len(self.mid_history) > self.DRIFT_WINDOW:
            self.mid_history.pop(0)

        inventory = state.position.get(self.symbol, 0)
        to_buy = self.limit - inventory
        to_sell = self.limit + inventory

        # 0-EV clearing: near position limits, skip passive quoting and
        # aggressively cross the spread to neutralize. Fixes the quote=0
        # no-op in the original AS (floor/ceil ignores quote for non-integer
        # true_value).
        if abs(inventory) >= self.limit - 5:
            if inventory > 0:
                for price, volume in buy_orders:
                    if to_sell <= 0:
                        break
                    qty = min(to_sell, volume)
                    self.sell(price, qty)
                    to_sell -= qty
            else:
                for price, volume in sell_orders:
                    if to_buy <= 0:
                        break
                    qty = min(to_buy, -volume)
                    self.buy(price, qty)
                    to_buy -= qty
            return

        total_time = 1_000_000
        time_left = max(0.0, (total_time - state.timestamp) / total_time)
        drift = self._compute_drift()
        reservation_price = popular_mid - (inventory * self.GAMMA * (self.SIGMA ** 2) * time_left) + drift

        # Suppress passive sells when trend is confirmed bullish — stops the
        # strategy from systematically selling into a rising market.
        ask_suppressed = drift > 0.05

        quote = 1
        max_buy_price = int(reservation_price) - quote if reservation_price % 1 == 0 else floor(reservation_price)
        min_sell_price = int(reservation_price) + quote if reservation_price % 1 == 0 else ceil(reservation_price)

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0:
            price = next((p + 1 for p, _ in buy_orders if p < max_buy_price), max_buy_price)
            self.buy(price, to_buy)

        if not ask_suppressed:
            for price, volume in buy_orders:
                if to_sell > 0 and price >= min_sell_price:
                    quantity = min(to_sell, volume)
                    self.sell(price, quantity)
                    to_sell -= quantity

            if to_sell > 0:
                price = next((p - 1 for p, _ in sell_orders if p > min_sell_price), min_sell_price)
                self.sell(price, to_sell)

    def save(self) -> list[float]:
        return self.mid_history

    def load(self, data: list[float]) -> None:
        self.mid_history = data


class Trader:
    def __init__(self) -> None:
        limits = {
            "ASH_COATED_OSMIUM": 80,
            "INTARIAN_PEPPER_ROOT": 80
        }

        self.strategies: dict[Symbol, Strategy] = {
            symbol: clazz(symbol, limits[symbol])
            for symbol, clazz in {
                "ASH_COATED_OSMIUM": OsmiumStrategy,
                "INTARIAN_PEPPER_ROOT": PepperRootStrategy,
            }.items()
        }

    def bid(self) -> int:
        """ For bid
            - 25 for default
        """
        return 25
    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        for symbol, strategy in self.strategies.items():
            if isinstance(strategy, StatefulStrategy) and symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            strategy_orders, strategy_conversions = strategy.run(state)
            orders[symbol] = strategy_orders
            conversions += strategy_conversions

            if isinstance(strategy, StatefulStrategy):
                new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)
        print(f"Submitting orders: {orders}")
        return orders, conversions, trader_data