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


class InvertedSignalStrategy(SignalStrategy):
    def __init__(self, symbol: Symbol, limit: int, underlying: SignalStrategy) -> None:
        super().__init__(symbol, limit)

        self.underlying = underlying

    def get_required_symbols(self) -> list[Symbol]:
        return [self.symbol, *self.underlying.get_required_symbols()]

    def get_signal(self, state: TradingState) -> Signal | None:
        signal = self.underlying.get_signal(state)

        if signal == Signal.LONG:
            return Signal.SHORT
        elif signal == Signal.SHORT:
            return Signal.LONG
        else:
            return signal


class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

    @abstractmethod
    def get_true_value(self, state: TradingState) -> float:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        max_buy_price = int(true_value) - 1 if true_value % 1 == 0 else floor(true_value)
        min_sell_price = int(true_value) + 1 if true_value % 1 == 0 else ceil(true_value)

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


class RollingZScoreStrategy(SignalStrategy, StatefulStrategy[dict[str, Any]]):

    def __init__(self, symbol: Symbol, limit: int, zscore_period: int, smoothing_period: int, threshold: float) -> None:
        super().__init__(symbol, limit)

        self.zscore_period = zscore_period
        self.smoothing_period = smoothing_period
        self.threshold = threshold

        self.history: list[float] = []

    def get_signal(self, state: TradingState) -> Signal | None:
        self.history.append(self.get_mid_price(state, self.symbol))

        required_history = self.zscore_period + self.smoothing_period
        if len(self.history) < required_history:
            return None
        if len(self.history) > required_history:
            self.history.pop(0)

        hist = pd.Series(self.history)
        score = (
            ((hist - hist.rolling(self.zscore_period).mean()) / hist.rolling(self.zscore_period).std())
            .rolling(self.smoothing_period)
            .mean()
            .iloc[-1]
        )

        if score < -self.threshold:
            return Signal.LONG

        if score > self.threshold:
            return Signal.SHORT

        return None

    def save(self) -> dict[str, Any]:  # type: ignore
        return {"signal": SignalStrategy.save(self), "history": self.history}

    def load(self, data: dict[str, Any]) -> None:  # type: ignore
        SignalStrategy.load(self, data["signal"])
        self.history = data["history"]


class DeanonymizedTradesStrategy(SignalStrategy):
    def __init__(self, symbol: Symbol, limit: int, trader_name: str) -> None:
        super().__init__(symbol, limit)

        self.trader_name = trader_name

    def get_signal(self, state: TradingState) -> Signal | None:
        trades = state.market_trades.get(self.symbol, [])
        trades = [t for t in trades if t.timestamp == state.timestamp - 100 and t.price > 0]

        has_buy_trade = any(t.buyer == self.trader_name for t in trades)
        has_sell_trade = any(t.seller == self.trader_name for t in trades)

        if has_buy_trade and not has_sell_trade:
            return Signal.LONG

        if has_sell_trade and not has_buy_trade:
            return Signal.SHORT

        return None

### STRATEGIES ROUND 0 ###

class OrderBookImbalanceStrategy(Strategy, StatefulStrategy[dict[str, Any]]):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        # Lowered entry threshold to trigger significantly more trades (65/35 split)
        self.entry_threshold = 0.3  
        # Lowered exit threshold to hold onto winners a bit longer
        self.exit_threshold = 0.05  
        
        # Array to hold the last few OBI readings for smoothing
        self.obi_history: list[float] = []

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        best_bid, bid_vol = sorted(order_depth.buy_orders.items(), reverse=True)[0]
        best_ask, ask_vol = sorted(order_depth.sell_orders.items())[0]
        
        bid_vol = abs(bid_vol)
        ask_vol = abs(ask_vol)
        total_vol = bid_vol + ask_vol
        
        if total_vol == 0:
            return

        # 1. Calculate the Raw OBI for this specific tick
        raw_obi = (bid_vol - ask_vol) / total_vol
        
        # 2. Smooth the signal (3-Tick Rolling Average)
        self.obi_history.append(raw_obi)
        if len(self.obi_history) > 3:
            self.obi_history.pop(0)
            
        smoothed_obi = sum(self.obi_history) / len(self.obi_history)
        
        # 3. Execution Logic using the Smoothed Signal
        if smoothed_obi > self.entry_threshold:
            to_buy = self.limit - position
            if to_buy > 0:
                self.buy(best_ask, to_buy)
                
        elif smoothed_obi < -self.entry_threshold:
            to_sell = self.limit + position
            if to_sell > 0:
                self.sell(best_bid, to_sell)
                
        else:
            if abs(smoothed_obi) < self.exit_threshold:
                if position > 0:
                    self.sell(best_ask, position)
                elif position < 0:
                    self.buy(best_bid, abs(position))

    # Save and Load state between ticks for the smoothing array
    def save(self) -> dict[str, Any]:
        return {"obi_history": self.obi_history}

    def load(self, data: dict[str, Any]) -> None:
        self.obi_history = data["obi_history"]


class EmeraldStrategy(MarketMakingStrategy, StatefulStrategy[dict[str, Any]]):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.history: list[float] = [] 

    def get_micro_price(self, state: TradingState) -> float:
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return self.get_mid_price(state, self.symbol)
            
        best_bid, bid_vol = sorted(order_depth.buy_orders.items(), reverse=True)[0]
        best_ask, ask_vol = sorted(order_depth.sell_orders.items())[0]
        
        bid_vol = abs(bid_vol)
        ask_vol = abs(ask_vol)
        
        return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

    def get_true_value(self, state: TradingState) -> float:
        micro_price = self.get_micro_price(state)
        inventory = state.position.get(self.symbol, 0)
        gamma = 0.015  
        
        self.history.append(micro_price)
        if len(self.history) > 100:
            self.history.pop(0)

        reservation_price = micro_price - (inventory * gamma)
        
        return round(reservation_price)

    def save(self) -> dict[str, Any]:
        return {"history": self.history}

    def load(self, data: dict[str, Any]) -> None:
        self.history = data["history"]


class Trader:
    def __init__(self) -> None:
        limits = {
            "TOMATOES": 80,
            "EMERALDS": 80
        }

        # Simplified Strategy Initialization (Fixes the Not Callable Error)
        self.strategies: dict[Symbol, Strategy] = {
            "TOMATOES": OrderBookImbalanceStrategy("TOMATOES", limits["TOMATOES"]),
            "EMERALDS": EmeraldStrategy("EMERALDS", limits["EMERALDS"])
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        for symbol, strategy in self.strategies.items():
            if isinstance(strategy, StatefulStrategy) and symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            strategy_orders, strategy_conversions = strategy.run(state)
            
            # Add safety check so we don't submit empty order lists
            if strategy_orders:
                orders[symbol] = strategy_orders
                
            conversions += strategy_conversions

            if isinstance(strategy, StatefulStrategy):
                new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)
        print(f"Submitting orders: {orders}")
        return orders, conversions, trader_data