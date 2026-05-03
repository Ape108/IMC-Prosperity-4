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



class AdvancedMMStrategy(StatefulStrategy[list[float]]):
    """
    High-Frequency Market Making strategy utilizing Order Book Imbalance (OBI),
    volume-weighted microprices, and momentum-adjusted take/make thresholds.
    """
    def __init__(self, symbol: Symbol, limit: int, window: int = 10) -> None:
        super().__init__(symbol, limit)
        self.window = window
        self.mid_history: list[float] = []

    def save(self) -> list[float]:
        return self.mid_history

    def load(self, data: list[float]) -> None:
        self.mid_history = data

    def best_bid_ask(self, od: OrderDepth) -> tuple[int | None, int | None]:
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return best_bid, best_ask

    def get_wall_mid(self, od: OrderDepth) -> float | None:
        """Finds the price levels with the heaviest volume walls and averages them."""
        if not od.buy_orders or not od.sell_orders:
            return None
        
        # Maximize by volume (remembering sell volumes are negative)
        popular_buy = max(od.buy_orders.items(), key=lambda x: x[1])[0]
        popular_sell = min(od.sell_orders.items(), key=lambda x: x[1])[0]
        return (popular_buy + popular_sell) / 2.0

    def microprice(self, od: OrderDepth, best_bid: int, best_ask: int, default: float) -> float:
        """Calculates the volume-weighted mid price."""
        bid_vol = od.buy_orders.get(best_bid, 0)
        ask_vol = -od.sell_orders.get(best_ask, 0) # Prosperity sell volumes are negative
        
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return default
            
        return (best_bid * ask_vol + best_ask * bid_vol) / total_vol

    def clip_buy_size(self, position: int, size: int) -> int:
        return min(size, self.limit - position)

    def clip_sell_size(self, position: int, size: int) -> int:
        return min(size, self.limit + position)

    def act(self, state: TradingState) -> None:
        od = state.order_depths[self.symbol]
        pos = state.position.get(self.symbol, 0)

        best_bid, best_ask = self.best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return

        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0
        
        wall_mid = self.get_wall_mid(od)
        if wall_mid is None:
            wall_mid = mid

        micro = self.microprice(od, best_bid, best_ask, mid)

        # 1. Fair Value Calculation (70% Microprice / 30% Wall Mid)
        fair_input = 0.7 * micro + 0.3 * wall_mid

        # 2. Update Rolling History
        self.mid_history.append(fair_input)
        if len(self.mid_history) > self.window:
            self.mid_history.pop(0)

        # 3. Linearly Weighted Moving Average (Heavier weight on recent ticks)
        weights = list(range(1, len(self.mid_history) + 1))
        fair = sum(w * x for w, x in zip(weights, self.mid_history)) / sum(weights)

        # 4. Short-term Momentum
        momentum = 0.0
        if len(self.mid_history) >= 5:
            momentum = self.mid_history[-1] - self.mid_history[-5]

        # 5. Top-of-book Imbalance (OBI)
        bid_vol = od.buy_orders.get(best_bid, 0)
        ask_vol = -od.sell_orders.get(best_ask, 0)
        imbalance = 0.0
        if bid_vol + ask_vol > 0:
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)

        # 6. Inventory Skew
        inv_skew = 1.5 * (pos / self.limit)

        # 7. Dynamic Take Thresholds
        take_buy_thresh = fair - 0.75
        take_sell_thresh = fair + 0.75

        # Momentum adjustment: Lean into the trend
        if momentum > 1.0:
            take_buy_thresh += 0.5
            take_sell_thresh += 1.0
        elif momentum < -1.0:
            take_buy_thresh -= 1.0
            take_sell_thresh -= 0.5

       # --- AGGRESSIVE TAKING ---
        # Uncapped taking size: Sweep up to 40 units at a time instead of 6
        aggressive_chunk = 40 

        if best_ask <= take_buy_thresh and imbalance > -0.6:
            buy_qty = self.clip_buy_size(pos, aggressive_chunk)
            if buy_qty > 0:
                self.buy(best_ask, buy_qty)
                pos += buy_qty  

        if best_bid >= take_sell_thresh and imbalance < 0.6:
            sell_qty = self.clip_sell_size(pos, aggressive_chunk)
            if sell_qty > 0:
                self.sell(best_bid, sell_qty)
                pos -= sell_qty

        # --- PASSIVE QUOTING ---
        allow_passive = spread >= 2
        if allow_passive:
            signal_shift = 0.5 * imbalance + 0.3 * momentum - inv_skew

            bid_px = int(floor(min(best_bid + 1, fair - 1 + signal_shift)))
            ask_px = int(ceil(max(best_ask - 1, fair + 1 + signal_shift)))

            # Scale quote size down when inventory is large
            # Scaled up for an 80-limit asset
            base_size = 40
            inventory_penalty = int(abs(pos) / 2) # Slowly scales down size as you approach 80
            passive_size = max(5, base_size - inventory_penalty)

            # Circuit breakers: Scaled up to trigger right before the 80 limit
            can_post_bid = not (pos > 70 and momentum < 0)
            can_post_ask = not (pos < -70 and momentum > 0)

            buy_size = self.clip_buy_size(pos, passive_size)
            sell_size = self.clip_sell_size(pos, passive_size)

            if can_post_bid and buy_size > 0:
                self.buy(bid_px, buy_size)
            if can_post_ask and sell_size > 0:
                self.sell(ask_px, sell_size) 

class Trader:
    def __init__(self) -> None:
        limits = {
            "ASH_COATED_OSMIUM": 80,
            "INTARIAN_PEPPER_ROOT": 80
        }

        self.strategies: dict[Symbol, Strategy] = {
            symbol: clazz(symbol, limits[symbol])
            for symbol, clazz in {
                "ASH_COATED_OSMIUM": AdvancedMMStrategy,
                "INTARIAN_PEPPER_ROOT": AdvancedMMStrategy,
            }.items()
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
            orders[symbol] = strategy_orders
            conversions += strategy_conversions

            if isinstance(strategy, StatefulStrategy):
                new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)
        print(f"Submitting orders: {orders}")
        return orders, conversions, trader_data


