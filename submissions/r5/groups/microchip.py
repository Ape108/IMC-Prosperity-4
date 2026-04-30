import json
from abc import abstractmethod
from math import ceil, floor
from typing import Any

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
            [],
            [],
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
                    [trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
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
                observation.sugarPrice,
                observation.sunlightIndex,
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


class R5BaseMMStrategy(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int, width: int = 1) -> None:
        super().__init__(symbol, limit)
        self.width = width

    def _microprice(self, state: TradingState) -> float:
        order_depth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        bid_vol = order_depth.buy_orders[best_bid]
        ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

    def get_true_value(self, state: TradingState) -> float:
        return self._microprice(state)

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
            passive_buy = max_buy_price - self.width + 1
            self.buy(passive_buy, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0:
            passive_sell = min_sell_price + self.width - 1
            self.sell(passive_sell, to_sell)


class R5LeadLagMMStrategy(R5BaseMMStrategy, StatefulStrategy[dict[str, Any]]):
    """
    Bias own quotes based on a leader product's tick return at lag-N.

    Operationalizes the cross-correlation result corr(ret_self[t], ret_leader[t-N]) = peak_corr.
    Maintains a rolling list of leader mids of length lag_ticks + 2; the single-tick leader
    return at t-N is the active signal each tick.

    Falls back gracefully if the leader book is missing/one-sided — leader_history is reset.
    """

    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        width: int,
        leader_symbol: Symbol,
        lag_ticks: int,
        k: float,
    ) -> None:
        super().__init__(symbol, limit, width)
        self.leader_symbol = leader_symbol
        self.lag_ticks = lag_ticks
        self.k = k
        self.leader_history: list[float] = []
        self.bias_fired = 0
        self.warmup_ticks = 0

    def get_true_value(self, state: TradingState) -> float:
        base = self._microprice(state)

        leader_depth = state.order_depths.get(self.leader_symbol)
        if leader_depth is None or not leader_depth.buy_orders or not leader_depth.sell_orders:
            self.leader_history = []
            self.warmup_ticks += 1
            return base

        leader_bid = max(leader_depth.buy_orders.keys())
        leader_ask = min(leader_depth.sell_orders.keys())
        leader_mid = (leader_bid + leader_ask) / 2

        self.leader_history.append(leader_mid)
        max_len = self.lag_ticks + 2
        if len(self.leader_history) > max_len:
            self.leader_history.pop(0)

        if len(self.leader_history) == max_len:
            mid_at_lag_minus_1 = self.leader_history[0]
            mid_at_lag = self.leader_history[1]
            if mid_at_lag_minus_1 != 0:
                leader_return = (mid_at_lag - mid_at_lag_minus_1) / mid_at_lag_minus_1
                base += self.k * leader_return * base
                self.bias_fired += 1
            else:
                self.warmup_ticks += 1
        else:
            self.warmup_ticks += 1

        return base

    def save(self) -> dict[str, Any]:
        return {
            "leader_history": list(self.leader_history),
            "bias_fired": self.bias_fired,
            "warmup_ticks": self.warmup_ticks,
        }

    def load(self, data: dict[str, Any]) -> None:
        self.leader_history = list(data.get("leader_history", []))
        self.bias_fired = int(data.get("bias_fired", 0))
        self.warmup_ticks = int(data.get("warmup_ticks", 0))


SYMBOLS = [
    "MICROCHIP_CIRCLE",
    "MICROCHIP_OVAL",
    "MICROCHIP_RECTANGLE",
    "MICROCHIP_SQUARE",
    "MICROCHIP_TRIANGLE",
]

LIMIT = 10


class Trader:
    def __init__(self) -> None:
        self.strategies: dict[Symbol, Strategy] = {}

        def mm_baseline():
            """Control: all 5 products at width=1. Reproduces strategy_h.py rows.

            MICROCHIP_CIRCLE                 -531.00   -1933.00   10477.00    8013.00
            MICROCHIP_OVAL                   3824.00    4264.00     556.00    8644.00
            MICROCHIP_RECTANGLE             11915.50    4787.00  -14246.00    2456.50
            MICROCHIP_SQUARE                  522.50   -2550.00    6549.00    4521.50
            MICROCHIP_TRIANGLE               4066.00   -1916.00    7457.00    9607.00
            
            w/ queue-penetration 0:
            MICROCHIP_CIRCLE                -1983.00   -3427.00    8881.00    3471.00
            MICROCHIP_OVAL                   2319.00    2771.00    -516.00    4574.00
            MICROCHIP_RECTANGLE             10411.50    3331.00  -15614.00   -1871.50
            MICROCHIP_SQUARE                -1207.50   -5108.00    4005.00   -2310.50
            MICROCHIP_TRIANGLE               2561.00   -3645.00    6019.00    4935.00
            """
            for sym in SYMBOLS:
                self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=1)

        def lead_lag_oval_k1():
            """Option A, k=1.0 (most aggressive — diagnostic test).

            MICROCHIP_OVAL                -227148.00 -217799.50 -169894.00 -614841.50
            """
            for sym in SYMBOLS:
                if sym == "MICROCHIP_OVAL":
                    self.strategies[sym] = R5LeadLagMMStrategy(
                        "MICROCHIP_OVAL", LIMIT, width=1,
                        leader_symbol="MICROCHIP_CIRCLE", lag_ticks=50, k=1.0,
                    )
                else:
                    self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=1)

        def lead_lag_oval_k05():
            """Option A, k=0.5 (moderate over-bet).

            MICROCHIP_OVAL                -106942.00 -105314.00  -81930.00 -294186.00
            """
            for sym in SYMBOLS:
                if sym == "MICROCHIP_OVAL":
                    self.strategies[sym] = R5LeadLagMMStrategy(
                        "MICROCHIP_OVAL", LIMIT, width=1,
                        leader_symbol="MICROCHIP_CIRCLE", lag_ticks=50, k=0.5,
                    )
                else:
                    self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=1)

        def lead_lag_oval_k005():
            """Option A, k=0.05 (calibrated to |xcorr|).

            MICROCHIP_OVAL                   3893.00    3859.50    -633.00    7119.50
            """
            for sym in SYMBOLS:
                if sym == "MICROCHIP_OVAL":
                    self.strategies[sym] = R5LeadLagMMStrategy(
                        "MICROCHIP_OVAL", LIMIT, width=1,
                        leader_symbol="MICROCHIP_CIRCLE", lag_ticks=50, k=0.05,
                    )
                else:
                    self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=1)

        def rectangle_widen():
            """Option B. RECTANGLE at width=2; others at width=1. Targets D+4 -14k outlier.

            MICROCHIP_RECTANGLE             12216.50    6093.00  -13029.00    5280.50
            
            queue-penetration 0:
            MICROCHIP_RECTANGLE             10712.50    4723.00  -14341.00    1094.50
            """
            for sym in SYMBOLS:
                if sym == "MICROCHIP_RECTANGLE":
                    self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=2)
                else:
                    self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=1)

        rectangle_widen()

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders: dict[Symbol, list[Order]] = {}
        conversions = 0

        raw = json.loads(state.traderData) if state.traderData not in ("", None) else {}
        old_trader_data = raw if isinstance(raw, dict) else {}
        new_trader_data: dict[str, Any] = {}

        for symbol, strategy in self.strategies.items():
            if isinstance(strategy, StatefulStrategy) and symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            strategy_orders, strategy_conversions = strategy.run(state)
            for order in strategy_orders:
                orders.setdefault(order.symbol, []).append(order)
            conversions += strategy_conversions

            if isinstance(strategy, StatefulStrategy):
                new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data


logger = Logger()
