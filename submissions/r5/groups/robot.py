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


class R5AutocorrMMStrategy(R5BaseMMStrategy, StatefulStrategy[dict[str, Any]]):
    def __init__(self, symbol: Symbol, limit: int, width: int, alpha: float) -> None:
        super().__init__(symbol, limit, width)
        self.alpha = alpha
        self.last_mid: float | None = None

    def get_true_value(self, state: TradingState) -> float:
        base = self._microprice(state)
        order_depth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        current_mid = (best_bid + best_ask) / 2

        if self.last_mid is not None:
            last_return = (current_mid - self.last_mid) / self.last_mid
            base -= self.alpha * last_return * base
        self.last_mid = current_mid
        return base

    def save(self) -> dict[str, Any]:
        return {"last_mid": self.last_mid}

    def load(self, data: dict[str, Any]) -> None:
        v = data.get("last_mid")
        self.last_mid = float(v) if v is not None else None


class R5PairTradeStrategy(StatefulStrategy[dict[str, Any]]):
    def __init__(
        self,
        symbol_a: Symbol,
        symbol_b: Symbol,
        limit: int,
        window: int,
        z_entry: float,
        z_exit: float,
        max_hold_ticks: int,
    ) -> None:
        super().__init__(symbol_a, limit)
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.window = window
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.max_hold_ticks = max_hold_ticks
        self.spread_history: list[float] = []
        self.entry_tick: int | None = None

    def get_required_symbols(self) -> list[Symbol]:
        return [self.symbol_a, self.symbol_b]

    def _mid(self, state: TradingState, symbol: Symbol) -> float:
        od = state.order_depths[symbol]
        return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2

    def _take(self, state: TradingState, symbol: Symbol, side: int, qty: int) -> None:
        """Cross the spread. side = +1 buy at ask, -1 sell at bid."""
        od = state.order_depths[symbol]
        if side > 0:
            price = min(od.sell_orders.keys())
            self.orders.append(Order(symbol, price, qty))
        else:
            price = max(od.buy_orders.keys())
            self.orders.append(Order(symbol, price, -qty))

    def _flatten(self, state: TradingState, symbol: Symbol, position: int) -> None:
        """Reverse an existing position: sell at bid if long, buy at ask if short."""
        if position > 0:
            self._take(state, symbol, side=-1, qty=position)
        elif position < 0:
            self._take(state, symbol, side=+1, qty=-position)

    def act(self, state: TradingState) -> None:
        mid_a = self._mid(state, self.symbol_a)
        mid_b = self._mid(state, self.symbol_b)
        spread = mid_a - mid_b

        self.spread_history.append(spread)
        if len(self.spread_history) > self.window:
            self.spread_history.pop(0)

        pos_a = state.position.get(self.symbol_a, 0)
        pos_b = state.position.get(self.symbol_b, 0)
        is_flat = pos_a == 0 and pos_b == 0

        # entry_tick is live iff we hold the pair. Reconcile at top of tick so
        # partial-fill exits don't strand the timer — residual position keeps it,
        # and a confirmed-flat tick clears it.
        if is_flat:
            self.entry_tick = None
        elif self.entry_tick is None:
            # Cold-start with a carried position (e.g. --carry without --persist).
            # Best we can do is start the hold timer now.
            self.entry_tick = state.timestamp

        # Time-stop is a risk-management override: fires regardless of z computability
        if not is_flat:
            held_delta = state.timestamp - self.entry_tick
            if held_delta > self.max_hold_ticks * 100:
                self._flatten(state, self.symbol_a, pos_a)
                self._flatten(state, self.symbol_b, pos_b)
                return

        if len(self.spread_history) < self.window:
            return  # warming up

        mean = sum(self.spread_history) / len(self.spread_history)
        var = sum((s - mean) ** 2 for s in self.spread_history) / len(self.spread_history)
        std = var ** 0.5
        if std == 0:
            return  # no signal — flat spread

        z = (spread - mean) / std

        if is_flat:
            if z > self.z_entry:
                self._take(state, self.symbol_a, side=-1, qty=self.limit)
                self._take(state, self.symbol_b, side=+1, qty=self.limit)
                self.entry_tick = state.timestamp
            elif z < -self.z_entry:
                self._take(state, self.symbol_a, side=+1, qty=self.limit)
                self._take(state, self.symbol_b, side=-1, qty=self.limit)
                self.entry_tick = state.timestamp
        else:
            if abs(z) < self.z_exit:
                self._flatten(state, self.symbol_a, pos_a)
                self._flatten(state, self.symbol_b, pos_b)

    def save(self) -> dict[str, Any]:
        return {"spread_history": list(self.spread_history), "entry_tick": self.entry_tick}

    def load(self, data: dict[str, Any]) -> None:
        self.spread_history = list(data.get("spread_history", []))
        v = data.get("entry_tick")
        self.entry_tick = int(v) if v is not None else None


SYMBOLS = [
    "ROBOT_VACUUMING",
    "ROBOT_MOPPING",
    "ROBOT_DISHES",
    "ROBOT_LAUNDRY",
    "ROBOT_IRONING",
]

LIMIT = 10


class Trader:
    def __init__(self) -> None:
        self.strategies: dict[Symbol, Strategy] = {}

        def mm_baseline():
            """ROBOT_DISHES                     6888.00    9049.50   -3505.00   12432.50
                ROBOT_IRONING                   13644.00   -1680.00    1005.00   12969.00
                ROBOT_LAUNDRY                   -3036.00     579.50    4963.50    2507.00
                ROBOT_MOPPING                  -12877.00  -12006.00    3547.50  -21335.50
                ROBOT_VACUUMING                 -4178.00    -124.00    -259.00   -4561.00
                
                w/ queue penetration 0:
                ROBOT_DISHES                     4686.00    6114.50   -5940.00    4860.50
                ROBOT_IRONING                   11564.00   -4347.00    -871.00    6346.00
                ROBOT_LAUNDRY                   -5266.00   -2355.50    2799.50   -4822.00
                ROBOT_MOPPING                  -15114.00  -15226.00     721.50  -29618.50
                ROBOT_VACUUMING                 -6408.00   -2740.00   -2143.00  -11291.00
            """
            for sym in SYMBOLS:
                self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=1)

        def autocorr_dishes():
            """ROBOT_DISHES                   -30059.00  -33662.00  382933.00  319212.00
                ROBOT_IRONING                   30399.00   -6836.00   -3889.00   19674.00
                ROBOT_LAUNDRY                   -3036.00     579.50    4963.50    2507.00
                ROBOT_MOPPING                  -12877.00  -12006.00    3547.50  -21335.50
                ROBOT_VACUUMING                 -4178.00    -124.00    -259.00   -4561.00
                
                w/ queue penetration 0:
                
                ROBOT_DISHES                   -75159.00  -78052.00  322490.00  169279.00
                ROBOT_IRONING                   21763.00  -12112.00   -7149.00    2502.00
                ROBOT_LAUNDRY                   -5266.00   -2355.50    2799.50   -4822.00
                ROBOT_MOPPING                  -15114.00  -15226.00     721.50  -29618.50
                ROBOT_VACUUMING                 -6408.00   -2740.00   -2143.00  -11291.00
            """
            for sym in SYMBOLS:
                if sym == "ROBOT_DISHES":
                    # α = combined-day lag-1 ACF from EDA (eda_triage_summary.md L113: DISHES = -0.222)
                    self.strategies[sym] = R5AutocorrMMStrategy(sym, LIMIT, width=1, alpha=0.222)
                elif sym == "ROBOT_IRONING":
                    # α = combined-day lag-1 ACF from EDA (eda_triage_summary.md L114: IRONING = -0.121)
                    self.strategies[sym] = R5AutocorrMMStrategy(sym, LIMIT, width=1, alpha=0.121)
                else:
                    self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=1)

        def pair_trade_laundry_vacuuming():
            """
                ROBOT_LAUNDRY                    4503.00   -7449.00   -6712.00   -9658.00
            """
            for sym in SYMBOLS:
                if sym in ("ROBOT_LAUNDRY", "ROBOT_VACUUMING"):
                    continue  # registered once below as a single pair-trade strategy;
                              # VACUUMING orders are emitted by LAUNDRY's R5PairTradeStrategy via
                              # Trader.run's setdefault().append() — no separate key needed
                self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=1)
            self.strategies["ROBOT_LAUNDRY"] = R5PairTradeStrategy(
                symbol_a="ROBOT_LAUNDRY",
                symbol_b="ROBOT_VACUUMING",
                limit=LIMIT,
                window=200,
                z_entry=2.0,
                z_exit=0.5,
                max_hold_ticks=500,
            )

        def width_tier_2():
            """ROBOT_DISHES                     7150.00    7866.50   -5421.00    9595.50
                ROBOT_IRONING                   14348.00   -3158.00    1154.00   12344.00
                ROBOT_LAUNDRY                   -1898.00      65.50   10148.50    8316.00
                ROBOT_MOPPING                  -12046.00  -10604.00    4247.00  -18403.00
                ROBOT_VACUUMING                 -5170.00    1391.00    1869.00   -1910.00
                
                w/ queue penetration 0:
                ROBOT_DISHES                     5219.00    5031.50   -7799.00    2451.50
                ROBOT_IRONING                   12298.00   -5761.00    -694.00    5843.00
                ROBOT_LAUNDRY                   -3933.00   -2779.50    8351.50    1639.00
                ROBOT_MOPPING                  -14081.00  -13616.00    1559.00  -26138.00
                ROBOT_VACUUMING                 -7135.00    -811.00     193.00   -7753.00
            """
            for sym in SYMBOLS:
                self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=2)

        pair_trade_laundry_vacuuming()

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
