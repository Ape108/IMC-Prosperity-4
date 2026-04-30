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


class R5CorrMMStrategy(R5BaseMMStrategy, StatefulStrategy[dict[str, Any]]):
    def __init__(self, symbol: Symbol, limit: int, width: int, partner_symbol: Symbol, beta: float) -> None:
        super().__init__(symbol, limit, width)
        self.partner_symbol = partner_symbol
        self.beta = beta
        self.last_partner_mid: float | None = None

    def get_true_value(self, state: TradingState) -> float:
        base = self._microprice(state)
        partner_depth = state.order_depths.get(self.partner_symbol)
        if partner_depth is None or not partner_depth.buy_orders or not partner_depth.sell_orders:
            self.last_partner_mid = None
            return base

        partner_bid = max(partner_depth.buy_orders.keys())
        partner_ask = min(partner_depth.sell_orders.keys())
        partner_mid = (partner_bid + partner_ask) / 2

        if self.last_partner_mid is not None:
            partner_return = (partner_mid - self.last_partner_mid) / self.last_partner_mid
            base -= self.beta * partner_return * base
        self.last_partner_mid = partner_mid
        return base

    def save(self) -> dict[str, Any]:
        return {"last_partner_mid": self.last_partner_mid}

    def load(self, data: dict[str, Any]) -> None:
        v = data.get("last_partner_mid")
        self.last_partner_mid = float(v) if v is not None else None


class R5TickResidualMMStrategy(R5BaseMMStrategy, StatefulStrategy[dict[str, Any]]):
    # Adjusts fair value by the within-tick unexplained residual:
    #   residual = beta * partner_return_this_tick - own_return_this_tick
    # Captures the gap when partner's book has moved but own book hasn't fully absorbed it yet.
    def __init__(self, symbol: Symbol, limit: int, width: int, partner_symbol: Symbol, beta: float) -> None:
        super().__init__(symbol, limit, width)
        self.partner_symbol = partner_symbol
        self.beta = beta
        self.last_partner_mid: float | None = None
        self.last_own_mid: float | None = None

    def get_true_value(self, state: TradingState) -> float:
        base = self._microprice(state)

        partner_depth = state.order_depths.get(self.partner_symbol)
        if partner_depth is None or not partner_depth.buy_orders or not partner_depth.sell_orders:
            self.last_partner_mid = None
            self.last_own_mid = None
            return base

        partner_bid = max(partner_depth.buy_orders.keys())
        partner_ask = min(partner_depth.sell_orders.keys())
        partner_mid = (partner_bid + partner_ask) / 2

        own_depth = state.order_depths[self.symbol]
        own_bid = max(own_depth.buy_orders.keys())
        own_ask = min(own_depth.sell_orders.keys())
        own_mid = (own_bid + own_ask) / 2

        if self.last_partner_mid is not None and self.last_own_mid is not None:
            partner_return = (partner_mid - self.last_partner_mid) / self.last_partner_mid
            own_return = (own_mid - self.last_own_mid) / self.last_own_mid
            residual = self.beta * partner_return - own_return
            base += residual * base

        self.last_partner_mid = partner_mid
        self.last_own_mid = own_mid
        return base

    def save(self) -> dict[str, Any]:
        return {"last_partner_mid": self.last_partner_mid, "last_own_mid": self.last_own_mid}

    def load(self, data: dict[str, Any]) -> None:
        v = data.get("last_partner_mid")
        self.last_partner_mid = float(v) if v is not None else None
        v = data.get("last_own_mid")
        self.last_own_mid = float(v) if v is not None else None


class R5BasketCapMMStrategy(R5BaseMMStrategy):
    """
    Basket-aware MM. Capacity skew driven by system-level factor exposure
    rather than per-leg position. Per-leg hard cap preserved.

    Phase 1 of B2 (basket-aware MM). Phase 2 will add price skew.
    """

    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        width: int,
        leg_sign: int,
        partners: dict[Symbol, int],
    ) -> None:
        super().__init__(symbol, limit, width)
        self.leg_sign = leg_sign
        self.partners = partners

    def get_required_symbols(self) -> list[Symbol]:
        return [self.symbol] + list(self.partners.keys())

    def act(self, state: TradingState) -> None:
        own_pos = state.position.get(self.symbol, 0)

        factor = self.leg_sign * own_pos
        for partner_symbol, partner_sign in self.partners.items():
            factor += partner_sign * state.position.get(partner_symbol, 0)

        system_size = 1 + len(self.partners)
        effective_pos = self.leg_sign * factor / system_size

        true_value = self.get_true_value(state)
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        to_buy = floor(self.limit - effective_pos)
        to_sell = floor(self.limit + effective_pos)
        # Per-leg hard cap (platform constraint):
        to_buy = max(0, min(to_buy, self.limit - own_pos))
        to_sell = max(0, min(to_sell, self.limit + own_pos))

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


SYMBOLS = [
    "SNACKPACK_CHOCOLATE",
    "SNACKPACK_VANILLA",
    "SNACKPACK_PISTACHIO",
    "SNACKPACK_STRAWBERRY",
    "SNACKPACK_RASPBERRY",
]

LIMIT = 10


class Trader:
    def __init__(self) -> None:
        self.strategies: dict[Symbol, Strategy] = {}

        # Corr overlays (lag-1 partner lean) — active for baseline comparison
        
        def corr():
            """SNACKPACK_CHOCOLATE             -6048.00    -787.00    5886.00    -949.00
                SNACKPACK_PISTACHIO            -11601.00  -10513.00  -15782.00  -37896.00
                SNACKPACK_RASPBERRY             -7061.00   -6431.00  -18146.00  -31638.00
                SNACKPACK_STRAWBERRY           -13596.00  -21576.00  -22633.00  -57805.00
                SNACKPACK_VANILLA               -8260.00     909.50  -13745.00  -21095.50
            """
            self.strategies["SNACKPACK_RASPBERRY"] = R5CorrMMStrategy(
                "SNACKPACK_RASPBERRY", LIMIT, width=3, partner_symbol="SNACKPACK_STRAWBERRY", beta=0.462,
            )
            self.strategies["SNACKPACK_STRAWBERRY"] = R5CorrMMStrategy(
                "SNACKPACK_STRAWBERRY", LIMIT, width=3, partner_symbol="SNACKPACK_RASPBERRY", beta=0.462,
            )
            self.strategies["SNACKPACK_CHOCOLATE"] = R5CorrMMStrategy(
                "SNACKPACK_CHOCOLATE", LIMIT, width=3, partner_symbol="SNACKPACK_VANILLA", beta=0.458,
            )
            self.strategies["SNACKPACK_VANILLA"] = R5CorrMMStrategy(
                "SNACKPACK_VANILLA", LIMIT, width=3, partner_symbol="SNACKPACK_CHOCOLATE", beta=0.458,
            )
            self.strategies["SNACKPACK_PISTACHIO"] = R5CorrMMStrategy(
                "SNACKPACK_PISTACHIO", LIMIT, width=3, partner_symbol="SNACKPACK_STRAWBERRY", beta=-0.457,
            )

        def mm():
            """SNACKPACK_CHOCOLATE              2177.00    6297.50    2907.00   11381.50
                SNACKPACK_PISTACHIO              1999.00    2239.00     315.00    4553.00
                SNACKPACK_RASPBERRY              1660.00    3409.50    2974.00    8043.50
                SNACKPACK_STRAWBERRY              121.00   -1800.00     582.00   -1097.00
                SNACKPACK_VANILLA               -1240.00   -2963.50     386.00   -3817.50
                
                w/ queue-penetration 0
                SNACKPACK_CHOCOLATE                 7.00    3402.50     577.00    3986.50
                SNACKPACK_PISTACHIO              -171.00    -656.00   -2015.00   -2842.00
                SNACKPACK_RASPBERRY              -510.00     514.50     644.00     648.50
                SNACKPACK_STRAWBERRY            -2049.00   -4761.00   -1944.00   -8754.00
                SNACKPACK_VANILLA               -3410.00   -5858.50   -1944.00  -11212.50
            """ 
            self.strategies["SNACKPACK_RASPBERRY"] = R5BaseMMStrategy("SNACKPACK_RASPBERRY", LIMIT, width=3)
            self.strategies["SNACKPACK_STRAWBERRY"] = R5BaseMMStrategy("SNACKPACK_STRAWBERRY", LIMIT, width=3)
            self.strategies["SNACKPACK_CHOCOLATE"] = R5BaseMMStrategy("SNACKPACK_CHOCOLATE", LIMIT, width=3)
            self.strategies["SNACKPACK_VANILLA"] = R5BaseMMStrategy("SNACKPACK_VANILLA", LIMIT, width=3)
            self.strategies["SNACKPACK_PISTACHIO"] = R5BaseMMStrategy("SNACKPACK_PISTACHIO", LIMIT, width=3)

        def tick_lean():
            """SNACKPACK_CHOCOLATE            -38598.00  -34968.00  -29685.00 -103251.00
                SNACKPACK_PISTACHIO              3756.00   -5544.00   -2406.00   -4194.00
                SNACKPACK_RASPBERRY            -67717.00  -79606.00  -78768.00 -226091.00
                SNACKPACK_STRAWBERRY           -62213.00  -49573.00  -45283.00 -157069.00
                SNACKPACK_VANILLA              -28511.00  -25201.00  -24455.00  -78167.00
            """
            # beta signs: negative for anti-correlated pairs (corr ~ -0.92), positive for co-moving (PIST/STRAW corr ~ +0.91)
            self.strategies["SNACKPACK_RASPBERRY"] = R5TickResidualMMStrategy(
                "SNACKPACK_RASPBERRY", LIMIT, width=3, partner_symbol="SNACKPACK_STRAWBERRY", beta=-0.462,
            )
            self.strategies["SNACKPACK_STRAWBERRY"] = R5TickResidualMMStrategy(
                "SNACKPACK_STRAWBERRY", LIMIT, width=3, partner_symbol="SNACKPACK_RASPBERRY", beta=-0.462,
            )
            self.strategies["SNACKPACK_CHOCOLATE"] = R5TickResidualMMStrategy(
                "SNACKPACK_CHOCOLATE", LIMIT, width=3, partner_symbol="SNACKPACK_VANILLA", beta=-0.458,
            )
            self.strategies["SNACKPACK_VANILLA"] = R5TickResidualMMStrategy(
                "SNACKPACK_VANILLA", LIMIT, width=3, partner_symbol="SNACKPACK_CHOCOLATE", beta=-0.458,
            )
            self.strategies["SNACKPACK_PISTACHIO"] = R5TickResidualMMStrategy(
                "SNACKPACK_PISTACHIO", LIMIT, width=3, partner_symbol="SNACKPACK_STRAWBERRY", beta=0.457,
            )

        def basket_cap():
            """SNACKPACK_CHOCOLATE              2177.00    6297.50    2907.00   11381.50
                SNACKPACK_PISTACHIO              1999.00    2239.00     315.00    4553.00
                SNACKPACK_RASPBERRY              1660.00    3409.50    2974.00    8043.50
                SNACKPACK_STRAWBERRY              121.00   -1800.00     582.00   -1097.00
                SNACKPACK_VANILLA               -1240.00   -2963.50     386.00   -3817.50
                
                w/ queue-penetration 0
                SSNACKPACK_CHOCOLATE                 7.00    3402.50     577.00    3986.50
                SNACKPACK_PISTACHIO              -171.00    -656.00   -2015.00   -2842.00
                SNACKPACK_RASPBERRY              -510.00     514.50     644.00     648.50
                SNACKPACK_STRAWBERRY            -2049.00   -4761.00   -1944.00   -8754.00
                SNACKPACK_VANILLA               -3410.00   -5858.50   -1944.00  -11212.50
            """
            self.strategies["SNACKPACK_CHOCOLATE"] = R5BasketCapMMStrategy(
                "SNACKPACK_CHOCOLATE", LIMIT, width=3,
                leg_sign=+1, partners={"SNACKPACK_VANILLA": -1},
            )
            self.strategies["SNACKPACK_VANILLA"] = R5BasketCapMMStrategy(
                "SNACKPACK_VANILLA", LIMIT, width=3,
                leg_sign=-1, partners={"SNACKPACK_CHOCOLATE": +1},
            )
            self.strategies["SNACKPACK_RASPBERRY"] = R5BasketCapMMStrategy(
                "SNACKPACK_RASPBERRY", LIMIT, width=3,
                leg_sign=-1, partners={"SNACKPACK_STRAWBERRY": +1, "SNACKPACK_PISTACHIO": +1},
            )
            self.strategies["SNACKPACK_STRAWBERRY"] = R5BasketCapMMStrategy(
                "SNACKPACK_STRAWBERRY", LIMIT, width=3,
                leg_sign=+1, partners={"SNACKPACK_RASPBERRY": -1, "SNACKPACK_PISTACHIO": +1},
            )
            self.strategies["SNACKPACK_PISTACHIO"] = R5BasketCapMMStrategy(
                "SNACKPACK_PISTACHIO", LIMIT, width=3,
                leg_sign=+1, partners={"SNACKPACK_RASPBERRY": -1, "SNACKPACK_STRAWBERRY": +1},
            )

        basket_cap()
            
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
            orders[symbol] = strategy_orders
            conversions += strategy_conversions

            if isinstance(strategy, StatefulStrategy):
                new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data


logger = Logger()
