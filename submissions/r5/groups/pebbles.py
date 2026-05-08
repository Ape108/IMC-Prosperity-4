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


class R5XLSkewMMStrategy(R5BaseMMStrategy, StatefulStrategy[dict[str, Any]]):
    """
    Asymmetric one-sided skew based on partner (XL) recent return.

    When |partner_return| <= threshold: standard symmetric quoting at width.
    When partner_return < -threshold (partner dropped, expect own to rise):
        bid raised by k_ticks above standard anchor; ask unchanged at standard.
    When partner_return > +threshold (partner rose, expect own to drop):
        ask lowered by k_ticks below standard anchor; bid unchanged at standard.

    Distinct from R5CorrMMStrategy (which moves both quotes together) — this
    commits to taking inventory in the predicted direction.
    """

    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        width: int,
        partner_symbol: Symbol,
        threshold: float,
        k_ticks: int,
    ) -> None:
        super().__init__(symbol, limit, width)
        self.partner_symbol = partner_symbol
        self.threshold = threshold
        self.k_ticks = k_ticks
        self.last_xl_mid: float | None = None

    def _partner_return(self, state: TradingState) -> float | None:
        partner_depth = state.order_depths.get(self.partner_symbol)
        if partner_depth is None or not partner_depth.buy_orders or not partner_depth.sell_orders:
            self.last_xl_mid = None
            return None
        partner_bid = max(partner_depth.buy_orders.keys())
        partner_ask = min(partner_depth.sell_orders.keys())
        partner_mid = (partner_bid + partner_ask) / 2
        if self.last_xl_mid is None:
            self.last_xl_mid = partner_mid
            return None
        ret = (partner_mid - self.last_xl_mid) / self.last_xl_mid
        self.last_xl_mid = partner_mid
        return ret

    def act(self, state: TradingState) -> None:
        true_value = self._microprice(state)
        partner_return = self._partner_return(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        max_buy_price = int(true_value) - 1 if true_value % 1 == 0 else floor(true_value)
        min_sell_price = int(true_value) + 1 if true_value % 1 == 0 else ceil(true_value)

        # Standard anchors (match R5BaseMMStrategy)
        passive_buy = max_buy_price - self.width + 1
        passive_sell = min_sell_price + self.width - 1

        # Asymmetric skew
        if partner_return is not None and abs(partner_return) > self.threshold:
            if partner_return < 0:
                passive_buy = passive_buy + self.k_ticks
            else:
                passive_sell = passive_sell - self.k_ticks

        # Take liquidity at or better than fair (standard MM behavior)
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0:
            self.buy(passive_buy, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0:
            self.sell(passive_sell, to_sell)

    def save(self) -> dict[str, Any]:
        return {"last_xl_mid": self.last_xl_mid}

    def load(self, data: dict[str, Any]) -> None:
        v = data.get("last_xl_mid")
        self.last_xl_mid = float(v) if v is not None else None


class R5XLTakerStrategy(StatefulStrategy[dict[str, Any]]):
    """
    Non-MM signal-gated taker. Reads partner (XL) recent return; on conviction
    moves crosses the spread on own symbol. Exits on signal flip or time stop.

    Distinct from MM-shaped strategies — pays the spread on entry and exit;
    only enters when |partner_return| > entry_threshold.
    """

    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        partner_symbol: Symbol,
        entry_threshold: float,
        exit_threshold: float,
        max_hold_ticks: int,
        k_clip: int,
    ) -> None:
        super().__init__(symbol, limit)
        self.partner_symbol = partner_symbol
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_hold_ticks = max_hold_ticks
        self.k_clip = k_clip
        self.last_xl_mid: float | None = None
        self.entry_tick: int | None = None
        self.tick: int = 0

    def _partner_return(self, state: TradingState) -> float | None:
        partner_depth = state.order_depths.get(self.partner_symbol)
        if partner_depth is None or not partner_depth.buy_orders or not partner_depth.sell_orders:
            self.last_xl_mid = None
            return None
        partner_bid = max(partner_depth.buy_orders.keys())
        partner_ask = min(partner_depth.sell_orders.keys())
        partner_mid = (partner_bid + partner_ask) / 2
        if self.last_xl_mid is None:
            self.last_xl_mid = partner_mid
            return None
        ret = (partner_mid - self.last_xl_mid) / self.last_xl_mid
        self.last_xl_mid = partner_mid
        return ret

    def act(self, state: TradingState) -> None:
        self.tick += 1
        partner_return = self._partner_return(state)
        position = state.position.get(self.symbol, 0)

        order_depth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        bid_vol = order_depth.buy_orders[best_bid]
        ask_vol = abs(order_depth.sell_orders[best_ask])

        # Time stop: flatten any held position (highest priority)
        if (
            position != 0
            and self.entry_tick is not None
            and (self.tick - self.entry_tick) > self.max_hold_ticks
        ):
            if position > 0:
                self.sell(best_bid, min(position, bid_vol))
            else:
                self.buy(best_ask, min(-position, ask_vol))
            self.entry_tick = None
            return

        if partner_return is None:
            return

        # Signal flip exit: position long & partner rising past exit_threshold,
        # or position short & partner falling past exit_threshold
        if position > 0 and partner_return > self.exit_threshold:
            self.sell(best_bid, min(position, bid_vol))
            self.entry_tick = None
            return
        if position < 0 and partner_return < -self.exit_threshold:
            self.buy(best_ask, min(-position, ask_vol))
            self.entry_tick = None
            return

        # Entry: only when flat and partner return above entry_threshold
        if position == 0 and abs(partner_return) > self.entry_threshold:
            if partner_return < 0:
                qty = min(self.k_clip, self.limit, ask_vol)
                if qty > 0:
                    self.buy(best_ask, qty)
                    self.entry_tick = self.tick
            else:
                qty = min(self.k_clip, self.limit, bid_vol)
                if qty > 0:
                    self.sell(best_bid, qty)
                    self.entry_tick = self.tick

    def save(self) -> dict[str, Any]:
        return {
            "last_xl_mid": self.last_xl_mid,
            "entry_tick": self.entry_tick,
            "tick": self.tick,
        }

    def load(self, data: dict[str, Any]) -> None:
        v = data.get("last_xl_mid")
        self.last_xl_mid = float(v) if v is not None else None
        v = data.get("entry_tick")
        self.entry_tick = int(v) if v is not None else None
        v = data.get("tick")
        self.tick = int(v) if v is not None else 0


class R5PairTradeStrategy(StatefulStrategy[dict[str, Any]]):
    """
    Z-score spread trade on a pair (symbol_a − symbol_b).

    Tracks rolling spread of size `window`. Once full, computes z = (spread − mean) / std.
    Entry at |z| > z_entry: positive z -> short A, long B at full limit. Negative z -> opposite.
    Exit at |z| < z_exit OR held_ticks > max_hold_ticks. Flatten both legs.

    Returns orders for both symbols. Trader.run dispatch must accumulate
    (orders.setdefault(symbol, []).append(order)).
    """

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
        self.tick: int = 0
        self.entry_tick: int | None = None

    def get_required_symbols(self) -> list[Symbol]:
        return [self.symbol_a, self.symbol_b]

    def _mid(self, state: TradingState, sym: Symbol) -> float:
        depth = state.order_depths[sym]
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2

    def _zscore(self, current: float) -> float | None:
        if len(self.spread_history) < self.window:
            return None
        mean = sum(self.spread_history) / len(self.spread_history)
        var = sum((x - mean) ** 2 for x in self.spread_history) / len(self.spread_history)
        std = var ** 0.5
        if std == 0:
            return None
        return (current - mean) / std

    def act(self, state: TradingState) -> None:
        self.tick += 1
        mid_a = self._mid(state, self.symbol_a)
        mid_b = self._mid(state, self.symbol_b)
        spread = mid_a - mid_b

        z = self._zscore(spread)

        # Maintain rolling history (FIFO)
        self.spread_history.append(spread)
        if len(self.spread_history) > self.window:
            self.spread_history.pop(0)

        pos_a = state.position.get(self.symbol_a, 0)
        pos_b = state.position.get(self.symbol_b, 0)
        depth_a = state.order_depths[self.symbol_a]
        depth_b = state.order_depths[self.symbol_b]
        best_bid_a = max(depth_a.buy_orders.keys())
        best_ask_a = min(depth_a.sell_orders.keys())
        best_bid_b = max(depth_b.buy_orders.keys())
        best_ask_b = min(depth_b.sell_orders.keys())

        held = pos_a != 0 or pos_b != 0

        # Exit conditions
        if held:
            time_stop = (
                self.entry_tick is not None
                and (self.tick - self.entry_tick) > self.max_hold_ticks
            )
            z_exit_hit = z is not None and abs(z) < self.z_exit

            if time_stop or z_exit_hit:
                if pos_a > 0:
                    self.orders.append(Order(self.symbol_a, best_bid_a, -pos_a))
                elif pos_a < 0:
                    self.orders.append(Order(self.symbol_a, best_ask_a, -pos_a))
                if pos_b > 0:
                    self.orders.append(Order(self.symbol_b, best_bid_b, -pos_b))
                elif pos_b < 0:
                    self.orders.append(Order(self.symbol_b, best_ask_b, -pos_b))
                self.entry_tick = None
            return

        # Entry conditions (only when flat)
        if z is None or abs(z) <= self.z_entry:
            return

        if z > 0:
            # Spread rich: short A, long B
            self.orders.append(Order(self.symbol_a, best_bid_a, -self.limit))
            self.orders.append(Order(self.symbol_b, best_ask_b, self.limit))
        else:
            # Spread cheap: long A, short B
            self.orders.append(Order(self.symbol_a, best_ask_a, self.limit))
            self.orders.append(Order(self.symbol_b, best_bid_b, -self.limit))
        self.entry_tick = self.tick

    def save(self) -> dict[str, Any]:
        return {
            "spread_history": list(self.spread_history),
            "tick": self.tick,
            "entry_tick": self.entry_tick,
        }

    def load(self, data: dict[str, Any]) -> None:
        v = data.get("spread_history")
        self.spread_history = [float(x) for x in v] if v else []
        v = data.get("tick")
        self.tick = int(v) if v is not None else 0
        v = data.get("entry_tick")
        self.entry_tick = int(v) if v is not None else None


SYMBOLS = [
    "PEBBLES_XS",
    "PEBBLES_S",
    "PEBBLES_M",
    "PEBBLES_L",
    "PEBBLES_XL",
]

LIMIT = 10


class Trader:
    def __init__(self) -> None:
        self.strategies: dict[Symbol, Strategy] = {}

        def mm_baseline():
            """PEBBLES_L                       -6871.00   14706.00  -10820.00   -2985.00
                PEBBLES_M                        3177.00  -14826.00  -13969.00  -25618.00
                PEBBLES_S                       16865.00    2148.00   21085.00   40098.00
                PEBBLES_XL                      -4781.00   10124.50    7121.00   12464.50
                PEBBLES_XS                      -3018.00   -6936.50    1693.00   -8261.50
                
                W/ Conservative
                PEBBLES_L                       -9886.00   11042.00  -14157.00  -13001.00
                PEBBLES_M                         114.00  -18489.00  -17402.00  -35777.00
                PEBBLES_S                       13836.00   -1284.00   18575.00   31127.00
                PEBBLES_XL                      -8361.00    5264.50    2484.00    -612.50
                PEBBLES_XS                      -5841.00   -9384.50    -181.00  -15406.50
            """
            for sym in SYMBOLS:
                self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=2)

        def xl_signal_skew():
            """ 
                PEBBLES_L                       -6228.00   14194.00  -11210.00   -3244.00
                PEBBLES_M                        2110.00  -15338.00  -14363.00  -27591.00
                PEBBLES_S                       17106.00    1698.00   19794.00   38598.00
                PEBBLES_XL                      -4781.00   10124.50    7121.00   12464.50
                PEBBLES_XS                      -4945.00    -412.50    3833.00   -1524.50
            """
            for sym in SYMBOLS:
                if sym == "PEBBLES_XL":
                    self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=2)
                else:
                    self.strategies[sym] = R5XLSkewMMStrategy(
                        symbol=sym, limit=LIMIT, width=2,
                        partner_symbol="PEBBLES_XL",
                        threshold=0.001, k_ticks=2,
                    )

        def xl_signal_taker():
            """PEBBLES_L                     -151063.00 -150106.00 -138220.00 -439389.00
                PEBBLES_M                     -142883.00 -137207.00 -147500.00 -427590.00
                PEBBLES_S                     -145557.00 -132774.00 -109084.00 -387415.00
                PEBBLES_XL                      -4781.00   10124.50    7121.00   12464.50
                PEBBLES_XS                    -145273.00  -94752.50  -93039.00 -333064.50
            """
            for sym in SYMBOLS:
                if sym == "PEBBLES_XL":
                    self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=2)
                else:
                    self.strategies[sym] = R5XLTakerStrategy(
                        symbol=sym, limit=LIMIT,
                        partner_symbol="PEBBLES_XL",
                        entry_threshold=0.0015,
                        exit_threshold=0.0005,
                        max_hold_ticks=50,
                        k_clip=5,
                    )

        def pair_trade_zscore():
            """PEBBLES_M                        4198.00    4618.00   10961.00   19777.00
                PEBBLES_XL                       5222.00   41125.00   53980.00  100327.00
                
                w/ queue penetration 0:
                PEBBLES_M                       -2512.00   -2292.00    2745.00   -2059.00
                PEBBLES_XL                      -2582.00   31885.00   42660.00   71963.00
            """
            self.strategies["PEBBLES_M"] = R5PairTradeStrategy(
                symbol_a="PEBBLES_M",
                symbol_b="PEBBLES_XL",
                limit=LIMIT,
                window=200,
                z_entry=2.0,
                z_exit=0.5,
                max_hold_ticks=500,
            )
            # PEBBLES_L, PEBBLES_S, PEBBLES_XS, PEBBLES_XL: no orders under this variant.

        mm_baseline()

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
