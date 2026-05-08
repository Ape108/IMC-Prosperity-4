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
        - Stable backtest performance with 16k-17k pnl across 3 days
        - Maintains backtest performance with pesimistic fill assumptions (queue-penetration = 0) with 3.2k-3.6k pnl across 3 days
        - IMC performance of 2,837.563 pnl - seems to be peak performance and upper percentile
    """
    def get_true_value(self, state: TradingState) -> float:
        expected_true_value = 10_000
        max_delta = 5
        mid_price = self.get_mid_price(state, self.symbol)
        if (expected_true_value - max_delta) <= mid_price <= (expected_true_value + max_delta):
            return expected_true_value
        return mid_price
        

class PepperRootStrategy(MarketMakingStrategy):
    """ 
        Notes:
        - logs in @logs/r1/sub5
        
        Results:
        - Stableish backtest performance with 16k-28k pnl across 3 days
        - Falls off considerably with pesimistic fill assumptions (queue-penetration = 0) with -1.5k to 16k pnl across 3 days
        - IMC performance of 1,331.297 pnl - very low percentile performance
    """
    def get_true_value(self, state: TradingState) -> float:
        mid_price = self.get_mid_price(state, self.symbol)
        inventory = state.position.get(self.symbol, 0)
        
        # sigma: per-tick mid-price volatility from round1 data (two-sided book ticks only)
        # gamma: calibrated so max inventory (80) at midday gives ~5 tick skew
        #        adj = q * gamma * sigma^2 * ticks_remaining
        #        80 * 4e-6 * 1.75^2 * 5000 ≈ 5.0 ticks at full position midday
        gamma = 4e-6
        sigma = 1.75  # per-tick std dev for INTARIAN_PEPPER_ROOT

        # Ticks remaining — consistent time unit with per-tick sigma
        # Each timestamp step = 100 units; day runs 0 → 999900 (9999 ticks)
        ticks_remaining = max(0.0, (999900 - state.timestamp) / 100)

        reservation_price = mid_price - (inventory * gamma * (sigma**2) * ticks_remaining)
        return reservation_price


class PepperRootMMStrategy(MarketMakingStrategy):
    """
        Notes:

        Results:
        - Not very stable backtest performance with 7.2k - 60k pnl across 3 days
        - Pesimistic fill assumptions (queue-penetration = 0) has significant negative impact with -3.2k - 42k pnl across 3 days
    """
    def get_true_value(self, state: TradingState) -> float:
        return self.get_mid_price(state, self.symbol)


class PepperRootAdaptiveMMStrategy(StatefulStrategy[list[float]]):
    """
    Regime-adaptive market-making for INTARIAN_PEPPER_ROOT.

    Fair value: EW-smoothed blend of microprice (70%) + volume-wall mid (30%).
    Quotes are shifted each tick by three signals:
      - Top-of-book imbalance  (+lean with order flow)
      - Short-term momentum    (+lean with trend)
      - Inventory skew         (-lean against overexposure)

    Momentum guards suppress passive bid/ask posting when inventory is large
    and price is moving against that inventory. This prevents catching
    falling knives in sustained bear frames — the failure mode identified
    in the AS strategy with a hardcoded bullish drift.

    Compared to PepperRootBuyHoldStrategy:
      - Lower ceiling in a confirmed bull run (doesn't go max-long immediately)
      - Much better floor in bear/choppy regimes (inventory self-corrects)

    Results:
      - Violatile backtest performance with 14.6k - 73k pnl across 3 days
      - Pessimistic fill assumptions (queue-penetration = 0) has significant negative impact with -452 - 65k pnl across 3 days
    """

    WINDOW = 20           # EW history length for fair value smoothing
    TAKE_SIZE = 6         # Max units per aggressive take
    PASSIVE_SIZE = 8      # Base passive quote size
    TAKE_EDGE = 0.75      # Min edge required to take aggressively
    INV_GUARD_THRESH = 35 # Inventory level that activates momentum guards

    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.fair_history: list[float] = []

    def _best_bid_ask(self, od: OrderDepth) -> tuple[int | None, int | None]:
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return best_bid, best_ask

    def _microprice(self, od: OrderDepth, best_bid: int, best_ask: int) -> float:
        bid_vol = od.buy_orders.get(best_bid, 0)
        ask_vol = -od.sell_orders.get(best_ask, 0)
        total = bid_vol + ask_vol
        if total == 0:
            return (best_bid + best_ask) / 2.0
        return (best_bid * ask_vol + best_ask * bid_vol) / total

    def _wall_mid(self, od: OrderDepth, volume_threshold: int = 10) -> float | None:
        """Mid-price anchored to the nearest large-volume levels on each side."""
        wall_bid = next(
            (p for p, v in sorted(od.buy_orders.items(), reverse=True) if v >= volume_threshold),
            None,
        )
        wall_ask = next(
            (p for p, v in sorted(od.sell_orders.items()) if -v >= volume_threshold),
            None,
        )
        if wall_bid is None or wall_ask is None:
            return None
        return (wall_bid + wall_ask) / 2.0

    def act(self, state: TradingState) -> None:
        od = state.order_depths[self.symbol]
        pos = state.position.get(self.symbol, 0)

        best_bid, best_ask = self._best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return

        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0

        wall_mid = self._wall_mid(od)
        micro = self._microprice(od, best_bid, best_ask)
        fair_input = 0.7 * micro + 0.3 * (wall_mid if wall_mid is not None else mid)

        self.fair_history.append(fair_input)
        if len(self.fair_history) > self.WINDOW:
            self.fair_history.pop(0)

        weights = list(range(1, len(self.fair_history) + 1))
        fair = sum(w * x for w, x in zip(weights, self.fair_history)) / sum(weights)

        momentum = 0.0
        if len(self.fair_history) >= 5:
            momentum = self.fair_history[-1] - self.fair_history[-5]

        bid_vol = od.buy_orders.get(best_bid, 0)
        ask_vol = -od.sell_orders.get(best_ask, 0)
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0.0

        inv_skew = 1.5 * (pos / self.limit)

        take_buy_thresh = fair - self.TAKE_EDGE
        take_sell_thresh = fair + self.TAKE_EDGE

        # Don't fade hard directional moves
        if momentum > 1.0:
            take_buy_thresh += 0.5
            take_sell_thresh += 1.0
        elif momentum < -1.0:
            take_buy_thresh -= 1.0
            take_sell_thresh -= 0.5

        to_buy = self.limit - pos
        to_sell = self.limit + pos

        # Aggressive takes — only when edge is confirmed and order flow isn't opposing
        if to_buy > 0 and best_ask <= take_buy_thresh and imbalance > -0.6:
            qty = min(to_buy, self.TAKE_SIZE)
            self.buy(best_ask, qty)
            pos += qty
            to_buy -= qty
            to_sell += qty

        if to_sell > 0 and best_bid >= take_sell_thresh and imbalance < 0.6:
            qty = min(to_sell, self.TAKE_SIZE)
            self.sell(best_bid, qty)
            pos -= qty
            to_sell -= qty
            to_buy += qty

        # Passive quoting — only meaningful when spread allows it
        if spread < 2:
            return

        signal_shift = 0.5 * imbalance + 0.3 * momentum - inv_skew

        bid_px = int(floor(min(best_bid + 1, fair - 1 + signal_shift)))
        ask_px = int(ceil(max(best_ask - 1, fair + 1 + signal_shift)))

        inventory_penalty = int(abs(pos) / 15)
        passive_size = max(2, self.PASSIVE_SIZE - inventory_penalty)

        # Momentum guards: stop adding to inventory that's already moving against us
        can_post_bid = not (pos > self.INV_GUARD_THRESH and momentum < 0)
        can_post_ask = not (pos < -self.INV_GUARD_THRESH and momentum > 0)

        buy_size = min(to_buy, passive_size)
        sell_size = min(to_sell, passive_size)

        if can_post_bid and buy_size > 0:
            self.buy(bid_px, buy_size)
        if can_post_ask and sell_size > 0:
            self.sell(ask_px, sell_size)

    def save(self) -> list[float]:
        return self.fair_history

    def load(self, data: list[float]) -> None:
        self.fair_history = data


class PepperRootBuyHoldStrategy(BuyHoldStrategy):
    """
        Notes:
        - Because pepper cosistently has a positive trend over every day, just max buy and hold lol
        - Big gamble on if the trend continues
        - logs in @logs/r1/sub9

        Results:
        - Incredibly high and stable backtest performance with 79.1k - 79.5k pnl across 3 days
        - Maintains same backtest performance with pesimistic fill assumptions (queue-penetration = 0) with 79.1k - 79.5k pnl across 3 days
        - IMC performance of 7,286 pnl - very high percentile performance
    """
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.daily_peak: float = 0.0
        self.stop_triggered: bool = False

    def act(self, state: TradingState) -> None:
        """
        Stop-loss: if mid price drops STOP_LOSS_TICKS below the daily peak we sell everything aggressively and stop re-buying for the rest of that day
        300 ticks ≈ 3.8 sigma over a 2,000-tick window (sigma=1.75/tick) - threshold for significant adverse price movement
        """  
        STOP_LOSS_TICKS = 300
        mid = self.get_mid_price(state, self.symbol)

        if mid > self.daily_peak:
            self.daily_peak = mid

        if self.daily_peak > 0 and mid < self.daily_peak - STOP_LOSS_TICKS:
            self.stop_triggered = True

        if self.stop_triggered:
            position = state.position.get(self.symbol, 0)
            if position > 0:
                order_depth = state.order_depths[self.symbol]
                buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
                to_sell = position
                for price, volume in buy_orders:
                    if to_sell <= 0:
                        break
                    qty = min(to_sell, volume)
                    self.sell(price, qty)
                    to_sell -= qty
            return

        # otherwise keep buy-holding
        super().act(state)


class PepperRootDynamicASStrategy(StatefulStrategy[list[float]]):
    """
    Avellaneda-Stoikov MM for INTARIAN_PEPPER_ROOT with data-driven drift.

    Replaces the hardcoded bullish_drift = 2.0 with a slope estimated via
    linear regression over recent mid-price observations. The directional lean
    is now derived from what the data actually shows, not assumed in advance:

      drift ≈ +0.1 ticks/tick  →  bullish regime, leans long
      drift ≈  0               →  flat/uncertain, inventory penalty dominates
      drift ≈ -0.1 ticks/tick  →  bearish regime, leans short

    At drift=0.1, the inventory penalty (0.3 ticks at pos=40) meaningfully
    opposes overexposure — unlike the hardcoded 2.0 which overwhelmed it.

    0-EV clearing is implemented as direct aggressive orders (not quote=0,
    which is a no-op for non-integer true_value in the base class).

    Results:
        - Stable backtest performance with 54k - 55k pnl across 3 days
        - Beats backtest performance with pesimistic fill assumptions (queue-penetration = 0) with 54k - 56k pnl across 3 days
        - IMC performance of 4,528.813 pnl - much better than prev MM
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
                "INTARIAN_PEPPER_ROOT": PepperRootAdaptiveMMStrategy,
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


