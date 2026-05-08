# Unshipped R4 strategies: Mark14FollowerStrategy, Mark14InformedMMStrategy,
# VelvetfruitSignalStrategy, VelvetfruitEMAStrategy.
# Self-contained — includes base classes and a runnable Trader.

import json
import math
from abc import abstractmethod
from enum import IntEnum
from math import ceil, floor
from typing import Any

import numpy as np
import pandas as pd
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

type JSON = dict[str, Any] | list[Any] | str | int | float | bool | None


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi)
    return S * sqrt_T * pdf_d1


def implied_vol(S: float, K: float, T: float, market_price: float, max_iter: int = 100) -> float | None:
    intrinsic = max(0.0, S - K)
    if market_price <= intrinsic + 1e-6:
        return None
    sigma = 0.5
    for _ in range(max_iter):
        price = bs_call_price(S, K, T, sigma)
        diff = price - market_price
        if abs(diff) < 1e-6:
            return sigma
        v = vega(S, K, T, sigma)
        if v < 1e-10:
            return None
        sigma -= diff / v
        if sigma <= 0:
            sigma = 1e-6
    return sigma


def fit_iv_smile(moneynesses: list[float], ivs: list[float]) -> np.ndarray | None:
    if len(moneynesses) < 3:
        return None
    return np.polyfit(moneynesses, ivs, 2)


def autocorr_1lag(series: list[float]) -> float:
    if len(series) < 2:
        return 0.0
    r = np.corrcoef(series[:-1], series[1:])[0, 1]
    return float(r) if not math.isnan(r) else 0.0


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions, "", "",
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp, trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            [], [], state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {sym: [od.buy_orders, od.sell_orders] for sym, od in order_depths.items()}

    def compress_observations(self, observations: Observation) -> list[Any]:
        return [observations.plainValueObservations, {}]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

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
            if len(json.dumps(candidate)) <= max_length:
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
        self.orders: list[Order] = []
        self.conversions = 0
        if all(
            sym in state.order_depths
            and len(state.order_depths[sym].buy_orders) > 0
            and len(state.order_depths[sym].sell_orders) > 0
            for sym in self.get_required_symbols()
        ):
            self.act(state)
        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        od = state.order_depths[symbol]
        popular_buy = max(od.buy_orders.items(), key=lambda t: t[1])[0]
        popular_sell = min(od.sell_orders.items(), key=lambda t: t[1])[0]
        return (popular_buy + popular_sell) / 2


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
        od = state.order_depths[self.symbol]

        if self.signal == Signal.NEUTRAL:
            if position < 0:
                self.buy(min(od.sell_orders.keys()), -position)
            elif position > 0:
                self.sell(max(od.buy_orders.keys()), position)
        elif self.signal == Signal.SHORT:
            self.sell(max(od.buy_orders.keys()), self.limit + position)
        elif self.signal == Signal.LONG:
            self.buy(min(od.sell_orders.keys()), self.limit - position)

    def save(self) -> int:
        return self.signal.value

    def load(self, data: int) -> None:
        self.signal = Signal(data)


class RollingZScoreStrategy(SignalStrategy, StatefulStrategy[dict[str, Any]]):
    def __init__(self, symbol: Symbol, limit: int, zscore_period: int, smoothing_period: int, threshold: float) -> None:
        super().__init__(symbol, limit)
        self.zscore_period = zscore_period
        self.smoothing_period = smoothing_period
        self.threshold = threshold
        self.history: list[float] = []

    def get_signal(self, state: TradingState) -> Signal | None:
        self.history.append(self.get_mid_price(state, self.symbol))

        required = self.zscore_period + self.smoothing_period
        if len(self.history) < required:
            return None
        if len(self.history) > required:
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

    def save(self) -> dict[str, Any]:
        return {"signal": SignalStrategy.save(self), "history": self.history}

    def load(self, data: dict[str, Any]) -> None:
        SignalStrategy.load(self, data["signal"])
        self.history = data["history"]


class MarketMakingStrategy(Strategy):
    @abstractmethod
    def get_true_value(self, state: TradingState) -> float:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)
        od = state.order_depths[self.symbol]
        buy_orders = sorted(od.buy_orders.items(), reverse=True)
        sell_orders = sorted(od.sell_orders.items())
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position
        max_buy_price = int(true_value) - 1 if true_value % 1 == 0 else floor(true_value)
        min_sell_price = int(true_value) + 1 if true_value % 1 == 0 else ceil(true_value)

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                qty = min(to_buy, -volume)
                self.buy(price, qty)
                to_buy -= qty

        if to_buy > 0:
            price = next((p + 1 for p, _ in buy_orders if p < max_buy_price), max_buy_price)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                qty = min(to_sell, volume)
                self.sell(price, qty)
                to_sell -= qty

        if to_sell > 0:
            price = next((p - 1 for p, _ in sell_orders if p > min_sell_price), min_sell_price)
            self.sell(price, to_sell)


MAX_CLIP = 40

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHER_SYMBOLS = [f"VEV_{k}" for k in STRIKES]
VEV_SPOT = "VELVETFRUIT_EXTRACT"
ROUND_START_TTE_DAYS = 4.0
TICKS_PER_DAY = 1_000_000

MARK14_INFORMED_BOT = "Mark 14"
MARK14_SIGNAL_SYMBOL = "VEV_5300"
MARK14_WINDOW_TICKS = 500
MARK14_TARGET_SIZE = 100
MARK14_MM_QUOTE_SIZE = 10
MARK14_MM_MAX_CLIP = 15
MARK14_MM_INV_K = 5.0
MARK14_MM_DELTA = 1.0
MARK14_MM_W = 1.0


class VelvetfruitSignalStrategy(SignalStrategy, StatefulStrategy[dict[str, Any]]):
    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        target_position: int = 60,
        zscore_period: int = 75,
        smoothing_period: int = 100,
        threshold: float = 0.5,
    ) -> None:
        super().__init__(symbol, limit)
        self.target_position = min(target_position, limit)
        self.zscore_period = zscore_period
        self.smoothing_period = smoothing_period
        self.threshold = threshold
        self.history: list[float] = []

    def get_signal(self, state: TradingState) -> Signal | None:
        mid = self.get_mid_price(state, self.symbol)
        self.history.append(mid)

        required = self.zscore_period + self.smoothing_period
        if len(self.history) > required:
            self.history.pop(0)
        if len(self.history) < required:
            return None

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
        return Signal.NEUTRAL

    def act(self, state: TradingState) -> None:
        new_signal = self.get_signal(state)
        if new_signal is not None:
            self.signal = new_signal

        position = state.position.get(self.symbol, 0)
        od = state.order_depths[self.symbol]

        if self.signal == Signal.NEUTRAL:
            if position > 0:
                self.sell(max(od.buy_orders.keys()), position)
            elif position < 0:
                self.buy(min(od.sell_orders.keys()), -position)
        elif self.signal == Signal.LONG:
            target = self.target_position
            if position < target:
                self.buy(min(od.sell_orders.keys()), target - position)
        elif self.signal == Signal.SHORT:
            target = -self.target_position
            if position > target:
                self.sell(max(od.buy_orders.keys()), position - target)

    def save(self) -> dict[str, Any]:
        return {"signal": SignalStrategy.save(self), "history": self.history}

    def load(self, data: dict[str, Any]) -> None:
        SignalStrategy.load(self, data["signal"])
        self.history = data.get("history", [])


class VelvetfruitEMAStrategy(MarketMakingStrategy, StatefulStrategy[dict[str, float | None]]):
    FAST_WINDOW = 20
    SLOW_WINDOW = 200
    LEAN_K = 0.5

    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.ema_fast: float | None = None
        self.ema_slow: float | None = None

    def get_true_value(self, state: TradingState) -> float:
        mid = self.get_mid_price(state, self.symbol)

        if self.ema_fast is None:
            self.ema_fast = mid
            self.ema_slow = mid
        else:
            alpha_f = 2.0 / (self.FAST_WINDOW + 1)
            alpha_s = 2.0 / (self.SLOW_WINDOW + 1)
            self.ema_fast = alpha_f * mid + (1.0 - alpha_f) * self.ema_fast
            self.ema_slow = alpha_s * mid + (1.0 - alpha_s) * self.ema_slow

        divergence = self.ema_fast - self.ema_slow
        return mid + self.LEAN_K * divergence

    def save(self) -> dict[str, float | None]:
        return {"ema_fast": self.ema_fast, "ema_slow": self.ema_slow}

    def load(self, data: dict[str, float | None]) -> None:
        self.ema_fast = data.get("ema_fast")
        self.ema_slow = data.get("ema_slow")


class Mark14FollowerStrategy(StatefulStrategy[dict]):
    def __init__(
        self,
        symbol: str,
        limit: int,
        direction: str = "follow_passive",
        size: int = 75,
    ) -> None:
        super().__init__(symbol, limit)
        if direction not in ("follow_passive", "fade_at_touch"):
            raise ValueError(f"unknown direction: {direction}")
        self.direction = direction
        self.size = size
        self.last_signal_ts: int | None = None

    def save(self) -> dict:
        return {"last_signal_ts": self.last_signal_ts}

    def load(self, data: dict) -> None:
        self.last_signal_ts = data.get("last_signal_ts")

    def act(self, state: TradingState) -> None:
        if self.last_signal_ts is not None and state.timestamp < self.last_signal_ts:
            self.last_signal_ts = None

        trades = state.market_trades.get(self.symbol, [])
        mark14_buy_ts = [t.timestamp for t in trades if t.buyer == MARK14_INFORMED_BOT]
        if mark14_buy_ts:
            self.last_signal_ts = max(mark14_buy_ts)

        in_window = (
            self.last_signal_ts is not None
            and state.timestamp - self.last_signal_ts <= MARK14_WINDOW_TICKS
        )

        if self.direction == "follow_passive":
            target = self.size if in_window else 0
        else:
            target = -self.size if in_window else 0

        position = state.position.get(self.symbol, 0)
        delta = target - position
        delta = max(-self.limit - position, min(self.limit - position, delta))
        od = state.order_depths[self.symbol]

        if delta > 0:
            best_bid = max(od.buy_orders.keys())
            self.buy(best_bid, delta)
        elif delta < 0:
            best_ask = min(od.sell_orders.keys())
            self.sell(best_ask, -delta)


class Mark14InformedMMStrategy(StatefulStrategy[dict]):
    def __init__(
        self,
        symbol: str,
        limit: int,
        strike: int,
        predicted_dir: str,
        max_pos: int,
        quote_size: int = MARK14_MM_QUOTE_SIZE,
        max_clip: int = MARK14_MM_MAX_CLIP,
        inv_k: float = MARK14_MM_INV_K,
        delta: float = MARK14_MM_DELTA,
        base_w: float = MARK14_MM_W,
        window_ticks: int = MARK14_WINDOW_TICKS,
    ) -> None:
        super().__init__(symbol, limit)
        if predicted_dir not in ("UP", "DOWN"):
            raise ValueError(f"unknown predicted_dir: {predicted_dir}")
        self.strike = strike
        self.predicted_dir = predicted_dir
        self.max_pos = max_pos
        self.quote_size = quote_size
        self.max_clip = max_clip
        self.inv_k = inv_k
        self.delta = delta
        self.base_w = base_w
        self.window_ticks = window_ticks
        self.last_signal_ts: int | None = None

    def get_required_symbols(self) -> list[Symbol]:
        return [VEV_SPOT] + VOUCHER_SYMBOLS

    def save(self) -> dict:
        return {"last_signal_ts": self.last_signal_ts}

    def load(self, data: dict) -> None:
        self.last_signal_ts = data.get("last_signal_ts")

    def act(self, state: TradingState) -> None:
        if self.last_signal_ts is not None and state.timestamp < self.last_signal_ts:
            self.last_signal_ts = None

        trades = state.market_trades.get(self.symbol, [])
        mark14_buy_ts = [t.timestamp for t in trades if t.buyer == MARK14_INFORMED_BOT]
        if mark14_buy_ts:
            self.last_signal_ts = max(mark14_buy_ts)

        in_window = (
            self.last_signal_ts is not None
            and state.timestamp - self.last_signal_ts <= self.window_ticks
        )

        fair = self._compute_fair(state)
        if fair is None:
            return

        position = state.position.get(self.symbol, 0)
        inv_skew = self._compute_inv_skew(position)
        fair_used = fair + inv_skew

        bid_offset, ask_offset = self._compute_offsets(in_window)

        od = state.order_depths[self.symbol]
        bid_price, ask_price = self._compute_quote_prices(od, fair_used, bid_offset, ask_offset)
        bid_qty, ask_qty = self._compute_quote_sizes(position)

        if bid_qty > 0:
            self.buy(bid_price, bid_qty)
        if ask_qty > 0:
            self.sell(ask_price, ask_qty)

    def _compute_offsets(self, in_window: bool) -> tuple[float, float]:
        if not in_window:
            return self.base_w, self.base_w
        if self.predicted_dir == "UP":
            return self.base_w - self.delta, self.base_w + self.delta
        return self.base_w + self.delta, self.base_w - self.delta

    def _compute_inv_skew(self, position: int) -> float:
        return -(position / self.limit) * self.inv_k

    def _compute_microprice(self, od: OrderDepth) -> float:
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_vol = od.buy_orders[best_bid]
        ask_vol = abs(od.sell_orders[best_ask])
        total_vol = bid_vol + ask_vol
        if total_vol > 0:
            return (best_bid * ask_vol + best_ask * bid_vol) / total_vol
        return (best_bid + best_ask) / 2.0

    def _compute_smile_theo(self, state: TradingState, spot: float, tte_years: float) -> float | None:
        moneynesses: list[float] = []
        ivs: list[float] = []
        for s, sym in zip(STRIKES, VOUCHER_SYMBOLS):
            od = state.order_depths[sym]
            mid = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
            intrinsic = max(0.0, spot - s)
            if mid <= intrinsic + 0.5:
                continue
            iv = implied_vol(spot, float(s), tte_years, mid)
            if iv is None:
                continue
            moneynesses.append(s / spot)
            ivs.append(iv)
        coeffs = fit_iv_smile(moneynesses, ivs)
        if coeffs is None:
            return None
        fitted_iv = float(np.polyval(coeffs, self.strike / spot))
        return bs_call_price(spot, float(self.strike), tte_years, fitted_iv)

    def _compute_fair(self, state: TradingState) -> float | None:
        od = state.order_depths.get(self.symbol)
        if od is None or not od.buy_orders or not od.sell_orders:
            return None
        microprice = self._compute_microprice(od)
        spot = (max(state.order_depths[VEV_SPOT].buy_orders.keys())
                + min(state.order_depths[VEV_SPOT].sell_orders.keys())) / 2.0
        tte_days = ROUND_START_TTE_DAYS - state.timestamp / TICKS_PER_DAY
        tte_years = max(tte_days, 0.001) / 365.0
        smile_theo = self._compute_smile_theo(state, spot, tte_years)
        if smile_theo is None:
            return microprice
        return 0.85 * microprice + 0.15 * smile_theo

    def _compute_quote_prices(
        self, od: OrderDepth, fair_used: float, bid_offset: float, ask_offset: float,
    ) -> tuple[int, int]:
        bid_price = floor(fair_used - bid_offset)
        ask_price = ceil(fair_used + ask_offset)
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_price = min(bid_price, best_ask - 1)
        ask_price = max(ask_price, best_bid + 1)
        return bid_price, ask_price

    def _compute_quote_sizes(self, position: int) -> tuple[int, int]:
        room_buy = self.max_pos - position
        room_sell = self.max_pos + position
        bid_qty = max(0, min(self.quote_size, self.max_clip, room_buy))
        ask_qty = max(0, min(self.quote_size, self.max_clip, room_sell))
        return bid_qty, ask_qty


class Trader:
    def __init__(self) -> None:
        limits = {
            "VELVETFRUIT_EXTRACT": 200,
            "VEV_5300": 300,
            "VEV_5400": 300,
            "VEV_5500": 300,
        }

        self.strategies: dict[Symbol, Strategy] = {
            "VELVETFRUIT_EXTRACT": VelvetfruitEMAStrategy(
                "VELVETFRUIT_EXTRACT", limits["VELVETFRUIT_EXTRACT"],
            ),
            "VEV_5300": Mark14FollowerStrategy(
                "VEV_5300", limits["VEV_5300"], direction="follow_passive", size=50,
            ),
            "VEV_5400": Mark14FollowerStrategy(
                "VEV_5400", limits["VEV_5400"], direction="fade_at_touch", size=60,
            ),
            "VEV_5500": Mark14FollowerStrategy(
                "VEV_5500", limits["VEV_5500"], direction="fade_at_touch", size=25,
            ),
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders: dict[Symbol, list[Order]] = {}
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData not in ("", None) else {}
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
