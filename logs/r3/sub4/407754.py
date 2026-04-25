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


# ── Black-Scholes math ───────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes call price with r=0."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def vega(S: float, K: float, T: float, sigma: float) -> float:
    """dC/dsigma with r=0."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi)
    return S * sqrt_T * pdf_d1


def implied_vol(S: float, K: float, T: float, market_price: float, max_iter: int = 100) -> float | None:
    """Newton-Raphson IV inversion. Returns None if price is below intrinsic or diverges."""
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
    """Fit quadratic IV = a*m^2 + b*m + c. Returns [a, b, c] or None if < 3 points."""
    if len(moneynesses) < 3:
        return None
    return np.polyfit(moneynesses, ivs, 2)


# ── Logger ───────────────────────────────────────────────────────────────────

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


# ── Strategy base classes ────────────────────────────────────────────────────

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
        
        # Sort asks (sell orders) lowest to highest, bids (buy orders) highest to lowest
        buy_orders = sorted(od.buy_orders.items(), reverse=True)
        sell_orders = sorted(od.sell_orders.items())
        
        position = state.position.get(self.symbol, 0)
        
        # --- THE FIX 1: Aggressive Quadratic Inventory Skew ---
        inventory_ratio = position / self.limit
        max_skew = 12.0  # Max price levels to adjust when at 100% capacity
        
        # Quadratic curve: tight at low inventory, aggressive at high inventory
        skew = math.copysign((abs(inventory_ratio) ** 2) * max_skew, inventory_ratio)
        skewed_true_value = true_value - skew
        
        to_buy = self.limit - position
        to_sell = self.limit + position
        
        # Calculate quoting prices using the skewed value
        max_buy_price = int(skewed_true_value) - 1 if skewed_true_value % 1 == 0 else floor(skewed_true_value)
        min_sell_price = int(skewed_true_value) + 1 if skewed_true_value % 1 == 0 else ceil(skewed_true_value)

        # 1. Take liquidity if available at favorable prices (Buy)
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                qty = min(to_buy, -volume)
                self.buy(price, qty)
                to_buy -= qty

        # 2. Provide liquidity for remaining buy limit
        if to_buy > 0:
            price = next((p + 1 for p, _ in buy_orders if p < max_buy_price), max_buy_price)
            self.buy(price, to_buy)

        # 3. Take liquidity if available at favorable prices (Sell)
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                qty = min(to_sell, volume)
                self.sell(price, qty)
                to_sell -= qty

        # 4. Provide liquidity for remaining sell limit
        if to_sell > 0:
            price = next((p - 1 for p, _ in sell_orders if p > min_sell_price), min_sell_price)
            self.sell(price, to_sell)

# ── Voucher constants ────────────────────────────────────────────────────────

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHER_SYMBOLS = [f"VEV_{k}" for k in STRIKES]
VEV_SPOT = "VELVETFRUIT_EXTRACT"
ROUND_START_TTE_DAYS = 5.0
TICKS_PER_DAY = 1_000_000


# ── Voucher IV smile scalper ─────────────────────────────────────────────────

class VoucherStrategy(Strategy):
    """
        Notes:
        -

        Results:
            Backtest:
            - 4000 strike has 0 pnl
            - 4500 strike has 0 pnl
            - 5000 strike has -114 to 162 pnl
            - 5100 strike has 6 to 49 pnl
            - 5200 strike has -106 to 1 pnl
            - 5300 strike has -563 to -13 pnl
            - 5400 strike has -3 to 0 pnl
            - 5500 strike has 0 pnl
            - 6000 strike has 0 pnl
            - 6500 strike has 0 pnl
        
    """
    def __init__(self, symbol: str, 
                 limit: int, strike: int, k: float = 300.0, min_residual: float = 0.01, max_otm_moneyness: float = 1.020) -> None:
        super().__init__(symbol, limit)
        self.strike = strike
        self.k = k
        self.min_residual = min_residual      # dead-band: ignore small residuals
        self.max_otm_moneyness = max_otm_moneyness  # skip far-OTM strikes (smile fit unreliable there)

    def get_required_symbols(self) -> list[Symbol]:
        return [VEV_SPOT] + VOUCHER_SYMBOLS

    def act(self, state: TradingState) -> None:
        spot = self.get_mid_price(state, VEV_SPOT)

        tte_days = ROUND_START_TTE_DAYS - state.timestamp / TICKS_PER_DAY
        tte_years = max(tte_days, 0.001) / 365.0

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
            return

        # Don't trade far-OTM strikes — smile fit is unreliable at the right wing
        if self.strike / spot > self.max_otm_moneyness:
            return

        od = state.order_depths[self.symbol]
        my_mid = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
        my_intrinsic = max(0.0, spot - self.strike)
        if my_mid <= my_intrinsic + 0.5:
            return
        my_iv = implied_vol(spot, float(self.strike), tte_years, my_mid)
        if my_iv is None:
            return
        fitted_iv = float(np.polyval(coeffs, self.strike / spot))
        residual = my_iv - fitted_iv

        # high IV residual → option overpriced → sell (negative target)
        # low IV residual → option underpriced → buy (positive target)
        if abs(residual) < self.min_residual:
            return
        position = state.position.get(self.symbol, 0)
        target = int(np.clip(-self.k * residual, -self.limit, self.limit))

        if target > position:
            qty_needed = target - position
            best_ask = min(od.sell_orders.keys())
            available = abs(od.sell_orders[best_ask])
            self.buy(best_ask, min(qty_needed, available))
        elif target < position:
            qty_needed = position - target
            best_bid = max(od.buy_orders.keys())
            available = od.buy_orders[best_bid]
            self.sell(best_bid, min(qty_needed, available))


# ── Delta-1 products ─────────────────────────────────────────────────────────

class HydrogelStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> float:
        od = state.order_depths.get(self.symbol)
        
        # Fallback to standard mid-price if the order book is unexpectedly empty
        if not od or not od.buy_orders or not od.sell_orders:
            return self.get_mid_price(state, self.symbol)
        
        buy_orders = sorted(od.buy_orders.items(), reverse=True)
        sell_orders = sorted(od.sell_orders.items())
        
        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]
        
        # --- THE FIX 2: Deep Book Volume Aggregation ---
        # Look at up to the top 3 levels to gauge true liquidity pressure
        depth = 3
        bid_vol = sum(vol for price, vol in buy_orders[:depth])
        ask_vol = sum(abs(vol) for price, vol in sell_orders[:depth])
        
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return (best_bid + best_ask) / 2.0
            
        # Cross-weight the top-of-book prices by the deep-book volume
        micro_price = (best_bid * ask_vol + best_ask * bid_vol) / total_vol
        
        return micro_price

class VelvetfruitStrategy(RollingZScoreStrategy):
    """
        Notes:
        -

        Results:
        - 0 pnl in backtest
    """
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit, zscore_period=75, smoothing_period=100, threshold=0.5)


# ── Trader ───────────────────────────────────────────────────────────────────

class Trader:
    def __init__(self) -> None:
        limits = {
            "HYDROGEL_PACK": 200,
            "VELVETFRUIT_EXTRACT": 200,
            "VEV_4000": 300,
            "VEV_4500": 300,
            "VEV_5000": 300,
            "VEV_5100": 300,
            "VEV_5200": 300,
            "VEV_5300": 300,
            "VEV_5400": 300,
            "VEV_5500": 300,
            "VEV_6000": 300,
            "VEV_6500": 300,
        }

        self.strategies: dict[Symbol, Strategy] = {
            "HYDROGEL_PACK": HydrogelStrategy("HYDROGEL_PACK", limits["HYDROGEL_PACK"]),
            "VELVETFRUIT_EXTRACT": HydrogelStrategy("VELVETFRUIT_EXTRACT", limits["VELVETFRUIT_EXTRACT"]),
            # Iterates over each voucher and creates a strategy for it with the appropriate strike
            **{
                f"VEV_{strike}": VoucherStrategy(f"VEV_{strike}", limits[f"VEV_{strike}"], strike)
                for strike in STRIKES
            },
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