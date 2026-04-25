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
    # High-precision polynomial approximation (faster than math.erf)
    # Saves compute time in tight loops
    b1 =  0.319381530
    b2 = -0.356563782
    b3 =  1.781477937
    b4 = -1.821255978
    b5 =  1.330274429
    p  =  0.2316419
    c  =  0.39894228
    
    if x >= 0.0:
        t = 1.0 / (1.0 + p * x)
        return (1.0 - c * math.exp(-x * x / 2.0) * t *
                (t *(t *(t *(t * b5 + b4) + b3) + b2) + b1))
    else:
        t = 1.0 / (1.0 - p * x)
        return (c * math.exp(-x * x / 2.0) * t *
                (t *(t *(t *(t * b5 + b4) + b3) + b2) + b1))

def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def implied_vol(S: float, K: float, T: float, market_price: float) -> float | None:
    """Bounded Bisection IV inversion. Guaranteed convergence, won't explode on low vega."""
    intrinsic = max(0.0, S - K)
    if market_price <= intrinsic + 1e-6:
        return None

    # Bounds for Bisection
    low, high = 1e-4, 5.0 
    
    # Quick check if price is outside our realistic volatility bounds
    if bs_call_price(S, K, T, high) < market_price:
        return None 
        
    for _ in range(60): # 60 iterations of bisection is ~1e-18 precision
        mid = (low + high) / 2.0
        price = bs_call_price(S, K, T, mid)
        diff = price - market_price
        
        if abs(diff) < 1e-6:
            return mid
            
        if diff > 0:
            high = mid
        else:
            low = mid
            
    return (low + high) / 2.0


def vega(S: float, K: float, T: float, sigma: float) -> float:
    """dC/dsigma with r=0."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi)
    return S * sqrt_T * pdf_d1


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
        position = state.position.get(self.symbol, 0)
        
        buy_orders = sorted(od.buy_orders.items(), reverse=True)
        sell_orders = sorted(od.sell_orders.items())
        
        # Base spread edge we want to capture
        edge = 1.0 
        
        # Inventory risk management (Passive skew)
        inventory_ratio = position / self.limit
        
        # Asymmetric quote shifting: 
        # If long (ratio > 0), drop bid aggressively to stop buying, drop ask slightly to get lifted.
        bid_shift = edge + (abs(inventory_ratio) ** 1.5) * 6 * (1.5 if inventory_ratio > 0 else -0.5)
        ask_shift = edge + (abs(inventory_ratio) ** 1.5) * 6 * (1.5 if inventory_ratio < 0 else -0.5)
        
        my_bid = int(floor(true_value - bid_shift))
        my_ask = int(ceil(true_value + ask_shift))

        to_buy = self.limit - position
        to_sell = self.limit + position

        # Arbitrage / Mispricing check: Only TAKE liquidity if the market is fundamentally wrong
        for price, volume in sell_orders:
            if to_buy > 0 and price < true_value - 0.5: # Market ask is below our true value
                qty = min(to_buy, -volume)
                self.buy(price, qty)
                to_buy -= qty

        for price, volume in buy_orders:
            if to_sell > 0 and price > true_value + 0.5: # Market bid is above our true value
                qty = min(to_sell, volume)
                self.sell(price, qty)
                to_sell -= qty

        # Provide passive liquidity at our skewed prices
        if to_buy > 0:
            # Ensure we don't cross the book with our maker orders
            best_ask = sell_orders[0][0] if sell_orders else my_bid + 2
            safe_bid = min(my_bid, best_ask - 1)
            self.buy(safe_bid, to_buy)

        if to_sell > 0:
            best_bid = buy_orders[0][0] if buy_orders else my_ask - 2
            safe_ask = max(my_ask, best_bid + 1)
            self.sell(safe_ask, to_sell)

# ── Voucher constants ────────────────────────────────────────────────────────

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHER_SYMBOLS = [f"VEV_{k}" for k in STRIKES]
VEV_SPOT = "VELVETFRUIT_EXTRACT"
ROUND_START_TTE_DAYS = 5.0
TICKS_PER_DAY = 1_000_000


# ── Voucher IV smile scalper ─────────────────────────────────────────────────

class VoucherStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, strike: int, k: float = 300.0) -> None:
        super().__init__(symbol, limit)
        self.strike = strike
        self.k = k

    def get_required_symbols(self) -> list[Symbol]:
        return [VEV_SPOT] + VOUCHER_SYMBOLS

    def act(self, state: TradingState) -> None:
        spot = self.get_mid_price(state, VEV_SPOT)
        tte_years = max(ROUND_START_TTE_DAYS - state.timestamp / TICKS_PER_DAY, 0.001) / 365.0

        moneynesses: list[float] = []
        ivs: list[float] = []
        
        for s, sym in zip(STRIKES, VOUCHER_SYMBOLS):
            od = state.order_depths[sym]
            if not od.buy_orders or not od.sell_orders:
                continue
                
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid = (best_bid + best_ask) / 2.0
            
            iv = implied_vol(spot, float(s), tte_years, mid)
            if iv is not None:
                moneynesses.append(s / spot)
                ivs.append(iv)

        coeffs = fit_iv_smile(moneynesses, ivs) if len(moneynesses) >= 3 else None
        if coeffs is None:
            return

        od = state.order_depths[self.symbol]
        if not od.buy_orders or not od.sell_orders:
            return
            
        my_best_bid = max(od.buy_orders.keys())
        my_best_ask = min(od.sell_orders.keys())
        my_mid = (my_best_bid + my_best_ask) / 2.0
        
        my_iv = implied_vol(spot, float(self.strike), tte_years, my_mid)
        if my_iv is None:
            return

        # CLAMP THE WINGS: Prevent polynomial explosion for deep OTM/ITM
        min_m, max_m = min(moneynesses), max(moneynesses)
        target_moneyness = np.clip(self.strike / spot, min_m, max_m)
        
        fitted_iv = float(np.polyval(coeffs, target_moneyness))
        residual = my_iv - fitted_iv

        # Dynamic dead-band based on the spread width (wider spread = less confidence)
        spread_width = my_best_ask - my_best_bid
        dynamic_residual_threshold = max(0.01, spread_width * 0.002) 

        if abs(residual) < dynamic_residual_threshold:
            return
            
        position = state.position.get(self.symbol, 0)
        target = int(np.clip(-self.k * residual, -self.limit, self.limit))

        if target > position:
            qty_needed = target - position
            available = abs(od.sell_orders[my_best_ask])
            self.buy(my_best_ask, min(qty_needed, available))
        elif target < position:
            qty_needed = position - target
            available = od.buy_orders[my_best_bid]
            self.sell(my_best_bid, min(qty_needed, available))

# ── Delta-1 products ─────────────────────────────────────────────────────────

class HydrogelStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> float:
        od = state.order_depths.get(self.symbol)
        
        if not od or not od.buy_orders or not od.sell_orders:
            return self.get_mid_price(state, self.symbol)
        
        buy_orders = sorted(od.buy_orders.items(), reverse=True)
        sell_orders = sorted(od.sell_orders.items())
        
        depth = 3
        
        # Calculate VWAP for bids and asks across depth
        bid_vol = 0
        bid_vwap_sum = 0.0
        for price, vol in buy_orders[:depth]:
            bid_vol += vol
            bid_vwap_sum += price * vol
            
        ask_vol = 0
        ask_vwap_sum = 0.0
        for price, vol in sell_orders[:depth]:
            abs_vol = abs(vol)
            ask_vol += abs_vol
            ask_vwap_sum += price * abs_vol
            
        if bid_vol == 0 or ask_vol == 0:
            return (buy_orders[0][0] + sell_orders[0][0]) / 2.0
            
        bid_vwap = bid_vwap_sum / bid_vol
        ask_vwap = ask_vwap_sum / ask_vol
        
        # Cross-weight the VWAPs by volume to find the micro-price
        total_vol = bid_vol + ask_vol
        micro_price = (bid_vwap * ask_vol + ask_vwap * bid_vol) / total_vol
        
        return micro_price


class VelvetfruitStrategy(StatefulStrategy[dict[str, Any]]):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.prev_bid_vol = 0
        self.prev_ask_vol = 0
        self.circuit_breaker_ticks = 0

    def get_required_symbols(self) -> list[Symbol]:
        # We need Hydrogel for lead-lag and VEV_5000 for options-implied volatility
        return [self.symbol, "HYDROGEL_PACK", "VEV_5000"]

    def get_true_value(self, state: TradingState) -> float:
        od = state.order_depths.get(self.symbol)
        if not od or not od.buy_orders or not od.sell_orders:
            return self.get_mid_price(state, self.symbol)

        buy_orders = sorted(od.buy_orders.items(), reverse=True)
        sell_orders = sorted(od.sell_orders.items())
        base_mid = (buy_orders[0][0] + sell_orders[0][0]) / 2.0

        # --- ALPHA 2: Lead-Lag Cross-Asset Skewing ---
        # Hydrogel momentum is used as a leading indicator for Velvetfruit
        hydro_od = state.order_depths.get("HYDROGEL_PACK")
        if hydro_od and hydro_od.buy_orders and hydro_od.sell_orders:
            hydro_bid_vol = sum(hydro_od.buy_orders.values())
            hydro_ask_vol = sum(abs(v) for v in hydro_od.sell_orders.values())
            
            # Calculate Hydrogel Order Book Imbalance [0.0 to 1.0]
            total_hydro_vol = hydro_bid_vol + hydro_ask_vol
            if total_hydro_vol > 0:
                hydro_imbalance = hydro_bid_vol / total_hydro_vol
                # Shift Velvetfruit true value by up to +/- 1 price level based on Hydrogel pressure
                lead_lag_shift = (hydro_imbalance - 0.5) * 2.0 
                return base_mid + lead_lag_shift

        return base_mid

    def act(self, state: TradingState) -> None:
        od = state.order_depths[self.symbol]
        
        current_bid_vol = sum(od.buy_orders.values()) if od.buy_orders else 0
        current_ask_vol = sum(abs(v) for v in od.sell_orders.values()) if od.sell_orders else 0

        # --- ALPHA 3: Adverse Selection Circuit Breaker (OFI) ---
        # If bid volume collapses and ask volume spikes, toxic sell flow is hitting the book
        if self.prev_bid_vol > 0:
            bid_drop = current_bid_vol / self.prev_bid_vol
            ask_spike = current_ask_vol / (self.prev_ask_vol + 1)
            
            if bid_drop < 0.3 and ask_spike > 2.0:
                self.circuit_breaker_ticks = 3  # Halt passive buying for 3 ticks to avoid the knife

        self.prev_bid_vol = current_bid_vol
        self.prev_ask_vol = current_ask_vol

        circuit_breaker_active = False
        if self.circuit_breaker_ticks > 0:
            self.circuit_breaker_ticks -= 1
            circuit_breaker_active = True

        true_value = self.get_true_value(state)
        position = state.position.get(self.symbol, 0)
        inventory_ratio = position / self.limit

        # --- ALPHA 1: Options-Implied Spread Widening ---
        # Use the ATM VEV_5000 options spread as a proxy for market volatility/uncertainty
        vev_od = state.order_depths.get("VEV_5000")
        iv_premium = 0.0
        if vev_od and vev_od.buy_orders and vev_od.sell_orders:
            vev_spread = min(vev_od.sell_orders.keys()) - max(vev_od.buy_orders.keys())
            if vev_spread > 15:  # Market makers are widening options; we should widen delta-1
                iv_premium = 1.0 

        base_edge = 1.0 + iv_premium

        # Asymmetric Inventory Skew
        bid_shift = base_edge + (abs(inventory_ratio) ** 1.5) * 6 * (1.5 if inventory_ratio > 0 else -0.5)
        ask_shift = base_edge + (abs(inventory_ratio) ** 1.5) * 6 * (1.5 if inventory_ratio < 0 else -0.5)

        my_bid = int(floor(true_value - bid_shift))
        my_ask = int(ceil(true_value + ask_shift))

        to_buy = self.limit - position
        to_sell = self.limit + position

        buy_orders = sorted(od.buy_orders.items(), reverse=True)
        sell_orders = sorted(od.sell_orders.items())

        # 1. Take liquidity (Arbitrage only)
        if not circuit_breaker_active:
            for price, volume in sell_orders:
                if to_buy > 0 and price < true_value - 0.5:
                    qty = min(to_buy, -volume)
                    self.buy(price, qty)
                    to_buy -= qty

        for price, volume in buy_orders:
            if to_sell > 0 and price > true_value + 0.5:
                qty = min(to_sell, volume)
                self.sell(price, qty)
                to_sell -= qty

        # 2. Provide passive liquidity
        if to_buy > 0 and not circuit_breaker_active:
            best_ask = sell_orders[0][0] if sell_orders else my_bid + 2
            safe_bid = min(my_bid, best_ask - 1)
            self.buy(safe_bid, to_buy)

        if to_sell > 0:
            best_bid = buy_orders[0][0] if buy_orders else my_ask - 2
            safe_ask = max(my_ask, best_bid + 1)
            self.sell(safe_ask, to_sell)

    def save(self) -> dict[str, Any]:
        return {
            "prev_bid_vol": self.prev_bid_vol,
            "prev_ask_vol": self.prev_ask_vol,
            "circuit_breaker_ticks": self.circuit_breaker_ticks
        }

    def load(self, data: dict[str, Any]) -> None:
        self.prev_bid_vol = data.get("prev_bid_vol", 0)
        self.prev_ask_vol = data.get("prev_ask_vol", 0)
        self.circuit_breaker_ticks = data.get("circuit_breaker_ticks", 0)

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
            "VELVETFRUIT_EXTRACT": VelvetfruitStrategy("VELVETFRUIT_EXTRACT", limits["VELVETFRUIT_EXTRACT"]),
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