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

def bs_call_delta(S: float, K: float, T: float, sigma: float) -> float:
    """Calculates the Delta (dC/dS) of a call option with r=0."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    return norm_cdf(d1)


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
        
        inventory_ratio = position / self.limit
        # If the timestamp is in the second half (H2), increase skew aggressiveness
        is_h2 = state.timestamp > (TICKS_PER_DAY / 2)
        dynamic_skew_factor = 2.0 if is_h2 else 1.0 
        skew = (inventory_ratio ** 3) * dynamic_skew_factor
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
    Voucher IV smile scalper utilizing a Maker execution style.
    Provides liquidity around a theoretical fair value derived from the 
    quadratic smile fit and manages risk with a cubic inventory skew.
    """
    def __init__(self, symbol: str, 
                 limit: int, strike: int, k: float = 300.0, 
                 min_residual: float = 0.005, 
                 max_otm_moneyness: float = 1.020, 
                 skew_factor: float = 0.5) -> None:
        super().__init__(symbol, limit)
        self.strike = strike
        self.k = k
        self.min_residual = min_residual      
        self.max_otm_moneyness = max_otm_moneyness
        self.skew_factor = skew_factor

    def get_required_symbols(self) -> list[Symbol]:
        return [VEV_SPOT] + VOUCHER_SYMBOLS

    def act(self, state: TradingState) -> None:
        spot = self.get_mid_price(state, VEV_SPOT)

        # Calculate Time-To-Expiry in years
        tte_days = ROUND_START_TTE_DAYS - state.timestamp / TICKS_PER_DAY
        tte_years = max(tte_days, 0.001) / 365.0

        # Construct the current IV smile from the option chain
        moneynesses: list[float] = []
        ivs: list[float] = []
        for s, sym in zip(STRIKES, VOUCHER_SYMBOLS):
            od = state.order_depths.get(sym)
            if not od or not od.buy_orders or not od.sell_orders:
                continue
            
            # Use mid-price as the basis for IV curve fitting
            mid = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
            intrinsic = max(0.0, spot - s)
            
            if mid <= intrinsic + 0.1:
                continue
                
            iv = implied_vol(spot, float(s), tte_years, mid)
            if iv is not None:
                moneynesses.append(s / spot)
                ivs.append(iv)

        # Fit quadratic smile: IV = a*m^2 + b*m + c
        coeffs = fit_iv_smile(moneynesses, ivs)
        if coeffs is None:
            return

        # Skip execution if the strike is too far Out-Of-The-Money (OTM)
        if self.strike / spot > self.max_otm_moneyness:
            return

        # Derive theoretical price from the fitted smile
        fitted_iv = float(np.polyval(coeffs, self.strike / spot))
        theo_price = bs_call_price(spot, float(self.strike), tte_years, fitted_iv)
        
        # Apply Cubic Inventory Skew to determine the fair value
        # This pressure increases non-linearly to prevent hitting the 300 position limit
        position = state.position.get(self.symbol, 0)
        inventory_ratio = position / self.limit
        skew = (inventory_ratio ** 3) * self.skew_factor
        skewed_theo = theo_price - skew
        
        # Quote Maker levels around the skewed theoretical price
        # Adjust quotes to integer tick sizes (1.0 sand dollar)
        width = 0.5 + (abs(inventory_ratio) * 2.0) # Spread widens as inventory grows
        bid_price = floor(skewed_theo - width)
        ask_price = ceil(skewed_theo + width)
        
        # Constraint checks for intrinsic value and minimum spread
        intrinsic = max(0.0, spot - self.strike)
        bid_price = max(bid_price, floor(intrinsic))
        ask_price = max(ask_price, bid_price + 1)

        to_buy = self.limit - position
        to_sell = self.limit + position

        # Provide liquidity at the calculated levels
        if to_buy > 0:
            self.buy(int(bid_price), to_buy)
        if to_sell > 0:
            self.sell(int(ask_price), to_sell)

    def get_current_delta(self, state: TradingState) -> float:
        """Returns the total delta exposure for this specific voucher."""
        position = state.position.get(self.symbol, 0)
        if position == 0:
            return 0.0

        spot = self.get_mid_price(state, VEV_SPOT)
        tte_days = ROUND_START_TTE_DAYS - state.timestamp / TICKS_PER_DAY
        tte_years = max(tte_days, 0.001) / 365.0
        
        # We use the mid-price to back out the current IV for the delta calculation
        od = state.order_depths[self.symbol]
        mid = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
        iv = implied_vol(spot, float(self.strike), tte_years, mid) or 0.5
        
        d = bs_call_delta(spot, float(self.strike), tte_years, iv)
        return position * d
    
# ── Delta-1 products ─────────────────────────────────────────────────────────

class HydrogelStrategy(MarketMakingStrategy):
    """
        Notes:
        - Replaced hardcoded 10k peg with an Order Book Imbalance (Microprice) calculation.
        - Dynamically tracks true value using top-of-book volume weights to predict short-term momentum.
    """
    def get_true_value(self, state: TradingState) -> float:
        od = state.order_depths.get(self.symbol)
        
        # Fallback to standard mid-price if the order book is unexpectedly empty
        if not od or not od.buy_orders or not od.sell_orders:
            return self.get_mid_price(state, self.symbol)
        
        # Get top of book (best bid and best ask)
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        
        # In the Prosperity engine, sell volumes are negative integers. We need absolute values.
        bid_vol = od.buy_orders[best_bid]
        ask_vol = abs(od.sell_orders[best_ask])
        
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return (best_bid + best_ask) / 2.0
            
        # The Microprice formula:
        # Cross-weights the price by the opposite volume.
        # Heavy buy volume pulls the true value closer to the ask.
        # Heavy sell volume pulls the true value closer to the bid.
        micro_price = (best_bid * ask_vol + best_ask * bid_vol) / total_vol
        
        return micro_price

class HedgedVelvetfruitStrategy(MarketMakingStrategy):
    """
    Market maker for the underlying that leans its quotes to offset 
    the aggregate delta of the options portfolio.
    """
    def get_true_value(self, state: TradingState) -> float:
        # Re-use your existing Microprice logic for the underlying
        od = state.order_depths.get(self.symbol)
        if not od or not od.buy_orders or not od.sell_orders:
            return self.get_mid_price(state, self.symbol)
        
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_vol = od.buy_orders[best_bid]
        ask_vol = abs(od.sell_orders[best_ask])
        
        total_vol = bid_vol + ask_vol
        return (best_bid * ask_vol + best_ask * bid_vol) / total_vol if total_vol > 0 else (best_bid + best_ask) / 2.0

    def act_hedged(self, state: TradingState, aggregate_delta: float) -> None:
        self.orders: list[Order] = []
        self.conversions = 0
        
        true_value = self.get_true_value(state)
        od = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        
        # The Hedge Target is the negative of your options delta
        # We clip it to the 200 position limit of the underlying
        hedge_target = np.clip(-aggregate_delta, -self.limit, self.limit)
        
        # --- THE FIX: Hedge-Aware Cubic Skew ---
        # Instead of skewing relative to 0, we skew relative to our hedge target
        inventory_error = position - hedge_target
        inventory_ratio = inventory_error / self.limit
        
        # Lean the true value to encourage trades that move us toward the hedge target
        # Increased skew_factor (1.5) to ensure hedging takes priority
        skew = (inventory_ratio ** 3) * 1.5 
        skewed_true_value = true_value - skew
        
        to_buy = self.limit - position
        to_sell = self.limit + position
        
        # Standard quoting logic around the skewed value
        max_buy_price = floor(skewed_true_value - 0.5)
        min_sell_price = ceil(skewed_true_value + 0.5)

        # 1. Take liquidity
        buy_orders = sorted(od.buy_orders.items(), reverse=True)
        sell_orders = sorted(od.sell_orders.items())

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                qty = min(to_buy, -volume)
                self.buy(price, qty)
                to_buy -= qty

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                qty = min(to_sell, volume)
                self.sell(price, qty)
                to_sell -= qty

        # 2. Post limit orders
        if to_buy > 0:
            price = next((p + 1 for p, _ in buy_orders if p < max_buy_price), max_buy_price)
            self.buy(int(price), to_buy)

        if to_sell > 0:
            price = next((p - 1 for p, _ in sell_orders if p > min_sell_price), min_sell_price)
            self.sell(int(price), to_sell)

# ── Trader ───────────────────────────────────────────────────────────────────

class Trader:
    def __init__(self) -> None:
        self.limits = {
            "HYDROGEL_PACK": 200,
            "VELVETFRUIT_EXTRACT": 200,
            **{f"VEV_{s}": 300 for s in STRIKES}
        }

        # Initialize strategy objects
        self.strategies: dict[Symbol, Strategy] = {
            "HYDROGEL_PACK": HydrogelStrategy("HYDROGEL_PACK", self.limits["HYDROGEL_PACK"]),
            "VELVETFRUIT_EXTRACT": HedgedVelvetfruitStrategy("VELVETFRUIT_EXTRACT", self.limits["VELVETFRUIT_EXTRACT"]),
            **{
                f"VEV_{s}": VoucherStrategy(f"VEV_{s}", self.limits[f"VEV_{s}"], s)
                for s in STRIKES
            },
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders: dict[Symbol, list[Order]] = {}
        conversions = 0
        
        # Safe deserialization for AWS Lambda environment
        old_data = json.loads(state.traderData) if state.traderData else {}
        new_trader_data: dict[str, Any] = {}

        # ── PASS 1: Execute Options & Aggregate Portfolio Delta ─────────────
        total_options_delta = 0.0
        for symbol in VOUCHER_SYMBOLS:
            strat = self.strategies[symbol]
            if isinstance(strat, StatefulStrategy) and symbol in old_data:
                strat.load(old_data[symbol])
            
            # Execute options trading
            strat_orders, _ = strat.run(state)
            orders[symbol] = strat_orders
            
            # Calculate delta of current (and newly updated) positions
            if hasattr(strat, 'get_current_delta'):
                total_options_delta += strat.get_current_delta(state)
            
            if isinstance(strat, StatefulStrategy):
                new_trader_data[symbol] = strat.save()

        # ── PASS 2: Execute Hedged Market Making for Underlying ──────────────
        vf_symbol = "VELVETFRUIT_EXTRACT"
        vf_strat = self.strategies[vf_symbol]
        if isinstance(vf_strat, HedgedVelvetfruitStrategy):
            if vf_symbol in old_data:
                vf_strat.load(old_data[vf_symbol])
            
            # Use aggregated delta to lean quotes
            vf_strat.act_hedged(state, total_options_delta)
            orders[vf_symbol] = vf_strat.orders
            
            if isinstance(vf_strat, StatefulStrategy):
                new_trader_data[vf_symbol] = vf_strat.save()

        # ── PASS 3: Execute Independent Products (Hydrogel) ─────────────────
        hg_symbol = "HYDROGEL_PACK"
        hg_strat = self.strategies[hg_symbol]
        if hg_symbol in old_data:
            hg_strat.load(old_data[hg_symbol])
            
        hg_orders, hg_conv = hg_strat.run(state)
        orders[hg_symbol] = hg_orders
        conversions += hg_conv
        
        if isinstance(hg_strat, StatefulStrategy):
            new_trader_data[hg_symbol] = hg_strat.save()

        # Serialize state for next tick (Lambda persistence)
        traderData = json.dumps(new_trader_data, separators=(",", ":"))
        
        # Log and return the required 3-tuple
        logger.flush(state, orders, conversions, traderData)
        return orders, conversions, traderData