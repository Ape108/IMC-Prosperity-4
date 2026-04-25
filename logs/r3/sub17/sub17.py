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
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def bs_call_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    return norm_cdf(d1)

def vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi)
    return S * sqrt_T * pdf_d1

def standard_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)

def bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    return standard_normal_pdf(d1) / (S * sigma * sqrt_T)

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
        self.orders: list[Order] = []

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def get_required_symbols(self) -> list[Symbol]:
        return [self.symbol]

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
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
        return (popular_buy + popular_sell) / 2.0


class StatefulStrategy[T: JSON](Strategy):
    @abstractmethod
    def save(self) -> T:
        raise NotImplementedError()

    @abstractmethod
    def load(self, data: T) -> None:
        raise NotImplementedError()


# ── Constants ────────────────────────────────────────────────────────────────

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHER_SYMBOLS = [f"VEV_{k}" for k in STRIKES]
VEV_SPOT = "VELVETFRUIT_EXTRACT"
ROUND_START_TTE_DAYS = 5.0
TICKS_PER_DAY = 1_000_000


# ── 1. VEV Options: Maker-Style Volatility Arbitrage ─────────────────────────

class VoucherStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, strike: int, max_otm_moneyness: float = 1.050) -> None:
        super().__init__(symbol, limit)
        self.strike = strike
        self.max_otm_moneyness = max_otm_moneyness

    def get_required_symbols(self) -> list[Symbol]:
        return [VEV_SPOT] + VOUCHER_SYMBOLS

    def act(self, state: TradingState) -> None:
        spot = self.get_mid_price(state, VEV_SPOT)
        tte_days = ROUND_START_TTE_DAYS - state.timestamp / TICKS_PER_DAY
        tte_years = max(tte_days, 0.001) / 365.0

        # Phase 1: Vega-Weighted Surface Fitting
        moneynesses, ivs, weights = [], [], []
        for s, sym in zip(STRIKES, VOUCHER_SYMBOLS):
            od = state.order_depths.get(sym)
            if not od or not od.buy_orders or not od.sell_orders: continue
            
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            
            # Illiquidity Dead-band: ignore garbage data
            if (best_ask - best_bid) > 15.0: 
                continue
                
            mid = (best_bid + best_ask) / 2.0
            iv = implied_vol(spot, float(s), tte_years, mid)
            
            if iv:
                v = vega(spot, float(s), tte_years, iv)
                if v > 1e-4:  # Only weight strikes with meaningful vega
                    moneynesses.append(s / spot)
                    ivs.append(iv)
                    weights.append(v)

        if len(moneynesses) < 3:
            return

        coeffs = np.polyfit(moneynesses, ivs, 2, w=weights)
        if self.strike / spot > self.max_otm_moneyness: return

        # Phase 2: Greek calculation & True Value
        fitted_iv = float(np.polyval(coeffs, self.strike / spot))
        theo_price = bs_call_price(spot, float(self.strike), tte_years, fitted_iv)
        
        o_vega = vega(spot, float(self.strike), tte_years, fitted_iv)
        o_gamma = bs_gamma(spot, float(self.strike), tte_years, fitted_iv)
        
        position = state.position.get(self.symbol, 0)
        inventory_ratio = position / self.limit

        # Phase 3: Risk-Adjusted Quoting Widths
        # Widens spreads for high gamma (pin risk) and high vega (vol sensitivity)
        gamma_penalty = o_gamma * 300.0
        vega_expansion = o_vega * 0.03
        
        # Time-to-Expiration (TTE) Scaling penalty
        tte_penalty = 1.0 if tte_days < 0.5 else 0.0

        dynamic_width = 1.0 + vega_expansion + gamma_penalty + tte_penalty + (abs(inventory_ratio) * 2.5)
        
        # Inventory Skew (Cubic)
        skew = (inventory_ratio ** 3) * 4.0
        skewed_theo = theo_price - skew
        
        bid_price = floor(skewed_theo - dynamic_width)
        ask_price = ceil(skewed_theo + dynamic_width)
        
        # Safety bounds
        intrinsic = max(0.0, spot - self.strike)
        bid_price = max(bid_price, floor(intrinsic))
        ask_price = max(ask_price, bid_price + 1)

        to_buy = self.limit - position
        to_sell = self.limit + position
        
        # Limit clip size on options to 50 per tick
        MAX_CLIP = 50
        to_buy = min(to_buy, MAX_CLIP)
        to_sell = min(to_sell, MAX_CLIP)

        if to_buy > 0: self.buy(int(bid_price), to_buy)
        if to_sell > 0: self.sell(int(ask_price), to_sell)

    def get_current_delta(self, state: TradingState) -> float:
        """Returns the total directional delta of our holding in this voucher."""
        position = state.position.get(self.symbol, 0)
        if position == 0: return 0.0
        
        spot = self.get_mid_price(state, VEV_SPOT)
        tte_days = ROUND_START_TTE_DAYS - state.timestamp / TICKS_PER_DAY
        tte_years = max(tte_days, 0.001) / 365.0
        
        od = state.order_depths[self.symbol]
        mid = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
        iv = implied_vol(spot, float(self.strike), tte_years, mid) or 0.5
        
        return position * bs_call_delta(spot, float(self.strike), tte_years, iv)


# ── 2. Velvetfruit: Hedged Stat-Arb Composite ────────────────────────────────

class HedgedVelvetfruitStrategy(StatefulStrategy[dict[str, Any]]):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.zscore_period = 75
        self.smoothing_period = 100
        self.history: list[float] = []

    def get_microprice(self, state: TradingState) -> float:
        od = state.order_depths[self.symbol]
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_vol = od.buy_orders[best_bid]
        ask_vol = abs(od.sell_orders[best_ask])
        total_vol = bid_vol + ask_vol
        return (best_bid * ask_vol + best_ask * bid_vol) / total_vol if total_vol > 0 else (best_bid + best_ask) / 2.0

    def get_stat_arb_target(self, state: TradingState) -> int:
        mid = self.get_mid_price(state, self.symbol)
        self.history.append(mid)

        required = self.zscore_period + self.smoothing_period
        if len(self.history) > required:
            self.history.pop(0)
        if len(self.history) < required:
            return 0

        hist = pd.Series(self.history)
        score = (
            ((hist - hist.rolling(self.zscore_period).mean()) / hist.rolling(self.zscore_period).std())
            .rolling(self.smoothing_period)
            .mean()
            .iloc[-1]
        )

        if pd.isna(score): return 0
        
        # Max stat arb allocation is 100 out of our 200 limit (saving room for hedging)
        # Score of +/- 1.5 equates to full allocation
        target = int(-score * 60)
        return max(-100, min(100, target))

    def act_hedged(self, state: TradingState, aggregate_delta: float) -> None:
        self.orders = []
        self.conversions = 0

        if not (self.symbol in state.order_depths and state.order_depths[self.symbol].buy_orders):
            return

        true_value = self.get_microprice(state)
        od = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        
        # Target 1: Inverse of Options Delta
        hedge_target = -aggregate_delta
        # Target 2: Mean-Reversion Alpha
        stat_arb_target = self.get_stat_arb_target(state)
        
        desired_position = np.clip(hedge_target + stat_arb_target, -self.limit, self.limit)
        
        # Skew to force inventory toward desired_position rather than 0
        inventory_error = position - desired_position
        inventory_error_ratio = inventory_error / self.limit
        
        # Aggressive Skew to hit target
        skew = (inventory_error_ratio ** 3) * 3.0 
        skewed_true_value = true_value - skew
        
        dynamic_width = 1.0 + (abs(inventory_error_ratio) * 2.5)
        
        max_buy_price = floor(skewed_true_value - dynamic_width)
        min_sell_price = ceil(skewed_true_value + dynamic_width)

        # Clip Limits: Size fading
        MAX_CLIP = 40
        to_buy = min(self.limit - position, MAX_CLIP)
        to_sell = min(self.limit + position, MAX_CLIP)

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

        if to_buy > 0:
            price = next((p + 1 for p, _ in buy_orders if p < max_buy_price), max_buy_price)
            self.buy(int(price), to_buy)

        if to_sell > 0:
            price = next((p - 1 for p, _ in sell_orders if p > min_sell_price), min_sell_price)
            self.sell(int(price), to_sell)

    def act(self, state: TradingState) -> None:
        pass # Overridden by act_hedged

    def save(self) -> dict[str, Any]:
        return {"history": self.history}

    def load(self, data: dict[str, Any]) -> None:
        self.history = data.get("history", [])


# ── 3. Hydrogel: Anchored Microprice Maker ───────────────────────────────────

class HydrogelStrategy(Strategy):
    def get_anchored_microprice(self, state: TradingState) -> float:
        od = state.order_depths[self.symbol]
        if not od.buy_orders or not od.sell_orders:
            return 10000.0

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_vol = od.buy_orders[best_bid]
        ask_vol = abs(od.sell_orders[best_ask])
        total_vol = bid_vol + ask_vol
        
        microprice = (best_bid * ask_vol + best_ask * bid_vol) / total_vol if total_vol > 0 else (best_bid + best_ask) / 2.0
        
        # Softened mean reversion pull (from 25% down to 10%)
        # Allows the model to ride short-term trends without fighting the tape too hard.
        ANCHOR = 10000.0
        return 0.90 * microprice + 0.10 * ANCHOR

    def act(self, state: TradingState) -> None:
        od = state.order_depths[self.symbol]
        if not od.buy_orders or not od.sell_orders:
            return

        true_value = self.get_anchored_microprice(state)
        position = state.position.get(self.symbol, 0)
        inventory_ratio = position / self.limit
        
        # SUPERCHARGED INVENTORY SKEW
        # Max skew is now 12.0 ticks. This heavily overpowers the anchor pull.
        # If we are long 200, we slash our fair value by 12 ticks to instantly dump inventory.
        skew = (abs(inventory_ratio) ** 3) * np.sign(inventory_ratio) * 12.0
        skewed_true_value = true_value - skew
        
        # PARABOLIC SPREAD WIDENING
        # Base width is 1.5. At max inventory, width expands by an extra 4 ticks.
        # This acts as an "airbag" against toxic momentum spikes.
        dynamic_width = 1.5 + ((inventory_ratio ** 2) * 4.0)
        
        max_buy_price = floor(skewed_true_value - dynamic_width)
        min_sell_price = ceil(skewed_true_value + dynamic_width)

        MAX_CLIP = 40
        to_buy = min(self.limit - position, MAX_CLIP)
        to_sell = min(self.limit + position, MAX_CLIP)

        # 1. Clear toxic flow by taking liquidity if our skew pushed us across the spread
        sell_orders = sorted(od.sell_orders.items())
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                qty = min(to_buy, -volume)
                self.buy(price, qty)
                to_buy -= qty

        buy_orders = sorted(od.buy_orders.items(), reverse=True)
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                qty = min(to_sell, volume)
                self.sell(price, qty)
                to_sell -= qty

        # 2. Provide liquidity (Maker)
        if to_buy > 0:
            price = next((p + 1 for p, _ in buy_orders if p < max_buy_price), max_buy_price)
            self.buy(int(price), to_buy)
        if to_sell > 0:
            price = next((p - 1 for p, _ in sell_orders if p > min_sell_price), min_sell_price)
            self.sell(int(price), to_sell)


# ── Trader Architecture ──────────────────────────────────────────────────────

class Trader:
    def __init__(self) -> None:
        self.limits = {
            "HYDROGEL_PACK": 200,
            "VELVETFRUIT_EXTRACT": 200,
            **{f"VEV_{s}": 300 for s in STRIKES}
        }

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
        
        old_data = json.loads(state.traderData) if state.traderData else {}
        new_trader_data: dict[str, Any] = {}

        # ── PASS 1: Execute Options & Aggregate Portfolio Delta ─────────────
        total_options_delta = 0.0
        for symbol in VOUCHER_SYMBOLS:
            strat = self.strategies[symbol]
            if isinstance(strat, StatefulStrategy) and symbol in old_data:
                strat.load(old_data[symbol])
            
            strat_orders, _ = strat.run(state)
            orders[symbol] = strat_orders
            
            if hasattr(strat, 'get_current_delta'):
                total_options_delta += strat.get_current_delta(state)
            
            if isinstance(strat, StatefulStrategy):
                new_trader_data[symbol] = strat.save()

        # ── PASS 2: Execute Hedged Velvetfruit (Alpha + Hedge) ───────────────
        vf_symbol = "VELVETFRUIT_EXTRACT"
        vf_strat = self.strategies[vf_symbol]
        if isinstance(vf_strat, HedgedVelvetfruitStrategy):
            if vf_symbol in old_data:
                vf_strat.load(old_data[vf_symbol])
            
            vf_strat.act_hedged(state, total_options_delta)
            orders[vf_symbol] = vf_strat.orders
            
            if isinstance(vf_strat, StatefulStrategy):
                new_trader_data[vf_symbol] = vf_strat.save()

        # ── PASS 3: Execute Independent Market Making (Hydrogel) ─────────────
        hg_symbol = "HYDROGEL_PACK"
        hg_strat = self.strategies[hg_symbol]
        if hg_symbol in old_data:
            hg_strat.load(old_data[hg_symbol])
            
        hg_orders, hg_conv = hg_strat.run(state)
        orders[hg_symbol] = hg_orders
        conversions += hg_conv
        
        if isinstance(hg_strat, StatefulStrategy):
            new_trader_data[hg_symbol] = hg_strat.save()

        # Serialize state and flush
        traderData = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, traderData)
        return orders, conversions, traderData