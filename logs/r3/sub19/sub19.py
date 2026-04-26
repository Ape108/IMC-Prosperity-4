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

# ── Black-Scholes math (Optimized) ───────────────────────────────────────────

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

def implied_vol(S: float, K: float, T: float, market_price: float, max_iter: int = 25) -> float | None:
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
MAX_CLIP = 40

# ── 1. VEV Options: Decoupled Volatility Arbitrage ───────────────────────────

class VoucherStrategy(StatefulStrategy[dict[str, Any]]):
    def __init__(self, symbol: str, limit: int, strike: int) -> None:
        super().__init__(symbol, limit)
        self.strike = strike
        self.iv_ema = 0.5
        # Injected by Trader
        self.spot = 0.0
        self.is_toxic_up = False
        self.is_toxic_down = False

    def save(self) -> dict[str, Any]:
        return {"iv_ema": self.iv_ema}

    def load(self, data: dict[str, Any]) -> None:
        self.iv_ema = data.get("iv_ema", 0.5)

    def get_required_symbols(self) -> list[Symbol]:
        return [VEV_SPOT, self.symbol]

    def act(self, state: TradingState) -> None:
        if self.spot == 0.0:
            return

        tte_days = max((ROUND_START_TTE_DAYS - state.timestamp / TICKS_PER_DAY), 0.001)
        tte_years = tte_days / 365.0

        v_mid = self.get_mid_price(state, self.symbol)
        current_iv = implied_vol(self.spot, float(self.strike), tte_years, v_mid)
        
        # EMA Tracking per strike avoids global polynomial lag
        if current_iv:
            self.iv_ema = 0.85 * self.iv_ema + 0.15 * current_iv

        theo_price = bs_call_price(self.spot, float(self.strike), tte_years, self.iv_ema)
        
        position = state.position.get(self.symbol, 0)
        od = state.order_depths[self.symbol]
        
        to_buy = self.limit - position
        to_sell = self.limit + position

        # --- AGGRESSIVE SNIPING (Taker) ---
        for price, volume in sorted(od.sell_orders.items()):
            if to_buy > 0 and price < theo_price - 1.2:  
                qty = min(to_buy, abs(volume), MAX_CLIP)
                self.buy(price, qty)
                to_buy -= qty

        for price, volume in sorted(od.buy_orders.items(), reverse=True):
            if to_sell > 0 and price > theo_price + 1.2:  
                qty = min(to_sell, abs(volume), MAX_CLIP)
                self.sell(price, qty)
                to_sell -= qty

        # --- TOXICITY-AWARE MAKER ---
        inventory_ratio = position / self.limit
        skew = (inventory_ratio * 3.5) + (np.sign(inventory_ratio) * (inventory_ratio ** 2) * 1.5)
        skewed_theo = theo_price - skew
        
        dynamic_width = 1.5 + (abs(inventory_ratio) * 2.0)
        
        bid_price = floor(skewed_theo - dynamic_width)
        ask_price = ceil(skewed_theo + dynamic_width)
        
        intrinsic = max(0.0, self.spot - self.strike)
        bid_price = max(bid_price, floor(intrinsic))
        ask_price = max(ask_price, bid_price + 1)

        # The PULL mechanic: don't provide liquidity if underlying is running against the side
        if to_buy > 0 and not self.is_toxic_down: 
            self.buy(int(bid_price), min(to_buy, 15))
        if to_sell > 0 and not self.is_toxic_up: 
            self.sell(int(ask_price), min(to_sell, 15))

    def get_current_delta(self, state: TradingState) -> float:
        position = state.position.get(self.symbol, 0)
        if position == 0 or self.spot == 0.0: return 0.0
        tte_years = max((ROUND_START_TTE_DAYS - state.timestamp / TICKS_PER_DAY), 0.001) / 365.0
        return position * bs_call_delta(self.spot, float(self.strike), tte_years, self.iv_ema)

# ── 2. Velvetfruit: Hedged Stat-Arb Composite ────────────────────────────────

class HedgedVelvetfruitStrategy(StatefulStrategy[dict[str, Any]]):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.zscore_period = 75
        self.smoothing_period = 100
        self.history: list[float] = []
        # Injected by Trader
        self.is_toxic_up = False
        self.is_toxic_down = False

    def get_microprice(self, state: TradingState) -> float:
        od = state.order_depths[self.symbol]
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_vol = od.buy_orders[best_bid]
        ask_vol = abs(od.sell_orders[best_ask])
        total_vol = bid_vol + ask_vol
        return (best_bid * ask_vol + best_ask * bid_vol) / total_vol if total_vol > 0 else (best_bid + best_ask) / 2.0

    def get_stat_arb_target(self, state: TradingState) -> int:
        od = state.order_depths[self.symbol]
        mid = self.get_mid_price(state, self.symbol)
        
        # --- DEPTH-WEIGHTED OBI ---
        weighted_bid_vol = sum(vol / (max(0.5, mid - price) ** 1.5) for price, vol in od.buy_orders.items())
        weighted_ask_vol = sum(abs(vol) / (max(0.5, price - mid) ** 1.5) for price, vol in od.sell_orders.items())
            
        total_weighted = weighted_bid_vol + weighted_ask_vol
        obi = (weighted_bid_vol - weighted_ask_vol) / total_weighted if total_weighted > 0 else 0.0
        obi_target = int(obi * 90) 

        self.history.append(mid)

        required = self.zscore_period + self.smoothing_period
        if len(self.history) > required:
            self.history.pop(0)
            
        z_target = 0
        if len(self.history) >= required:
            hist = pd.Series(self.history)
            score = (
                ((hist - hist.rolling(self.zscore_period).mean()) / hist.rolling(self.zscore_period).std())
                .rolling(self.smoothing_period)
                .mean()
                .iloc[-1]
            )
            if not pd.isna(score):
                z_target = int(-score * 60) 
        
        return max(-100, min(100, z_target + obi_target))

    def act_hedged(self, state: TradingState, aggregate_delta: float) -> None:
        self.orders = []
        self.conversions = 0

        if not (self.symbol in state.order_depths and state.order_depths[self.symbol].buy_orders):
            return

        true_value = self.get_microprice(state)
        od = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        
        hedge_target = -aggregate_delta
        stat_arb_target = self.get_stat_arb_target(state)
        
        desired_position = np.clip(hedge_target + stat_arb_target, -self.limit, self.limit)
        inventory_error = position - desired_position
        inventory_error_ratio = inventory_error / self.limit
        
        # Linear + Quadratic skew to fight inventory buildup immediately 
        skew = (inventory_error_ratio * 4.0) + (np.sign(inventory_error_ratio) * (inventory_error_ratio ** 2) * 2.0)
        skewed_true_value = true_value - skew
        dynamic_width = 1.0 + (abs(inventory_error_ratio) * 2.5)
        
        max_buy_price = floor(skewed_true_value - dynamic_width)
        min_sell_price = ceil(skewed_true_value + dynamic_width)

        to_buy = min(self.limit - position, MAX_CLIP)
        to_sell = min(self.limit + position, MAX_CLIP)

        # Atomic Hedge Sweep: If we are far from desired delta, sweep the book.
        for price, volume in sorted(od.sell_orders.items()):
            if to_buy > 0 and price <= max_buy_price:
                qty = min(to_buy, -volume)
                self.buy(price, qty)
                to_buy -= qty

        for price, volume in sorted(od.buy_orders.items(), reverse=True):
            if to_sell > 0 and price >= min_sell_price:
                qty = min(to_sell, volume)
                self.sell(price, qty)
                to_sell -= qty

        # Maker: Pull quotes if toxic
        if to_buy > 0 and not self.is_toxic_down:
            price = next((p + 1 for p, _ in sorted(od.buy_orders.items(), reverse=True) if p < max_buy_price), max_buy_price)
            self.buy(int(price), min(to_buy, 20))

        if to_sell > 0 and not self.is_toxic_up:
            price = next((p - 1 for p, _ in sorted(od.sell_orders.items()) if p > min_sell_price), min_sell_price)
            self.sell(int(price), min(to_sell, 20))

    def act(self, state: TradingState) -> None: pass 
    def save(self) -> dict[str, Any]: return {"history": self.history}
    def load(self, data: dict[str, Any]) -> None: self.history = data.get("history", [])

# ── 3. Hydrogel: Anchored Microprice Maker ───────────────────────────────────

class HydrogelStrategy(Strategy):
    def __init__(self, symbol: str, limit: int) -> None:
        super().__init__(symbol, limit)
        # Injected by Trader
        self.cross_asset_signal = 0.0

    def get_anchored_microprice(self, state: TradingState) -> float:
        od = state.order_depths[self.symbol]
        if not od.buy_orders or not od.sell_orders: return 10000.0

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_vol = od.buy_orders[best_bid]
        ask_vol = abs(od.sell_orders[best_ask])
        total_vol = bid_vol + ask_vol
        
        microprice = (best_bid * ask_vol + best_ask * bid_vol) / total_vol if total_vol > 0 else (best_bid + best_ask) / 2.0
        
        # Lead-Lag anchor adjustment based on Velvetfruit
        dynamic_anchor = 10000.0 + (self.cross_asset_signal * 1.25)
        return 0.85 * microprice + 0.15 * dynamic_anchor

    def act(self, state: TradingState) -> None:
        od = state.order_depths[self.symbol]
        if not od.buy_orders or not od.sell_orders: return

        true_value = self.get_anchored_microprice(state)
        position = state.position.get(self.symbol, 0)
        inventory_ratio = position / self.limit
        
        # Avellaneda-style linear penalty
        skew = inventory_ratio * 10.0 
        skewed_true_value = true_value - skew
        dynamic_width = 1.5 + (abs(inventory_ratio) * 3.5)
        
        max_buy_price = floor(skewed_true_value - dynamic_width)
        min_sell_price = ceil(skewed_true_value + dynamic_width)

        to_buy = min(self.limit - position, MAX_CLIP)
        to_sell = min(self.limit + position, MAX_CLIP)

        for price, volume in sorted(od.sell_orders.items()):
            if to_buy > 0 and price <= max_buy_price:
                qty = min(to_buy, -volume)
                self.buy(price, qty)
                to_buy -= qty

        for price, volume in sorted(od.buy_orders.items(), reverse=True):
            if to_sell > 0 and price >= min_sell_price:
                qty = min(to_sell, volume)
                self.sell(price, qty)
                to_sell -= qty

        if to_buy > 0:
            price = next((p + 1 for p, _ in sorted(od.buy_orders.items(), reverse=True) if p < max_buy_price), max_buy_price)
            self.buy(int(price), to_buy)
        if to_sell > 0:
            price = next((p - 1 for p, _ in sorted(od.sell_orders.items()) if p > min_sell_price), min_sell_price)
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

        # ── Global State Tracking & Toxicity Detection ───────────────
        vf_symbol = "VELVETFRUIT_EXTRACT"
        vf_mid_history = old_data.get("vf_mid_history", [])
        
        spot = 0.0
        current_vf_momentum = 0.0
        is_toxic_up, is_toxic_down = False, False
        
        od = state.order_depths.get(vf_symbol)
        if od and od.buy_orders and od.sell_orders:
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            spot = (best_bid + best_ask) / 2.0
            
            vf_mid_history.append(spot)
            if len(vf_mid_history) > 15:
                vf_mid_history.pop(0)
            
            if len(vf_mid_history) >= 5:
                velocity = spot - vf_mid_history[-5]
                is_toxic_up = velocity > 3.0
                is_toxic_down = velocity < -3.0
                
            if len(vf_mid_history) >= 10:
                current_vf_momentum = spot - (sum(vf_mid_history[-10:]) / 10.0)
                
            new_trader_data["vf_mid_history"] = vf_mid_history

        # ── PASS 1: Execute Options & Aggregate Portfolio Delta ─────────────
        total_options_delta = 0.0
        for symbol in VOUCHER_SYMBOLS:
            strat = self.strategies[symbol]
            
            if isinstance(strat, VoucherStrategy):
                # Inject global state before running
                strat.spot = spot
                strat.is_toxic_up = is_toxic_up
                strat.is_toxic_down = is_toxic_down
                if symbol in old_data:
                    strat.load(old_data[symbol])
            
            strat_orders, _ = strat.run(state)
            orders[symbol] = strat_orders
            
            if hasattr(strat, 'get_current_delta'):
                total_options_delta += strat.get_current_delta(state)
            
            if isinstance(strat, StatefulStrategy):
                new_trader_data[symbol] = strat.save()

        # ── PASS 2: Execute Hedged Velvetfruit (Alpha + Hedge) ───────────────
        vf_strat = self.strategies[vf_symbol]
        if isinstance(vf_strat, HedgedVelvetfruitStrategy):
            vf_strat.is_toxic_up = is_toxic_up
            vf_strat.is_toxic_down = is_toxic_down
            
            if vf_symbol in old_data:
                vf_strat.load(old_data[vf_symbol])
            
            vf_strat.act_hedged(state, total_options_delta)
            orders[vf_symbol] = vf_strat.orders
            
            if isinstance(vf_strat, StatefulStrategy):
                new_trader_data[vf_symbol] = vf_strat.save()

        # ── PASS 3: Execute Independent Market Making (Hydrogel) ─────────────
        hg_symbol = "HYDROGEL_PACK"
        hg_strat = self.strategies[hg_symbol]
        if isinstance(hg_strat, HydrogelStrategy):
            hg_strat.cross_asset_signal = current_vf_momentum
            
        hg_orders, hg_conv = hg_strat.run(state)
        orders[hg_symbol] = hg_orders
        conversions += hg_conv
        
        traderData = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, traderData)
        return orders, conversions, traderData