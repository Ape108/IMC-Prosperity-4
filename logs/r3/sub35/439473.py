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


# ── 1. Velvetfruit: Hedged Stat-Arb Composite ────────────────────────────────

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
        
        hedge_target = -aggregate_delta
        stat_arb_target = self.get_stat_arb_target(state)
        
        desired_position = np.clip(hedge_target + stat_arb_target, -self.limit, self.limit)
        
        inventory_error = position - desired_position
        inventory_error_ratio = inventory_error / self.limit
        
        skew = (inventory_error_ratio ** 3) * 3.0 
        skewed_true_value = true_value - skew
        
        dynamic_width = 1.0 + (abs(inventory_error_ratio) * 2.5)
        
        max_buy_price = floor(skewed_true_value - dynamic_width)
        min_sell_price = ceil(skewed_true_value + dynamic_width)

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
        pass 

    def save(self) -> dict[str, Any]:
        return {"history": self.history}

    def load(self, data: dict[str, Any]) -> None:
        self.history = data.get("history", [])


# ── 2. Hydrogel: Anchored Microprice Maker ───────────────────────────────────

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
        
        ANCHOR = 10000.0
        return 0.90 * microprice + 0.10 * ANCHOR

    def act(self, state: TradingState) -> None:
        od = state.order_depths[self.symbol]
        if not od.buy_orders or not od.sell_orders:
            return

        true_value = self.get_anchored_microprice(state)
        position = state.position.get(self.symbol, 0)
        inventory_ratio = position / self.limit
        
        skew = (abs(inventory_ratio) ** 3) * np.sign(inventory_ratio) * 12.0
        skewed_true_value = true_value - skew
        
        dynamic_width = 1.5 + ((inventory_ratio ** 2) * 4.0)
        
        max_buy_price = floor(skewed_true_value - dynamic_width)
        min_sell_price = ceil(skewed_true_value + dynamic_width)

        MAX_CLIP = 40
        to_buy = min(self.limit - position, MAX_CLIP)
        to_sell = min(self.limit + position, MAX_CLIP)

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
        }

        self.strategies: dict[Symbol, Strategy] = {
            "HYDROGEL_PACK": HydrogelStrategy("HYDROGEL_PACK", self.limits["HYDROGEL_PACK"]),
            "VELVETFRUIT_EXTRACT": HedgedVelvetfruitStrategy("VELVETFRUIT_EXTRACT", self.limits["VELVETFRUIT_EXTRACT"]),
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders: dict[Symbol, list[Order]] = {}
        conversions = 0
        
        old_data = json.loads(state.traderData) if state.traderData else {}
        new_trader_data: dict[str, Any] = {}

        # ── Options Logic Removed: Delta hardcoded to 0.0 ─────────────
        total_options_delta = 0.0

        # ── PASS 1: Execute Hedged Velvetfruit (Alpha + Hedge) ───────────────
        vf_symbol = "VELVETFRUIT_EXTRACT"
        vf_strat = self.strategies[vf_symbol]
        if isinstance(vf_strat, HedgedVelvetfruitStrategy):
            if vf_symbol in old_data:
                vf_strat.load(old_data[vf_symbol])
            
            vf_strat.act_hedged(state, total_options_delta)
            orders[vf_symbol] = vf_strat.orders
            
            if isinstance(vf_strat, StatefulStrategy):
                new_trader_data[vf_symbol] = vf_strat.save()

        # ── PASS 2: Execute Independent Market Making (Hydrogel) ─────────────
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