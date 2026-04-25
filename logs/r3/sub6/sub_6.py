import json
import math
from abc import abstractmethod
from enum import IntEnum
from math import ceil, floor
from typing import Any
from collections import deque

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

type JSON = dict[str, Any] | list[Any] | str | int | float | bool | None

# ── Fast Math & Pricing Utilities ────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """Highly optimized rational approximation of Normal CDF."""
    b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    p, c = 0.2316419, 0.39894228
    
    if x >= 0.0:
        t = 1.0 / (1.0 + p * x)
        return (1.0 - c * math.exp(-x * x / 2.0) * t * (t *(t *(t *(t * b5 + b4) + b3) + b2) + b1))
    else:
        t = 1.0 / (1.0 - p * x)
        return (c * math.exp(-x * x / 2.0) * t * (t *(t *(t *(t * b5 + b4) + b3) + b2) + b1))

def bs_call_price_and_delta(S: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    """Returns (Price, Delta)."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K), (1.0 if S > K else 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    price = S * norm_cdf(d1) - K * norm_cdf(d2)
    delta = norm_cdf(d1)
    return price, delta

def fast_implied_vol(S: float, K: float, T: float, market_price: float) -> float | None:
    """15-iteration bounded bisection. Fast, safe, no division by zero."""
    intrinsic = max(0.0, S - K)
    if market_price <= intrinsic + 0.1:
        return None

    low, high = 1e-4, 5.0 
    price_high, _ = bs_call_price_and_delta(S, K, T, high)
    if price_high < market_price:
        return None 
        
    for _ in range(15):
        mid = (low + high) / 2.0
        price, _ = bs_call_price_and_delta(S, K, T, mid)
        if price > market_price:
            high = mid
        else:
            low = mid
            
    return (low + high) / 2.0


# ── Logger (Kept Intact) ─────────────────────────────────────────────────────
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
        max_item_length = max(0, (self.max_log_length - base_length) // 3)
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [state.timestamp, trader_data, [[l.symbol, l.product, l.denomination] for l in state.listings.values()], {sym: [od.buy_orders, od.sell_orders] for sym, od in state.order_depths.items()}, [], [], state.position, [state.observations.plainValueObservations, {}]]
    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]
    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))
    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length: return value
        return value[:max_length-3] + "..."

logger = Logger()

# ── Strategy Base Classes ────────────────────────────────────────────────────

class Strategy[T: JSON]:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState, shared_state: dict[str, Any]) -> None:
        raise NotImplementedError()

    def get_required_symbols(self) -> list[Symbol]:
        return [self.symbol]

    def run(self, state: TradingState, shared_state: dict[str, Any]) -> tuple[list[Order], int]:
        self.orders: list[Order] = []
        self.conversions = 0
        if all(sym in state.order_depths and state.order_depths[sym].buy_orders and state.order_depths[sym].sell_orders for sym in self.get_required_symbols()):
            self.act(state, shared_state)
        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        od = state.order_depths[symbol]
        return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0


class StatefulStrategy[T: JSON](Strategy):
    @abstractmethod
    def save(self) -> T: raise NotImplementedError()
    @abstractmethod
    def load(self, data: T) -> None: raise NotImplementedError()


# ── Constants ────────────────────────────────────────────────────────────────
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHER_SYMBOLS = [f"VEV_{k}" for k in STRIKES]
VEV_SPOT = "VELVETFRUIT_EXTRACT"
ROUND_START_TTE_DAYS = 5.0
TICKS_PER_DAY = 1_000_000


# ── Strategies ───────────────────────────────────────────────────────────────

class HydrogelStrategy(Strategy):
    def act(self, state: TradingState, shared_state: dict[str, Any]) -> None:
        od = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        
        buy_orders = sorted(od.buy_orders.items(), reverse=True)
        sell_orders = sorted(od.sell_orders.items())
        
        # Depth VWAP Microprice
        bid_vol, bid_vwap_sum = 0, 0.0
        for price, vol in buy_orders[:3]:
            bid_vol += vol
            bid_vwap_sum += price * vol
            
        ask_vol, ask_vwap_sum = 0, 0.0
        for price, vol in sell_orders[:3]:
            abs_vol = abs(vol)
            ask_vol += abs_vol
            ask_vwap_sum += price * abs_vol
            
        micro_price = (buy_orders[0][0] + sell_orders[0][0]) / 2.0
        if bid_vol > 0 and ask_vol > 0:
            bid_vwap, ask_vwap = bid_vwap_sum / bid_vol, ask_vwap_sum / ask_vol
            micro_price = (bid_vwap * ask_vol + ask_vwap * bid_vol) / (bid_vol + ask_vol)
            
        inventory_ratio = position / self.limit
        bid_shift = 1.0 + (abs(inventory_ratio) ** 1.5) * 6 * (1.5 if inventory_ratio > 0 else -0.5)
        ask_shift = 1.0 + (abs(inventory_ratio) ** 1.5) * 6 * (1.5 if inventory_ratio < 0 else -0.5)
        
        my_bid = int(floor(micro_price - bid_shift))
        my_ask = int(ceil(micro_price + ask_shift))

        to_buy, to_sell = self.limit - position, self.limit + position

        if to_buy > 0:
            best_ask = sell_orders[0][0]
            safe_bid = min(my_bid, best_ask - 1)
            self.buy(safe_bid, to_buy)

        if to_sell > 0:
            best_bid = buy_orders[0][0]
            safe_ask = max(my_ask, best_bid + 1)
            self.sell(safe_ask, to_sell)


class VelvetfruitStrategy(StatefulStrategy[dict[str, Any]]):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.history_bid_vol = deque(maxlen=3)
        self.history_ask_vol = deque(maxlen=3)

    def get_required_symbols(self) -> list[Symbol]:
        return [self.symbol, "HYDROGEL_PACK", "VEV_5000"]

    def act(self, state: TradingState, shared_state: dict[str, Any]) -> None:
        od = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        
        buy_orders = sorted(od.buy_orders.items(), reverse=True)
        sell_orders = sorted(od.sell_orders.items())
        base_mid = (buy_orders[0][0] + sell_orders[0][0]) / 2.0

        # --- Lead/Lag Skew ---
        hydro_od = state.order_depths.get("HYDROGEL_PACK")
        lead_lag_shift = 0.0
        if hydro_od and hydro_od.buy_orders and hydro_od.sell_orders:
            hydro_bid = sum(hydro_od.buy_orders.values())
            hydro_ask = sum(abs(v) for v in hydro_od.sell_orders.values())
            total = hydro_bid + hydro_ask
            if total > 0:
                lead_lag_shift = ((hydro_bid / total) - 0.5) * 2.0 
        
        true_value = base_mid + lead_lag_shift

        # --- Rolling OFI Circuit Breaker ---
        current_bid_vol = sum(od.buy_orders.values())
        current_ask_vol = sum(abs(v) for v in od.sell_orders.values())
        self.history_bid_vol.append(current_bid_vol)
        self.history_ask_vol.append(current_ask_vol)

        circuit_breaker_active = False
        if len(self.history_bid_vol) == 3:
            avg_bid = sum(self.history_bid_vol) / 3.0
            avg_ask = sum(self.history_ask_vol) / 3.0
            # If sustained imbalance is overwhelmingly toxic
            if avg_bid < (avg_ask * 0.15): circuit_breaker_active = True

        # --- The Delta Band ---
        options_delta = shared_state.get("options_net_delta", 0.0)
        total_delta = position + options_delta
        
        # Base edge scales with options IV
        base_edge = 1.0
        vev_od = state.order_depths.get("VEV_5000")
        if vev_od and vev_od.buy_orders and vev_od.sell_orders:
            if min(vev_od.sell_orders.keys()) - max(vev_od.buy_orders.keys()) > 15:
                base_edge += 1.0 

        # Aggressively skew quotes if we breach the Delta Band [-40, +40]
        delta_skew_bid = 0.0
        delta_skew_ask = 0.0
        if total_delta > 40:
            delta_skew_ask = -2.0  # Drop asks to dump spot and reduce long delta
            delta_skew_bid = 3.0   # Drop bids to stop buying
        elif total_delta < -40:
            delta_skew_ask = 3.0
            delta_skew_bid = -2.0  # Raise bids to acquire spot and reduce short delta

        inventory_ratio = position / self.limit
        bid_shift = base_edge + delta_skew_bid + (abs(inventory_ratio) ** 1.5) * 4 * (1.5 if inventory_ratio > 0 else -0.5)
        ask_shift = base_edge + delta_skew_ask + (abs(inventory_ratio) ** 1.5) * 4 * (1.5 if inventory_ratio < 0 else -0.5)

        my_bid = int(floor(true_value - bid_shift))
        my_ask = int(ceil(true_value + ask_shift))

        to_buy, to_sell = self.limit - position, self.limit + position

        # Provide passive liquidity
        if to_buy > 0 and not circuit_breaker_active:
            safe_bid = min(my_bid, sell_orders[0][0] - 1)
            self.buy(safe_bid, to_buy)

        if to_sell > 0:
            safe_ask = max(my_ask, buy_orders[0][0] + 1)
            self.sell(safe_ask, to_sell)

    def save(self) -> dict[str, Any]:
        return {"bids": list(self.history_bid_vol), "asks": list(self.history_ask_vol)}

    def load(self, data: dict[str, Any]) -> None:
        self.history_bid_vol = deque(data.get("bids", []), maxlen=3)
        self.history_ask_vol = deque(data.get("asks", []), maxlen=3)


class VoucherStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, strike: int, k: float = 300.0) -> None:
        super().__init__(symbol, limit)
        self.strike = strike
        self.k = k

    def get_required_symbols(self) -> list[Symbol]:
        return [VEV_SPOT] + VOUCHER_SYMBOLS

    def act(self, state: TradingState, shared_state: dict[str, Any]) -> None:
        spot = self.get_mid_price(state, VEV_SPOT)
        tte_years = max(ROUND_START_TTE_DAYS - state.timestamp / TICKS_PER_DAY, 0.001) / 365.0

        # Smile Construction (Pure Python List Comprehensions, NO NUMPY)
        moneynesses, ivs = [], []
        for s, sym in zip(STRIKES, VOUCHER_SYMBOLS):
            od = state.order_depths[sym]
            if not od.buy_orders or not od.sell_orders: continue
            mid = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
            iv = fast_implied_vol(spot, float(s), tte_years, mid)
            if iv is not None:
                moneynesses.append(s / spot)
                ivs.append(iv)

        if len(moneynesses) < 3: return

        # Simple Linear Extrapolation/Clamping for the target IV (Replacing Polyfit)
        target_m = self.strike / spot
        if target_m <= moneynesses[0]:
            fitted_iv = ivs[0] # Clamp left wing
        elif target_m >= moneynesses[-1]:
            fitted_iv = ivs[-1] # Clamp right wing
        else:
            # Linear Interpolation between the closest points
            for i in range(len(moneynesses)-1):
                if moneynesses[i] <= target_m <= moneynesses[i+1]:
                    x0, x1 = moneynesses[i], moneynesses[i+1]
                    y0, y1 = ivs[i], ivs[i+1]
                    fitted_iv = y0 + (y1 - y0) * ((target_m - x0) / (x1 - x0))
                    break

        od = state.order_depths[self.symbol]
        my_best_bid = max(od.buy_orders.keys())
        my_best_ask = min(od.sell_orders.keys())
        my_iv = fast_implied_vol(spot, float(self.strike), tte_years, (my_best_bid + my_best_ask) / 2.0)
        
        if my_iv is None: return

        residual = my_iv - fitted_iv
        spread_width = my_best_ask - my_best_bid
        if abs(residual) < max(0.01, spread_width * 0.002): return
            
        position = state.position.get(self.symbol, 0)
        target = int(max(-self.limit, min(self.limit, -self.k * residual)))

        if target > position:
            self.buy(my_best_ask, min(target - position, abs(od.sell_orders[my_best_ask])))
        elif target < position:
            self.sell(my_best_bid, min(position - target, od.buy_orders[my_best_bid]))


# ── Trader Orchestration ─────────────────────────────────────────────────────

class Trader:
    def __init__(self) -> None:
        limits = {
            "HYDROGEL_PACK": 200,
            "VELVETFRUIT_EXTRACT": 200,
            **{f"VEV_{strike}": 300 for strike in STRIKES}
        }
        self.strategies: dict[Symbol, Strategy] = {
            "HYDROGEL_PACK": HydrogelStrategy("HYDROGEL_PACK", limits["HYDROGEL_PACK"]),
            "VELVETFRUIT_EXTRACT": VelvetfruitStrategy("VELVETFRUIT_EXTRACT", limits["VELVETFRUIT_EXTRACT"]),
            **{f"VEV_{strike}": VoucherStrategy(f"VEV_{strike}", limits[f"VEV_{strike}"], strike) for strike in STRIKES},
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders: dict[Symbol, list[Order]] = {}
        conversions = 0
        shared_state: dict[str, Any] = {}

        # 1. Calculate Global Options Delta BEFORE running strategies
        options_net_delta = 0.0
        if VEV_SPOT in state.order_depths and state.order_depths[VEV_SPOT].buy_orders and state.order_depths[VEV_SPOT].sell_orders:
            spot_price = (max(state.order_depths[VEV_SPOT].buy_orders.keys()) + min(state.order_depths[VEV_SPOT].sell_orders.keys())) / 2.0
            tte_years = max(ROUND_START_TTE_DAYS - state.timestamp / TICKS_PER_DAY, 0.001) / 365.0
            
            for strike, sym in zip(STRIKES, VOUCHER_SYMBOLS):
                pos = state.position.get(sym, 0)
                if pos != 0 and sym in state.order_depths and state.order_depths[sym].sell_orders and state.order_depths[sym].buy_orders:
                    opt_mid = (max(state.order_depths[sym].buy_orders.keys()) + min(state.order_depths[sym].sell_orders.keys())) / 2.0
                    iv = fast_implied_vol(spot_price, float(strike), tte_years, opt_mid)
                    if iv:
                        _, delta = bs_call_price_and_delta(spot_price, float(strike), tte_years, iv)
                        options_net_delta += pos * delta

        shared_state["options_net_delta"] = options_net_delta

        # 2. Execute Strategies
        old_trader_data = json.loads(state.traderData) if state.traderData not in ("", None) else {}
        new_trader_data: dict[str, Any] = {}

        for symbol, strategy in self.strategies.items():
            if isinstance(strategy, StatefulStrategy) and symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            strategy_orders, strategy_conversions = strategy.run(state, shared_state)
            orders[symbol] = strategy_orders
            conversions += strategy_conversions

            if isinstance(strategy, StatefulStrategy):
                new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data