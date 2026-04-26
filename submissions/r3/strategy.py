import json
import math
from typing import Any
import numpy as np

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

type JSON = dict[str, Any] | list[Any] | str | int | float | bool | None

# ── Black-Scholes Math (Optimized) ───────────────────────────────────────────

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0: return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

# ── Constants ────────────────────────────────────────────────────────────────

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHER_SYMBOLS = [f"VEV_{k}" for k in STRIKES]
VEV_SPOT = "VELVETFRUIT_EXTRACT"
ROUND_START_TTE_DAYS = 5.0
TICKS_PER_DAY = 1_000_000
MAX_CLIP = 40

# ── Trader Architecture ──────────────────────────────────────────────────────

class Trader:
    def __init__(self) -> None:
        self.limits = {
            "HYDROGEL_PACK": 200,
            "VELVETFRUIT_EXTRACT": 200,
            **{f"VEV_{s}": 300 for s in STRIKES}
        }
    
    def get_mid(self, state: TradingState, symbol: str) -> float | None:
        od = state.order_depths.get(symbol)
        if not od or not od.buy_orders or not od.sell_orders: return None
        return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # Engine Compliance: Initialize the required return dictionary
        orders: dict[Symbol, list[Order]] = {sym: [] for sym in self.limits.keys()}
        conversions = 0
        trader_data = json.loads(state.traderData) if state.traderData else {}

        vf_mid = self.get_mid(state, VEV_SPOT)
        if not vf_mid:
            # Safe exit if underlying order book is empty
            return orders, conversions, json.dumps(trader_data)

        tte_years = max((ROUND_START_TTE_DAYS - state.timestamp / TICKS_PER_DAY), 0.001) / 365.0

        # =====================================================================
        # PHASE 1: AGGREGATE DELTA ORDER BOOK IMBALANCE (AD-OBI)
        # =====================================================================
        aggregate_market_delta_demand = 0.0
        
        for s, sym in zip(STRIKES, VOUCHER_SYMBOLS):
            od = state.order_depths.get(sym)
            if not od: continue
            
            # Rough delta estimate using a static vol (0.5) to weight the book
            strike_delta = bs_call_delta(vf_mid, float(s), tte_years, 0.5)
            
            bid_vol = sum(od.buy_orders.values())
            ask_vol = sum(abs(v) for v in od.sell_orders.values())
            
            # Net pressure on this specific strike
            net_vol_pressure = bid_vol - ask_vol 
            aggregate_market_delta_demand += (net_vol_pressure * strike_delta)

        # =====================================================================
        # PHASE 2: PREDATORY VELVETFRUIT FRONT-RUNNING
        # =====================================================================
        vf_od = state.order_depths[VEV_SPOT]
        vf_pos = state.position.get(VEV_SPOT, 0)
        
        # Predictive Target based on AD-OBI
        predictive_target = np.clip(int(aggregate_market_delta_demand * 2.5), -self.limits[VEV_SPOT], self.limits[VEV_SPOT])
        
        # Local Mean Reversion OBI
        vf_bid_vol = sum(vf_od.buy_orders.values())
        vf_ask_vol = sum(abs(v) for v in vf_od.sell_orders.values())
        vf_obi = (vf_bid_vol - vf_ask_vol) / (vf_bid_vol + vf_ask_vol + 1)
        local_obi_target = int(vf_obi * 50)
        
        final_vf_target = np.clip(predictive_target + local_obi_target, -self.limits[VEV_SPOT], self.limits[VEV_SPOT])
        to_trade_vf = final_vf_target - vf_pos

        # Aggressively cross the spread to front-run the AMM hedge
        if to_trade_vf > 15: 
            for price, vol in sorted(vf_od.sell_orders.items()):
                if to_trade_vf <= 0: break
                take = min(to_trade_vf, abs(vol))
                orders[VEV_SPOT].append(Order(VEV_SPOT, price, take))
                to_trade_vf -= take
                
        elif to_trade_vf < -15: 
            for price, vol in sorted(vf_od.buy_orders.items(), reverse=True):
                if to_trade_vf >= 0: break
                take = min(abs(to_trade_vf), abs(vol))
                orders[VEV_SPOT].append(Order(VEV_SPOT, price, -take))
                to_trade_vf += take

        # =====================================================================
        # PHASE 3: HYDROGEL (Anchored Microprice Maker)
        # =====================================================================
        hg_od = state.order_depths.get("HYDROGEL_PACK")
        if hg_od and hg_od.buy_orders and hg_od.sell_orders:
            hg_mid = self.get_mid(state, "HYDROGEL_PACK")
            hg_pos = state.position.get("HYDROGEL_PACK", 0)
            
            best_bid = max(hg_od.buy_orders.keys())
            best_ask = min(hg_od.sell_orders.keys())
            bid_vol = hg_od.buy_orders[best_bid]
            ask_vol = abs(hg_od.sell_orders[best_ask])
            total_vol = bid_vol + ask_vol
            
            microprice = (best_bid * ask_vol + best_ask * bid_vol) / total_vol if total_vol > 0 else hg_mid
            true_value = 0.85 * microprice + 0.15 * 10000.0 # Restored 10k anchor
            
            # Aggressive Linear Avellaneda Skew
            inventory_ratio = hg_pos / self.limits["HYDROGEL_PACK"]
            skew = inventory_ratio * 10.0 
            skewed_true_value = true_value - skew
            dynamic_width = 1.5 + (abs(inventory_ratio) * 3.5)
            
            max_buy_price = math.floor(skewed_true_value - dynamic_width)
            min_sell_price = math.ceil(skewed_true_value + dynamic_width)

            to_buy_hg = min(self.limits["HYDROGEL_PACK"] - hg_pos, MAX_CLIP)
            to_sell_hg = min(self.limits["HYDROGEL_PACK"] + hg_pos, MAX_CLIP)

            for price, volume in sorted(hg_od.sell_orders.items()):
                if to_buy_hg > 0 and price <= max_buy_price:
                    qty = min(to_buy_hg, -volume)
                    orders["HYDROGEL_PACK"].append(Order("HYDROGEL_PACK", price, qty))
                    to_buy_hg -= qty

            for price, volume in sorted(hg_od.buy_orders.items(), reverse=True):
                if to_sell_hg > 0 and price >= min_sell_price:
                    qty = min(to_sell_hg, volume)
                    orders["HYDROGEL_PACK"].append(Order("HYDROGEL_PACK", price, -qty))
                    to_sell_hg -= qty

            if to_buy_hg > 0:
                price = next((p + 1 for p, _ in sorted(hg_od.buy_orders.items(), reverse=True) if p < max_buy_price), max_buy_price)
                orders["HYDROGEL_PACK"].append(Order("HYDROGEL_PACK", int(price), to_buy_hg))
            if to_sell_hg > 0:
                price = next((p - 1 for p, _ in sorted(hg_od.sell_orders.items()) if p > min_sell_price), min_sell_price)
                orders["HYDROGEL_PACK"].append(Order("HYDROGEL_PACK", int(price), -to_sell_hg))

        # Engine Compliance: Return strict tuple format
        return orders, conversions, json.dumps(trader_data)