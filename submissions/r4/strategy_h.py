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


def autocorr_1lag(series: list[float]) -> float:
    if len(series) < 2:
        return 0.0
    r = np.corrcoef(series[:-1], series[1:])[0, 1]
    return float(r) if not math.isnan(r) else 0.0


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


# ── Voucher constants ────────────────────────────────────────────────────────

MAX_CLIP = 40

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHER_SYMBOLS = [f"VEV_{k}" for k in STRIKES]
VEV_SPOT = "VELVETFRUIT_EXTRACT"
ROUND_START_TTE_DAYS = 4.0
TICKS_PER_DAY = 1_000_000

# ── Mark 14 follower constants ───────────────────────────────────────────────

# EDA (submissions/r4/eda_mark_bots.py): Mark 14 is the only bot with
# |lead_5_corr|>0.15 and |net_dir|>0.3. Counterparty name format is
# space-separated zero-padded ("Mark 14"), not "Mark_14".
MARK14_INFORMED_BOT = "Mark 14"

# Only Mark 14 product with positive lead-corr (+0.225). VEV_5400/5500 had
# negative corr (likely Mark 14 pushing illiquid OTM strikes that revert).
MARK14_SIGNAL_SYMBOL = "VEV_5300"

# EDA lead-5 corr peak. Price rows are 100 timestamp units apart, so 5 rows
# = 500 ticks. Matches TimoDiehm's KELP Olivia-window of 500 ticks.
MARK14_WINDOW_TICKS = 500

# 1/3 of the 300 position limit. The +0.225 lead-5 corr is modest — full
# conviction is not justified. Scale up if backtest is consistently positive
# across all 3 days.
MARK14_TARGET_SIZE = 100

# ── Mark 14 Informed-MM bias constants ───────────────────────────────────────

# Per-side lots posted each tick on VEV_5300/5400/5500. ~5 matches Mark 14's
# typical trade clip; 10 sits naturally in queue without dwarfing it.
MARK14_MM_QUOTE_SIZE = 10

# Hard cap on per-tick fills per side. Prevents single sweep from blowing past
# max_pos.
MARK14_MM_MAX_CLIP = 15

# Inventory skew constant (Hydrogel pattern: inv_skew = -(pos/limit) * inv_k).
# Half of HydrogelStrategy's 10 — voucher books are thinner; start cautious.
MARK14_MM_INV_K = 5.0

# Spread asymmetry tilt magnitude (in ticks). When in window, the target side's
# offset becomes (W - delta) and the away side's becomes (W + delta).
MARK14_MM_DELTA = 1.0

# Base symmetric half-width when out of window. With delta=1 and W=1, the
# tightened side is at offset 0 (touch) and the loosened side at offset 2.
MARK14_MM_W = 1.0


# ── Voucher IV smile scalper ─────────────────────────────────────────────────

class VoucherStrategy(StatefulStrategy[dict[str, Any]]):
    def __init__(self, symbol: str,
                 limit: int, strike: int, k: float = 300.0, min_residual: float = 0.01,
                 max_otm_moneyness: float = 1.020,
                 carry_window: int = 100, carry_threshold: float = 0.020,
                 autocorr_window: int = 30, autocorr_threshold: float = -0.05) -> None:
        super().__init__(symbol, limit)
        self.strike = strike
        self.k = k
        self.min_residual = min_residual
        self.max_otm_moneyness = max_otm_moneyness
        self.carry_window = carry_window
        self.carry_threshold = carry_threshold
        self.autocorr_window = autocorr_window
        self.autocorr_threshold = autocorr_threshold
        self.residual_history: list[float] = []

    def _apply_carry(self, scalper_target: int) -> int:
        if len(self.residual_history) < self.carry_window:
            return scalper_target
        mean_residual = float(np.mean(self.residual_history))
        if mean_residual > self.carry_threshold:
            return min(scalper_target, 0)
        if mean_residual < -self.carry_threshold:
            return max(scalper_target, 0)
        return scalper_target

    def save(self) -> dict[str, Any]:
        return {"residual_history": self.residual_history}

    def load(self, data: dict[str, Any]) -> None:
        self.residual_history = data.get("residual_history", [])

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

        # Update carry layer history (always, even if dead-band filters this tick)
        self.residual_history.append(residual)
        if len(self.residual_history) > self.carry_window:
            self.residual_history.pop(0)

        if abs(residual) < self.min_residual:
            return
        if len(self.residual_history) < self.autocorr_window:
            return
        if autocorr_1lag(self.residual_history[-self.autocorr_window:]) >= self.autocorr_threshold:
            return
        position = state.position.get(self.symbol, 0)
        scalper_target = int(np.clip(-self.k * residual, -self.limit, self.limit))
        target = self._apply_carry(scalper_target)

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


# ── Mark 14 follower ─────────────────────────────────────────────────────────

class Mark14FollowerStrategy(StatefulStrategy[dict]):
    """Mark 14 follower with direction-aware passive entry.

    direction:
        "follow_passive": when in window, post BUY at best_bid (our touch).
                          No spread cross. Cancel-and-repost each tick.
        "fade_at_touch":  when in window, post SELL at best_ask.

    size: target |position| while in window.

    Out-of-window: target = 0; reconcile passively (post at our touch).
    Strikes absent from mark14_config fall through to VoucherStrategy.
    """

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
        # Day-boundary guard: timestamp resets each day → drop stale ts.
        if self.last_signal_ts is not None and state.timestamp < self.last_signal_ts:
            self.last_signal_ts = None

        # Detect signal: scan market_trades for Mark 14 buys on this symbol.
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
        else:  # fade_at_touch
            target = -self.size if in_window else 0

        position = state.position.get(self.symbol, 0)
        delta = target - position
        # Clamp delta so the resulting position never exceeds [-limit, +limit].
        delta = max(-self.limit - position, min(self.limit - position, delta))
        od = state.order_depths[self.symbol]

        if delta > 0:
            best_bid = max(od.buy_orders.keys())
            self.buy(best_bid, delta)
        elif delta < 0:
            best_ask = min(od.sell_orders.keys())
            self.sell(best_ask, -delta)


# ── Mark 14 Informed-MM bias ─────────────────────────────────────────────────


class Mark14InformedMMStrategy(StatefulStrategy[dict]):
    """Pure-passive market maker on illiquid OTM voucher strikes with quotes
    biased by Mark 14 signal direction.

    When in window (recent Mark 14 buy on this symbol), the bid/ask offsets
    around fair value become asymmetric per the strike's predicted price
    direction:
      - predicted UP  (e.g. VEV_5300, lead-5 +0.225): tighten bid + loosen ask
      - predicted DOWN (e.g. VEV_5400/5500, lead-5 < 0): tighten ask + loosen bid

    Out of window: symmetric quotes.

    Inventory skew (Hydrogel pattern) stacks on top: the fair value anchor
    shifts opposite the position to drive unwind.

    Never crosses the spread.
    """

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
        # Day-boundary guard: timestamp resets each day → drop stale ts.
        if self.last_signal_ts is not None and state.timestamp < self.last_signal_ts:
            self.last_signal_ts = None

        # Detect signal: scan market_trades for Mark 14 buys on this symbol.
        trades = state.market_trades.get(self.symbol, [])
        mark14_buy_ts = [t.timestamp for t in trades if t.buyer == MARK14_INFORMED_BOT]
        if mark14_buy_ts:
            self.last_signal_ts = max(mark14_buy_ts)

        in_window = (
            self.last_signal_ts is not None
            and state.timestamp - self.last_signal_ts <= self.window_ticks
        )

        # Layer 1: fair value blend.
        fair = self._compute_fair(state)
        if fair is None:
            return

        # Layer 3: inventory skew on the anchor.
        position = state.position.get(self.symbol, 0)
        inv_skew = self._compute_inv_skew(position)
        fair_used = fair + inv_skew

        # Layer 4: spread asymmetry on the offsets.
        bid_offset, ask_offset = self._compute_offsets(in_window)

        # Layer 5: quote prices (with book clamp) + sizes (with cap).
        od = state.order_depths[self.symbol]
        bid_price, ask_price = self._compute_quote_prices(
            od, fair_used, bid_offset, ask_offset,
        )
        bid_qty, ask_qty = self._compute_quote_sizes(position)

        if bid_qty > 0:
            self.buy(bid_price, bid_qty)
        if ask_qty > 0:
            self.sell(ask_price, ask_qty)

    def _compute_offsets(self, in_window: bool) -> tuple[float, float]:
        """Return (bid_offset, ask_offset). Out-of-window: symmetric.
        In-window: tighten the side aligned with predicted direction, loosen
        the other.

        Predicted UP  → tighten BID (raise it, more competitive on buy side).
        Predicted DOWN → tighten ASK (lower it, more competitive on sell side).
        """
        if not in_window:
            return self.base_w, self.base_w
        if self.predicted_dir == "UP":
            return self.base_w - self.delta, self.base_w + self.delta
        # DOWN
        return self.base_w + self.delta, self.base_w - self.delta

    def _compute_inv_skew(self, position: int) -> float:
        """Hydrogel-style inventory skew: long position → negative skew (pulls
        anchor down → bid less competitive, ask more competitive → unwind)."""
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
        """Fit quadratic IV smile across all available voucher strikes, then
        compute Black-Scholes call price for our strike at the fitted IV.
        Returns None when the fit has fewer than 3 valid points.

        Note: this duplicates a few lines from VoucherStrategy.act() to keep
        VoucherStrategy untouched per the spec.
        """
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
        """fair = 0.85 * microprice + 0.15 * smile_theo, falling back to pure
        microprice when smile is unfittable."""
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
        """Compute integer-tick bid/ask prices from the shifted fair value
        and the bid/ask offsets. Clamp inside the existing book so we never
        cross — pure passive."""
        bid_price = floor(fair_used - bid_offset)
        ask_price = ceil(fair_used + ask_offset)
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_price = min(bid_price, best_ask - 1)
        ask_price = max(ask_price, best_bid + 1)
        return bid_price, ask_price

    def _compute_quote_sizes(self, position: int) -> tuple[int, int]:
        """Per-side quote quantities, capped by quote_size, max_clip, and the
        remaining headroom to ±max_pos. Position cap is hard."""
        room_buy = self.max_pos - position
        room_sell = self.max_pos + position
        bid_qty = max(0, min(self.quote_size, self.max_clip, room_buy))
        ask_qty = max(0, min(self.quote_size, self.max_clip, room_sell))
        return bid_qty, ask_qty


# ── VEV_4000 inside-spread MM ────────────────────────────────────────────────


class Vev4000MMStrategy(Strategy):
    """Stateless inside-spread market maker for VEV_4000.

    Quotes are anchored to the observed touch (best_bid + offset / best_ask -
    offset), not to a fair-value formula. Inventory protection is a three-region
    state machine on the position:

        |pos| <= band_inner           : quote both sides
        band_inner < pos <= band_cap  : ask only (reduce long)
        -band_cap <= pos < -band_inner: bid only (reduce short)
        |pos| > band_cap              : no quotes

    Per-side quantity = min(max_per_tick, remaining_band_capacity).

    This is intentionally NOT Avellaneda-style: there is no risk-aversion
    coefficient, no width function, no volatility input. The user vetoed
    continuous inventory skew after prior submissions performed poorly with
    that pattern (see cerebrum 2026-04-27).
    """

    def __init__(
        self,
        symbol: str,
        limit: int,
        offset: int = 5,
        band_inner: int = 20,
        band_cap: int = 30,
        max_per_tick: int = 10,
    ) -> None:
        super().__init__(symbol, limit)
        self.offset = offset
        self.band_inner = band_inner
        self.band_cap = band_cap
        self.max_per_tick = max_per_tick

    def _compute_sizes(self, position: int) -> tuple[int, int]:
        """Return (bid_qty, ask_qty) for this tick given the current position."""
        if abs(position) > self.band_cap:
            return 0, 0

        bid_qty = max(0, min(self.max_per_tick, self.band_cap - position))
        ask_qty = max(0, min(self.max_per_tick, self.band_cap + position))

        if position > self.band_inner:
            bid_qty = 0
        elif position < -self.band_inner:
            ask_qty = 0

        return bid_qty, ask_qty

    def _compute_prices(self, od: OrderDepth) -> tuple[int, int]:
        """Return (bid_price, ask_price) — inside-spread when the spread is
        wide enough, at-touch otherwise. Never crosses.

        Precondition: both sides of `od` are non-empty. The base
        `Strategy.run()` guards this before calling `act()`.
        """
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        spread = best_ask - best_bid
        # Need `spread - 2*offset >= 1` for inside-spread quotes to leave at
        # least one tick of gap between them.
        if spread - 2 * self.offset >= 1:
            return best_bid + self.offset, best_ask - self.offset
        return best_bid, best_ask

    def act(self, state: TradingState) -> None:
        od = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)

        bid_price, ask_price = self._compute_prices(od)
        bid_qty, ask_qty = self._compute_sizes(position)

        if bid_qty > 0:
            self.buy(bid_price, bid_qty)
        if ask_qty > 0:
            self.sell(ask_price, ask_qty)


# ── VelvetfruitExtract z-score signal ────────────────────────────────────────


class VelvetfruitSignalStrategy(SignalStrategy, StatefulStrategy[dict[str, Any]]):
    """Aggressive mean-reversion signal strategy for VELVETFRUIT_EXTRACT.

    Computes a smoothed rolling z-score of mid prices. Crosses the spread on
    deviation (buys at best_ask on LONG, sells at best_bid on SHORT). Returns
    Signal.NEUTRAL (not None) when score reverts between thresholds — this
    flattens position on reversion rather than holding until the opposite
    signal fires (correct for a lag-1 autocorr signal).

    All five parameters are Optuna-tunable. Starting defaults from jmerle's
    VolcanicRock (P3 underlying equivalent): zscore_period=75,
    smoothing_period=100, threshold=0.5.
    """

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
            return None  # warmup: keep last signal, don't update

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
        return Signal.NEUTRAL  # explicit exit — closes position on reversion

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


# ── Delta-1 products ─────────────────────────────────────────────────────────

class HydrogelStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> float:
        return self.get_mid_price(state, self.symbol)

    def act(self, state: TradingState) -> None:
        od = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_vol = od.buy_orders[best_bid]
        ask_vol = abs(od.sell_orders[best_ask])
        total_vol = bid_vol + ask_vol

        microprice = (
            (best_bid * ask_vol + best_ask * bid_vol) / total_vol
            if total_vol > 0
            else (best_bid + best_ask) / 2.0
        )
        base_value = 0.85 * microprice + 0.15 * 10_000
        inventory_ratio = position / self.limit
        skewed_value = base_value - inventory_ratio * 10.0
        dynamic_width = 1.5 + abs(inventory_ratio) * 3.5

        max_buy_price = math.floor(skewed_value - dynamic_width)
        min_sell_price = math.ceil(skewed_value + dynamic_width)

        buy_orders = sorted(od.buy_orders.items(), reverse=True)
        sell_orders = sorted(od.sell_orders.items())

        to_buy = min(self.limit - position, MAX_CLIP)
        to_sell = min(self.limit + position, MAX_CLIP)

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                qty = min(to_buy, -volume)
                self.buy(price, qty)
                to_buy -= qty

        if to_buy > 0:
            price = next(
                (p + 1 for p, _ in buy_orders if p < max_buy_price),
                max_buy_price,
            )
            self.buy(int(price), to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                qty = min(to_sell, volume)
                self.sell(price, qty)
                to_sell -= qty

        if to_sell > 0:
            price = next(
                (p - 1 for p, _ in sell_orders if p > min_sell_price),
                min_sell_price,
            )
            self.sell(int(price), to_sell)


class VelvetfruitStrategy(MarketMakingStrategy):
    """Neutral mid-price MM. Backtests at 0 (passive orders never fill in
    historical data). Kept as safe fallback — does not accumulate directional
    inventory risk unlike aggressive strategies."""

    def get_true_value(self, state: TradingState) -> float:
        return self.get_mid_price(state, self.symbol)


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

        # Direction + size per strike from submissions/r4/mark14_direction.md.
        # Strikes absent from this dict fall through to VoucherStrategy.
        # To exclude a strike entirely, remove its entry.
        #
        # Track B (Mark14InformedMMStrategy) was implemented and tested on
        # VEV_5300/5400/5500 on 2026-04-27. Backtest produced 0.00 fills on
        # all three strikes — identical to Track A's default-mode result.
        # Diagnosis: spread=1 books on 5400/5500 leave no room for spread-
        # asymmetric quoting to differentiate from at-touch quotes; structural
        # smile-theo bias pushes blended fair off mid; queue priority means
        # at-touch quotes don't fill on these illiquid OTM books. Reverted
        # to Track A. The InformedMM class is retained in this file for
        # revert-ability and future reference. See cerebrum Decision Log
        # [2026-04-27] and submissions/r4/backtest_summary.md for full notes.
        mark14_config = {
            "VEV_5300": {"direction": "follow_passive", "size": 50},
            "VEV_5400": {"direction": "fade_at_touch", "size": 60},
            "VEV_5500": {"direction": "fade_at_touch", "size": 25},
        }

        self.strategies: dict[Symbol, Strategy] = {
            "HYDROGEL_PACK": HydrogelStrategy("HYDROGEL_PACK", limits["HYDROGEL_PACK"]),
            "VELVETFRUIT_EXTRACT": VelvetfruitStrategy(
                "VELVETFRUIT_EXTRACT", limits["VELVETFRUIT_EXTRACT"]
            ),
        }

        for strike in STRIKES:
            sym = f"VEV_{strike}"
            if sym == "VEV_4000":
                # Inside-spread MM with hard position bands. Wide-spread, deep-ITM
                self.strategies[sym] = Vev4000MMStrategy(sym, limits[sym])
                continue
            cfg = mark14_config.get(sym)
            if cfg is not None:
                self.strategies[sym] = Mark14FollowerStrategy(
                    sym, limits[sym],
                    direction=cfg["direction"],
                    size=cfg["size"],
                )
            else:
                self.strategies[sym] = VoucherStrategy(
                    sym, limits[sym], strike,
                    k=150, min_residual=0.01, max_otm_moneyness=0.996,
                    carry_window=100, carry_threshold=0.020,
                    autocorr_window=30, autocorr_threshold=-0.05,
                )

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