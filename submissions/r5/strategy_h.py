import json
from abc import abstractmethod
from math import ceil, floor
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

type JSON = dict[str, Any] | list[Any] | str | int | float | bool | None


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            [],
            [],
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

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
            encoded_candidate = json.dumps(candidate)
            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


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
        self.orders = list[Order]()
        self.conversions = 0

        if all(
            (
                v in state.order_depths
                and len(state.order_depths[v].buy_orders) > 0
                and len(state.order_depths[v].sell_orders) > 0
            )
            for v in self.get_required_symbols()
        ):
            self.act(state)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2


class StatefulStrategy[T: JSON](Strategy):
    @abstractmethod
    def save(self) -> T:
        raise NotImplementedError()

    @abstractmethod
    def load(self, data: T) -> None:
        raise NotImplementedError()


class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

    @abstractmethod
    def get_true_value(self, state: TradingState) -> float:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        max_buy_price = int(true_value) - 1 if true_value % 1 == 0 else floor(true_value)
        min_sell_price = int(true_value) + 1 if true_value % 1 == 0 else ceil(true_value)

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0:
            price = next((price + 1 for price, _ in buy_orders if price < max_buy_price), max_buy_price)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0:
            price = next((price - 1 for price, _ in sell_orders if price > min_sell_price), min_sell_price)
            self.sell(price, to_sell)


class R5BaseMMStrategy(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int, width: int = 1) -> None:
        super().__init__(symbol, limit)
        self.width = width

    def _microprice(self, state: TradingState) -> float:
        order_depth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        bid_vol = order_depth.buy_orders[best_bid]
        ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

    def get_true_value(self, state: TradingState) -> float:
        return self._microprice(state)

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        max_buy_price = int(true_value) - 1 if true_value % 1 == 0 else floor(true_value)
        min_sell_price = int(true_value) + 1 if true_value % 1 == 0 else ceil(true_value)

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0:
            passive_buy = max_buy_price - self.width + 1
            self.buy(passive_buy, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0:
            passive_sell = min_sell_price + self.width - 1
            self.sell(passive_sell, to_sell)


class R5AutocorrMMStrategy(R5BaseMMStrategy, StatefulStrategy[dict[str, Any]]):
    def __init__(self, symbol: Symbol, limit: int, width: int, alpha: float) -> None:
        super().__init__(symbol, limit, width)
        self.alpha = alpha
        self.last_mid: float | None = None

    def get_true_value(self, state: TradingState) -> float:
        base = self._microprice(state)
        order_depth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        current_mid = (best_bid + best_ask) / 2

        if self.last_mid is not None:
            last_return = (current_mid - self.last_mid) / self.last_mid
            base -= self.alpha * last_return * base
        self.last_mid = current_mid
        return base

    def save(self) -> dict[str, Any]:
        return {"last_mid": self.last_mid}

    def load(self, data: dict[str, Any]) -> None:
        v = data.get("last_mid")
        self.last_mid = float(v) if v is not None else None


class R5CorrMMStrategy(R5BaseMMStrategy, StatefulStrategy[dict[str, Any]]):
    def __init__(self, symbol: Symbol, limit: int, width: int, partner_symbol: Symbol, beta: float) -> None:
        super().__init__(symbol, limit, width)
        self.partner_symbol = partner_symbol
        self.beta = beta
        self.last_partner_mid: float | None = None

    def get_true_value(self, state: TradingState) -> float:
        base = self._microprice(state)
        partner_depth = state.order_depths.get(self.partner_symbol)
        if partner_depth is None or not partner_depth.buy_orders or not partner_depth.sell_orders:
            self.last_partner_mid = None
            return base

        partner_bid = max(partner_depth.buy_orders.keys())
        partner_ask = min(partner_depth.sell_orders.keys())
        partner_mid = (partner_bid + partner_ask) / 2

        if self.last_partner_mid is not None:
            partner_return = (partner_mid - self.last_partner_mid) / self.last_partner_mid
            base -= self.beta * partner_return * base
        self.last_partner_mid = partner_mid
        return base

    def save(self) -> dict[str, Any]:
        return {"last_partner_mid": self.last_partner_mid}

    def load(self, data: dict[str, Any]) -> None:
        v = data.get("last_partner_mid")
        self.last_partner_mid = float(v) if v is not None else None


class R5XLSkewMMStrategy(R5BaseMMStrategy, StatefulStrategy[dict[str, Any]]):
    """
    Asymmetric one-sided skew based on partner (XL) recent return.

    When |partner_return| <= threshold: standard symmetric quoting at width.
    When partner_return < -threshold (partner dropped, expect own to rise):
        bid raised by k_ticks above standard anchor; ask unchanged at standard.
    When partner_return > +threshold (partner rose, expect own to drop):
        ask lowered by k_ticks below standard anchor; bid unchanged at standard.

    Distinct from R5CorrMMStrategy (which moves both quotes together) — this
    commits to taking inventory in the predicted direction.
    """

    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        width: int,
        partner_symbol: Symbol,
        threshold: float,
        k_ticks: int,
    ) -> None:
        super().__init__(symbol, limit, width)
        self.partner_symbol = partner_symbol
        self.threshold = threshold
        self.k_ticks = k_ticks
        self.last_xl_mid: float | None = None

    def _partner_return(self, state: TradingState) -> float | None:
        partner_depth = state.order_depths.get(self.partner_symbol)
        if partner_depth is None or not partner_depth.buy_orders or not partner_depth.sell_orders:
            self.last_xl_mid = None
            return None
        partner_bid = max(partner_depth.buy_orders.keys())
        partner_ask = min(partner_depth.sell_orders.keys())
        partner_mid = (partner_bid + partner_ask) / 2
        if self.last_xl_mid is None:
            self.last_xl_mid = partner_mid
            return None
        ret = (partner_mid - self.last_xl_mid) / self.last_xl_mid
        self.last_xl_mid = partner_mid
        return ret

    def act(self, state: TradingState) -> None:
        true_value = self._microprice(state)
        partner_return = self._partner_return(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        max_buy_price = int(true_value) - 1 if true_value % 1 == 0 else floor(true_value)
        min_sell_price = int(true_value) + 1 if true_value % 1 == 0 else ceil(true_value)

        # Standard anchors (match R5BaseMMStrategy)
        passive_buy = max_buy_price - self.width + 1
        passive_sell = min_sell_price + self.width - 1

        # Asymmetric skew
        if partner_return is not None and abs(partner_return) > self.threshold:
            if partner_return < 0:
                passive_buy = passive_buy + self.k_ticks
            else:
                passive_sell = passive_sell - self.k_ticks

        # Take liquidity at or better than fair (standard MM behavior)
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0:
            self.buy(passive_buy, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0:
            self.sell(passive_sell, to_sell)

    def save(self) -> dict[str, Any]:
        return {"last_xl_mid": self.last_xl_mid}

    def load(self, data: dict[str, Any]) -> None:
        v = data.get("last_xl_mid")
        self.last_xl_mid = float(v) if v is not None else None


class R5PairTradeStrategy(StatefulStrategy[dict[str, Any]]):
    """
    Z-score spread trade on a pair (symbol_a − symbol_b).

    Tracks rolling spread of size `window`. Once full, computes z = (spread − mean) / std.
    Entry at |z| > z_entry: positive z -> short A, long B at full limit. Negative z -> opposite.
    Exit at |z| < z_exit OR held_ticks > max_hold_ticks. Flatten both legs.

    Returns orders for both symbols. Trader.run dispatch must accumulate
    (orders.setdefault(symbol, []).append(order)).
    """

    def __init__(
        self,
        symbol_a: Symbol,
        symbol_b: Symbol,
        limit: int,
        window: int,
        z_entry: float,
        z_exit: float,
        max_hold_ticks: int,
        limit_b: int | None = None,
    ) -> None:
        super().__init__(symbol_a, limit)
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.window = window
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.max_hold_ticks = max_hold_ticks
        # Per-leg sizing: limit_b defaults to limit for symmetric pair trades.
        # Asymmetric sizing (e.g. PEBBLES_M=5, PEBBLES_XL=10) is used when leg
        # decomposition shows one leg drives all alpha — see Phase B audit.
        self.limit_b = limit_b if limit_b is not None else limit
        self.spread_history: list[float] = []
        self.entry_tick: int | None = None

    def get_required_symbols(self) -> list[Symbol]:
        return [self.symbol_a, self.symbol_b]

    def _mid(self, state: TradingState, symbol: Symbol) -> float:
        od = state.order_depths[symbol]
        return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2

    def _take(self, state: TradingState, symbol: Symbol, side: int, qty: int) -> None:
        """Cross the spread. side = +1 buy at ask, -1 sell at bid."""
        od = state.order_depths[symbol]
        if side > 0:
            price = min(od.sell_orders.keys())
            self.orders.append(Order(symbol, price, qty))
        else:
            price = max(od.buy_orders.keys())
            self.orders.append(Order(symbol, price, -qty))

    def _flatten(self, state: TradingState, symbol: Symbol, position: int) -> None:
        """Reverse an existing position: sell at bid if long, buy at ask if short."""
        if position > 0:
            self._take(state, symbol, side=-1, qty=position)
        elif position < 0:
            self._take(state, symbol, side=+1, qty=-position)

    def act(self, state: TradingState) -> None:
        mid_a = self._mid(state, self.symbol_a)
        mid_b = self._mid(state, self.symbol_b)
        spread = mid_a - mid_b

        self.spread_history.append(spread)
        if len(self.spread_history) > self.window:
            self.spread_history.pop(0)

        pos_a = state.position.get(self.symbol_a, 0)
        pos_b = state.position.get(self.symbol_b, 0)
        is_flat = pos_a == 0 and pos_b == 0

        # entry_tick is live iff we hold the pair. Reconcile at top of tick so
        # partial-fill exits don't strand the timer — residual position keeps it,
        # and a confirmed-flat tick clears it.
        if is_flat:
            self.entry_tick = None
        elif self.entry_tick is None:
            # Cold-start with a carried position (e.g. --carry without --persist).
            # Best we can do is start the hold timer now.
            self.entry_tick = state.timestamp

        # Time-stop is a risk-management override: fires regardless of z computability
        if not is_flat:
            held_delta = state.timestamp - self.entry_tick
            if held_delta > self.max_hold_ticks * 100:
                self._flatten(state, self.symbol_a, pos_a)
                self._flatten(state, self.symbol_b, pos_b)
                return

        if len(self.spread_history) < self.window:
            return  # warming up

        mean = sum(self.spread_history) / len(self.spread_history)
        var = sum((s - mean) ** 2 for s in self.spread_history) / len(self.spread_history)
        std = var ** 0.5
        if std == 0:
            return  # no signal — flat spread

        z = (spread - mean) / std

        if is_flat:
            if z > self.z_entry:
                self._take(state, self.symbol_a, side=-1, qty=self.limit)
                self._take(state, self.symbol_b, side=+1, qty=self.limit_b)
                self.entry_tick = state.timestamp
            elif z < -self.z_entry:
                self._take(state, self.symbol_a, side=+1, qty=self.limit)
                self._take(state, self.symbol_b, side=-1, qty=self.limit_b)
                self.entry_tick = state.timestamp
        else:
            if abs(z) < self.z_exit:
                self._flatten(state, self.symbol_a, pos_a)
                self._flatten(state, self.symbol_b, pos_b)

    def save(self) -> dict[str, Any]:
        return {"spread_history": list(self.spread_history), "entry_tick": self.entry_tick}

    def load(self, data: dict[str, Any]) -> None:
        self.spread_history = list(data.get("spread_history", []))
        v = data.get("entry_tick")
        self.entry_tick = int(v) if v is not None else None


# Phase A drops (2026-04-30): 9 alpha-negative products removed from wiring.
# Drop list (alpha = conservative-backtest PnL, all 3 days combined):
#   TIGHT  — SLEEP_POD_LAMB_WOOL (-36,725), PANEL_1X2 (-30,661),
#            TRANSLATOR_SPACE_GRAY (-16,719), PANEL_4X4 (-16,387),
#            TRANSLATOR_GRAPHITE_MIST (-13,946)
#   MEDIUM — GALAXY_SOUNDS_SOLAR_FLAMES (-23,644), UV_VISOR_MAGENTA (-22,796),
#            OXYGEN_SHAKE_MINT (-14,704), OXYGEN_SHAKE_GARLIC (-9,630)
# Expected swing: +185k conservative / +115k default. See CLAUDE.md decision log.
#
# UV_VISOR dive (2026-04-30): EG kill criterion fired — no pair p < 0.05 across all 3 days.
# ORANGE (-526 conservative), RED (-3,893), YELLOW (-5,625) dropped from MEDIUM_TIER.
# AMBER (+7,878 conservative) retained at width=2. Group exhausted.
#
# PANEL dive (2026-04-30): EG kill criterion fired — no pair p < 0.05 across all 3 days.
# All pair trade variants confirmed negative in backtest. PANEL_2X2 dropped:
# conservative -4,433 (Days 3+4 systematic losses). 1X4 and 2X4 retained at width=1.
#
# SLEEP_POD dive (2026-04-30): EG kill criterion fired (0/6 pairs cointegrated).
# SUEDE width=2 rescues conservative (+4,438 delta, −357 → +4,081). NYLON also improves
# at width=2 (+1,724 delta). POLYESTER (best −1,601 conservative) and COTTON (best −2,134)
# dropped — net-negative at every tested width. Per-product wiring in Group Sleep Pod block.
TIGHT_TIER = [
    "ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES", "ROBOT_LAUNDRY", "ROBOT_IRONING",
    "TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_VOID_BLUE",
    "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE", "MICROCHIP_RECTANGLE",
    "MICROCHIP_TRIANGLE",
    "SLEEP_POD_NYLON", "SLEEP_POD_SUEDE",
    "PANEL_1X4", "PANEL_2X4",
]

MEDIUM_TIER = [
    "OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH", "OXYGEN_SHAKE_CHOCOLATE",
    "PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL",
    "UV_VISOR_AMBER",  # YELLOW/ORANGE/RED dropped: EG kill criterion fired 2026-04-30
    "GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_SOLAR_WINDS",
]

WIDE_TIER = [
    "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO",
    "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
]

LIMIT = 10

# Parameter safeguards (reference):
#   beta  — use OLS from returns regression: beta = corr(own, partner) * (std_own / std_partner).
#            Negative for anti-correlated pairs (CHOC/VAN, RASP/STRAW), positive for co-moving pairs (PIST/STRAW).
#   width — set from spread economics: width >= 2 * sigma_microprice_error (how far microprice drifts
#            from realized fill mid). Wider = less adverse selection, fewer fills. Do not tune to PnL.
class Trader:
    def __init__(self) -> None:
        self.strategies: dict[Symbol, Strategy] = {}

        # ── Tight tier (width=1) ─────────────────────────────────────────
        for sym in TIGHT_TIER:
            if sym.startswith("ROBOT_"):
                continue  # per-product widths wired in Group Robots block below
            if sym.startswith("MICROCHIP_"):
                continue  # per-product widths wired in Group Microchip block below
            if sym.startswith("SLEEP_POD_"):
                continue  # per-product widths wired in Group Sleep Pod block below
            self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=1)

        # ── Medium tier (width=2) ────────────────────────────────────────
        for sym in MEDIUM_TIER:
            if sym == "OXYGEN_SHAKE_EVENING_BREATH":
                self.strategies[sym] = R5AutocorrMMStrategy(sym, LIMIT, width=2, alpha=0.118)
            elif sym == "OXYGEN_SHAKE_CHOCOLATE":
                self.strategies[sym] = R5AutocorrMMStrategy(sym, LIMIT, width=2, alpha=0.082)
            elif sym == "GALAXY_SOUNDS_PLANETARY_RINGS":
                self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=3)
            elif sym.startswith("PEBBLES_"):
                continue # PEBBLES pairs and skew wired in PEBBLES pair trade and skew blocks below
            else:
                self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=2)

        # ── PEBBLES pair trade (M ↔ XL) ──────────────────────────────────
        # Registered under "PEBBLES_M" key; emits orders for both legs.
        # Validated in pebbles.py: combined +120k default / +70k conservative
        # at symmetric 10/10 sizing.
        # Phase B audit (2026-04-30): leg decomposition classified as
        # Hypothesis B (structural — XL drives all alpha; M is anchor).
        # Median sigma-adjusted ratio: 3.47 default / 5.33 conservative.
        # Resized M to 5 (frees 5 position units for C.1/C.2 pair-trade nominations).
        # Expected cost: ~-8.3k default; conservative slightly improves +1.6k.
        self.strategies["PEBBLES_M"] = R5PairTradeStrategy(
            symbol_a="PEBBLES_M",
            symbol_b="PEBBLES_XL",
            limit=5,
            limit_b=LIMIT,
            window=200,
            z_entry=2.0,
            z_exit=0.5,
            max_hold_ticks=500,
        )

        self.strategies["PEBBLES_S"] = R5XLSkewMMStrategy(
                        symbol="PEBBLES_S", limit=LIMIT, width=2,
                        partner_symbol="PEBBLES_XL",
                        threshold=0.001, k_ticks=2,
        )
        
        # Group Snacks - per product widths from snackpack.py
        self.strategies["SNACKPACK_CHOCOLATE"] = R5BaseMMStrategy("SNACKPACK_CHOCOLATE", LIMIT, width=3)
        self.strategies["SNACKPACK_RASPBERRY"] = R5BaseMMStrategy("SNACKPACK_RASPBERRY", LIMIT, width=3)
        self.strategies["SNACKPACK_PISTACHIO"] = R5BaseMMStrategy("SNACKPACK_PISTACHIO", LIMIT, width=3)
        
        # Group Robots — per-product widths per CLAUDE.md ROBOT deep-dive
        self.strategies["ROBOT_DISHES"] = R5BaseMMStrategy("ROBOT_DISHES", LIMIT, width=1) # -4.6k on IMC submission 
        self.strategies["ROBOT_IRONING"] = R5BaseMMStrategy("ROBOT_IRONING", LIMIT, width=1)
        self.strategies["ROBOT_LAUNDRY"] = R5BaseMMStrategy("ROBOT_LAUNDRY", LIMIT, width=2)

        # Group Microchip — RECTANGLE width=2 per CLAUDE.md MICROCHIP deep-dive
        # rectangle_widen variant: +2,824 default / +2,966 conservative vs width=1 baseline.
        # Lead-lag CIRCLE→OVAL angle dropped (all three k values net-negative; see eda_gaps.md).
        self.strategies["MICROCHIP_RECTANGLE"] = R5BaseMMStrategy("MICROCHIP_RECTANGLE", LIMIT, width=2)

        # Group Sleep Pod — width=2 per CLAUDE.md SLEEP_POD dive (2026-04-30)
        # EG screen: 0/6 pairs cointegrated. Width=2 rescues SUEDE (+4,438 delta conservative)
        # and improves NYLON (+1,724 delta). POLYESTER/COTTON dropped (net-negative at all widths).
        self.strategies["SLEEP_POD_NYLON"] = R5BaseMMStrategy("SLEEP_POD_NYLON", LIMIT, width=2)
        self.strategies["SLEEP_POD_SUEDE"] = R5BaseMMStrategy("SLEEP_POD_SUEDE", LIMIT, width=2)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders: dict[Symbol, list[Order]] = {}
        conversions = 0

        raw = json.loads(state.traderData) if state.traderData not in ("", None) else {}
        old_trader_data = raw if isinstance(raw, dict) else {}
        new_trader_data: dict[str, Any] = {}

        for symbol, strategy in self.strategies.items():
            if isinstance(strategy, StatefulStrategy) and symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            strategy_orders, strategy_conversions = strategy.run(state)
            for order in strategy_orders:
                orders.setdefault(order.symbol, []).append(order)
            conversions += strategy_conversions

            if isinstance(strategy, StatefulStrategy):
                new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data


logger = Logger()
