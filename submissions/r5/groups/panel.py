"""PANEL group — area cointegration analysis and strategy variants.

Hypothesis: Panel area is the price-determining variable. Products with the
same area should trade at the same equilibrium price.

Areas: 1X4=4, 2X2=4 (primary pair), 2X4=8 (fallback pair leg).
1X2 and 4X4 dropped in Phase A.

Kill criterion: No pair has Engle-Granger p < 0.05 across all 3 days
→ ship 1X4 + 2X4 base MM only; declare group complete.

Phase B freed 5 position units from PEBBLES_M resize (10→5). The PANEL
pair trade sizes at limit=5 to fit within this freed headroom.

Risk #2 (from spec): geometric cointegration may be lag-0 (like SNACKPACK
mirror pairs). If lag-N matrix shows lag-0 peak, pair trade is still viable
as spread-divergence z-score, but NO partner-lean overlay (avoids SNACKPACK
tick-residual catastrophe). See cerebrum.md Do-Not-Repeat.
"""

import json
from abc import abstractmethod
from math import ceil, floor
from pathlib import Path
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

type JSON = dict[str, Any] | list[Any] | str | int | float | bool | None

DATASET_DIR = Path(__file__).resolve().parents[3] / "datasets" / "round5"
SYMBOLS = ["PANEL_1X4", "PANEL_2X4", "PANEL_2X2"]
LIMIT = 10


# ── Platform boilerplate ──────────────────────────────────────────────────────

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]
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

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for arr in trades.values() for t in arr
        ]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv = {}
        for product, o in observations.conversionObservations.items():
            conv[product] = [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff,
                             o.importTariff, o.sugarPrice, o.sunlightIndex]
        return [observations.plainValueObservations, conv]

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
            v in state.order_depths
            and len(state.order_depths[v].buy_orders) > 0
            and len(state.order_depths[v].sell_orders) > 0
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
        od = state.order_depths[self.symbol]
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_vol = od.buy_orders[best_bid]
        ask_vol = abs(od.sell_orders[best_ask])
        return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

    def get_true_value(self, state: TradingState) -> float:
        return self._microprice(state)

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
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity
        if to_buy > 0:
            self.buy(max_buy_price - self.width + 1, to_buy)
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity
        if to_sell > 0:
            self.sell(min_sell_price + self.width - 1, to_sell)


class R5PairTradeStrategy(StatefulStrategy[dict[str, Any]]):
    """Z-score spread trade on a cointegrated pair (symbol_a − symbol_b).

    Tracks rolling spread of size `window`. Entry at |z| > z_entry.
    Exit at |z| < z_exit OR timestamp delta > max_hold_ticks * 100.

    Both legs' orders are emitted from this strategy. Trader.run accumulates
    by order.symbol so both legs get routed correctly.
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
        self.limit_b = limit_b if limit_b is not None else limit
        self.spread_history: list[float] = []
        self.entry_tick: int | None = None

    def get_required_symbols(self) -> list[Symbol]:
        return [self.symbol_a, self.symbol_b]

    def _mid(self, state: TradingState, symbol: Symbol) -> float:
        od = state.order_depths[symbol]
        return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2

    def _take(self, state: TradingState, symbol: Symbol, side: int, qty: int) -> None:
        od = state.order_depths[symbol]
        if side > 0:
            self.orders.append(Order(symbol, min(od.sell_orders.keys()), qty))
        else:
            self.orders.append(Order(symbol, max(od.buy_orders.keys()), -qty))

    def _flatten(self, state: TradingState, symbol: Symbol, position: int) -> None:
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

        if is_flat:
            self.entry_tick = None
        elif self.entry_tick is None:
            self.entry_tick = state.timestamp

        # Time-stop: risk-management override, fires regardless of z computability
        if not is_flat:
            if (state.timestamp - self.entry_tick) > self.max_hold_ticks * 100:
                self._flatten(state, self.symbol_a, pos_a)
                self._flatten(state, self.symbol_b, pos_b)
                return

        if len(self.spread_history) < self.window:
            return

        mean = sum(self.spread_history) / len(self.spread_history)
        var = sum((s - mean) ** 2 for s in self.spread_history) / len(self.spread_history)
        std = var ** 0.5
        if std == 0:
            return

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


# ── Strategy variants ─────────────────────────────────────────────────────────

class Trader:
    def __init__(self) -> None:
        self.strategies: dict[Symbol, Strategy] = {}

        def baseline():
            """Baseline: 1X4, 2X4, 2X2 all at base MM width=1.
            Expected default: 1X4 +25,767 / 2X4 +14,005 / 2X2 +2,706.
            Conservative: 1X4 +18,891 / 2X4 +5,578 / 2X2 -4,433.
            
            Default:
            PANEL_1X4                       15049.00    3028.00    7689.50   25766.50
            PANEL_2X2                        5758.00   -1933.00   -1119.00    2706.00
            PANEL_2X4                         -87.00    9172.00    4919.50   14004.50
            
            Conservative:
            PANEL_1X4                       12831.00     286.00    5773.50   18890.50
            PANEL_2X2                        3528.00   -4868.00   -3093.00   -4433.00
            PANEL_2X4                       -2426.00    5867.00    2136.50    5577.50
            """
            for sym in SYMBOLS:
                self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=1)

        def pair_1x4_2x2():
            """Variant B: 1X4 ↔ 2X2 pair trade (both area=4), limit=5.
            2X4 keeps base MM width=1. 2X2 absorbed into pair (no standalone MM).
            Break-even bar: +8.5k default over baseline (Phase B freed headroom cost).
            PEBBLES analog: +120k default / +70k conservative at 10/10. At 5/5: ~half.
            NO partner-lean overlay regardless of lag-N result (Risk #2 mitigation).
            
            Default:
            PANEL_1X4                       -2645.00   -7795.00   -3903.00  -14343.00
            PANEL_2X2                       -3970.00   -1659.00    4710.00    -919.00
            
            Conservative: N/A
            """
            self.strategies["PANEL_1X4"] = R5PairTradeStrategy(
                symbol_a="PANEL_1X4",
                symbol_b="PANEL_2X2",
                limit=5,
                limit_b=5,
                window=200,
                z_entry=2.0,
                z_exit=0.5,
                max_hold_ticks=500,
            )
            self.strategies["PANEL_2X4"] = R5BaseMMStrategy("PANEL_2X4", LIMIT, width=1)

        def pair_1x4_2x4():
            """Variant C-fallback: 1X4 ↔ 2X4 pair (areas 4 vs 8), limit=5.
            2X2 keeps base MM width=1.
            
            Default:
            PANEL_1X4                       -9565.00   -9409.00   -3056.50  -22030.50
            PANEL_2X4                       -9970.00     791.00    -981.50  -10160.50
            
            Conservative: N/A
            """
            self.strategies["PANEL_1X4"] = R5PairTradeStrategy(
                symbol_a="PANEL_1X4",
                symbol_b="PANEL_2X4",
                limit=5,
                limit_b=5,
                window=200,
                z_entry=2.0,
                z_exit=0.5,
                max_hold_ticks=500,
            )
            self.strategies["PANEL_2X2"] = R5BaseMMStrategy("PANEL_2X2", LIMIT, width=1)

        def pair_2x2_2x4():
            """Variant C-fallback2: 2X2 ↔ 2X4 pair (areas 4 vs 8), limit=5.
            1X4 keeps base MM width=1.
            
            Default:
            PANEL_2X2                       -6705.00   -3345.00   -1723.00  -11773.00
            PANEL_2X4                       -7125.00   -2279.00    -973.50  -10377.50
            
            Conservative: N/A
            """
            self.strategies["PANEL_2X2"] = R5PairTradeStrategy(
                symbol_a="PANEL_2X2",
                symbol_b="PANEL_2X4",
                limit=5,
                limit_b=5,
                window=200,
                z_entry=2.0,
                z_exit=0.5,
                max_hold_ticks=500,
            )
            self.strategies["PANEL_1X4"] = R5BaseMMStrategy("PANEL_1X4", LIMIT, width=1)

        drop_2x2()  # Shipped: 1X4 + 2X4 only; 2X2 dropped (conservative alpha -4,433; days 3+4 systematic losses)

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


# ── Analysis (run with: python panel.py) ─────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from statsmodels.tsa.stattools import adfuller, coint

    DAYS = [2, 3, 4]
    PAIRS = [
        ("PANEL_1X4", "PANEL_2X2"),
        ("PANEL_1X4", "PANEL_2X4"),
        ("PANEL_2X2", "PANEL_2X4"),
    ]

    def load_mids(day: int, product: str) -> pd.Series:
        path = DATASET_DIR / f"prices_round_5_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        return df[df["product"] == product].set_index("timestamp")["mid_price"].sort_index()

    def adf_screen() -> None:
        print("=" * 70)
        print("ADF stationarity check (I(1) precondition for EG)")
        print("p > 0.05 = non-stationary (I(1) ok for EG; I(0) invalid for EG)")
        print("=" * 70)
        print(f"  {'Product':>25} {'Day':>5} {'ADF p':>10}  {'I(1)?':>6}")
        for sym in SYMBOLS:
            for day in DAYS:
                mids = load_mids(day, sym).dropna()
                p = adfuller(mids)[1]
                i1 = "YES" if p > 0.05 else "NO"
                print(f"  {sym:>25} {day:>5} {p:>10.4f}  {i1:>6}")

    def eg_screen() -> None:
        print("\n" + "=" * 70)
        print("Engle-Granger cointegration screen")
        print("p < 0.05 = cointegrated (kill criterion: all 3 days must pass)")
        print("=" * 70)
        print(f"  {'Pair':>33} {'D2':>10} {'D3':>10} {'D4':>10} {'Pass all?':>10}")
        results = {}
        for sym_a, sym_b in PAIRS:
            pvals = []
            for day in DAYS:
                s_a = load_mids(day, sym_a).dropna()
                s_b = load_mids(day, sym_b).dropna()
                idx = s_a.index.intersection(s_b.index)
                _, p, _ = coint(s_a.loc[idx], s_b.loc[idx])
                pvals.append(p)
            all_pass = all(p < 0.05 for p in pvals)
            results[(sym_a, sym_b)] = (pvals, all_pass)
            label = f"{sym_a.replace('PANEL_','').replace('PANEL_','')} ↔ {sym_b.replace('PANEL_','')}"
            print(f"  {label:>33} {pvals[0]:>10.4f} {pvals[1]:>10.4f} {pvals[2]:>10.4f} {'YES' if all_pass else 'no':>10}")
        return results

    def lag_n_matrix(sym_a: str, sym_b: str, max_lag: int = 3) -> None:
        print(f"\n  Lag-N cross-correlation: {sym_a} ↔ {sym_b}")
        print(f"  corr(ret_{sym_a.replace('PANEL_','')}, ret_{sym_b.replace('PANEL_','')}.shift(k))")
        print(f"  {'k':>6} {'D2':>10} {'D3':>10} {'D4':>10}  note")
        for k in range(-max_lag, max_lag + 1):
            vals = []
            for day in DAYS:
                r_a = load_mids(day, sym_a).pct_change().dropna()
                r_b = load_mids(day, sym_b).pct_change().dropna()
                idx = r_a.index.intersection(r_b.index)
                c = r_a.loc[idx].corr(r_b.loc[idx].shift(k))
                vals.append(c)
            note = ""
            if k == 0:
                note = "<-- lag-0 (mirror?)"
            elif abs(k) == abs(max(range(-max_lag, max_lag+1), key=lambda kk: abs(sum(r_a.loc[idx].corr(load_mids(d, sym_b).pct_change().dropna().loc[r_a.index.intersection(load_mids(d, sym_b).pct_change().dropna().index)].shift(kk)) for d in DAYS)))):
                note = "<-- peak"
            print(f"  {k:>6} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}  {note}")

    def lag_n_matrix_clean(sym_a: str, sym_b: str, max_lag: int = 3) -> None:
        """Cleaner lag-N matrix without the inline peak detection bug."""
        print(f"\n  Lag-N cross-correlation: {sym_a} ↔ {sym_b}")
        a_lbl = sym_a.replace("PANEL_", "")
        b_lbl = sym_b.replace("PANEL_", "")
        print(f"  corr(ret_{a_lbl}, ret_{b_lbl}.shift(k)) — positive k means {b_lbl} leads {a_lbl}")
        print(f"  {'k':>6} {'D2':>10} {'D3':>10} {'D4':>10}")
        all_vals = {}
        for k in range(-max_lag, max_lag + 1):
            vals = []
            for day in DAYS:
                r_a = load_mids(day, sym_a).pct_change().dropna()
                r_b = load_mids(day, sym_b).pct_change().dropna()
                idx = r_a.index.intersection(r_b.index)
                c = r_a.loc[idx].corr(r_b.loc[idx].shift(k))
                vals.append(c)
            all_vals[k] = vals
            print(f"  {k:>6} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}")
        # Identify peak lag by mean |corr| across days
        peak_k = max(all_vals, key=lambda k: sum(abs(v) for v in all_vals[k]) / len(DAYS))
        peak_mean = sum(abs(v) for v in all_vals[peak_k]) / len(DAYS)
        lag0_mean = sum(abs(v) for v in all_vals[0]) / len(DAYS)
        print(f"  Peak lag: k={peak_k} (mean |corr|={peak_mean:.4f})")
        print(f"  Lag-0 mean |corr|: {lag0_mean:.4f}")
        if peak_k == 0:
            print(f"  → LAG-0 PEAK: mirror relationship; spread-divergence trade OK, NO partner-lean overlay")
        else:
            direction = f"{b_lbl} leads {a_lbl}" if peak_k > 0 else f"{a_lbl} leads {b_lbl}"
            print(f"  → NON-ZERO PEAK (k={peak_k}): {direction}")
            print(f"    Partner-lean overlay may be viable (verify per-day stability before wiring)")

    print("=" * 70)
    print("PANEL group — area cointegration analysis")
    print("=" * 70)

    adf_screen()
    results = eg_screen()

    print("\n" + "=" * 70)
    print("Lag-N cross-correlation matrices")
    print("=" * 70)
    for sym_a, sym_b in PAIRS:
        lag_n_matrix_clean(sym_a, sym_b)

    print("\n" + "=" * 70)
    print("Kill criterion outcome")
    print("=" * 70)
    passing_pairs = [(a, b) for (a, b), (_, ok) in results.items() if ok]
    if not passing_pairs:
        print("KILL CRITERION MET: No pair has EG p < 0.05 across all 3 days.")
        print("→ Ship 1X4 + 2X4 base MM only. 2X2 status deferred to adverse-selection check.")
        print("→ No wiring changes needed for pair trade.")
    else:
        print(f"PAIRS PASSING: {[(a.replace('PANEL_',''), b.replace('PANEL_','')) for a,b in passing_pairs]}")
        print("→ Kill criterion NOT met. Proceed to backtest variants.")
        print()
        print("Recommended variant order:")
        for rank, (a, b) in enumerate(passing_pairs, 1):
            a_s = a.replace("PANEL_", "")
            b_s = b.replace("PANEL_", "")
            print(f"  {rank}. {a_s} ↔ {b_s}")
        print()
        print("Backtest commands (run in WSL2):")
        print()
        print("  # Variant A — baseline confirmation")
        print("  cd ~/prosperity_rust_backtester")
        print('  rust_backtester --trader "$PROSP4/submissions/r5/groups/panel.py" --dataset round5 --persist --carry')
        print()
        print("  # Variant B — pair trade 1X4 ↔ 2X2 (change baseline() → pair_1x4_2x2() in Trader.__init__)")
        print('  rust_backtester --trader "$PROSP4/submissions/r5/groups/panel.py" --dataset round5 --persist --carry')
        print('  rust_backtester --trader "$PROSP4/submissions/r5/groups/panel.py" --dataset round5 \\')
        print('    --queue-penetration 0 --price-slippage-bps 5 --persist --carry')
        print()
        print("  # Per-day consistency")
        print("  for day in 2 3 4; do")
        print('    rust_backtester --trader "$PROSP4/submissions/r5/groups/panel.py" --dataset round5 \\')
        print('      --day $day --persist --carry')
        print("  done")
