"""SLEEP_POD group — material substitutability cointegration analysis.

Hypothesis: Material substitutability (NYLON, SUEDE, POLYESTER, COTTON) creates
cointegrating relationships. LAMB_WOOL dropped Phase A (alpha −36,725).

Kill criterion: No pair has Engle-Granger p < 0.05 across all 3 days AND
SUEDE adverse-selection diagnostic shows structural loss (not pickoff-rescuable).
→ Ship NYLON as base MM only; drop SUEDE/POLYESTER/COTTON from TIGHT_TIER.

Per-product alpha state (post-Phase-A, conservative):
  NYLON     +1,797  → keep, ship base MM width=1
  SUEDE      −357   → marginal; width-tune or drop
  POLYESTER −4,048  → drop unless pair leg or width-tune rescues
  COTTON    −5,316  → drop unless pair leg or width-tune rescues
"""
import json
from abc import abstractmethod
from math import ceil, floor
from pathlib import Path
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

type JSON = dict[str, Any] | list[Any] | str | int | float | bool | None

DATASET_DIR = Path(__file__).resolve().parents[3] / "datasets" / "round5"
SYMBOLS = ["SLEEP_POD_NYLON", "SLEEP_POD_SUEDE", "SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON"]
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


class R5BaseMMStrategy(Strategy):
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

    def act(self, state: TradingState) -> None:
        true_value = self._microprice(state)
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
    Both legs' orders are emitted from this strategy.
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
            """Baseline: all 4 surviving products at width=1.
            Matches current strategy_h.py wiring (post-Phase-A, LAMB_WOOL dropped).
            Expected per-product (default / conservative):
              NYLON     +9,115 /  +1,797
              SUEDE     +8,271 /   −357
              POLYESTER +4,808 / −4,048
              COTTON    +3,265 / −5,316
            """
            for sym in SYMBOLS:
                self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=1)

        def suede_w3():
            """Variant C2: SUEDE at width=3, others at width=1.
            
            Default:
            SLEEP_POD_SUEDE                 11160.00    4890.00   -5280.00   10770.00
            
            Conservative:
            SLEEP_POD_SUEDE                  8973.00    1428.00   -8076.00    2325.00
            """
            for sym in SYMBOLS:
                w = 3 if sym == "SLEEP_POD_SUEDE" else 1
                self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=w)

        def all_wider_w2():
            """Variant C3: All 4 products at width=2.
            Tests whether POLYESTER/COTTON alpha loss is also pickoff-rescuable.
            
            Default:
            SLEEP_POD_COTTON                -2023.00     788.00    7550.00    6315.00
            SLEEP_POD_NYLON                  1683.00    5521.50    3372.00   10576.50
            SLEEP_POD_POLYESTER             -1264.00    5376.50    3007.00    7119.50
            SLEEP_POD_SUEDE                 14044.00    4278.00   -5746.00   12576.00
            
            Conservative:
            SLEEP_POD_COTTON                -4293.00   -2512.00    4671.00   -2134.00
            SLEEP_POD_LAMB_WOOL                 0.00       0.00       0.00       0.00
            SLEEP_POD_NYLON                  -413.00    2874.50    1060.00    3521.50
            SLEEP_POD_POLYESTER             -3560.00    1879.50      79.00   -1601.50
            SLEEP_POD_SUEDE                 11807.00     816.00   -8542.00    4081.00
            """
            for sym in SYMBOLS:
                self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=2)

        def pair_trade(sym_a: str, sym_b: str) -> None:
            """Variant B: pair trade on an EG-passing pair (limit=5 each leg).
            Break-even bar: must add ≥ 0 conservative vs dropping both products.
            Freed headroom: 5 units from PEBBLES_M resize (Phase B).
            NO partner-lean overlay — pair may be lag-0 (SNACKPACK lesson).
            Remaining products get base MM width=1.
            
            Default:
            
            Conservative:
            """
            self.strategies[sym_a] = R5PairTradeStrategy(
                symbol_a=sym_a,
                symbol_b=sym_b,
                limit=5,
                limit_b=5,
                window=200,
                z_entry=2.0,
                z_exit=0.5,
                max_hold_ticks=500,
            )
            for sym in SYMBOLS:
                if sym not in (sym_a, sym_b):
                    self.strategies[sym] = R5BaseMMStrategy(sym, LIMIT, width=1)

        # Active variant — swap to test others:
        #   suede_w3()           → Variant C2 (SUEDE pickoff test, width=3)
        #   all_wider_w2()       → Variant C3 (all at width=2)
        #   pair_trade(A, B)     → Variant B (replace A/B with passing EG pair)
        pair_trade()

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

# ── Analysis (run with: python sleep_pod.py) ─────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from statsmodels.tsa.stattools import adfuller, coint

    DAYS = [2, 3, 4]
    PAIRS = [
        ("SLEEP_POD_NYLON",     "SLEEP_POD_SUEDE"),
        ("SLEEP_POD_NYLON",     "SLEEP_POD_POLYESTER"),
        ("SLEEP_POD_NYLON",     "SLEEP_POD_COTTON"),
        ("SLEEP_POD_SUEDE",     "SLEEP_POD_POLYESTER"),
        ("SLEEP_POD_SUEDE",     "SLEEP_POD_COTTON"),
        ("SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON"),
    ]

    def load_mids(day: int, product: str) -> pd.Series:
        path = DATASET_DIR / f"prices_round_5_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        return df[df["product"] == product].set_index("timestamp")["mid_price"].sort_index()

    def short(sym: str) -> str:
        return sym.replace("SLEEP_POD_", "")

    def adf_screen() -> None:
        print("=" * 70)
        print("ADF stationarity check (I(1) precondition for EG)")
        print("p > 0.05 = non-stationary (I(1) — ok for EG)")
        print("p < 0.05 = stationary (I(0) — invalid for EG; already mean-reverts)")
        print("=" * 70)
        print(f"  {'Product':>15} {'Day':>5} {'ADF p':>10}  {'I(1)?':>6}")
        for sym in SYMBOLS:
            for day in DAYS:
                mids = load_mids(day, sym).dropna()
                p = adfuller(mids)[1]
                i1 = "YES" if p > 0.05 else "NO "
                print(f"  {short(sym):>15} {day:>5} {p:>10.4f}  {i1:>6}")
        print()

    def eg_screen() -> dict[tuple[str, str], tuple[list[float], bool]]:
        print("=" * 70)
        print("Engle-Granger cointegration screen — 6 within-group pairs")
        print("p < 0.05 = cointegrated (kill criterion: all 3 days must pass)")
        print("=" * 70)
        print(f"  {'Pair':>28} {'D2':>10} {'D3':>10} {'D4':>10} {'Pass all?':>10}")
        results: dict[tuple[str, str], tuple[list[float], bool]] = {}
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
            label = f"{short(sym_a)} ↔ {short(sym_b)}"
            print(f"  {label:>28} {pvals[0]:>10.4f} {pvals[1]:>10.4f} {pvals[2]:>10.4f} {'YES' if all_pass else 'no':>10}")
        print()
        return results

    def lag_n_matrix(sym_a: str, sym_b: str, max_lag: int = 3) -> None:
        a_lbl = short(sym_a)
        b_lbl = short(sym_b)
        print(f"  Lag-N cross-correlation: {a_lbl} ↔ {b_lbl}")
        print(f"  corr(ret_{a_lbl}, ret_{b_lbl}.shift(k)) — positive k means {b_lbl} leads {a_lbl}")
        print(f"  {'k':>6} {'D2':>10} {'D3':>10} {'D4':>10}")
        all_vals: dict[int, list[float]] = {}
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
        peak_k = max(all_vals, key=lambda k: sum(abs(v) for v in all_vals[k]) / len(DAYS))
        peak_mean = sum(abs(v) for v in all_vals[peak_k]) / len(DAYS)
        lag0_mean = sum(abs(v) for v in all_vals[0]) / len(DAYS)
        print(f"  Peak lag: k={peak_k} (mean |corr|={peak_mean:.4f}), lag-0 mean |corr|={lag0_mean:.4f}")
        if peak_k == 0:
            print(f"  → LAG-0 PEAK: spread-divergence trade viable; NO partner-lean overlay")
        else:
            direction = f"{b_lbl} leads {a_lbl}" if peak_k > 0 else f"{a_lbl} leads {b_lbl}"
            print(f"  → NON-ZERO PEAK (k={peak_k}): {direction}")
        print()

    def suede_volatility_analysis() -> None:
        """Per-day volatility characterization to classify SUEDE D4 loss.

        Known PnL from Phase A backtests:
          Default:      D2 +13,223 / D3 +3,163 / D4 −8,116
          Conservative: D2 +10,920 / D3 −347   / D4 −10,929

        Conservative D4 is WORSE than default D4 (−10.9k vs −8.1k) → genuine
        adverse selection, not queue-priority leakage.

        This analysis computes intraday trend strength and lag-1 autocorrelation
        per day to classify: strong trend + positive autocorr = regime loss;
        high noise + negative autocorr = pickoff (width-rescuable).
        """
        print("=" * 70)
        print("SUEDE per-day volatility analysis")
        print("Known PnL: default  D2 +13223 / D3 +3163 / D4 −8116")
        print("           conserv. D2 +10920 / D3  −347  / D4 −10929")
        print("D4 conservative WORSE than default → genuine adverse selection")
        print("=" * 70)
        print(f"  {'Day':>5} {'σ_ret':>10} {'mean_ret':>12} {'trend/σ':>10} {'ACF_lag1':>10}  interpretation")
        for day in DAYS:
            mids = load_mids(day, "SLEEP_POD_SUEDE").dropna()
            rets = mids.pct_change().dropna()
            sigma = rets.std()
            mean_r = rets.mean()
            trend_noise = mean_r / sigma if sigma > 0 else 0.0
            acf1 = rets.autocorr(lag=1) if len(rets) > 2 else float("nan")

            if abs(trend_noise) > 0.10:
                interp = "TRENDING (regime loss)"
            elif acf1 < -0.05:
                interp = "MEAN-REVERTING (pickoff-shaped)"
            else:
                interp = "MIXED/NEUTRAL"
            print(f"  {day:>5} {sigma:>10.6f} {mean_r:>12.6f} {trend_noise:>10.4f} {acf1:>10.4f}  {interp}")
        print()
        print("  Pickoff-driven loss → wider width may rescue (test Variants C1/C2)")
        print("  Regime/structural  → drop SUEDE from TIGHT_TIER")
        print()

    def print_kill_criterion(results: dict[tuple[str, str], tuple[list[float], bool]]) -> None:
        print("=" * 70)
        print("Kill criterion outcome")
        print("=" * 70)
        passing = [(a, b) for (a, b), (_, ok) in results.items() if ok]
        if not passing:
            print("KILL CRITERION (EG side) MET: No pair has p < 0.05 across all 3 days.")
            print()
            print("→ Proceed to SUEDE volatility analysis above to check structural flag.")
            print("→ If SUEDE shows structural loss (regime): ship NYLON only.")
            print("  Remove SUEDE, POLYESTER, COTTON from TIGHT_TIER in strategy_h.py.")
            print()
            print("→ If SUEDE shows pickoff: test width=2/3 variants below.")
            print()
        else:
            print(f"EG PAIRS PASSING: {[(short(a), short(b)) for a, b in passing]}")
            print("Kill criterion NOT met — proceed to lag-N matrix (already printed above).")
            print()
            print("Recommended next step:")
            for rank, (a, b) in enumerate(passing, 1):
                print(f"  {rank}. Backtest Variant B — pair_trade('{a}', '{b}')")
            print()

    def print_backtest_commands() -> None:
        print("=" * 70)
        print("Backtest commands (run in WSL2)")
        print("=" * 70)
        print()
        print("  # Variant A — baseline confirmation (all 4 products width=1)")
        print("  # Active: baseline() in Trader.__init__")
        print("  cd ~/prosperity_rust_backtester")
        print('  rust_backtester --trader "$PROSP4/submissions/r5/groups/sleep_pod.py" --dataset round5 --persist --carry')
        print('  rust_backtester --trader "$PROSP4/submissions/r5/groups/sleep_pod.py" --dataset round5 \\')
        print('    --queue-penetration 0 --price-slippage-bps 5 --persist --carry')
        print()
        print("  # Variant C1 — SUEDE width=2 (pickoff diagnostic)")
        print("  # Change: baseline() → suede_w2()")
        print('  rust_backtester --trader "$PROSP4/submissions/r5/groups/sleep_pod.py" --dataset round5 --persist --carry')
        print('  for day in 2 3 4; do')
        print('    rust_backtester --trader "$PROSP4/submissions/r5/groups/sleep_pod.py" --dataset round5 --day $day --persist --carry')
        print('  done')
        print()
        print("  # Variant C2 — SUEDE width=3")
        print("  # Change: baseline() → suede_w3()")
        print('  rust_backtester --trader "$PROSP4/submissions/r5/groups/sleep_pod.py" --dataset round5 --persist --carry')
        print()
        print("  # Variant C3 — all products width=2 (POLYESTER/COTTON pickoff test)")
        print("  # Change: baseline() → all_wider_w2()")
        print('  rust_backtester --trader "$PROSP4/submissions/r5/groups/sleep_pod.py" --dataset round5 --persist --carry')
        print()
        print("  # Variant D — NYLON + SUEDE only (drop POLYESTER/COTTON)")
        print("  # Change: baseline() → drop_poly_cotton()")
        print('  rust_backtester --trader "$PROSP4/submissions/r5/groups/sleep_pod.py" --dataset round5 --persist --carry')
        print('  rust_backtester --trader "$PROSP4/submissions/r5/groups/sleep_pod.py" --dataset round5 \\')
        print('    --queue-penetration 0 --price-slippage-bps 5 --persist --carry')
        print()
        print("  # Kill criterion outcome — NYLON only")
        print("  # Change: baseline() → nylon_only()")
        print('  rust_backtester --trader "$PROSP4/submissions/r5/groups/sleep_pod.py" --dataset round5 --persist --carry')
        print()
        print("  # Variant B — pair trade (only if EG passed)")
        print("  # Change: baseline() → pair_trade('SLEEP_POD_X', 'SLEEP_POD_Y')")
        print('  rust_backtester --trader "$PROSP4/submissions/r5/groups/sleep_pod.py" --dataset round5 --persist --carry')
        print('  rust_backtester --trader "$PROSP4/submissions/r5/groups/sleep_pod.py" --dataset round5 \\')
        print('    --queue-penetration 0 --price-slippage-bps 5 --persist --carry')
        print()

    print("=" * 70)
    print("SLEEP_POD group — material substitutability cointegration analysis")
    print("=" * 70)
    print()

    adf_screen()
    results = eg_screen()

    passing_pairs = [(a, b) for (a, b), (_, ok) in results.items() if ok]
    if passing_pairs:
        print("=" * 70)
        print("Lag-N cross-correlation matrices (EG candidates only)")
        print("=" * 70)
        print()
        for sym_a, sym_b in passing_pairs:
            lag_n_matrix(sym_a, sym_b)
    else:
        print("No EG-passing pairs — skipping lag-N matrix.")
        print()

    suede_volatility_analysis()
    print_kill_criterion(results)
    print_backtest_commands()
