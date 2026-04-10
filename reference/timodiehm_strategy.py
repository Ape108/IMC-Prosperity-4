"""
TimoDiehm/imc-prosperity-3 — FrankfurtHedgehogs_polished.py
Source: https://github.com/TimoDiehm/imc-prosperity-3
2nd place overall (1,433,876 SeaShells), 12,000+ teams

Key insights from their writeup:
- "Olivia" bot consistently buys at daily lows/sells at daily highs
- Speed and order cancellation are irrelevant (sequential processing within timesteps)
- Fixed threshold models for ETF basket mean reversion
- IV scalping on options by analyzing volatility smile deviations
- Round 4: hidden "taker bot" fills at approx int(externalBid + 0.5)
- Round 5: trader IDs revealed, refined Olivia detection to direct ID check

Strategy types:
  StaticTrader     — Market making on RAINFOREST_RESIN (quotes around wall_mid)
  DynamicTrader    — KELP market making + Olivia signal following
  InkTrader        — SQUID_INK pure Olivia signal following
  EtfTrader        — ETF basket arbitrage (basket vs synthetic constituents)
  OptionTrader     — IV scalping + mean reversion on VOLCANIC_ROCK options
  CommodityTrader  — MAGNIFICENT_MACARONS conversion arbitrage
"""

from datamodel import OrderDepth, TradingState, Order
import json
import numpy as np
import math
from statistics import NormalDist

_N = NormalDist()

####### GENERAL #######

ETF_BASKET_SYMBOLS = ['PICNIC_BASKET1', 'PICNIC_BASKET2']
ETF_CONSTITUENT_SYMBOLS = ['CROISSANTS', 'JAMS', 'DJEMBES']

STATIC_SYMBOL = 'RAINFOREST_RESIN'
DYNAMIC_SYMBOL = 'KELP'
INK_SYMBOL = 'SQUID_INK'

OPTION_UNDERLYING_SYMBOL = 'VOLCANIC_ROCK'

COMMODITY_SYMBOL = 'MAGNIFICENT_MACARONS'

OPTION_SYMBOLS = [
    'VOLCANIC_ROCK_VOUCHER_9500',
    'VOLCANIC_ROCK_VOUCHER_9750',
    'VOLCANIC_ROCK_VOUCHER_10000',
    'VOLCANIC_ROCK_VOUCHER_10250',
    'VOLCANIC_ROCK_VOUCHER_10500'
]

POS_LIMITS = {
    STATIC_SYMBOL: 50,
    DYNAMIC_SYMBOL: 50,
    INK_SYMBOL: 50,
    ETF_BASKET_SYMBOLS[0]: 60,
    ETF_BASKET_SYMBOLS[1]: 100,
    ETF_CONSTITUENT_SYMBOLS[0]: 250,
    ETF_CONSTITUENT_SYMBOLS[1]: 350,
    ETF_CONSTITUENT_SYMBOLS[2]: 60,
    OPTION_UNDERLYING_SYMBOL: 400,
    **{os: 200 for os in OPTION_SYMBOLS},
    COMMODITY_SYMBOL: 75,
}

CONVERSION_LIMIT = 10

LONG, NEUTRAL, SHORT = 1, 0, -1

INFORMED_TRADER_ID = 'Olivia'


####### ETF PARAMETERS #######

ETF_CONSTITUENT_FACTORS = [[6, 3, 1], [4, 2, 0]]
BASKET_THRESHOLDS = [80, 50]
n_hist_samples = 60_000
INITIAL_ETF_PREMIUMS = [5, 53]
ETF_INFORMED_CONSTITUENT = ETF_CONSTITUENT_SYMBOLS[0]
ETF_THR_INFORMED_ADJS = [90, 90]
ETF_CLOSE_AT_ZERO = True
CALCULATE_RUNNING_ETF_PREMIUM = True
ETF_HEDGE_FACTOR = 0.5


####### OPTIONS PARAMETERS #######

DAY = 5
DAYS_PER_YEAR = 365
THR_OPEN, THR_CLOSE = 0.5, 0
LOW_VEGA_THR_ADJ = 0.5
THEO_NORM_WINDOW = 20
IV_SCALPING_THR = 0.7
IV_SCALPING_WINDOW = 100
underlying_mean_reversion_thr = 15
underlying_mean_reversion_window = 10
options_mean_reversion_thr = 5
options_mean_reversion_window = 30


class ProductTrader:
    def __init__(self, name, state, prints, new_trader_data, product_group=None):
        self.orders = []
        self.name = name
        self.state = state
        self.prints = prints
        self.new_trader_data = new_trader_data
        self.product_group = name if product_group is None else product_group

        self.last_traderData = self.get_last_traderData()
        self.position_limit = POS_LIMITS.get(self.name, 0)
        self.initial_position = self.state.position.get(self.name, 0)
        self.expected_position = self.initial_position

        self.mkt_buy_orders, self.mkt_sell_orders = self.get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self.get_walls()
        self.best_bid, self.best_ask = self.get_best_bid_ask()
        self.max_allowed_buy_volume, self.max_allowed_sell_volume = self.get_max_allowed_volume()
        self.total_mkt_buy_volume, self.total_mkt_sell_volume = self.get_total_market_buy_sell_volume()

    def get_last_traderData(self):
        last_traderData = {}
        try:
            if self.state.traderData != '':
                last_traderData = json.loads(self.state.traderData)
        except: self.log("ERROR", 'td')
        return last_traderData

    def get_best_bid_ask(self):
        best_bid = best_ask = None
        try:
            if len(self.mkt_buy_orders) > 0:
                best_bid = max(self.mkt_buy_orders.keys())
            if len(self.mkt_sell_orders) > 0:
                best_ask = min(self.mkt_sell_orders.keys())
        except: pass
        return best_bid, best_ask

    def get_walls(self):
        bid_wall = wall_mid = ask_wall = None
        try: bid_wall = min([x for x,_ in self.mkt_buy_orders.items()])
        except: pass
        try: ask_wall = max([x for x,_ in self.mkt_sell_orders.items()])
        except: pass
        try: wall_mid = (bid_wall + ask_wall) / 2
        except: pass
        return bid_wall, wall_mid, ask_wall

    def get_total_market_buy_sell_volume(self):
        market_bid_volume = market_ask_volume = 0
        try:
            market_bid_volume = sum([v for p, v in self.mkt_buy_orders.items()])
            market_ask_volume = sum([v for p, v in self.mkt_sell_orders.items()])
        except: pass
        return market_bid_volume, market_ask_volume

    def get_max_allowed_volume(self):
        max_allowed_buy_volume = self.position_limit - self.initial_position
        max_allowed_sell_volume = self.position_limit + self.initial_position
        return max_allowed_buy_volume, max_allowed_sell_volume

    def get_order_depth(self):
        order_depth, buy_orders, sell_orders = {}, {}, {}
        try: order_depth: OrderDepth = self.state.order_depths[self.name]
        except: pass
        try: buy_orders = {bp: abs(bv) for bp, bv in sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)}
        except: pass
        try: sell_orders = {sp: abs(sv) for sp, sv in sorted(order_depth.sell_orders.items(), key=lambda x: x[0])}
        except: pass
        return buy_orders, sell_orders

    def bid(self, price, volume, logging=True):
        abs_volume = min(abs(int(volume)), self.max_allowed_buy_volume)
        order = Order(self.name, int(price), abs_volume)
        if logging: self.log("BUYO", {"p":price, "s":self.name, "v":int(volume)}, product_group='ORDERS')
        self.max_allowed_buy_volume -= abs_volume
        self.orders.append(order)

    def ask(self, price, volume, logging=True):
        abs_volume = min(abs(int(volume)), self.max_allowed_sell_volume)
        order = Order(self.name, int(price), -abs_volume)
        if logging: self.log("SELLO", {"p":price, "s":self.name, "v":int(volume)}, product_group='ORDERS')
        self.max_allowed_sell_volume -= abs_volume
        self.orders.append(order)

    def log(self, kind, message, product_group=None):
        if product_group is None: product_group = self.product_group
        if product_group == 'ORDERS':
            group = self.prints.get(product_group, [])
            group.append({kind: message})
        else:
            group = self.prints.get(product_group, {})
            group[kind] = message
        self.prints[product_group] = group

    def check_for_informed(self):
        informed_direction, informed_bought_ts, informed_sold_ts = NEUTRAL, None, None
        informed_bought_ts, informed_sold_ts = self.last_traderData.get(self.name, [None, None])
        trades = self.state.market_trades.get(self.name, []) + self.state.own_trades.get(self.name, [])
        for trade in trades:
            if trade.buyer == INFORMED_TRADER_ID:
                informed_bought_ts = trade.timestamp
            if trade.seller == INFORMED_TRADER_ID:
                informed_sold_ts = trade.timestamp
        self.new_trader_data[self.name] = [informed_bought_ts, informed_sold_ts]

        informed_sold = informed_sold_ts is not None
        informed_bought = informed_bought_ts is not None

        if not informed_bought and not informed_sold:
            informed_direction = NEUTRAL
        elif not informed_bought and informed_sold:
            informed_direction = SHORT
        elif informed_bought and not informed_sold:
            informed_direction = LONG
        elif informed_bought and informed_sold:
            if informed_sold_ts > informed_bought_ts:
                informed_direction = SHORT
            elif informed_sold_ts < informed_bought_ts:
                informed_direction = LONG
            else:
                informed_direction = NEUTRAL

        self.log('TD', self.new_trader_data[self.name])
        self.log('ID', informed_direction)
        return informed_direction, informed_bought_ts, informed_sold_ts

    def get_orders(self):
        return {}


class StaticTrader(ProductTrader):
    """Market making on RAINFOREST_RESIN — quotes around wall_mid, takes mispriced orders."""
    def __init__(self, state, prints, new_trader_data):
        super().__init__(STATIC_SYMBOL, state, prints, new_trader_data)

    def get_orders(self):
        if self.wall_mid is not None:
            # 1. TAKING
            for sp, sv in self.mkt_sell_orders.items():
                if sp <= self.wall_mid - 1:
                    self.bid(sp, sv, logging=False)
                elif sp <= self.wall_mid and self.initial_position < 0:
                    volume = min(sv, abs(self.initial_position))
                    self.bid(sp, volume, logging=False)

            for bp, bv in self.mkt_buy_orders.items():
                if bp >= self.wall_mid + 1:
                    self.ask(bp, bv, logging=False)
                elif bp >= self.wall_mid and self.initial_position > 0:
                    volume = min(bv, self.initial_position)
                    self.ask(bp, volume, logging=False)

            # 2. MAKING
            bid_price = int(self.bid_wall + 1)
            ask_price = int(self.ask_wall - 1)

            for bp, bv in self.mkt_buy_orders.items():
                overbidding_price = bp + 1
                if bv > 1 and overbidding_price < self.wall_mid:
                    bid_price = max(bid_price, overbidding_price)
                    break
                elif bp < self.wall_mid:
                    bid_price = max(bid_price, bp)
                    break

            for sp, sv in self.mkt_sell_orders.items():
                underbidding_price = sp - 1
                if sv > 1 and underbidding_price > self.wall_mid:
                    ask_price = min(ask_price, underbidding_price)
                    break
                elif sp > self.wall_mid:
                    ask_price = min(ask_price, sp)
                    break

            self.bid(bid_price, self.max_allowed_buy_volume)
            self.ask(ask_price, self.max_allowed_sell_volume)

        return {self.name: self.orders}


class DynamicTrader(ProductTrader):
    """KELP market making with Olivia signal following."""
    def __init__(self, state, prints, new_trader_data):
        super().__init__(DYNAMIC_SYMBOL, state, prints, new_trader_data)
        self.informed_direction, self.informed_bought_ts, self.informed_sold_ts = self.check_for_informed()

    def get_orders(self):
        if self.wall_mid is not None:
            bid_price = self.bid_wall + 1
            bid_volume = self.max_allowed_buy_volume

            if self.informed_bought_ts is not None and self.informed_bought_ts + 5_00 >= self.state.timestamp:
                if self.initial_position < 40:
                    bid_price = self.ask_wall
                    bid_volume = 40 - self.initial_position
            else:
                if self.wall_mid - bid_price < 1 and (self.informed_direction == SHORT and self.initial_position > -40):
                    bid_price = self.bid_wall

            self.bid(bid_price, bid_volume)

            ask_price = self.ask_wall - 1
            ask_volume = self.max_allowed_sell_volume

            if self.informed_sold_ts is not None and self.informed_sold_ts + 5_00 >= self.state.timestamp:
                if self.initial_position > -40:
                    ask_price = self.bid_wall
                    ask_volume = 40 + self.initial_position

            if ask_price - self.wall_mid < 1 and (self.informed_direction == LONG and self.initial_position < 40):
                ask_price = self.ask_wall

            self.ask(ask_price, ask_volume)

        return {self.name: self.orders}


class InkTrader(ProductTrader):
    """SQUID_INK — pure Olivia signal following."""
    def __init__(self, state, prints, new_trader_data):
        super().__init__(INK_SYMBOL, state, prints, new_trader_data)
        self.informed_direction, _, _ = self.check_for_informed()

    def get_orders(self):
        expected_position = 0
        if self.informed_direction == LONG:
            expected_position = self.position_limit
        elif self.informed_direction == SHORT:
            expected_position = -self.position_limit

        remaining_volume = expected_position - self.initial_position

        if remaining_volume > 0 and self.ask_wall is not None:
            self.bid(self.ask_wall, remaining_volume)
        elif remaining_volume < 0 and self.bid_wall is not None:
            self.ask(self.bid_wall, -remaining_volume)

        return {self.name: self.orders}


class EtfTrader:
    """ETF basket arbitrage — basket vs synthetic constituent value."""
    def __init__(self, state, prints, new_trader_data):
        self.baskets = [ProductTrader(s, state, prints, new_trader_data, product_group='ETF') for s in ETF_BASKET_SYMBOLS]
        self.informed_constituent = ProductTrader(ETF_INFORMED_CONSTITUENT, state, prints, new_trader_data, product_group='ETF')
        self.hedging_constituents = [ProductTrader(s, state, prints, new_trader_data, product_group='ETF') for s in ETF_CONSTITUENT_SYMBOLS if s != ETF_INFORMED_CONSTITUENT]
        self.state = state
        self.last_traderData = self.informed_constituent.last_traderData
        self.new_trader_data = new_trader_data
        self.spreads = self.calculate_spreads()
        self.informed_direction, _, _ = self.informed_constituent.check_for_informed()

    def calculate_spreads(self):
        return [self.calculate_spread(basket) for basket in self.baskets]

    def calculate_spread(self, basket):
        spread = None
        b_idx = ETF_BASKET_SYMBOLS.index(basket.name)
        try:
            constituents = [self.informed_constituent] + self.hedging_constituents
            const_prices = [const.wall_mid for const in constituents.sort(key=lambda c: {s: i for i, s in enumerate(ETF_CONSTITUENT_SYMBOLS)}[c.name])]
            index_price = np.asarray(const_prices) @ np.asarray(ETF_CONSTITUENT_FACTORS[b_idx])
            etf_price = basket.wall_mid
            raw_spread = etf_price - index_price

            if CALCULATE_RUNNING_ETF_PREMIUM:
                old_etf_mean_premium = self.last_traderData.get(f'ETF_{b_idx}_P', [INITIAL_ETF_PREMIUMS[b_idx], n_hist_samples])
                mean_premium, n = old_etf_mean_premium
                n += 1
                mean_premium += (raw_spread - mean_premium) / n
                self.new_trader_data[f'ETF_{b_idx}_P'] = [mean_premium, n]
            else:
                mean_premium = INITIAL_ETF_PREMIUMS[b_idx]

            spread = raw_spread - mean_premium
        except:
            old_etf_mean_premium = self.last_traderData.get(f'{basket.name[-1]}_P', [INITIAL_ETF_PREMIUMS[b_idx], n_hist_samples])
            self.new_trader_data[f'{basket.name[-1]}_P'] = old_etf_mean_premium
        return spread

    def get_basket_orders(self):
        out = {}
        for b_idx, basket in enumerate(self.baskets):
            if self.spreads[b_idx] is None: continue
            informed_thr_adj = {LONG: ETF_THR_INFORMED_ADJS[b_idx], SHORT: -ETF_THR_INFORMED_ADJS[b_idx]}.get(self.informed_direction, 0)

            if self.spreads[basket.name] > (BASKET_THRESHOLDS[b_idx] + informed_thr_adj) and basket.max_allowed_sell_volume > 0:
                basket.ask(basket.bid_wall, basket.max_allowed_sell_volume)
                basket.expected_position -= min(basket.total_mkt_sell_volume, basket.max_allowed_sell_volume)
            elif self.spreads[basket.name] < (-BASKET_THRESHOLDS[b_idx] + informed_thr_adj) and basket.max_allowed_buy_volume > 0:
                basket.bid(basket.ask_wall, basket.max_allowed_buy_volume)
                basket.expected_position += min(basket.total_mkt_buy_volume, basket.max_allowed_buy_volume)
            elif ETF_CLOSE_AT_ZERO:
                if self.spreads[b_idx] > informed_thr_adj and basket.initial_position > 0:
                    basket.ask(basket.bid_wall, basket.initial_position)
                    basket.expected_position -= min(basket.total_mkt_sell_volume, basket.initial_position)
                elif self.spreads[b_idx] < informed_thr_adj and basket.initial_position < 0:
                    basket.bid(basket.ask_wall, -basket.initial_position)
                    basket.expected_position += min(basket.total_mkt_buy_volume, -basket.initial_position)

            out.update({basket.name: basket.orders})
        return out

    def get_constituent_orders(self):
        expected_position = {LONG: self.informed_constituent.position_limit, SHORT: -self.informed_constituent.position_limit}.get(self.informed_direction, 0)
        remaining_volume = expected_position - self.informed_constituent.initial_position

        if remaining_volume > 0:
            self.informed_constituent.bid(self.informed_constituent.ask_wall, remaining_volume)
        elif remaining_volume < 0:
            self.informed_constituent.ask(self.informed_constituent.bid_wall, -remaining_volume)

        out = {self.informed_constituent.name: self.informed_constituent.orders}

        for hedging_constituent in self.hedging_constituents:
            expected_hedge_position = 0
            for b_idx, basket in enumerate(self.baskets):
                etf_const_factor = ETF_CONSTITUENT_FACTORS[b_idx][ETF_CONSTITUENT_SYMBOLS.index(hedging_constituent.name)]
                expected_hedge_position += -basket.expected_position * etf_const_factor * ETF_HEDGE_FACTOR
            remaining_volume = round(expected_hedge_position - hedging_constituent.initial_position)

            if remaining_volume > 0:
                hedging_constituent.bid(hedging_constituent.ask_wall, remaining_volume)
            elif remaining_volume < 0:
                hedging_constituent.ask(hedging_constituent.bid_wall, -remaining_volume)
            out[hedging_constituent.name] = hedging_constituent.orders
        return out

    def get_orders(self):
        return {**self.get_basket_orders(), **self.get_constituent_orders()}


class OptionTrader:
    """IV scalping + mean reversion on VOLCANIC_ROCK options."""
    def __init__(self, state, prints, new_trader_data):
        self.options = [ProductTrader(os, state, prints, new_trader_data, product_group='OPTION') for os in OPTION_SYMBOLS]
        self.underlying = ProductTrader(OPTION_UNDERLYING_SYMBOL, state, prints, new_trader_data, product_group='OPTION')
        self.state = state
        self.last_traderData = self.underlying.last_traderData
        self.new_trader_data = new_trader_data
        self.indicators = self.calculate_indicators()

    def get_option_values(self, S, K, TTE):
        def bs_call(S, K, TTE, s, r=0):
            d1 = (math.log(S/K) + (r + 0.5 * s**2) * TTE) / (s * TTE**0.5)
            d2 = d1 - s * TTE**0.5
            return S * _N.cdf(d1) - K * math.exp(-r * TTE) * _N.cdf(d2), _N.cdf(d1)

        def bs_vega(S, K, TTE, s, r=0):
            d1 = (math.log(S/K) + (r + 0.5*s**2) * TTE) / (s * TTE**0.5)
            return S * _N.pdf(d1) * TTE**0.5

        def get_iv(St, K, TTE):
            m_t_k = np.log(K/St) / TTE**0.5
            coeffs = [0.27362531, 0.01007566, 0.14876677]
            iv = np.poly1d(coeffs)(m_t_k)
            return iv

        iv = get_iv(S, K, TTE)
        bs_call_value, delta = bs_call(S, K, TTE, iv)
        vega = bs_vega(S, K, TTE, iv)
        return bs_call_value, delta, vega

    def calculate_ema(self, td_key, window, value):
        old_mean = self.last_traderData.get(td_key, 0)
        alpha = 2/(window+1)
        new_mean = alpha * value + (1 - alpha) * old_mean
        self.new_trader_data[td_key] = new_mean
        return new_mean

    def calculate_indicators(self):
        indicators = {
            'ema_u_dev': None, 'ema_o_dev': None,
            'mean_theo_diffs': {}, 'current_theo_diffs': {},
            'switch_means': {}, 'deltas': {}, 'vegas': {},
        }

        if self.underlying.wall_mid is not None:
            new_mean_price = self.calculate_ema('ema_u', underlying_mean_reversion_window, self.underlying.wall_mid)
            indicators['ema_u_dev'] = self.underlying.wall_mid - new_mean_price
            new_mean_price = self.calculate_ema('ema_o', options_mean_reversion_window, self.underlying.wall_mid)
            indicators['ema_o_dev'] = self.underlying.wall_mid - new_mean_price

            for option in self.options:
                k = int(option.name.split('_')[-1])
                if option.wall_mid is None:
                    if option.ask_wall is not None:
                        option.wall_mid = option.ask_wall - 0.5
                        option.bid_wall = option.ask_wall - 1
                        option.best_bid = option.ask_wall - 1
                    elif option.bid_wall is not None:
                        option.wall_mid = option.bid_wall + 0.5
                        option.ask_wall = option.bid_wall + 1
                        option.best_ask = option.bid_wall + 1

                if option.wall_mid is not None:
                    tte = 1 - (DAYS_PER_YEAR - 8 + DAY + self.state.timestamp // 100 / 10_000) / DAYS_PER_YEAR
                    underlying = self.underlying.best_bid * 0.5 + self.underlying.best_ask * 0.5
                    option_theo, option_delta, option_vega = self.get_option_values(underlying, k, tte)
                    option_theo_diff = option.wall_mid - option_theo

                    indicators['current_theo_diffs'][option.name] = option_theo_diff
                    indicators['deltas'][option.name] = option_delta
                    indicators['vegas'][option.name] = option_vega

                    new_mean_diff = self.calculate_ema(f'{option.name}_theo_diff', THEO_NORM_WINDOW, option_theo_diff)
                    indicators['mean_theo_diffs'][option.name] = new_mean_diff

                    new_mean_avg_dev = self.calculate_ema(f'{option.name}_avg_devs', IV_SCALPING_WINDOW, abs(option_theo_diff - new_mean_diff))
                    indicators['switch_means'][option.name] = new_mean_avg_dev

        return indicators

    def get_iv_scalping_orders(self, options):
        out = {}
        for option in options:
            if option.name in self.indicators['mean_theo_diffs'] and option.name in self.indicators['current_theo_diffs'] and option.name in self.new_switch_mean:
                if self.new_switch_mean[option.name] >= IV_SCALPING_THR:
                    current_theo_diff = self.indicators['current_theo_diffs'][option.name]
                    mean_theo_diff = self.indicators['mean_theo_diffs'][option.name]
                    low_vega_adj = 0
                    if self.vegas.get(option.name, 0) <= 1:
                        low_vega_adj = LOW_VEGA_THR_ADJ

                    if current_theo_diff - option.wall_mid + option.best_bid - mean_theo_diff >= (THR_OPEN + low_vega_adj) and option.max_allowed_sell_volume > 0:
                        option.ask(option.best_bid, option.max_allowed_sell_volume)
                    if current_theo_diff - option.wall_mid + option.best_bid - mean_theo_diff >= THR_CLOSE and option.initial_position > 0:
                        option.ask(option.best_bid, option.initial_position)
                    elif current_theo_diff - option.wall_mid + option.best_ask - mean_theo_diff <= -(THR_OPEN + low_vega_adj) and option.max_allowed_buy_volume > 0:
                        option.bid(option.best_ask, option.max_allowed_buy_volume)
                    if current_theo_diff - option.wall_mid + option.best_ask - mean_theo_diff <= -THR_CLOSE and option.initial_position < 0:
                        option.bid(option.best_ask, -option.initial_position)
                else:
                    if option.initial_position > 0:
                        option.ask(option.best_bid, option.initial_position)
                    elif option.initial_position < 0:
                        option.bid(option.best_ask, -option.initial_position)
            out[option.name] = option.orders
        return out

    def get_mr_orders(self, options):
        out = {}
        for option in options:
            if option.name in self.indicators['current_theo_diffs'] and option.name in self.indicators['mean_theo_diffs'] and self.indicators.get('ema_o_dev') is not None:
                current_deviation = self.indicators['ema_o_dev']
                iv_deviation = self.indicators['current_theo_diffs'][option.name] - self.indicators['mean_theo_diffs'][option.name]
                current_deviation += iv_deviation

                if current_deviation > options_mean_reversion_thr and option.max_allowed_sell_volume > 0:
                    option.ask(option.best_bid, option.max_allowed_sell_volume)
                elif current_deviation < -options_mean_reversion_thr and option.max_allowed_buy_volume > 0:
                    option.bid(option.best_ask, option.max_allowed_buy_volume)
                out[option.name] = option.orders
        return out

    def get_option_orders(self):
        if self.state.timestamp / 100 < min([THEO_NORM_WINDOW, underlying_mean_reversion_window, options_mean_reversion_window]): return {}
        iv_scalping_options = [o for o in self.options if int(o.name.split('_')[-1]) >= 9750]
        mr_options = [o for o in self.options if o.name.endswith('9500')]
        return {**self.get_iv_scalping_orders(iv_scalping_options), **self.get_mr_orders(mr_options)}

    def get_underlying_orders(self):
        if self.state.timestamp / 100 < underlying_mean_reversion_window: return {}
        if self.indicators.get('ema_u_dev') is not None:
            current_deviation = self.indicators['ema_o_dev']
            if current_deviation > underlying_mean_reversion_thr and self.underlying.max_allowed_sell_volume > 0:
                self.underlying.ask(self.underlying.bid_wall + 1, self.underlying.max_allowed_sell_volume)
            elif current_deviation < -underlying_mean_reversion_thr and self.underlying.max_allowed_buy_volume > 0:
                self.underlying.bid(self.underlying.ask_wall - 1, self.underlying.max_allowed_buy_volume)
        return {self.underlying.name: self.underlying.orders}

    def get_orders(self):
        return {**self.get_option_orders(), **self.get_underlying_orders()}


class CommodityTrader(ProductTrader):
    """MAGNIFICENT_MACARONS — conversion arbitrage."""
    def __init__(self, state, prints, new_trader_data):
        super().__init__(COMMODITY_SYMBOL, state, prints, new_trader_data)
        self.conversions = 0

    def get_orders(self):
        conv_obs = self.state.observations.conversionObservations[self.name]
        ex_raw_bid, ex_raw_ask = conv_obs.bidPrice, conv_obs.askPrice
        transport_fees = conv_obs.transportFees
        export_tariff = conv_obs.exportTariff
        import_tariff = conv_obs.importTariff

        local_sell_price = math.floor(ex_raw_bid + 0.5)
        local_buy_price = math.ceil(ex_raw_ask - 0.5)
        ex_ask = (ex_raw_ask + import_tariff + transport_fees)
        ex_bid = (ex_raw_bid - export_tariff - transport_fees)

        short_arbitrage = round(local_sell_price - ex_ask, 1)
        long_arbitrage = round(ex_bid - local_buy_price - 0.1, 1)

        short_arbs_hist = self.last_traderData.get('SA', [])
        long_arbs_hist = self.last_traderData.get('LA', [])
        if len(short_arbs_hist) > 10:
            short_arbs_hist.pop(0)
            long_arbs_hist.pop(0)
        short_arbs_hist.append(short_arbitrage)
        long_arbs_hist.append(long_arbitrage)
        self.new_trader_data['SA'] = short_arbs_hist
        self.new_trader_data['LA'] = long_arbs_hist

        mean_short_arb_hist = np.mean(short_arbs_hist)
        mean_long_arb_hist = np.mean(long_arbs_hist)

        if short_arbitrage > long_arbitrage:
            if short_arbitrage >= 0 and mean_short_arb_hist > 0:
                remaining_volume = CONVERSION_LIMIT
                for bp, bv in self.mkt_buy_orders.items():
                    if (short_arbitrage - (local_sell_price - bp)) > (0.58 * short_arbitrage):
                        v = min(remaining_volume, bv)
                        self.ask(bp, v)
                        remaining_volume -= v
                    else: break
                if remaining_volume > 0:
                    self.ask(local_sell_price, remaining_volume)
        else:
            if long_arbitrage >= 0 and mean_long_arb_hist > 0:
                remaining_volume = CONVERSION_LIMIT
                for ap, av in self.mkt_sell_orders.items():
                    if (long_arbitrage - (ap - local_buy_price)) > (0.58 * long_arbitrage):
                        v = min(remaining_volume, av)
                        self.bid(ap, v)
                        remaining_volume -= v
                    else: break
                if remaining_volume > 0:
                    self.bid(local_buy_price, remaining_volume)

        self.conversions = max(min(-self.initial_position, CONVERSION_LIMIT), -CONVERSION_LIMIT)
        return {self.name: self.orders}

    def get_conversions(self):
        return self.conversions


class Trader:
    def run(self, state: TradingState):
        result = {}
        new_trader_data = {}
        prints = {"GENERAL": {"TIMESTAMP": state.timestamp, "POSITIONS": state.position}}

        def export(prints):
            try: print(json.dumps(prints))
            except: pass

        product_traders = {
            STATIC_SYMBOL: StaticTrader,
            DYNAMIC_SYMBOL: DynamicTrader,
            INK_SYMBOL: InkTrader,
            ETF_BASKET_SYMBOLS[0]: EtfTrader,
            OPTION_UNDERLYING_SYMBOL: OptionTrader,
            COMMODITY_SYMBOL: CommodityTrader,
        }

        result, conversions = {}, 0
        for symbol, product_trader in product_traders.items():
            if symbol in state.order_depths:
                try:
                    trader = product_trader(state, prints, new_trader_data)
                    result.update(trader.get_orders())
                    if symbol == COMMODITY_SYMBOL:
                        conversions = trader.get_conversions()
                except: pass

        try: final_trader_data = json.dumps(new_trader_data)
        except: final_trader_data = ''

        export(prints)
        return result, conversions, final_trader_data
