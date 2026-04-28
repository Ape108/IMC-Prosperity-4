"""Unit tests for eda_imitation_pnl helpers and decision rule."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from eda_imitation_pnl import (
    BOTS,
    HORIZONS,
    PRODUCTS,
    THRESHOLD_N,
    THRESHOLD_T_STAT,
)


def _make_trades(rows: list[dict]) -> pd.DataFrame:
    """Build a synthetic trades DataFrame with the canonical columns."""
    return pd.DataFrame(
        rows,
        columns=["ts", "day", "bot", "product", "signed_qty"],
    )


def _make_prices(rows: list[dict]) -> pd.DataFrame:
    """Build a synthetic prices DataFrame with the canonical columns."""
    return pd.DataFrame(
        rows,
        columns=["ts", "day", "product", "bid_price_1", "ask_price_1", "mid_price"],
    )


def test_cell_passes_true_when_all_gates_clear():
    from eda_imitation_pnl import cell_passes
    # n=10, mean=2.0, std=1.0, cost=1.0 → t = sqrt(10) * 2.0 / 1.0 = 6.32
    # n>=10 ✓, mean>=cost ✓, |t|>=2.0 ✓
    assert cell_passes(n=10, mean_pnl=2.0, std_pnl=1.0, cost=1.0) is True


def test_cell_passes_true_when_mean_equals_cost_exactly():
    from eda_imitation_pnl import cell_passes
    # Boundary: mean exactly equals cost. Per spec rule (mean >= cost), passes.
    # n=10, mean=1.5, std=0.5, cost=1.5 → t = sqrt(10) * 1.5 / 0.5 = 9.49
    assert cell_passes(n=10, mean_pnl=1.5, std_pnl=0.5, cost=1.5) is True


def test_cell_passes_false_when_n_below_threshold():
    from eda_imitation_pnl import cell_passes
    assert cell_passes(n=9, mean_pnl=2.0, std_pnl=1.0, cost=1.0) is False


def test_cell_passes_false_when_mean_just_below_cost():
    from eda_imitation_pnl import cell_passes
    # mean fractionally below cost
    assert cell_passes(n=10, mean_pnl=0.999, std_pnl=0.5, cost=1.0) is False


def test_cell_passes_false_when_t_stat_below_threshold():
    from eda_imitation_pnl import cell_passes
    # n=10, mean=1.0, std=10.0, cost=1.0 → t = sqrt(10)*1.0/10.0 ≈ 0.316
    assert cell_passes(n=10, mean_pnl=1.0, std_pnl=10.0, cost=1.0) is False


def test_cell_passes_false_when_std_zero():
    from eda_imitation_pnl import cell_passes
    # Degenerate: std=0 → can't compute t-stat, never a pass.
    assert cell_passes(n=10, mean_pnl=2.0, std_pnl=0.0, cost=1.0) is False


def test_cell_passes_false_when_negative_mean():
    from eda_imitation_pnl import cell_passes
    # Sign-aware mean is negative → imitation loses → never passes a positive cost gate.
    assert cell_passes(n=10, mean_pnl=-2.0, std_pnl=1.0, cost=1.0) is False


def test_cell_passes_true_when_t_stat_equals_threshold_exactly():
    from eda_imitation_pnl import cell_passes
    # t = sqrt(10) * mean / std = 2.0 exactly → mean/std = 2/sqrt(10)
    # Use mean=2.0, std=math.sqrt(10) so t-stat is exactly 2.0.
    # Boundary: |t|>=2.0 must be inclusive at 2.0 → True.
    std = math.sqrt(10)
    assert cell_passes(n=10, mean_pnl=2.0, std_pnl=std, cost=1.0) is True


def test_cell_passes_false_when_t_stat_just_below_threshold():
    from eda_imitation_pnl import cell_passes
    # Probe `>=` vs `>` near the t-stat boundary. Construct t-stat ≈ 1.999.
    # t = sqrt(10) * mean / std = 1.999 → std = sqrt(10) * mean / 1.999
    mean = 2.0
    std = math.sqrt(10) * mean / 1.999
    assert cell_passes(n=10, mean_pnl=mean, std_pnl=std, cost=1.0) is False


def test_median_round_trip_cost_returns_median_of_book_spreads():
    from eda_imitation_pnl import median_round_trip_cost
    prices = _make_prices([
        {"ts": 100, "day": 1, "product": "VEV_5300",
         "bid_price_1": 50, "ask_price_1": 51, "mid_price": 50.5},
        {"ts": 200, "day": 1, "product": "VEV_5300",
         "bid_price_1": 50, "ask_price_1": 52, "mid_price": 51.0},
        {"ts": 300, "day": 1, "product": "VEV_5300",
         "bid_price_1": 50, "ask_price_1": 53, "mid_price": 51.5},
    ])
    # Spreads: [1, 2, 3] → median 2.0
    assert median_round_trip_cost(prices, "VEV_5300") == 2.0


def test_median_round_trip_cost_filters_by_product():
    from eda_imitation_pnl import median_round_trip_cost
    prices = _make_prices([
        {"ts": 100, "day": 1, "product": "VEV_5300",
         "bid_price_1": 50, "ask_price_1": 51, "mid_price": 50.5},
        {"ts": 100, "day": 1, "product": "VEV_5400",
         "bid_price_1": 30, "ask_price_1": 35, "mid_price": 32.5},
    ])
    assert median_round_trip_cost(prices, "VEV_5400") == 5.0


def test_median_round_trip_cost_returns_nan_for_empty_product():
    from eda_imitation_pnl import median_round_trip_cost
    prices = _make_prices([
        {"ts": 100, "day": 1, "product": "VEV_5300",
         "bid_price_1": 50, "ask_price_1": 51, "mid_price": 50.5},
    ])
    assert math.isnan(median_round_trip_cost(prices, "DOES_NOT_EXIST"))


def test_median_round_trip_cost_handles_single_tick():
    from eda_imitation_pnl import median_round_trip_cost
    prices = _make_prices([
        {"ts": 100, "day": 1, "product": "VEV_5300",
         "bid_price_1": 50, "ask_price_1": 51, "mid_price": 50.5},
    ])
    assert median_round_trip_cost(prices, "VEV_5300") == 1.0


def test_median_round_trip_cost_returns_nan_when_all_spreads_are_nan():
    from eda_imitation_pnl import median_round_trip_cost
    prices = _make_prices([
        {"ts": 100, "day": 1, "product": "VEV_5300",
         "bid_price_1": float("nan"), "ask_price_1": float("nan"),
         "mid_price": 50.5},
    ])
    assert math.isnan(median_round_trip_cost(prices, "VEV_5300"))


def test_compute_imitation_pnl_buy_returns_positive_when_price_rises():
    from eda_imitation_pnl import compute_imitation_pnl
    trades = _make_trades([
        {"ts": 100, "day": 1, "bot": "Mark 14", "product": "VEV_5300",
         "signed_qty": 5},
    ])
    prices = _make_prices([
        {"ts": 100, "day": 1, "product": "VEV_5300",
         "bid_price_1": 9, "ask_price_1": 11, "mid_price": 10.0},
        {"ts": 110, "day": 1, "product": "VEV_5300",
         "bid_price_1": 10, "ask_price_1": 12, "mid_price": 11.0},
        {"ts": 120, "day": 1, "product": "VEV_5300",
         "bid_price_1": 11, "ask_price_1": 13, "mid_price": 12.0},
    ])
    # Trade at ts=100; entry tick is first ts > 100 (=110, mid=11);
    # exit tick at entry_idx+1 = idx 2 (mid=12). PnL = +1 * (12 - 11) = 1.0.
    pnls = compute_imitation_pnl(trades, prices, bot="Mark 14",
                                 product="VEV_5300", horizon=1)
    assert pnls == [1.0]


def test_compute_imitation_pnl_sell_inverts_sign():
    from eda_imitation_pnl import compute_imitation_pnl
    trades = _make_trades([
        {"ts": 100, "day": 1, "bot": "Mark 14", "product": "VEV_5300",
         "signed_qty": -5},
    ])
    prices = _make_prices([
        {"ts": 100, "day": 1, "product": "VEV_5300",
         "bid_price_1": 9, "ask_price_1": 11, "mid_price": 10.0},
        {"ts": 110, "day": 1, "product": "VEV_5300",
         "bid_price_1": 10, "ask_price_1": 12, "mid_price": 11.0},
        {"ts": 120, "day": 1, "product": "VEV_5300",
         "bid_price_1": 11, "ask_price_1": 13, "mid_price": 12.0},
    ])
    # Bot sold; we sell at entry, buy back at exit. Same up-move → -1.0.
    pnls = compute_imitation_pnl(trades, prices, bot="Mark 14",
                                 product="VEV_5300", horizon=1)
    assert pnls == [-1.0]


def test_compute_imitation_pnl_uses_first_tick_strictly_after_trade_ts():
    from eda_imitation_pnl import compute_imitation_pnl
    # Trade at ts=110 (an exact price-tick timestamp). 1-tick lag means entry
    # is the next tick (120, mid=12), not the same tick (110, mid=11).
    trades = _make_trades([
        {"ts": 110, "day": 1, "bot": "Mark 14", "product": "VEV_5300",
         "signed_qty": 5},
    ])
    prices = _make_prices([
        {"ts": 100, "day": 1, "product": "VEV_5300",
         "bid_price_1": 9, "ask_price_1": 11, "mid_price": 10.0},
        {"ts": 110, "day": 1, "product": "VEV_5300",
         "bid_price_1": 10, "ask_price_1": 12, "mid_price": 11.0},
        {"ts": 120, "day": 1, "product": "VEV_5300",
         "bid_price_1": 11, "ask_price_1": 13, "mid_price": 12.0},
        {"ts": 130, "day": 1, "product": "VEV_5300",
         "bid_price_1": 12, "ask_price_1": 14, "mid_price": 13.0},
    ])
    # Entry = ts=120 (mid=12), exit = entry_idx+1 = idx 3 (ts=130, mid=13).
    # PnL = +1 * (13 - 12) = 1.0.
    pnls = compute_imitation_pnl(trades, prices, bot="Mark 14",
                                 product="VEV_5300", horizon=1)
    assert pnls == [1.0]


def test_compute_imitation_pnl_skips_when_exit_past_last_tick():
    from eda_imitation_pnl import compute_imitation_pnl
    trades = _make_trades([
        {"ts": 110, "day": 1, "bot": "Mark 14", "product": "VEV_5300",
         "signed_qty": 5},
    ])
    prices = _make_prices([
        {"ts": 100, "day": 1, "product": "VEV_5300",
         "bid_price_1": 9, "ask_price_1": 11, "mid_price": 10.0},
        {"ts": 110, "day": 1, "product": "VEV_5300",
         "bid_price_1": 10, "ask_price_1": 12, "mid_price": 11.0},
        {"ts": 120, "day": 1, "product": "VEV_5300",
         "bid_price_1": 11, "ask_price_1": 13, "mid_price": 12.0},
    ])
    # Entry = idx 2 (ts=120). horizon=2 → exit_idx=4, len=3 → skip.
    pnls = compute_imitation_pnl(trades, prices, bot="Mark 14",
                                 product="VEV_5300", horizon=2)
    assert pnls == []


def test_compute_imitation_pnl_does_not_span_day_boundary():
    from eda_imitation_pnl import compute_imitation_pnl
    # Day 1 trade near end of day 1 prices. Day 2 prices exist with very
    # different mid, but should NOT be used as exit.
    trades = _make_trades([
        {"ts": 1_000_100, "day": 1, "bot": "Mark 14",
         "product": "VEV_5300", "signed_qty": 5},
    ])
    prices = _make_prices([
        # Day 1
        {"ts": 1_000_100, "day": 1, "product": "VEV_5300",
         "bid_price_1": 9, "ask_price_1": 11, "mid_price": 10.0},
        {"ts": 1_000_200, "day": 1, "product": "VEV_5300",
         "bid_price_1": 10, "ask_price_1": 12, "mid_price": 11.0},
        # Day 2 (large jump)
        {"ts": 2_000_000, "day": 2, "product": "VEV_5300",
         "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100.0},
        {"ts": 2_000_100, "day": 2, "product": "VEV_5300",
         "bid_price_1": 100, "ask_price_1": 102, "mid_price": 101.0},
    ])
    # Per-day slicing: day 1 has 2 ticks. Trade at ts=1_000_100 → entry is
    # first ts > 1_000_100 within day 1 = idx 1 (mid=11). horizon=1 → exit
    # idx=2 within day 1 prices, len=2 → skip. Day 2 prices are NEVER used.
    pnls = compute_imitation_pnl(trades, prices, bot="Mark 14",
                                 product="VEV_5300", horizon=1)
    assert pnls == []


def test_compute_imitation_pnl_filters_by_bot_and_product():
    from eda_imitation_pnl import compute_imitation_pnl
    trades = _make_trades([
        {"ts": 100, "day": 1, "bot": "Mark 22", "product": "VEV_5300",
         "signed_qty": 5},
        {"ts": 100, "day": 1, "bot": "Mark 14", "product": "VEV_5400",
         "signed_qty": 5},
    ])
    prices = _make_prices([
        {"ts": 100, "day": 1, "product": "VEV_5300",
         "bid_price_1": 9, "ask_price_1": 11, "mid_price": 10.0},
        {"ts": 110, "day": 1, "product": "VEV_5300",
         "bid_price_1": 10, "ask_price_1": 12, "mid_price": 11.0},
        {"ts": 120, "day": 1, "product": "VEV_5300",
         "bid_price_1": 11, "ask_price_1": 13, "mid_price": 12.0},
    ])
    # Querying bot=Mark 14, product=VEV_5300 → Mark 22's VEV_5300 trade
    # is excluded by bot, Mark 14's VEV_5400 trade is excluded by product.
    pnls = compute_imitation_pnl(trades, prices, bot="Mark 14",
                                 product="VEV_5300", horizon=1)
    assert pnls == []


def test_emit_report_on_synthetic_dataset_emits_a_verdict_token():
    """Tiny synthetic dataset (3 bots × 2 products × 2 horizons): assert the
    verdict text contains either 'PASS:' or 'NO PASS'. Confirms the wiring
    from compute_cells → emit_report is intact end-to-end.
    """
    from eda_imitation_pnl import compute_cells, emit_report

    trades = _make_trades([
        # Mark 14 buys VEV_5300 across two days
        {"ts": 100, "day": 1, "bot": "Mark 14", "product": "VEV_5300",
         "signed_qty": 5},
        {"ts": 1_000_100, "day": 2, "bot": "Mark 14", "product": "VEV_5300",
         "signed_qty": 5},
        # Mark 22 buys HYDROGEL_PACK once
        {"ts": 100, "day": 1, "bot": "Mark 22", "product": "HYDROGEL_PACK",
         "signed_qty": 3},
        # Mark 38 sells VEV_5300 once
        {"ts": 200, "day": 1, "bot": "Mark 38", "product": "VEV_5300",
         "signed_qty": -3},
    ])

    rows = []
    for day_offset, day in [(0, 1), (1_000_000, 2)]:
        for product, base in [("VEV_5300", 100.0), ("HYDROGEL_PACK", 10000.0)]:
            for i in range(20):
                ts = day_offset + 100 + i * 10
                mid = base + i * 0.5
                rows.append({
                    "ts": ts, "day": day, "product": product,
                    "bid_price_1": mid - 0.5,
                    "ask_price_1": mid + 0.5,
                    "mid_price": mid,
                })
    prices = _make_prices(rows)

    cells = compute_cells(
        trades, prices,
        bots=["Mark 14", "Mark 22", "Mark 38"],
        products=["VEV_5300", "HYDROGEL_PACK"],
        horizons=[1, 2],
    )
    report = emit_report(cells)

    assert "=== VERDICT ===" in report
    assert ("PASS:" in report) or ("NO PASS" in report)


def test_constants_exported_match_spec_scope():
    # Sanity: 7 bots, 11 products (VEF excluded), 4 horizons.
    assert len(BOTS) == 7
    assert len(PRODUCTS) == 11
    assert "VELVETFRUIT_EXTRACT" not in PRODUCTS
    assert HORIZONS == [5, 10, 20, 50]


def test_compute_imitation_pnl_returns_empty_for_empty_inputs():
    from eda_imitation_pnl import compute_imitation_pnl
    empty_trades = _make_trades([])
    empty_prices = _make_prices([])
    assert compute_imitation_pnl(empty_trades, empty_prices,
                                 bot="Mark 14", product="VEV_5300",
                                 horizon=5) == []

    # Non-empty trades but empty prices.
    trades = _make_trades([
        {"ts": 100, "day": 1, "bot": "Mark 14", "product": "VEV_5300",
         "signed_qty": 5},
    ])
    assert compute_imitation_pnl(trades, empty_prices,
                                 bot="Mark 14", product="VEV_5300",
                                 horizon=5) == []


def test_compute_imitation_pnl_pools_pnls_across_days():
    from eda_imitation_pnl import compute_imitation_pnl
    trades = _make_trades([
        {"ts": 1_000_100, "day": 1, "bot": "Mark 14",
         "product": "VEV_5300", "signed_qty": 5},
        {"ts": 2_000_100, "day": 2, "bot": "Mark 14",
         "product": "VEV_5300", "signed_qty": 5},
    ])
    prices = _make_prices([
        {"ts": 1_000_100, "day": 1, "product": "VEV_5300",
         "bid_price_1": 9, "ask_price_1": 11, "mid_price": 10.0},
        {"ts": 1_000_200, "day": 1, "product": "VEV_5300",
         "bid_price_1": 10, "ask_price_1": 12, "mid_price": 11.0},
        {"ts": 1_000_300, "day": 1, "product": "VEV_5300",
         "bid_price_1": 11, "ask_price_1": 13, "mid_price": 12.0},
        {"ts": 2_000_100, "day": 2, "product": "VEV_5300",
         "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100.0},
        {"ts": 2_000_200, "day": 2, "product": "VEV_5300",
         "bid_price_1": 100, "ask_price_1": 102, "mid_price": 101.0},
        {"ts": 2_000_300, "day": 2, "product": "VEV_5300",
         "bid_price_1": 101, "ask_price_1": 103, "mid_price": 102.0},
    ])
    # Day 1: entry mid=11, exit mid=12 → +1.0. Day 2: entry mid=101, exit
    # mid=102 → +1.0. Both pooled into a 2-element list.
    pnls = compute_imitation_pnl(trades, prices, bot="Mark 14",
                                 product="VEV_5300", horizon=1)
    assert pnls == [1.0, 1.0]


def test_compute_imitation_pnl_dedups_same_ts_trades_to_one_observation():
    from eda_imitation_pnl import compute_imitation_pnl
    # Two rows at the same (bot, product, ts) — multi-level lift of the
    # book in one tick. Should yield ONE PnL observation, not two.
    trades = _make_trades([
        {"ts": 100, "day": 1, "bot": "Mark 14", "product": "VEV_5300",
         "signed_qty": 5},
        {"ts": 100, "day": 1, "bot": "Mark 14", "product": "VEV_5300",
         "signed_qty": 3},
    ])
    prices = _make_prices([
        {"ts": 100, "day": 1, "product": "VEV_5300",
         "bid_price_1": 9, "ask_price_1": 11, "mid_price": 10.0},
        {"ts": 110, "day": 1, "product": "VEV_5300",
         "bid_price_1": 10, "ask_price_1": 12, "mid_price": 11.0},
        {"ts": 120, "day": 1, "product": "VEV_5300",
         "bid_price_1": 11, "ask_price_1": 13, "mid_price": 12.0},
    ])
    pnls = compute_imitation_pnl(trades, prices, bot="Mark 14",
                                 product="VEV_5300", horizon=1)
    assert pnls == [1.0]


def test_compute_imitation_pnl_skips_zero_quantity_trade():
    from eda_imitation_pnl import compute_imitation_pnl
    # signed_qty=0 has no defined direction — skip silently.
    trades = _make_trades([
        {"ts": 100, "day": 1, "bot": "Mark 14", "product": "VEV_5300",
         "signed_qty": 0},
    ])
    prices = _make_prices([
        {"ts": 100, "day": 1, "product": "VEV_5300",
         "bid_price_1": 9, "ask_price_1": 11, "mid_price": 10.0},
        {"ts": 110, "day": 1, "product": "VEV_5300",
         "bid_price_1": 10, "ask_price_1": 12, "mid_price": 11.0},
        {"ts": 120, "day": 1, "product": "VEV_5300",
         "bid_price_1": 11, "ask_price_1": 13, "mid_price": 12.0},
    ])
    pnls = compute_imitation_pnl(trades, prices, bot="Mark 14",
                                 product="VEV_5300", horizon=1)
    assert pnls == []
