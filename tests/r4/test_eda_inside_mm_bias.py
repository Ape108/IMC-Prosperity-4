"""Unit tests for eda_inside_mm_bias helpers and decision rule."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


def test_load_market_trades_columns_and_ts(tmp_path):
    from eda_inside_mm_bias import load_market_trades
    csv = tmp_path / "trades_round_4_day_1.csv"
    csv.write_text(
        "timestamp;buyer;seller;symbol;currency;price;quantity\n"
        "100;Mark 14;Mark 38;HYDROGEL_PACK;Seashell;9999;5\n"
        "200;Mark 38;Mark 14;HYDROGEL_PACK;Seashell;10001;3\n",
        encoding="utf-8",
    )
    df = load_market_trades(days=[1], dataset_dir=str(tmp_path))
    assert list(df.columns) == ["ts", "day", "product", "price"]
    assert len(df) == 2
    # ts = day * 1_000_000 + raw_timestamp
    assert df["ts"].tolist() == [1_000_100, 1_000_200]
    assert df["product"].tolist() == ["HYDROGEL_PACK", "HYDROGEL_PACK"]
    assert df["price"].tolist() == [9999.0, 10001.0]


def test_parse_top_k_cells_filters_and_assigns_direction(tmp_path):
    from eda_inside_mm_bias import parse_top_k_cells, TopKCell
    verdict = tmp_path / "v.txt"
    verdict.write_text(
        "Mark 01   VEV_5400        horizon=50  n= 262  mean_t=  +0.160  cost= 1.000  net_t=  -0.840  t=  +2.26  hit= 0.42  PASS=no\n"
        "Mark 22   VEV_5400        horizon=50  n= 275  mean_t=  -0.133  cost= 1.000  net_t=  -1.133  t=  -1.83  hit= 0.32  PASS=no\n"
        "Mark 14   VEV_5500        horizon= 5  n=   7  mean_t=  +0.071  cost= 1.000  net_t=  -0.929  t=  +0.55  hit= 0.29  PASS=no\n"
        "Mark 38   VEV_5300        horizon= 5  n=   2  mean_t=  -1.000  cost= 2.000  net_t=  -3.000  t=   nan   hit= 0.00  PASS=no\n",
        encoding="utf-8",
    )
    cells = parse_top_k_cells(verdict_path=str(verdict), n_min=30, t_min=1.5)
    assert len(cells) == 2
    by_bot = {c.bot: c for c in cells}
    assert by_bot["Mark 01"] == TopKCell(
        bot="Mark 01", product="VEV_5400", horizon=50,
        mean_t=0.160, t_stat=2.26, direction="follow",
    )
    assert by_bot["Mark 22"] == TopKCell(
        bot="Mark 22", product="VEV_5400", horizon=50,
        mean_t=-0.133, t_stat=-1.83, direction="fade",
    )


def test_parse_top_k_cells_excludes_mark38_hydrogel_when_mark14_present(tmp_path):
    from eda_inside_mm_bias import parse_top_k_cells
    verdict = tmp_path / "v.txt"
    verdict.write_text(
        "Mark 14   HYDROGEL_PACK   horizon=50  n=1002  mean_t=  +0.732  cost=16.000  net_t= -15.268  t=  +1.74  hit= 0.52  PASS=no\n"
        "Mark 38   HYDROGEL_PACK   horizon=50  n=1021  mean_t=  -0.658  cost=16.000  net_t= -16.658  t=  -1.58  hit= 0.47  PASS=no\n",
        encoding="utf-8",
    )
    cells = parse_top_k_cells(verdict_path=str(verdict), n_min=30, t_min=1.5)
    assert len(cells) == 1
    assert cells[0].bot == "Mark 14"


def test_parse_top_k_cells_keeps_mark38_hydrogel_when_mark14_absent(tmp_path):
    from eda_inside_mm_bias import parse_top_k_cells
    verdict = tmp_path / "v.txt"
    verdict.write_text(
        "Mark 38   HYDROGEL_PACK   horizon=50  n=1021  mean_t=  -0.658  cost=16.000  net_t= -16.658  t=  -1.58  hit= 0.47  PASS=no\n",
        encoding="utf-8",
    )
    cells = parse_top_k_cells(verdict_path=str(verdict), n_min=30, t_min=1.5)
    assert len(cells) == 1
    assert cells[0].bot == "Mark 38"
    assert cells[0].direction == "fade"


def test_parse_top_k_cells_skips_nan_t_stat_even_with_high_n(tmp_path):
    """Regression: a row with n>=n_min but t=nan must be excluded.

    Real verdict data contains rows like:
        Mark 01   VEV_6500   horizon=50  n=316  ...  t=   nan
    These must not slip through with t_stat=nan.
    """
    from eda_inside_mm_bias import parse_top_k_cells
    verdict = tmp_path / "v.txt"
    verdict.write_text(
        "Mark 01   VEV_6500        horizon=50  n= 316  mean_t=  +0.000  cost= 1.000  net_t=  -1.000  t=   nan   hit= 0.00  PASS=no\n",
        encoding="utf-8",
    )
    cells = parse_top_k_cells(verdict_path=str(verdict), n_min=30, t_min=1.5)
    assert cells == []


def _make_events(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ts", "day", "bot", "product", "signed_qty"])


def _make_prices(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["ts", "day", "product", "bid_price_1", "ask_price_1", "mid_price"],
    )


def _make_market_trades(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ts", "day", "product", "price"])


def test_simulate_long_bias_optimistic_fills_at_touch():
    from eda_inside_mm_bias import simulate_passive_fills
    # Long-bias event at ts=100 with day=1, signed_qty=+1.
    # Price ticks at ts=200, 300, 400, 500 with bid=99, ask=101.
    # Market trade at ts=250, price=99 → optimistic fill at first window tick.
    events = _make_events([{"ts": 100, "day": 1, "bot": "B", "product": "X", "signed_qty": 1}])
    prices = _make_prices([
        {"ts": 200, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 300, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 400, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 500, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
    ])
    market = _make_market_trades([{"ts": 250, "day": 1, "product": "X", "price": 99}])
    fills = simulate_passive_fills(
        events=events, prices=prices, market_trades=market,
        product="X", horizon=3, direction="follow", regime="optimistic",
    )
    assert len(fills) == 1
    assert fills[0].fill_price == 99.0


def test_simulate_long_bias_conservative_requires_strict_penetration():
    from eda_inside_mm_bias import simulate_passive_fills
    events = _make_events([{"ts": 100, "day": 1, "bot": "B", "product": "X", "signed_qty": 1}])
    prices = _make_prices([
        {"ts": 200, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 300, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 400, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 500, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
    ])
    # Trade at price=99 (AT bid) — does NOT cross conservative gate.
    market_at = _make_market_trades([{"ts": 250, "day": 1, "product": "X", "price": 99}])
    fills_at = simulate_passive_fills(
        events=events, prices=prices, market_trades=market_at,
        product="X", horizon=3, direction="follow", regime="conservative",
    )
    assert fills_at == []
    # Trade at price=98 (BELOW bid) — penetrates queue, conservative fill.
    market_below = _make_market_trades([{"ts": 250, "day": 1, "product": "X", "price": 98}])
    fills_below = simulate_passive_fills(
        events=events, prices=prices, market_trades=market_below,
        product="X", horizon=3, direction="follow", regime="conservative",
    )
    assert len(fills_below) == 1
    assert fills_below[0].fill_price == 99.0  # we'd be posted at bid=99


def test_simulate_short_bias_optimistic_fills_at_ask():
    from eda_inside_mm_bias import simulate_passive_fills
    events = _make_events([{"ts": 100, "day": 1, "bot": "B", "product": "X", "signed_qty": -1}])  # follow short
    prices = _make_prices([
        {"ts": 200, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 300, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 400, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 500, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
    ])
    market = _make_market_trades([{"ts": 250, "day": 1, "product": "X", "price": 101}])
    fills = simulate_passive_fills(
        events=events, prices=prices, market_trades=market,
        product="X", horizon=3, direction="follow", regime="optimistic",
    )
    assert len(fills) == 1
    assert fills[0].fill_price == 101.0


def test_simulate_fade_direction_inverts_bias():
    from eda_inside_mm_bias import simulate_passive_fills
    # signed_qty=+1 with direction="fade" → short bias (post ask).
    events = _make_events([{"ts": 100, "day": 1, "bot": "B", "product": "X", "signed_qty": 1}])
    prices = _make_prices([
        {"ts": 200, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 300, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 400, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 500, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
    ])
    market = _make_market_trades([{"ts": 250, "day": 1, "product": "X", "price": 101}])
    fills = simulate_passive_fills(
        events=events, prices=prices, market_trades=market,
        product="X", horizon=3, direction="fade", regime="optimistic",
    )
    assert len(fills) == 1
    assert fills[0].fill_price == 101.0


def test_simulate_skips_events_with_window_past_day_end():
    from eda_inside_mm_bias import simulate_passive_fills
    events = _make_events([{"ts": 400, "day": 1, "bot": "B", "product": "X", "signed_qty": 1}])
    # Only 2 price ticks after the event — horizon=3 needs exit at ts[entry+3].
    prices = _make_prices([
        {"ts": 500, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 600, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
    ])
    market = _make_market_trades([{"ts": 500, "day": 1, "product": "X", "price": 99}])
    fills = simulate_passive_fills(
        events=events, prices=prices, market_trades=market,
        product="X", horizon=3, direction="follow", regime="optimistic",
    )
    assert fills == []


def test_simulate_no_qualifying_trade_in_window():
    from eda_inside_mm_bias import simulate_passive_fills
    events = _make_events([{"ts": 100, "day": 1, "bot": "B", "product": "X", "signed_qty": 1}])
    prices = _make_prices([
        {"ts": 200, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 300, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 400, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 500, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
    ])
    # All trades are above the bid — long-bias (post at 99) never fills.
    market = _make_market_trades([
        {"ts": 250, "day": 1, "product": "X", "price": 100},
        {"ts": 350, "day": 1, "product": "X", "price": 100},
    ])
    fills = simulate_passive_fills(
        events=events, prices=prices, market_trades=market,
        product="X", horizon=3, direction="follow", regime="optimistic",
    )
    assert fills == []


def test_simulate_returns_empty_when_inputs_empty():
    from eda_inside_mm_bias import simulate_passive_fills
    empty_events = _make_events([])
    empty_prices = _make_prices([])
    empty_trades = _make_market_trades([])
    assert simulate_passive_fills(
        empty_events, empty_prices, empty_trades, "X", 5, "follow", "optimistic"
    ) == []


def test_compute_pnls_long_bias_positive_move():
    from eda_inside_mm_bias import compute_pnls, Fill
    events = _make_events([{"ts": 100, "day": 1, "bot": "B", "product": "X", "signed_qty": 1}])
    prices = _make_prices([
        {"ts": 200, "day": 1, "product": "X", "bid_price_1": 99,  "ask_price_1": 101, "mid_price": 100},
        {"ts": 300, "day": 1, "product": "X", "bid_price_1": 99,  "ask_price_1": 101, "mid_price": 100},
        {"ts": 400, "day": 1, "product": "X", "bid_price_1": 99,  "ask_price_1": 101, "mid_price": 100},
        {"ts": 500, "day": 1, "product": "X", "bid_price_1": 105, "ask_price_1": 107, "mid_price": 106},
    ])
    fills = [Fill(event_idx=0, fill_tick=300, fill_price=99.0)]
    pnls = compute_pnls(
        fills=fills, events=events, prices=prices,
        product="X", horizon=3, direction="follow",
    )
    assert len(pnls) == 1
    pnl_mid, pnl_flat = pnls[0]
    # event ts=100 → entry_idx=0 (ts=200) → exit_idx = 0 + 3 = 3 (ts=500)
    # long bias: pnl_mid = mid(500) - fill_price = 106 - 99 = 7
    #            pnl_flat = best_bid(500) - fill_price = 105 - 99 = 6
    assert pnl_mid == 7.0
    assert pnl_flat == 6.0


def test_compute_pnls_short_bias_negative_move():
    from eda_inside_mm_bias import compute_pnls, Fill
    # signed_qty=-1, direction="follow" → short bias.
    events = _make_events([{"ts": 100, "day": 1, "bot": "B", "product": "X", "signed_qty": -1}])
    prices = _make_prices([
        {"ts": 200, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 300, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 400, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 500, "day": 1, "product": "X", "bid_price_1": 93, "ask_price_1": 95,  "mid_price": 94},
    ])
    fills = [Fill(event_idx=0, fill_tick=300, fill_price=101.0)]
    pnls = compute_pnls(
        fills=fills, events=events, prices=prices,
        product="X", horizon=3, direction="follow",
    )
    assert len(pnls) == 1
    pnl_mid, pnl_flat = pnls[0]
    # short bias: pnl_mid = fill_price - mid(500) = 101 - 94 = 7
    #             pnl_flat = fill_price - best_ask(500) = 101 - 95 = 6
    assert pnl_mid == 7.0
    assert pnl_flat == 6.0


def test_cell_passes_true_when_all_gates_clear():
    from eda_inside_mm_bias import CellMetrics, cell_passes
    m = CellMetrics(
        n_events=100, n_fills=50, fill_rate=0.50,
        mean_pnl_mid=0.5, std_pnl_mid=1.0, t_stat_mid=5.0,
        mean_pnl_flat=0.2, std_pnl_flat=1.0, t_stat_flat=2.0,
        hit_rate_flat=0.55,
    )
    assert cell_passes(m) is True


def test_cell_passes_false_when_n_fills_below_30():
    from eda_inside_mm_bias import CellMetrics, cell_passes
    m = CellMetrics(
        n_events=100, n_fills=29, fill_rate=0.29,
        mean_pnl_mid=0.5, std_pnl_mid=1.0, t_stat_mid=5.0,
        mean_pnl_flat=0.2, std_pnl_flat=1.0, t_stat_flat=2.0,
        hit_rate_flat=0.55,
    )
    assert cell_passes(m) is False


def test_cell_passes_false_when_fill_rate_below_10pct():
    from eda_inside_mm_bias import CellMetrics, cell_passes
    m = CellMetrics(
        n_events=1000, n_fills=99, fill_rate=0.099,
        mean_pnl_mid=0.5, std_pnl_mid=1.0, t_stat_mid=5.0,
        mean_pnl_flat=0.2, std_pnl_flat=1.0, t_stat_flat=2.0,
        hit_rate_flat=0.55,
    )
    assert cell_passes(m) is False


def test_cell_passes_false_when_pnl_mid_not_positive():
    from eda_inside_mm_bias import CellMetrics, cell_passes
    m = CellMetrics(
        n_events=100, n_fills=50, fill_rate=0.50,
        mean_pnl_mid=0.0, std_pnl_mid=1.0, t_stat_mid=0.0,
        mean_pnl_flat=0.2, std_pnl_flat=1.0, t_stat_flat=2.0,
        hit_rate_flat=0.55,
    )
    assert cell_passes(m) is False


def test_cell_passes_false_when_t_mid_below_2():
    from eda_inside_mm_bias import CellMetrics, cell_passes
    m = CellMetrics(
        n_events=100, n_fills=50, fill_rate=0.50,
        mean_pnl_mid=0.5, std_pnl_mid=1.0, t_stat_mid=1.99,
        mean_pnl_flat=0.2, std_pnl_flat=1.0, t_stat_flat=2.0,
        hit_rate_flat=0.55,
    )
    assert cell_passes(m) is False


def test_cell_passes_false_when_pnl_flat_not_positive():
    from eda_inside_mm_bias import CellMetrics, cell_passes
    m = CellMetrics(
        n_events=100, n_fills=50, fill_rate=0.50,
        mean_pnl_mid=0.5, std_pnl_mid=1.0, t_stat_mid=5.0,
        mean_pnl_flat=0.0, std_pnl_flat=1.0, t_stat_flat=0.0,
        hit_rate_flat=0.55,
    )
    assert cell_passes(m) is False


def test_cell_passes_true_at_exact_thresholds():
    from eda_inside_mm_bias import CellMetrics, cell_passes
    # n_fills=30, fill_rate=0.10, t_mid=2.0 — boundary inclusive.
    m = CellMetrics(
        n_events=300, n_fills=30, fill_rate=0.10,
        mean_pnl_mid=0.5, std_pnl_mid=1.0, t_stat_mid=2.0,
        mean_pnl_flat=0.001, std_pnl_flat=1.0, t_stat_flat=0.5,
        hit_rate_flat=0.55,
    )
    assert cell_passes(m) is True


def test_compute_cell_metrics_aggregates_both_regimes():
    from eda_inside_mm_bias import TopKCell, compute_cell_metrics
    # 2 events, both long-bias follow; one fills optimistically (price=bid),
    # one fills conservatively (price < bid).
    events = _make_events([
        {"ts": 100, "day": 1, "bot": "B", "product": "X", "signed_qty": 1},
        {"ts": 600, "day": 1, "bot": "B", "product": "X", "signed_qty": 1},
    ])
    prices = _make_prices([
        {"ts": 200,  "day": 1, "product": "X", "bid_price_1": 99,  "ask_price_1": 101, "mid_price": 100},
        {"ts": 300,  "day": 1, "product": "X", "bid_price_1": 99,  "ask_price_1": 101, "mid_price": 100},
        {"ts": 400,  "day": 1, "product": "X", "bid_price_1": 99,  "ask_price_1": 101, "mid_price": 100},
        {"ts": 500,  "day": 1, "product": "X", "bid_price_1": 105, "ask_price_1": 107, "mid_price": 106},
        {"ts": 700,  "day": 1, "product": "X", "bid_price_1": 99,  "ask_price_1": 101, "mid_price": 100},
        {"ts": 800,  "day": 1, "product": "X", "bid_price_1": 99,  "ask_price_1": 101, "mid_price": 100},
        {"ts": 900,  "day": 1, "product": "X", "bid_price_1": 99,  "ask_price_1": 101, "mid_price": 100},
        {"ts": 1000, "day": 1, "product": "X", "bid_price_1": 102, "ask_price_1": 104, "mid_price": 103},
    ])
    market = _make_market_trades([
        {"ts": 250, "day": 1, "product": "X", "price": 99},  # AT bid → optimistic only
        {"ts": 750, "day": 1, "product": "X", "price": 98},  # below bid → both regimes
    ])
    cell = TopKCell(bot="B", product="X", horizon=3, mean_t=0.5, t_stat=2.0, direction="follow")
    opt, cons = compute_cell_metrics(cell, events, prices, market)
    assert opt.n_events == 2
    assert opt.n_fills == 2  # both events fill optimistically
    assert cons.n_fills == 1  # only the second event fills conservatively
    assert opt.fill_rate == 1.0
    assert cons.fill_rate == 0.5


def test_emit_report_no_pass_verdict():
    from eda_inside_mm_bias import CellMetrics, TopKCell, emit_report
    cell = TopKCell(bot="B", product="X", horizon=5, mean_t=0.5, t_stat=2.0, direction="follow")
    failing = CellMetrics(
        n_events=10, n_fills=5, fill_rate=0.5,
        mean_pnl_mid=-0.1, std_pnl_mid=1.0, t_stat_mid=-0.5,
        mean_pnl_flat=-0.5, std_pnl_flat=1.0, t_stat_flat=-1.5,
        hit_rate_flat=0.4,
    )
    report = emit_report([(cell, failing, failing)])
    assert "NO PASS" in report
    assert "PASS:" not in report.split("=== VERDICT ===", 1)[1]


def test_emit_report_pass_verdict_lists_passing_cells():
    from eda_inside_mm_bias import CellMetrics, TopKCell, emit_report
    cell = TopKCell(bot="B", product="X", horizon=5, mean_t=0.5, t_stat=2.0, direction="follow")
    passing = CellMetrics(
        n_events=100, n_fills=50, fill_rate=0.50,
        mean_pnl_mid=0.5, std_pnl_mid=1.0, t_stat_mid=5.0,
        mean_pnl_flat=0.2, std_pnl_flat=1.0, t_stat_flat=2.0,
        hit_rate_flat=0.55,
    )
    failing = CellMetrics(
        n_events=100, n_fills=10, fill_rate=0.10,
        mean_pnl_mid=-0.1, std_pnl_mid=1.0, t_stat_mid=-0.5,
        mean_pnl_flat=-0.5, std_pnl_flat=1.0, t_stat_flat=-1.5,
        hit_rate_flat=0.4,
    )
    # Pass on conservative; optimistic identical here.
    report = emit_report([(cell, passing, passing)])
    verdict_section = report.split("=== VERDICT ===", 1)[1]
    assert "PASS: B X horizon=5 direction=follow" in verdict_section
    assert "NO PASS" not in verdict_section


def test_compute_cell_metrics_filters_events_by_product():
    """Regression: events for OTHER products by the same bot must not contaminate the cell.

    Bug previously: bot_events filter only matched on bot, so Mark 14's VEF trades
    got used as triggers for Mark 14 x HYDROGEL fills.
    """
    from eda_inside_mm_bias import TopKCell, compute_cell_metrics
    # Two events for bot B: one on product X (the cell), one on product Y (cross-product).
    events = _make_events([
        {"ts": 100, "day": 1, "bot": "B", "product": "X", "signed_qty": 1},
        {"ts": 600, "day": 1, "bot": "B", "product": "Y", "signed_qty": 1},
    ])
    prices = _make_prices([
        {"ts": 200, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 300, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 400, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 500, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 700, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 800, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
        {"ts": 900, "day": 1, "product": "X", "bid_price_1": 99, "ask_price_1": 101, "mid_price": 100},
    ])
    market = _make_market_trades([
        {"ts": 250, "day": 1, "product": "X", "price": 99},
        {"ts": 750, "day": 1, "product": "X", "price": 99},
    ])
    cell = TopKCell(bot="B", product="X", horizon=3, mean_t=0.5, t_stat=2.0, direction="follow")
    opt, _cons = compute_cell_metrics(cell, events, prices, market)
    # Only the product=X event should count. n_events = 1, not 2.
    assert opt.n_events == 1
