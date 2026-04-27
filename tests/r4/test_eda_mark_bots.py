import io
import pandas as pd
import pytest
from eda_mark_bots import load_trades, load_prices, lead_lag_corr, classify_bot


def _make_trades_csv(rows: list[str]) -> str:
    header = "timestamp;buyer;seller;symbol;currency;price;quantity"
    return "\n".join([header] + rows)


def test_load_trades_extracts_mark_buyer(tmp_path):
    csv = _make_trades_csv([
        "100;Mark 01;SomeOther;HYDROGEL_PACK;XIRECS;10000.0;5",
    ])
    day_file = tmp_path / "trades_round_4_day_1.csv"
    day_file.write_text(csv)
    df = load_trades(days=[1], dataset_dir=str(tmp_path))
    assert len(df) == 1
    assert df.iloc[0]["bot"] == "Mark 01"
    assert df.iloc[0]["signed_qty"] == 5
    assert df.iloc[0]["product"] == "HYDROGEL_PACK"
    assert df.iloc[0]["timestamp"] == 100


def test_load_trades_extracts_mark_seller(tmp_path):
    csv = _make_trades_csv([
        "200;SomeOther;Mark 02;VELVETFRUIT_EXTRACT;XIRECS;5250.0;3",
    ])
    day_file = tmp_path / "trades_round_4_day_1.csv"
    day_file.write_text(csv)
    df = load_trades(days=[1], dataset_dir=str(tmp_path))
    assert len(df) == 1
    assert df.iloc[0]["bot"] == "Mark 02"
    assert df.iloc[0]["signed_qty"] == -3


def test_load_trades_skips_non_mark_rows(tmp_path):
    csv = _make_trades_csv([
        "100;Alice;Bob;HYDROGEL_PACK;XIRECS;10000.0;5",
    ])
    day_file = tmp_path / "trades_round_4_day_1.csv"
    day_file.write_text(csv)
    df = load_trades(days=[1], dataset_dir=str(tmp_path))
    assert len(df) == 0


def test_load_trades_concatenates_multiple_days(tmp_path):
    csv1 = _make_trades_csv(["100;Mark 01;;HYDROGEL_PACK;XIRECS;10000.0;2"])
    csv2 = _make_trades_csv(["200;Mark 01;;HYDROGEL_PACK;XIRECS;10001.0;3"])
    (tmp_path / "trades_round_4_day_1.csv").write_text(csv1)
    (tmp_path / "trades_round_4_day_2.csv").write_text(csv2)
    df = load_trades(days=[1, 2], dataset_dir=str(tmp_path))
    assert len(df) == 2
    assert list(df["signed_qty"]) == [2, 3]


def test_load_trades_both_mark_recorded_twice(tmp_path):
    # In real R4, both buyer AND seller are Mark bots.
    # Each trade should appear twice: once as a buy for the buyer bot,
    # once as a sell for the seller bot.
    csv = _make_trades_csv([
        "100;Mark 01;Mark 22;HYDROGEL_PACK;XIRECS;10000.0;5",
    ])
    day_file = tmp_path / "trades_round_4_day_1.csv"
    day_file.write_text(csv)
    df = load_trades(days=[1], dataset_dir=str(tmp_path))
    assert len(df) == 2
    buy_row = df[df["bot"] == "Mark 01"].iloc[0]
    sell_row = df[df["bot"] == "Mark 22"].iloc[0]
    assert buy_row["signed_qty"] == 5
    assert sell_row["signed_qty"] == -5


def _make_prices_csv(rows: list[str]) -> str:
    header = "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss"
    return "\n".join([header] + rows)


def test_load_prices_extracts_mid_price(tmp_path):
    csv = _make_prices_csv([
        "1;100;HYDROGEL_PACK;9999;10;;;;;10001;10;;;;;10000.0;0.0",
        "1;200;HYDROGEL_PACK;9999;10;;;;;10001;10;;;;;10000.5;0.0",
    ])
    (tmp_path / "prices_round_4_day_1.csv").write_text(csv)
    df = load_prices(days=[1], dataset_dir=str(tmp_path))
    hg = df[df["product"] == "HYDROGEL_PACK"].reset_index(drop=True)
    assert len(hg) == 2
    assert hg.iloc[0]["mid_price"] == pytest.approx(10000.0)
    assert hg.iloc[1]["mid_price"] == pytest.approx(10000.5)
    assert hg.iloc[0]["timestamp"] == 100


def test_load_prices_includes_all_products(tmp_path):
    csv = _make_prices_csv([
        "1;100;HYDROGEL_PACK;9999;5;;;;;10001;5;;;;;10000.0;0.0",
        "1;100;VELVETFRUIT_EXTRACT;5249;5;;;;;5251;5;;;;;5250.0;0.0",
    ])
    (tmp_path / "prices_round_4_day_1.csv").write_text(csv)
    df = load_prices(days=[1], dataset_dir=str(tmp_path))
    assert set(df["product"].unique()) == {"HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"}


def test_lead_lag_corr_perfect_predictor():
    # Varying buy sizes that perfectly predict price moves
    trades = pd.DataFrame({
        "timestamp": [0, 100, 200, 300],
        "bot": ["Mark 01"] * 4,
        "product": ["HYDROGEL_PACK"] * 4,
        "signed_qty": [1.0, 2.0, 1.0, 2.0],
    })
    prices = pd.DataFrame({
        "timestamp": [0, 100, 200, 300, 400],
        "product": ["HYDROGEL_PACK"] * 5,
        "mid_price": [100.0, 101.0, 103.0, 104.0, 106.0],
    })
    corrs = lead_lag_corr(trades, prices, "Mark 01", "HYDROGEL_PACK", lags=[1])
    assert corrs[0] == pytest.approx(1.0, abs=1e-6)


def test_lead_lag_corr_no_signal():
    # signed_qty=[1,-1,1,-1], price_changes at lag=1=[1,1,-1,-1] → corr=0
    trades = pd.DataFrame({
        "timestamp": [0, 100, 200, 300],
        "bot": ["Mark 01"] * 4,
        "product": ["HYDROGEL_PACK"] * 4,
        "signed_qty": [1.0, -1.0, 1.0, -1.0],
    })
    prices = pd.DataFrame({
        "timestamp": [0, 100, 200, 300, 400],
        "product": ["HYDROGEL_PACK"] * 5,
        "mid_price": [100.0, 101.0, 102.0, 101.0, 100.0],
    })
    corrs = lead_lag_corr(trades, prices, "Mark 01", "HYDROGEL_PACK", lags=[1])
    assert abs(corrs[0]) < 0.1


def test_lead_lag_corr_returns_zeros_for_empty_bot():
    trades = pd.DataFrame(columns=["timestamp", "bot", "product", "signed_qty"])
    prices = pd.DataFrame({
        "timestamp": [0, 100],
        "product": ["HYDROGEL_PACK", "HYDROGEL_PACK"],
        "mid_price": [100.0, 101.0],
    })
    corrs = lead_lag_corr(trades, prices, "Mark 99", "HYDROGEL_PACK", lags=[1, 5])
    assert corrs == [0.0, 0.0]


def test_classify_informed():
    stats = {"net_direction": 0.5, "lead_5_corr": 0.2, "trade_count": 30}
    assert classify_bot(stats) == "INFORMED"


def test_classify_informed_negative_corr():
    # Negative lead corr + strong short bias also qualifies
    stats = {"net_direction": -0.4, "lead_5_corr": -0.18, "trade_count": 20}
    assert classify_bot(stats) == "INFORMED"


def test_classify_market_maker():
    stats = {"net_direction": 0.05, "lead_5_corr": 0.02, "trade_count": 200}
    assert classify_bot(stats) == "MARKET_MAKER"


def test_classify_noise():
    stats = {"net_direction": 0.5, "lead_5_corr": 0.04, "trade_count": 5}
    assert classify_bot(stats) == "NOISE"
