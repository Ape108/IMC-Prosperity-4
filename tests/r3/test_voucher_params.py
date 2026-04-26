import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'submissions', 'r3'))

from strategy import Trader

def test_voucher_strategy_params():
    trader = Trader()
    strat = trader.strategies["VEV_5000"]
    assert strat.k == 150
    assert strat.min_residual == 0.01
    assert strat.max_otm_moneyness == 1.005

def test_all_vouchers_use_new_params():
    trader = Trader()
    for strike in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]:
        strat = trader.strategies[f"VEV_{strike}"]
        assert strat.k == 150, f"VEV_{strike}: expected k=150, got {strat.k}"
        assert strat.max_otm_moneyness == 1.005, f"VEV_{strike}: expected gate=1.005"
