import math
import pytest
from strategy import norm_cdf, bs_call_price


def test_norm_cdf_at_zero():
    assert abs(norm_cdf(0.0) - 0.5) < 1e-10


def test_norm_cdf_at_positive_one():
    assert abs(norm_cdf(1.0) - 0.8413) < 1e-4


def test_norm_cdf_at_negative_one():
    assert abs(norm_cdf(-1.0) - 0.1587) < 1e-4


def test_norm_cdf_symmetry():
    assert abs(norm_cdf(1.5) + norm_cdf(-1.5) - 1.0) < 1e-10


def test_bs_call_deep_itm_near_intrinsic():
    price = bs_call_price(S=5247.0, K=4000.0, T=0.001, sigma=0.3)
    intrinsic = 5247.0 - 4000.0
    assert abs(price - intrinsic) < 5.0


def test_bs_call_deep_otm_near_zero():
    price = bs_call_price(S=5247.0, K=6500.0, T=0.001, sigma=0.3)
    assert price < 1.0


def test_bs_call_atm_positive():
    price = bs_call_price(S=5247.0, K=5247.0, T=5 / 365, sigma=0.30)
    assert price > 0


def test_bs_call_increases_with_sigma():
    low = bs_call_price(S=5247.0, K=5200.0, T=5 / 365, sigma=0.20)
    high = bs_call_price(S=5247.0, K=5200.0, T=5 / 365, sigma=0.40)
    assert high > low


from strategy import vega, implied_vol


def test_vega_positive_for_valid_inputs():
    v = vega(S=5247.0, K=5200.0, T=5 / 365, sigma=0.30)
    assert v > 0


def test_implied_vol_roundtrip_atm():
    S, K, T, true_sigma = 5247.0, 5200.0, 5 / 365, 0.30
    market_price = bs_call_price(S, K, T, true_sigma)
    recovered = implied_vol(S, K, T, market_price)
    assert recovered is not None
    assert abs(recovered - true_sigma) < 0.001


def test_implied_vol_roundtrip_itm():
    S, K, T, true_sigma = 5247.0, 5000.0, 5 / 365, 0.28
    market_price = bs_call_price(S, K, T, true_sigma)
    recovered = implied_vol(S, K, T, market_price)
    assert recovered is not None
    assert abs(recovered - true_sigma) < 0.001


def test_implied_vol_returns_none_for_below_intrinsic():
    result = implied_vol(S=5247.0, K=5000.0, T=5 / 365, market_price=200.0)
    assert result is None


def test_implied_vol_roundtrip_otm():
    S, K, T, true_sigma = 5247.0, 5400.0, 5 / 365, 0.32
    market_price = bs_call_price(S, K, T, true_sigma)
    recovered = implied_vol(S, K, T, market_price)
    assert recovered is not None
    assert abs(recovered - true_sigma) < 0.001
