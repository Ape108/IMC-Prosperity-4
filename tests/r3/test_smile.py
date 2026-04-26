import numpy as np
import pytest
from strategy import fit_iv_smile


def test_fit_returns_coefficients_for_valid_input():
    moneynesses = [0.76, 0.86, 0.95, 0.97, 0.99, 1.01, 1.03, 1.05, 1.14, 1.24]
    ivs = [0.50, 0.42, 0.34, 0.32, 0.30, 0.31, 0.32, 0.35, 0.44, 0.54]
    coeffs = fit_iv_smile(moneynesses, ivs)
    assert coeffs is not None
    assert len(coeffs) == 3


def test_fit_returns_none_for_fewer_than_3_points():
    assert fit_iv_smile([0.95, 1.05], [0.30, 0.31]) is None


def test_fitted_curve_evaluates_correctly():
    moneynesses = [0.76, 0.86, 0.95, 0.97, 0.99, 1.01, 1.03, 1.05, 1.14, 1.24]
    ivs = [0.50, 0.42, 0.34, 0.32, 0.30, 0.31, 0.32, 0.35, 0.44, 0.54]
    coeffs = fit_iv_smile(moneynesses, ivs)
    mean_m = float(np.mean(moneynesses))
    fitted = float(np.polyval(coeffs, mean_m))
    assert 0.20 < fitted < 0.60


def test_residuals_sum_near_zero():
    moneynesses = [0.8, 0.9, 1.0, 1.1, 1.2]
    ivs = [0.5 * (m - 1) ** 2 + 0.3 for m in moneynesses]
    coeffs = fit_iv_smile(moneynesses, ivs)
    residuals = [iv - float(np.polyval(coeffs, m)) for iv, m in zip(ivs, moneynesses)]
    assert all(abs(r) < 1e-10 for r in residuals)
