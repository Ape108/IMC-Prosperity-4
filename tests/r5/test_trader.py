import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r5"))

from strategy import Trader, R5BaseMMStrategy, TIGHT_TIER, MEDIUM_TIER, WIDE_TIER
from strategy import R5AutocorrMMStrategy, R5CorrMMStrategy


def test_trader_has_50_strategies():
    t = Trader()
    assert len(t.strategies) == 50


def test_all_product_symbols_registered():
    t = Trader()
    all_syms = set(TIGHT_TIER + MEDIUM_TIER + WIDE_TIER)
    assert set(t.strategies.keys()) == all_syms


def test_tight_tier_width_1():
    t = Trader()
    for sym in TIGHT_TIER:
        s = t.strategies[sym]
        assert isinstance(s, R5BaseMMStrategy)
        assert s.width == 1


def test_medium_tier_width_2():
    t = Trader()
    for sym in MEDIUM_TIER:
        s = t.strategies[sym]
        assert isinstance(s, R5BaseMMStrategy)
        assert s.width == 2


def test_wide_tier_width_3():
    t = Trader()
    for sym in WIDE_TIER:
        s = t.strategies[sym]
        assert isinstance(s, R5BaseMMStrategy)
        assert s.width == 3


def test_all_limits_are_10():
    t = Trader()
    for s in t.strategies.values():
        assert s.limit == 10


def test_ironing_uses_autocorr_overlay():
    t = Trader()
    s = t.strategies["ROBOT_IRONING"]
    assert isinstance(s, R5AutocorrMMStrategy)
    assert s.alpha == 0.121


def test_snackpack_raspberry_uses_corr_overlay():
    t = Trader()
    s = t.strategies["SNACKPACK_RASPBERRY"]
    assert isinstance(s, R5CorrMMStrategy)
    assert s.partner_symbol == "SNACKPACK_STRAWBERRY"
    assert abs(s.beta - 0.462) < 0.001


def test_pistachio_has_negative_beta():
    t = Trader()
    s = t.strategies["SNACKPACK_PISTACHIO"]
    assert isinstance(s, R5CorrMMStrategy)
    assert s.beta < 0  # follow, not fade
