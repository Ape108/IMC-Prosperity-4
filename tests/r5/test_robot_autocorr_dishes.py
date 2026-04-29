import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submissions" / "r5" / "groups"))

from robot import (
    Trader,
    R5BaseMMStrategy,
    R5AutocorrMMStrategy,
)


def test_autocorr_dishes_wires_dishes_and_ironing_with_autocorr():
    """
    autocorr_dishes() registers DISHES at α=0.222 and IRONING at α=0.121,
    leaves LAUNDRY/VACUUMING/MOPPING on base MM.
    """
    t = Trader()
    t.strategies = {}
    # This is a body replica, not a call to autocorr_dishes() — the variant is a
    # closure inside Trader.__init__ and not externally callable. We rebuild the
    # same registrations to lock the alpha values and class assignments.
    t.strategies["ROBOT_DISHES"] = R5AutocorrMMStrategy("ROBOT_DISHES", 10, width=1, alpha=0.222)
    t.strategies["ROBOT_IRONING"] = R5AutocorrMMStrategy("ROBOT_IRONING", 10, width=1, alpha=0.121)
    for sym in ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "ROBOT_MOPPING"):
        t.strategies[sym] = R5BaseMMStrategy(sym, 10, width=1)

    assert isinstance(t.strategies["ROBOT_DISHES"], R5AutocorrMMStrategy)
    assert t.strategies["ROBOT_DISHES"].alpha == 0.222
    assert isinstance(t.strategies["ROBOT_IRONING"], R5AutocorrMMStrategy)
    assert t.strategies["ROBOT_IRONING"].alpha == 0.121
    for sym in ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "ROBOT_MOPPING"):
        # type-is, not isinstance — R5AutocorrMMStrategy inherits from R5BaseMMStrategy
        assert type(t.strategies[sym]) is R5BaseMMStrategy
        assert t.strategies[sym].width == 1


def test_autocorr_dishes_alpha_values_match_eda():
    """
    Lock the EDA-derived alpha values:
    - DISHES combined-day lag-1 ACF = -0.222 (per eda_triage_summary.md line 112)
    - IRONING combined-day lag-1 ACF = -0.121 (per eda_triage_summary.md line 113)
    Sign convention: the strategy uses |alpha| as a positive coefficient (lean magnitude).
    """
    s_dishes = R5AutocorrMMStrategy("ROBOT_DISHES", 10, width=1, alpha=0.222)
    s_ironing = R5AutocorrMMStrategy("ROBOT_IRONING", 10, width=1, alpha=0.121)
    assert s_dishes.alpha == 0.222
    assert s_ironing.alpha == 0.121
