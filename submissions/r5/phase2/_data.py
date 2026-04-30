"""Shared data-loading helpers for Phase 2 EDA scripts.

Centralizes the dataset path, group/product registry, and mid-price loader
used by every analysis script. Mirrors the load pattern from
submissions/r5/groups/<group>.py (semicolon-delimited CSV, pivot on product).
"""

from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parents[3] / "datasets" / "round5"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DAYS = (2, 3, 4)

# Round 5 group/product registry. Groups of 5 each, position limit 10.
GROUPS: dict[str, tuple[str, ...]] = {
    "GALAXY_SOUNDS": ("DARK_MATTER", "BLACK_HOLES", "PLANETARY_RINGS", "SOLAR_WINDS", "SOLAR_FLAMES"),
    "SLEEP_POD": ("SUEDE", "LAMB_WOOL", "POLYESTER", "NYLON", "COTTON"),
    "MICROCHIP": ("CIRCLE", "OVAL", "SQUARE", "RECTANGLE", "TRIANGLE"),
    "PEBBLES": ("XS", "S", "M", "L", "XL"),
    "ROBOT": ("VACUUMING", "MOPPING", "DISHES", "LAUNDRY", "IRONING"),
    "UV_VISOR": ("YELLOW", "AMBER", "ORANGE", "RED", "MAGENTA"),
    "TRANSLATOR": ("SPACE_GRAY", "ASTRO_BLACK", "ECLIPSE_CHARCOAL", "GRAPHITE_MIST", "VOID_BLUE"),
    "PANEL": ("1X2", "2X2", "1X4", "2X4", "4X4"),
    "OXYGEN_SHAKE": ("MORNING_BREATH", "EVENING_BREATH", "MINT", "CHOCOLATE", "GARLIC"),
    "SNACKPACK": ("CHOCOLATE", "VANILLA", "PISTACHIO", "STRAWBERRY", "RASPBERRY"),
}


def full_symbol(group: str, product: str) -> str:
    """Compose 'GROUP_PRODUCT' as it appears in the price CSV's `product` column."""
    return f"{group}_{product}"


def all_symbols() -> list[str]:
    """Flat list of all 50 fully-qualified symbols."""
    return [full_symbol(g, p) for g, prods in GROUPS.items() for p in prods]


def load_mids(day: int, symbol: str) -> pd.Series:
    """Return mid_price series for a fully-qualified symbol on a given day.

    Index is timestamp, sorted ascending. Returned series may have gaps if
    the product was not active for some ticks; callers should `.dropna()`
    or `.intersection()` indices when joining across products.
    """
    path = DATASET_DIR / f"prices_round_5_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    return df[df["product"] == symbol].set_index("timestamp")["mid_price"].sort_index()


def per_day_sigma(symbol: str) -> dict[int, float]:
    """Per-day std of pct_change of mid_price."""
    return {d: load_mids(d, symbol).pct_change().std() for d in DAYS}
