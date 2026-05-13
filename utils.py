"""
utils.py — shared helpers: feature ordering, array builder.
"""

import numpy as np

# ── Canonical feature order (must match training) ──────────────
FEATURE_NAMES = [
    "daily_ship_transits",
    "oil_throughput_mbpd",
    "war_risk_insurance_pct",
    "vessels_attacked_cumulative",
    "cape_reroute_extra_days",
    "is_war_crisis",
    "days_since_closure",
    "transit_pct_of_prewar_avg",
    "vessels_stranded_in_gulf",
    "lng_throughput_bcfd",
    "diesel_usd_gallon_us",
]


def build_feature_array(request) -> np.ndarray:
    """
    Convert a Pydantic schema instance into a (1, n_features) numpy array
    in the exact order the trained model expects.
    """
    values = [getattr(request, name) for name in FEATURE_NAMES]
    return np.array(values, dtype=float).reshape(1, -1)
