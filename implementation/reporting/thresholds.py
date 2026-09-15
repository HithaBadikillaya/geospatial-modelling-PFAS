from __future__ import annotations

from math import isfinite

EXCEEDANCE_THRESHOLDS = {
    "LOW": 0.35,
    "MEDIUM": 0.65,
}


def normalize_probability(value: object, default: float = 0.0) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return default
    if not isfinite(probability):
        return default
    return min(max(probability, 0.0), 1.0)


def exceedance_category(probability: object) -> str:
    normalized = normalize_probability(probability)
    if normalized < EXCEEDANCE_THRESHOLDS["LOW"]:
        return "LOW"
    if normalized < EXCEEDANCE_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "HIGH"