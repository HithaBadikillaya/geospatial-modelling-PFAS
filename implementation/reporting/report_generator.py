from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .report_models import AnalyticsReport
from .thresholds import exceedance_category, normalize_probability

COMPOUNDS = ["PFOS", "PFOA", "PFNA", "PFDA", "PFHXS", "PFHPA", "PFBS"]
COMPOUND_INFO = {
    "PFOS": ("Sulfonate", "C8", "High"), "PFOA": ("Carboxylate", "C8", "High"),
    "PFNA": ("Carboxylate", "C9", "Medium"), "PFDA": ("Carboxylate", "C10", "Medium"),
    "PFHXS": ("Sulfonate", "C6", "Watch"), "PFHPA": ("Carboxylate", "C7", "Watch"),
    "PFBS": ("Sulfonate", "C4", "Monitor"),
}


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def build_report(scan: dict[str, Any], xai: dict[str, Any] | None = None, simulation: dict[str, Any] | None = None) -> AnalyticsReport:
    now = datetime.now(timezone.utc).isoformat()
    probability = normalize_probability(scan.get("exceedance_prob"))
    concentration = float(scan.get("predicted_value_ngl", 0.0))
    threshold = 100.0
    selected = str(scan.get("substance", "GENERAL")).upper()
    compounds = []
    compound_outputs = scan.get("compound_results", {})
    for compound in COMPOUNDS:
        info = COMPOUND_INFO[compound]
        output = compound_outputs.get(compound, {}) if isinstance(compound_outputs, dict) else {}
        value = output.get("predicted_value_ngl") if isinstance(output, dict) else None
        prob = output.get("exceedance_prob") if isinstance(output, dict) else None
        if compound == selected and not output:
            value, prob = concentration, probability
        compounds.append({
            "compound": compound, "type": info[0], "carbon_chain": info[1], "priority": info[2],
            "predicted_concentration_ngl": value, "exceedance_probability": prob,
            "exceedance_category": exceedance_category(prob) if prob is not None else None,
            "evaluation_status": "Evaluated" if value is not None or prob is not None else "Not evaluated",
            "threshold_status": ("Exceeds" if value is not None and float(value) >= threshold else "Below") if value is not None else None,
        })

    top_features = list((xai or {}).get("top_features", []))
    location = {
        "latitude": scan.get("lat"), "longitude": scan.get("lon"),
        "year": scan.get("year"), "media_type": scan.get("media_type"),
        "airport_distance_km": scan.get("dist_to_airport_km"),
        "nearest_training_point_km": scan.get("dist_to_nearest_sample_km"),
        "model_features": {key: _json_value(value) for key, value in (scan.get("feature_vector") or {}).items()},
    }
    return AnalyticsReport(
        metadata={
            "report_id": f"RPT-{datetime.now(timezone.utc):%Y%m%d}-{uuid4().hex[:8]}",
            "generated_timestamp": now, "prediction_timestamp": now,
            "latitude": scan.get("lat"), "longitude": scan.get("lon"),
            "model_version": "LightGBM (loaded artifact)", "target_definition": "PFAS concentration >= 100 ng/L (exceedance)",
            "report_generation_version": "1.0",
        },
        executive_summary={
            "exceedance_category": exceedance_category(probability),
            "exceedance_probability": probability, "predicted_concentration_ngl": concentration,
            "confidence_level": scan.get("confidence_level"), "confidence_note": scan.get("confidence_note"),
            "threshold_ngl": threshold, "interpretation": (xai or {}).get("headline", scan.get("confidence_note", "Model-derived site estimate.")),
            "primary_factors": [item.get("label", item.get("feature", "Unknown")) for item in top_features[:5]],
        },
        location_context=location,
        compounds=compounds,
        explainability={
            "top_features": top_features, "prediction_drivers": (xai or {}).get("risk_drivers", []),
            "protective_factors": (xai or {}).get("protective_factors", []),
            "data_quality_note": (xai or {}).get("data_quality_note", "SHAP explanation unavailable for this prediction."),
        },
        model={"algorithm": "LightGBM", "feature_count": len(scan.get("feature_vector", {})), "target": "above_100_ng_l", "calibration": "Isotonic regression (when present)", "oversampling": "ADASYN (training only)", "preprocessing": "Existing project feature pipeline"},
        validation={"method": "5-fold Spatial Block GroupKFold", "note": "Spatial blocks reduce geographic leakage between nearby samples.", "metrics_available": False, "metrics_note": "This report describes the validation methodology; metrics are not available for the individual scan."},
        simulations=[simulation] if simulation else [],
        provenance={"dataset": "dataset/pfas_golden.parquet", "record_count": "Available from dashboard dataset metadata", "prediction_timestamp": now},
        limitations=[
            "This is a model-derived estimate, not a laboratory measurement.",
            "A high exceedance probability does not prove contamination; a low probability does not guarantee its absence.",
            "Reliability depends on training-data coverage, and spatial extrapolation may increase uncertainty.",
            "Interpret predictions alongside available observations and confirm important decisions with laboratory testing.",
        ],
    )