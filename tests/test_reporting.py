from implementation.reporting import build_report
from implementation.reporting.export import report_csv, report_json, report_pdf
from implementation.reporting.thresholds import exceedance_category, normalize_probability


def complete_scan():
    return {
        "lat": 51.5,
        "lon": -0.12,
        "year": 2024,
        "media_type": "Surface Water",
        "substance": "PFOS",
        "exceedance_prob": 0.87,
        "predicted_value_ngl": 142.6,
        "dist_to_airport_km": 3.2,
        "dist_to_nearest_sample_km": 12.0,
        "confidence_level": "HIGH",
        "confidence_note": "Nearby measurements support this estimate.",
        "feature_vector": {"dist_to_airport_km": 3.2},
    }


def test_report_preserves_scan_values_and_exports():
    report = build_report(complete_scan(), {"headline": "Observed drivers.", "top_features": []})
    payload = report.to_dict()
    assert payload["executive_summary"]["predicted_concentration_ngl"] == 142.6
    assert payload["executive_summary"]["exceedance_probability"] == 0.87
    assert payload["compounds"][0]["predicted_concentration_ngl"] == 142.6
    assert payload["compounds"][1]["predicted_concentration_ngl"] is None
    assert "PFOS" in report_csv(payload)
    assert "executive_summary" in report_json(payload)
    assert report_pdf(payload).startswith(b"%PDF")


def test_report_handles_missing_optional_sections():
    payload = build_report({"lat": None, "lon": None, "exceedance_prob": 0.1, "predicted_value_ngl": 2.0}, None, None).to_dict()
    assert payload["explainability"]["top_features"] == []
    assert payload["simulations"] == []
    assert all(row["predicted_concentration_ngl"] is None for row in payload["compounds"])


def test_report_marks_only_selected_compound_as_evaluated():
    payload = build_report(complete_scan()).to_dict()
    evaluated = [row for row in payload["compounds"] if row["evaluation_status"] == "Evaluated"]
    assert [row["compound"] for row in evaluated] == ["PFOS"]
    assert all(row["evaluation_status"] == "Not evaluated" for row in payload["compounds"][1:])
    assert payload["executive_summary"]["exceedance_category"] == "HIGH"
    assert "prediction_drivers" in payload["explainability"]


def test_thresholds_handle_boundaries_and_invalid_numbers():
    assert exceedance_category(0.0) == "LOW"
    assert exceedance_category(0.35) == "MEDIUM"
    assert exceedance_category(0.65) == "HIGH"
    assert normalize_probability(-1) == 0.0
    assert normalize_probability(2) == 1.0
    assert normalize_probability("invalid") == 0.0


def test_exports_handle_long_rows_and_missing_sections():
    payload = build_report(
        {"substance": "PFOA", "exceedance_prob": float("nan"), "predicted_value_ngl": float("inf")},
        {"prediction_drivers": ["A very long driver " * 20], "top_features": []},
        {"scenario_label": "Long scenario " * 20, "base_prob": 0.2, "scenario_prob": 0.4, "delta_pts": 20},
    ).to_dict()
    assert payload["executive_summary"]["exceedance_probability"] == 0.0
    assert "evaluation_status" in report_csv(payload).splitlines()[0]
    assert "Geospatial exceedance assessment" not in report_json(payload)
    assert report_pdf(payload).startswith(b"%PDF")