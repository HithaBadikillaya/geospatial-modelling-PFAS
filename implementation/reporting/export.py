from __future__ import annotations

import csv
from html import escape
import io
import json
from typing import Any


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, default=str)


def report_csv(report: dict[str, Any]) -> str:
    rows = report.get("compounds", [])
    fields = ["compound", "type", "carbon_chain", "priority", "evaluation_status", "predicted_concentration_ngl", "exceedance_probability", "exceedance_category", "threshold_status"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row.get(field) for field in fields} for row in rows)
    return output.getvalue()


def report_pdf(report: dict[str, Any]) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle
    from reportlab.lib import colors

    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    body_style = styles["BodyText"]
    body_style.leading = 12

    def text(value: Any) -> Paragraph:
        return Paragraph(escape("Not available" if value is None else str(value)).replace("\n", "<br/>"), body_style)

    def table(rows: list[list[Any]], widths: list[int]) -> LongTable:
        wrapped = [[text(cell) for cell in row] for row in rows]
        return LongTable(wrapped, colWidths=widths, repeatRows=1, splitByRow=1, hAlign="LEFT", style=TableStyle([
            ["BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F28C62")],
            ["GRID", (0, 0), (-1, -1), 0.35, colors.grey],
            ["VALIGN", (0, 0), (-1, -1), "TOP"],
            ["LEFTPADDING", (0, 0), (-1, -1), 5],
            ["RIGHTPADDING", (0, 0), (-1, -1), 5],
        ]))

    story = [Paragraph("PFAS Analytics Report", styles["Title"]), Paragraph("Geospatial exceedance assessment generated from the PFAS Scanner", styles["Normal"]), Spacer(1, 12)]
    metadata = report.get("metadata", {})
    story.append(Paragraph("Report Metadata", styles["Heading2"]))
    story.append(table([["Field", "Value"]] + [[key.replace("_", " ").title(), value] for key, value in metadata.items()], [150, 350]))
    story.append(Spacer(1, 12))
    summary = report.get("executive_summary", {})
    story.append(Paragraph("1. Executive Summary", styles["Heading2"]))
    story.append(Paragraph(f"Exceedance category: {summary.get('exceedance_category', 'Unavailable')} | Exceedance probability: {summary.get('exceedance_probability', 'Unavailable')} | Predicted concentration: {summary.get('predicted_concentration_ngl', 'Unavailable')} ng/L", body_style))
    story.append(Paragraph(f"Interpretation: {escape(str(summary.get('interpretation', 'Not available')))}", body_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("2. Location & Environmental Context", styles["Heading2"]))
    context = report.get("location_context", {})
    story.append(table([["Field", "Value"]] + [[key.replace("_", " ").title(), value] for key, value in context.items() if key != "model_features"], [180, 320]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("3. PFAS Compound Analysis", styles["Heading2"]))
    data = [["Compound", "Type", "Chain", "Status", "Concentration (ng/L)", "Exceedance", "Category"]] + [[row.get("compound"), row.get("type"), row.get("carbon_chain"), row.get("evaluation_status"), row.get("predicted_concentration_ngl"), row.get("exceedance_probability"), row.get("exceedance_category")] for row in report.get("compounds", [])]
    story.append(table(data, [65, 75, 45, 65, 90, 75, 85]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("4. Why This Prediction?", styles["Heading2"]))
    explainability = report.get("explainability", {})
    story.append(Paragraph("Prediction drivers: " + ("; ".join(explainability.get("prediction_drivers", [])) or "Not available"), body_style))
    story.append(Paragraph("Protective factors: " + ("; ".join(explainability.get("protective_factors", [])) or "Not available"), styles["BodyText"]))
    shap_data = [["Feature", "Raw value", "SHAP", "Effect"]] + [[item.get("label", item.get("feature", "Unknown")), item.get("value", "Not available"), item.get("shap", "Not available"), "Increases exceedance probability" if float(item.get("shap", 0)) > 0 else "Decreases exceedance probability"] for item in explainability.get("top_features", [])]
    if len(shap_data) > 1:
        story.append(table(shap_data, [190, 130, 70, 110]))
    else:
        story.append(Paragraph("SHAP explanation unavailable for this prediction.", styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("5. Model & Validation", styles["Heading2"]))
    story.append(table([["Item", "Value"]] + [[key.replace("_", " ").title(), value] for key, value in {**report.get("model", {}), **report.get("validation", {})}.items()], [180, 320]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("6. Scenario / What-If Analysis", styles["Heading2"]))
    simulations = report.get("simulations", [])
    if simulations:
        story.append(table([["Scenario", "Baseline", "Scenario", "Change"]] + [[item.get("scenario_label"), item.get("base_prob"), item.get("scenario_prob"), item.get("delta_pts")] for item in simulations], [190, 100, 100, 110]))
    else:
        story.append(Paragraph("No simulation was performed for this assessment.", styles["BodyText"]))
    story.append(Paragraph("7. Data Provenance", styles["Heading2"]))
    story.append(table([["Item", "Value"]] + [[key.replace("_", " ").title(), value] for key, value in report.get("provenance", {}).items()], [180, 320]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("8. Limitations & Interpretation", styles["Heading2"]))
    story.extend(Paragraph(f"• {item}", styles["BodyText"]) for item in report.get("limitations", []))
    document.build(story)
    return buffer.getvalue()