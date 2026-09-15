from __future__ import annotations

import csv
import io
import json
from typing import Any


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, default=str)


def report_csv(report: dict[str, Any]) -> str:
    rows = report.get("compounds", [])
    fields = ["compound", "type", "carbon_chain", "priority", "predicted_concentration_ngl", "exceedance_probability", "risk_category", "threshold_status"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row.get(field) for field in fields} for row in rows)
    return output.getvalue()


def report_pdf(report: dict[str, Any]) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors

    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [Paragraph("PFAS Analytics Report", styles["Title"]), Paragraph("Research assessment generated from the PFAS Risk Scanner", styles["Normal"]), Spacer(1, 12)]
    metadata = report.get("metadata", {})
    story.append(Paragraph("Report Metadata", styles["Heading2"]))
    story.append(Table([[key.replace("_", " ").title(), value] for key, value in metadata.items()], colWidths=[150, 350], style=TableStyle([["GRID", (0, 0), (-1, -1), 0.35, colors.grey], ["VALIGN", (0, 0), (-1, -1), "TOP"]])))
    story.append(Spacer(1, 12))
    summary = report.get("executive_summary", {})
    story.append(Paragraph("1. Executive Summary", styles["Heading2"]))
    story.append(Paragraph(f"Overall risk: {summary.get('overall_risk', 'Unavailable')} | Risk score: {summary.get('risk_score', 'Unavailable')} / 100 | Exceedance probability: {summary.get('exceedance_probability', 'Unavailable')} | Predicted concentration: {summary.get('predicted_concentration_ngl', 'Unavailable')} ng/L", styles["BodyText"]))
    story.append(Paragraph(f"Interpretation: {summary.get('interpretation', 'Not available')}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("2. Location & Environmental Context", styles["Heading2"]))
    context = report.get("location_context", {})
    story.append(Table([[key.replace("_", " ").title(), value] for key, value in context.items() if key != "model_features"], colWidths=[180, 320], style=TableStyle([["GRID", (0, 0), (-1, -1), 0.35, colors.grey], ["VALIGN", (0, 0), (-1, -1), "TOP"]])))
    story.append(Spacer(1, 12))
    story.append(Paragraph("3. PFAS Compound Analysis", styles["Heading2"]))
    data = [["Compound", "Type", "Chain", "Priority", "Concentration (ng/L)", "Exceedance", "Status"]] + [[row.get("compound"), row.get("type"), row.get("carbon_chain"), row.get("priority"), row.get("predicted_concentration_ngl", "Not available"), row.get("exceedance_probability", "Not available"), row.get("threshold_status", "Not available")] for row in report.get("compounds", [])]
    table = Table(data)
    table.setStyle(TableStyle([["BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F28C62")], ["GRID", (0, 0), (-1, -1), 0.5, colors.grey], ["VALIGN", (0, 0), (-1, -1), "TOP"]]))
    story.append(table)
    story.append(Spacer(1, 12))
    story.append(Paragraph("4. Why This Prediction?", styles["Heading2"]))
    explainability = report.get("explainability", {})
    story.append(Paragraph("Risk drivers: " + ("; ".join(explainability.get("risk_drivers", [])) or "Not available"), styles["BodyText"]))
    story.append(Paragraph("Protective factors: " + ("; ".join(explainability.get("protective_factors", [])) or "Not available"), styles["BodyText"]))
    shap_data = [["Feature", "Raw value", "SHAP", "Effect"]] + [[item.get("label", item.get("feature", "Unknown")), item.get("value", "Not available"), item.get("shap", "Not available"), "Increases risk" if float(item.get("shap", 0)) > 0 else "Decreases risk"] for item in explainability.get("top_features", [])]
    if len(shap_data) > 1:
        story.append(Table(shap_data, style=TableStyle([["BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F28C62")], ["GRID", (0, 0), (-1, -1), 0.35, colors.grey]])))
    else:
        story.append(Paragraph("SHAP explanation unavailable for this prediction.", styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("5. Model & Validation", styles["Heading2"]))
    story.append(Table([[key.replace("_", " ").title(), value] for key, value in {**report.get("model", {}), **report.get("validation", {})}.items()], colWidths=[180, 320], style=TableStyle([["GRID", (0, 0), (-1, -1), 0.35, colors.grey], ["VALIGN", (0, 0), (-1, -1), "TOP"]])))
    story.append(Spacer(1, 12))
    story.append(Paragraph("6. Scenario / What-If Analysis", styles["Heading2"]))
    simulations = report.get("simulations", [])
    if simulations:
        story.append(Table([["Scenario", "Baseline", "Scenario", "Change"]] + [[item.get("scenario_label"), item.get("base_prob"), item.get("scenario_prob"), item.get("delta_pts")] for item in simulations], style=TableStyle([["BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F28C62")], ["GRID", (0, 0), (-1, -1), 0.35, colors.grey]])))
    else:
        story.append(Paragraph("No simulation was performed for this assessment.", styles["BodyText"]))
    story.append(PageBreak())
    story.append(Paragraph("7. Data Provenance", styles["Heading2"]))
    story.append(Table([[key.replace("_", " ").title(), value] for key, value in report.get("provenance", {}).items()], colWidths=[180, 320], style=TableStyle([["GRID", (0, 0), (-1, -1), 0.35, colors.grey], ["VALIGN", (0, 0), (-1, -1), "TOP"]])))
    story.append(Spacer(1, 12))
    story.append(Paragraph("8. Limitations & Interpretation", styles["Heading2"]))
    story.extend(Paragraph(f"• {item}", styles["BodyText"]) for item in report.get("limitations", []))
    document.build(story)
    return buffer.getvalue()