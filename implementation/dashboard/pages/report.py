from __future__ import annotations

import dash
import plotly.graph_objects as go
from dash import Input, Output, ctx, dcc, html, no_update

from dash_common import ACCENT, SUCCESS, base_figure_layout, glass_card, page_header, section_title

dash.register_page(__name__, path="/report", name="Report", title="PFAS Analytics Report")


def _value(value, suffix=""):
    return "Not available" if value is None else f"{value}{suffix}"


def _table(headers, rows):
    return html.Table([html.Thead(html.Tr([html.Th(item) for item in headers])), html.Tbody([html.Tr([html.Td(item) for item in row]) for row in rows])], className="report-table")


def _metadata_section(report):
    metadata = report["metadata"]
    return glass_card([section_title("Report Metadata"), _table(["Report ID", "Generated", "Model", "Target"], [[metadata.get("report_id"), metadata.get("generated_timestamp"), metadata.get("model_version"), metadata.get("target_definition")]])], "span-12")


def _summary_section(report):
    summary = report["executive_summary"]
    return glass_card([section_title("1. Executive Summary"), html.Div([html.Div("Exceedance category", className="metric-label"), html.Div(summary.get("exceedance_category", "Not available"), className="report-risk"), html.P(f"Exceedance probability: {_value(summary.get('exceedance_probability'))} | Predicted concentration: {_value(summary.get('predicted_concentration_ngl'), ' ng/L')}", className="status-note"), html.P(summary.get("interpretation", "Not available"), className="status-note")])], "span-6")


def _context_section(report):
    context = report["location_context"]
    features = context.get("model_features", {})
    return glass_card([section_title("2. Location & Environmental Context"), _table(["Field", "Value"], [[key.replace("_", " ").title(), value] for key, value in context.items() if key != "model_features"]), html.P(f"Model features available: {len(features)}", className="status-note")], "span-6")


def _compound_section(report):
    rows = [[row["compound"], row["evaluation_status"], row["type"], row["carbon_chain"], _value(row.get("predicted_concentration_ngl"), " ng/L"), _value(row.get("exceedance_probability")), _value(row.get("exceedance_category"))] for row in report["compounds"]]
    return glass_card([section_title("3. PFAS Compound Analysis"), _table(["Compound", "Evaluation", "Type", "Chain", "Predicted concentration", "Exceedance probability", "Category"], rows)], "span-12")


def _explainability_section(report):
    explainability = report["explainability"]
    top_features = explainability.get("top_features", [])
    shap_rows = [(item.get("label", item.get("feature", "Unknown")), item.get("value", "Not available"), f"{float(item.get('shap', 0)):+.3f}", "Increases exceedance probability" if float(item.get("shap", 0)) > 0 else "Decreases exceedance probability") for item in top_features]
    shap_fig = go.Figure(go.Bar(x=[float(item.get("shap", 0)) for item in top_features], y=[item.get("label", item.get("feature", "Unknown")) for item in top_features], orientation="h", marker_color=[ACCENT if float(item.get("shap", 0)) > 0 else SUCCESS for item in top_features]))
    shap_fig.update_layout(base_figure_layout(300), yaxis={"autorange": "reversed"}, xaxis={"title": "SHAP impact on exceedance probability"}, showlegend=False)
    return html.Div([glass_card([section_title("4. Why This Prediction?"), dcc.Graph(figure=shap_fig, config={"displayModeBar": False})], "span-7"), glass_card([section_title("Prediction drivers"), _table(["Feature", "Value", "SHAP", "Effect"], shap_rows or [["Not available", "", "", "SHAP explanation unavailable for this prediction."]])], "span-5")], className="dashboard-grid")


def _validation_section(report):
    values = {**report["model"], **report["validation"]}
    return glass_card([section_title("5. Model & Validation"), _table(["Item", "Value"], [[key.replace("_", " ").title(), value] for key, value in values.items()])], "span-6")


def _simulation_section(report):
    rows = [[item.get("scenario_label", "Scenario"), item.get("base_prob", "Not available"), item.get("scenario_prob", "Not available"), item.get("delta_pts", "Not available")] for item in report["simulations"]] or [["No simulation was performed for this assessment.", "", "", ""]]
    return glass_card([section_title("6. Scenario / What-If Analysis"), _table(["Scenario", "Baseline probability", "Scenario probability", "Change"], rows)], "span-6")


def _provenance_section(report):
    return glass_card([section_title("7. Data Provenance"), _table(["Item", "Value"], [[key.replace("_", " ").title(), value] for key, value in report["provenance"].items()])], "span-6")


def _limitations_section(report):
    return glass_card([section_title("8. Limitations & Interpretation"), html.Ul([html.Li(item) for item in report["limitations"]], className="status-note")], "span-6")


def layout():
    return html.Div([
        page_header("Report", "PFAS Analytics Report", "A structured assessment assembled from the latest Scanner prediction, explanation, and executed scenarios."),
        dcc.Download(id="report-download-pdf"), dcc.Download(id="report-download-json"), dcc.Download(id="report-download-csv"),
        html.Div(id="report-content"),
    ])


@dash.callback(Output("report-content", "children"), Input("report-store", "data"))
def render_report(report):
    if not report:
        return glass_card([section_title("No report available"), html.P("Run a Scanner prediction, then choose Generate Detailed Report.", className="status-note")], "span-12")
    return html.Div([
        html.Div([html.Button("Download PDF", id="download-pdf", className="secondary-button"), html.Button("Download JSON", id="download-json", className="secondary-button"), html.Button("Download CSV", id="download-csv", className="secondary-button")], className="button-row report-actions"),
        _metadata_section(report),
        html.Div([_summary_section(report), _context_section(report)], className="dashboard-grid"),
        _compound_section(report),
        _explainability_section(report),
        html.Div([_validation_section(report), _simulation_section(report)], className="dashboard-grid"),
        html.Div([_provenance_section(report), _limitations_section(report)], className="dashboard-grid"),
    ], className="report-page")


@dash.callback(Output("report-download-json", "data"), Input("download-json", "n_clicks"), Input("report-store", "data"), prevent_initial_call=True)
def download_json(_clicks, report):
    if not report or ctx.triggered_id != "download-json":
        return no_update
    from reporting.export import report_json
    return dcc.send_string(report_json(report), "pfas_analytics_report.json", type="application/json")


@dash.callback(Output("report-download-csv", "data"), Input("download-csv", "n_clicks"), Input("report-store", "data"), prevent_initial_call=True)
def download_csv(_clicks, report):
    if not report or ctx.triggered_id != "download-csv":
        return no_update
    from reporting.export import report_csv
    return dcc.send_string(report_csv(report), "pfas_compound_analysis.csv", type="text/csv")


@dash.callback(Output("report-download-pdf", "data"), Input("download-pdf", "n_clicks"), Input("report-store", "data"), prevent_initial_call=True)
def download_pdf(_clicks, report):
    if not report or ctx.triggered_id != "download-pdf":
        return no_update
    from reporting.export import report_pdf
    return dcc.send_bytes(lambda buffer: buffer.write(report_pdf(report)), "pfas_analytics_report.pdf")