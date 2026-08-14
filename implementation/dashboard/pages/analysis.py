from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import dash
from dash import Input, Output, dcc, html

from dash_common import ACCENT, CHART_MUTED, base_figure_layout, glass_card, page_header, section_title

dash.register_page(__name__, path="/analysis", name="Analysis", title="PFAS Analysis")


def layout():
    return html.Div(
        [
            page_header(
                "Analysis",
                "Factor attribution",
                "Inspect site-specific model drivers after a Scanner prediction, without recomputing the full dashboard.",
            ),
            html.Div(id="analysis-content"),
        ]
    )


@dash.callback(Output("analysis-content", "children"), Input("xai-store", "data"))
def render_analysis(xai_payload):
    if not xai_payload:
        return glass_card(
            [
                section_title("No active attribution"),
                html.P("Run a Scanner prediction first to populate SHAP contributors and data quality notes.", className="status-note"),
            ],
            "span-12",
        )

    shap_df = pd.DataFrame(xai_payload.get("top_features", [])).head(10)
    fig = go.Figure()
    if not shap_df.empty:
        colors = [ACCENT if value > 0 else CHART_MUTED for value in shap_df["shap"]]
        fig.add_bar(
            x=shap_df["shap"],
            y=shap_df["label"],
            orientation="h",
            marker_color=colors,
            text=[f"{value:+.3f}" for value in shap_df["shap"]],
            textposition="outside",
        )
    fig.update_layout(base_figure_layout(420), yaxis={"autorange": "reversed"}, showlegend=False)

    return html.Div(
        [
            glass_card([section_title("TreeSHAP impact values"), dcc.Graph(figure=fig, config={"displayModeBar": False})], "span-12"),
            glass_card(
                [
                    section_title("Risk contributors"),
                    html.Ul([html.Li(item) for item in xai_payload.get("risk_drivers", [])] or [html.Li("None identified.")], className="status-note"),
                ],
                "span-6",
            ),
            glass_card(
                [
                    section_title("Protective factors"),
                    html.Ul([html.Li(item) for item in xai_payload.get("protective_factors", [])] or [html.Li("None identified.")], className="status-note"),
                ],
                "span-6",
            ),
            glass_card([section_title("Data quality"), html.P(xai_payload.get("data_quality_note", ""), className="status-note")], "span-12"),
        ],
        className="dashboard-grid",
    )
