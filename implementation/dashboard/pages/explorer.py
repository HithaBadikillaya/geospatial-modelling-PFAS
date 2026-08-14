from __future__ import annotations

import plotly.express as px
import dash
from dash import Input, Output, dcc, html

from dash_common import ACCENT, CHART_SEQUENCE, base_figure_layout, empty_figure, glass_card, load_trend_data, page_header, section_title

dash.register_page(__name__, path="/explorer", name="Explorer", title="PFAS Explorer")


def layout():
    trend_df = load_trend_data()
    compounds = sorted(trend_df["substance"].dropna().unique()) if not trend_df.empty and "substance" in trend_df else []
    return html.Div(
        [
            page_header(
                "Explorer",
                "Temporal explorer",
                "Filter compounds and update only the relevant Plotly figures through Dash callbacks.",
            ),
            html.Div(
                [
                    glass_card(
                        [
                            section_title("Compound filter"),
                            dcc.Dropdown(
                                id="explorer-compounds",
                                options=[{"label": item, "value": item} for item in compounds],
                                value=compounds[:5],
                                multi=True,
                            ),
                        ],
                        "span-12",
                    ),
                    glass_card([section_title("Concentration progression"), dcc.Graph(id="explorer-trend", config={"displayModeBar": False})], "span-12"),
                    glass_card([section_title("Exceedance rate"), dcc.Graph(id="explorer-exceedance", config={"displayModeBar": False})], "span-6"),
                    glass_card([section_title("Value distribution"), dcc.Graph(id="explorer-distribution", config={"displayModeBar": False})], "span-6"),
                ],
                className="dashboard-grid",
            ),
        ]
    )


@dash.callback(
    Output("explorer-trend", "figure"),
    Output("explorer-exceedance", "figure"),
    Output("explorer-distribution", "figure"),
    Input("explorer-compounds", "value"),
)
def update_explorer(selected):
    trend_df = load_trend_data()
    if trend_df.empty:
        empty = empty_figure("Dataset is not available")
        return empty, empty, empty
    selected = selected or sorted(trend_df["substance"].dropna().unique())[:5]
    filtered = trend_df[trend_df["substance"].isin(selected)]
    if filtered.empty:
        empty = empty_figure("No records match the selected compounds")
        return empty, empty, empty

    yearly = filtered.groupby(["year", "substance"])["log_value"].median().reset_index()
    yearly.columns = ["Year", "Compound", "Median Log Conc."]
    trend_fig = px.line(
        yearly,
        x="Year",
        y="Median Log Conc.",
        color="Compound",
        markers=True,
        color_discrete_sequence=CHART_SEQUENCE,
    )
    trend_fig.update_layout(base_figure_layout(390))

    exc_rate = filtered.groupby("substance")["above_100_ng_l"].mean().reset_index()
    exc_rate.columns = ["Compound", "Rate"]
    exc_fig = px.bar(exc_rate.sort_values("Rate", ascending=False), x="Compound", y="Rate", color_discrete_sequence=[ACCENT])
    exc_fig.update_layout(base_figure_layout(320), yaxis_tickformat=".0%")

    sample = filtered.sample(min(5000, len(filtered)), random_state=42)
    dist_fig = px.violin(sample, x="substance", y="log_value", box=True, color_discrete_sequence=[ACCENT])
    dist_fig.update_layout(base_figure_layout(320))
    return trend_fig, exc_fig, dist_fig
