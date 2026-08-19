from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import dash
from dash import Input, Output, State, dcc, html, no_update

from dash_common import (
    ACCENT,
    SUCCESS,
    CHART_MUTED,
    CHART_SEQUENCE,
    SCENARIO_PRESETS,
    base_figure_layout,
    gauge_figure,
    get_backend,
    glass_card,
    metric,
    page_header,
    section_title,
    serialize_sim_result,
)

dash.register_page(__name__, path="/simulation", name="Simulation", title="PFAS Simulation")


def layout():
    preset_options = [{"label": value["label"], "value": key} for key, value in SCENARIO_PRESETS.items()]
    return html.Div(
        [
            page_header(
                "Simulation",
                "Scenario laboratory",
                "Select one or more scenarios to compare them side-by-side. Combine presets and fine-tune sliders for a full researcher playground.",
            ),
            html.Div(
                [
                    glass_card(
                        [
                            section_title("Scenario design"),
                            html.Div(
                                [
                                    html.Label("SELECT SCENARIOS (MULTI-CHOICE)", style={"display": "block", "marginBottom": "8px", "color": "#6E6760", "fontSize": "0.72rem", "fontWeight": "800", "letterSpacing": "0.08em"}),
                                    dcc.Dropdown(
                                        id="sim-preset",
                                        options=preset_options,
                                        value=["baseline"],
                                        multi=True,
                                        clearable=False,
                                        placeholder="Select one or more scenarios to compare…",
                                    ),
                                ],
                                style={"marginBottom": "16px"},
                            ),
                            dcc.Loading(
                                html.Div(id="sim-description", className="status-note"),
                                type="dot",
                                color=ACCENT,
                            ),
                            html.Div(
                                [
                                    html.Div("FINE-TUNE PARAMETERS", style={"color": "#6E6760", "fontSize": "0.70rem", "fontWeight": "800", "letterSpacing": "0.10em", "textTransform": "uppercase", "marginBottom": "14px", "marginTop": "8px"}),
                                    html.Div(
                                        [
                                            html.Label("Industrial intensity", style={"fontSize": "0.82rem", "fontWeight": "700", "color": "#191715", "display": "block", "marginBottom": "6px"}),
                                            dcc.Slider(id="sim-industrial", min=0, max=300, step=5, value=100,
                                                       marks={0: "0%", 100: "100%", 300: "300%"},
                                                       tooltip={"placement": "bottom", "always_visible": False}),
                                        ],
                                        style={"marginBottom": "20px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Airport distance (km)", style={"fontSize": "0.82rem", "fontWeight": "700", "color": "#191715", "display": "block", "marginBottom": "6px"}),
                                            dcc.Slider(id="sim-airport", min=1, max=300, step=1, value=50,
                                                       marks={1: "1 km", 150: "150 km", 300: "300 km"},
                                                       tooltip={"placement": "bottom", "always_visible": False}),
                                        ],
                                        style={"marginBottom": "20px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Intervention efficiency", style={"fontSize": "0.82rem", "fontWeight": "700", "color": "#191715", "display": "block", "marginBottom": "6px"}),
                                            dcc.Slider(id="sim-cleanup", min=0, max=90, step=5, value=0,
                                                       marks={0: "0%", 45: "45%", 90: "90%"},
                                                       tooltip={"placement": "bottom", "always_visible": False}),
                                        ],
                                        style={"marginBottom": "4px"},
                                    ),
                                ],
                                className="assessment-area",
                            ),
                            html.Div(style={"height": "16px"}),
                            html.Button("Execute scenario", id="sim-run", className="primary-button"),
                            dcc.Loading(
                                html.Div(id="sim-status", className="status-note"),
                                type="dot",
                                color=ACCENT,
                            ),
                        ],
                        "span-4",
                    ),
                    html.Div(
                        dcc.Loading(
                            html.Div(id="sim-result"),
                            type="circle",
                            color=ACCENT,
                            style={"minHeight": "240px"},
                        ),
                        className="span-8",
                    ),
                ],
                className="dashboard-grid",
            ),
        ]
    )


@dash.callback(Output("sim-description", "children"), Input("sim-preset", "value"))
def describe_preset(selected_presets):
    if not selected_presets:
        return "No preset selected. Choose one or more scenarios to compare."
    items = []
    for preset_key in selected_presets:
        preset = SCENARIO_PRESETS.get(preset_key, SCENARIO_PRESETS.get("baseline", {}))
        items.append(html.Div(
            [
                html.Span(f"▶ {preset.get('label', preset_key)}: ", style={"fontWeight": "700", "color": "#191715"}),
                html.Span(preset.get("description", "")),
            ],
            style={"marginBottom": "6px", "fontSize": "0.86rem", "lineHeight": "1.5"},
        ))
    return html.Div(items)


@dash.callback(
    Output("sim-store", "data"),
    Output("sim-status", "children"),
    Input("sim-run", "n_clicks"),
    State("scan-store", "data"),
    State("sim-preset", "value"),
    State("sim-industrial", "value"),
    State("sim-airport", "value"),
    State("sim-cleanup", "value"),
    prevent_initial_call=True,
)
def run_simulation(_clicks, scan, selected_presets, industrial_pct, airport_km, cleanup_pct):
    if not scan:
        return no_update, "Run Scanner first to establish a baseline feature vector."
    _, simulator, _ = get_backend()
    if not simulator:
        return no_update, "Simulation engine is offline."

    selected_presets = selected_presets or ["baseline"]
    composite_results = []
    labels = []

    for preset_key in selected_presets:
        preset = SCENARIO_PRESETS.get(preset_key, SCENARIO_PRESETS["baseline"])
        mods = preset["mods"].copy()
        mods["spatial_density_boost"] = float(industrial_pct) / 100.0
        mods["airport_distance_km"] = float(airport_km)
        if cleanup_pct and cleanup_pct > 0:
            mods["mean_log_value_reduction"] = float(cleanup_pct) / 100.0
        if preset_key == "baseline":
            mods = {}
        result = simulator.run_custom(pd.DataFrame([scan["feature_vector"]]), mods, label=preset["label"])
        composite_results.append(result)
        labels.append(preset["label"])

    if len(composite_results) == 1:
        payload = serialize_sim_result(composite_results[0])
        payload["all_results"] = [{"label": composite_results[0].scenario_label, "score": composite_results[0].scenario_score, "delta": composite_results[0].delta_pts, "risk_level": composite_results[0].risk_level}]
        return payload, "Scenario outcome updated."

    # Multi-scenario aggregation
    aggregate_prob = float(sum(r.scenario_prob for r in composite_results) / len(composite_results))
    aggregate_delta = float(sum(r.delta_pts for r in composite_results) / len(composite_results))
    aggregate_result = composite_results[0]
    aggregate_result.scenario_prob = aggregate_prob
    aggregate_result.delta_pts = aggregate_delta
    aggregate_result.scenario_label = " + ".join(labels)
    aggregate_result.plain_explanation = (
        f"Comparing {len(composite_results)} scenarios: {', '.join(labels)}. "
        f"Average exceedance probability across all scenarios: {aggregate_prob * 100:.1f}% "
        f"(mean change: {aggregate_delta:+.1f} pts from baseline)."
    )
    payload = serialize_sim_result(aggregate_result)
    payload["all_results"] = [
        {"label": r.scenario_label, "score": r.scenario_score, "delta": r.delta_pts, "risk_level": r.risk_level}
        for r in composite_results
    ]
    return payload, f"{len(composite_results)} scenarios evaluated and compared."


@dash.callback(Output("sim-result", "children"), Input("scan-store", "data"), Input("sim-store", "data"))
def render_simulation_result(scan, sim):
    if not scan:
        return glass_card(
            [
                section_title("Baseline required"),
                html.P("Run a Scanner prediction first. The simulation lab uses that feature vector as its baseline.", className="status-note"),
            ],
            "span-12",
        )
    if not sim:
        return glass_card(
            [
                section_title("Awaiting scenario"),
                html.P(
                    f"Current baseline risk is {float(scan['exceedance_prob'])*100:.1f}%. Choose one or more scenarios and execute them.",
                    className="status-note",
                ),
            ],
            "span-12",
        )

    all_results = sim.get("all_results", [])
    delta = float(sim["delta_pts"])
    is_multi = len(all_results) > 1

    children = []

    # Gauge (aggregate or single)
    gauge_color = "#2E8B57" if float(sim["scenario_score"]) < 35 else ACCENT if float(sim["scenario_score"]) < 65 else "#D96B34"
    children.append(
        glass_card(
            [section_title("Simulated risk score"), dcc.Graph(figure=gauge_figure(float(sim["scenario_score"]), "Simulated Risk", gauge_color), config={"displayModeBar": False})],
            "span-5",
        )
    )
    children.append(
        glass_card([section_title("Outcome narrative"), html.P(sim["plain_explanation"], className="status-note")], "span-7")
    )

    # If multi-scenario: render a comparative bar chart
    if is_multi and len(all_results) > 1:
        baseline_score = float(scan["exceedance_prob"]) * 100
        bar_labels = ["Baseline"] + [r["label"] for r in all_results]
        bar_scores = [baseline_score] + [r["score"] for r in all_results]
        bar_colors = []
        for s in bar_scores:
            if s < 35:
                bar_colors.append("#2E8B57")
            elif s < 65:
                bar_colors.append("#F28C62")
            else:
                bar_colors.append("#D96B34")

        fig = go.Figure(go.Bar(
            x=bar_labels,
            y=bar_scores,
            marker_color=bar_colors,
            text=[f"{s:.1f}%" for s in bar_scores],
            textposition="outside",
            cliponaxis=False,
        ))
        layout_upd = base_figure_layout(280)
        layout_upd.update({
            "showlegend": False,
            "yaxis": {**layout_upd["yaxis"], "range": [0, 115], "title": "Risk Score (%)"},
            "xaxis": {**layout_upd["xaxis"], "title": ""},
            "bargap": 0.35,
        })
        fig.update_layout(layout_upd)
        children.append(
            glass_card(
                [section_title("Scenario comparison"), dcc.Graph(figure=fig, config={"displayModeBar": False})],
                "span-12",
            )
        )

        # Per-scenario cards
        comparison_cards = []
        for r in all_results:
            delta_val = r["delta"]
            card_color = "#2E8B57" if r["score"] < 35 else "#F28C62" if r["score"] < 65 else "#D96B34"
            comparison_cards.append(
                html.Div(
                    [
                        html.Div(r["label"], style={"fontSize": "0.72rem", "fontWeight": "800", "letterSpacing": "0.10em", "textTransform": "uppercase", "color": "#6E6760", "marginBottom": "10px"}),
                        html.Div(f"{r['score']:.1f}%", style={"fontSize": "1.55rem", "fontWeight": "800", "letterSpacing": "-0.03em", "color": card_color}),
                        html.Div(f"{delta_val:+.1f} pts", style={"fontSize": "0.82rem", "marginTop": "6px", "color": "#6E6760"}),
                        html.Div(r["risk_level"], style={"fontSize": "0.78rem", "fontWeight": "700", "marginTop": "4px", "color": card_color}),
                    ],
                    style={
                        "padding": "16px 18px",
                        "background": "#fff",
                        "border": f"1px solid rgba(25,23,21,0.10)",
                        "borderLeft": f"4px solid {card_color}",
                        "borderRadius": "10px",
                    },
                )
            )
        if comparison_cards:
            cols = min(len(comparison_cards), 4)
            children.append(
                html.Div(
                    comparison_cards,
                    style={"display": "grid", "gridTemplateColumns": f"repeat({cols}, 1fr)", "gap": "14px"},
                    className="span-12",
                )
            )
    else:
        # Single scenario: standard metric 4-column row across full width
        children.append(
            html.Div(
                [
                    metric("Baseline", f"{float(sim['base_score']):.1f}%", "Original score"),
                    metric("Scenario", f"{float(sim['scenario_score']):.1f}%", sim["risk_level"]),
                    metric("Variance", f"{delta:+.1f} pts", "Probability delta"),
                    metric("Scenario", sim["scenario_label"], "Preset applied"),
                ],
                className="metric-grid-4 span-12",
            )
        )

    return html.Div(children, className="dashboard-grid")
