from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import dash
from dash import Input, Output, State, clientside_callback, ctx, dcc, html, no_update

from dash_common import (
    ACCENT,
    SUCCESS,
    CHART_MUTED,
    XAI_SUGGESTED_QUESTIONS,
    base_figure_layout,
    get_backend,
    restore_xai_context,
)

dash.register_page(__name__, path="/ai-assistant", name="AI Assistant", title="PFAS AI Assistant")


def layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Suggested questions", className="assistant-card-title"),
                            html.Div(
                                [
                                    html.Button(prompt, id=f"ai-prompt-{idx}", className="prompt-button")
                                    for idx, prompt in enumerate(XAI_SUGGESTED_QUESTIONS)
                                ],
                                className="prompt-grid",
                            ),
                            html.Div("Conversation", className="assistant-card-title"),
                            dcc.Loading(
                                html.Div(id="chat-log", className="chat-log"),
                                type="circle",
                                color=ACCENT,
                                style={"minHeight": "200px"},
                            ),
                            html.Div(
                                [
                                    dcc.Input(
                                        id="chat-input",
                                        placeholder="Ask anything — risk, compounds, health effects, remediation, model accuracy…",
                                        type="text",
                                    ),
                                    html.Button("Send", id="chat-send", className="primary-button"),
                                ],
                                className="chat-compose",
                            ),
                            html.Div(style={"height": "12px"}),
                            html.Button("Clear conversation", id="chat-clear", className="secondary-button"),
                            html.Div(id="chat-scroll-sentinel", style={"display": "none"}),
                        ],
                        className="assistant-main-panel",
                    ),
                    html.Div(id="assistant-context", className="assistant-context"),
                ],
                className="assistant-layout",
            )
        ],
        className="assistant-page",
        style={"padding": "24px 32px"},
    )


@dash.callback(
    Output("chat-store", "data"),
    Output("chat-input", "value"),
    Input("chat-send", "n_clicks"),
    Input("chat-input", "n_submit"),
    Input("chat-clear", "n_clicks"),
    *[Input(f"ai-prompt-{idx}", "n_clicks") for idx in range(len(XAI_SUGGESTED_QUESTIONS))],
    State("chat-input", "value"),
    State("chat-store", "data"),
    State("xai-store", "data"),
    prevent_initial_call=True,
)
def update_chat(_send, _submit, _clear, *args):
    message_value = args[-3]
    history = args[-2] or []
    xai_payload = args[-1]
    triggered = ctx.triggered_id

    if triggered == "chat-clear":
        return [], ""

    prompt_text = None
    if isinstance(triggered, str) and triggered.startswith("ai-prompt-"):
        prompt_text = XAI_SUGGESTED_QUESTIONS[int(triggered.rsplit("-", 1)[-1])]

    user_text = prompt_text or (message_value or "").strip()
    if not user_text:
        return no_update, no_update

    _, _, xai = get_backend()
    if not xai:
        ai_text = "AI engine is offline. Run the model pipeline first."
    else:
        restore_xai_context(xai, xai_payload)
        ai_text = xai.chat(user_text)

    return history + [{"role": "user", "text": user_text}, {"role": "ai", "text": ai_text}], ""


@dash.callback(Output("chat-log", "children"), Input("chat-store", "data"))
def render_chat(history):
    history = history or []
    if not history:
        return [
            html.Div(
                "Hello! I'm your PFAS risk assistant. Run a Scanner analysis to get site-specific insights, or ask me anything about PFAS science, health effects, compounds, model accuracy, or remediation options.",
                className="message ai",
            )
        ]
    return [html.Div(item["text"], className=f"message {item['role']}") for item in history]


clientside_callback(
    """
    function(children) {
      setTimeout(function() {
        const log = document.getElementById("chat-log");
        if (log) { log.scrollTop = log.scrollHeight; }
      }, 40);
      return "";
    }
    """,
    Output("chat-scroll-sentinel", "children"),
    Input("chat-log", "children"),
)


@dash.callback(Output("assistant-context", "children"), Input("scan-store", "data"), Input("xai-store", "data"))
def render_context(scan, xai_payload):
    cards = []
    if scan:
        prob = float(scan["exceedance_prob"]) * 100
        risk_color = "#2E8B57" if prob < 35 else "#F28C62" if prob < 65 else "#D96B34"
        cards.append(
            html.Div(
                [
                    html.Div("Active site", className="assistant-card-title"),
                    html.Div(
                        [
                            html.Div(f"{prob:.1f}%", style={"fontSize": "1.8rem", "fontWeight": "800", "color": risk_color, "letterSpacing": "-0.04em"}),
                            html.Div("exceedance probability", style={"fontSize": "0.78rem", "color": "#6E6760", "marginTop": "4px"}),
                        ],
                        style={"marginBottom": "10px"},
                    ),
                    html.P(
                        f"Lat {float(scan['lat']):.4f}, Lon {float(scan['lon']):.4f}",
                        className="status-note",
                        style={"marginTop": "0"},
                    ),
                ],
                className="assistant-context-card",
            )
        )
    else:
        cards.append(
            html.Div(
                [
                    html.Div("Active site", className="assistant-card-title"),
                    html.P("No scanner result attached yet. Run a scan first.", className="status-note"),
                ],
                className="assistant-context-card",
            )
        )

    if xai_payload and xai_payload.get("top_features"):
        shap_df = pd.DataFrame(xai_payload["top_features"]).head(6)
        fig = go.Figure(
            go.Bar(
                x=shap_df["shap"],
                y=shap_df["label"],
                orientation="h",
                marker_color=["#F28C62" if value > 0 else "#2E8B57" for value in shap_df["shap"]],
                text=shap_df["shap"].round(3),
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig.update_layout(
            base_figure_layout(280),
            margin={"l": 24, "r": 40, "t": 18, "b": 32},
            showlegend=False,
            yaxis={"autorange": "reversed", "title": "", "tickfont": {"size": 11}},
            xaxis={"title": "SHAP impact", "automargin": True, "tickfont": {"size": 10}},
        )
        cards.append(
            html.Div(
                [
                    html.Div("Top SHAP drivers", className="assistant-card-title"),
                    dcc.Graph(figure=fig, config={"displayModeBar": False}, className="assistant-chart"),
                    html.P(
                        "Orange bars increase risk · Green bars reduce risk",
                        style={"fontSize": "0.72rem", "color": "#6E6760", "marginTop": "8px", "textAlign": "center"},
                    ),
                ],
                className="assistant-context-card",
            )
        )
    else:
        cards.append(
            html.Div(
                [
                    html.Div("Attribution", className="assistant-card-title"),
                    html.P("Run Scanner to attach SHAP contributors.", className="status-note"),
                ],
                className="assistant-context-card",
            )
        )

    cards.append(
        html.Div(
            [
                html.Div("Reference thresholds", className="assistant-card-title"),
                html.Div(
                    [
                        html.Div([html.Span("100 ng/L", style={"fontWeight": "700", "color": "#F28C62"}), html.Span(" · EU screening threshold (20 priority PFAS)", style={"fontSize": "0.80rem"})]),
                        html.Div([html.Span("10 ng/L", style={"fontWeight": "700", "color": "#D96B34"}), html.Span(" · EU total PFAS limit", style={"fontSize": "0.80rem"})]),
                        html.Div([html.Span("70 ng/L", style={"fontWeight": "700", "color": "#2E8B57"}), html.Span(" · US EPA advisory (PFOS + PFOA combined)", style={"fontSize": "0.80rem"})]),
                    ],
                    style={"display": "flex", "flexDirection": "column", "gap": "8px", "fontSize": "0.82rem", "color": "#191715"},
                ),
            ],
            className="assistant-context-card",
        )
    )
    return cards
