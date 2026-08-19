from __future__ import annotations

from dash import Dash, Input, Output, State, dcc, html, page_container

from dash_common import get_backend

NAV_ITEMS = [
    ("Overview", "/", "⌁"),
    ("Scanner", "/scanner", "⌖"),
    ("Simulation", "/simulation", "⇄"),
    ("Analysis", "/analysis", "≋"),
    ("Explorer", "/explorer", "◌"),
    ("AI Assistant", "/ai-assistant", "✦"),
]

app = Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    suppress_callback_exceptions=True,
    title="PFAS Risk Intelligence",
)
server = app.server


def nav_link(label: str, href: str, icon: str) -> dcc.Link:
    return dcc.Link(
        [html.Span(icon, className="nav-icon"), html.Span(label)],
        href=href,
        id={"type": "nav-link", "href": href},
        className="nav-pill",
    )


def app_shell() -> html.Div:
    predictor, _, _ = get_backend()
    status_text = "Operational" if predictor else "Offline"
    return html.Div(
        [
            dcc.Location(id="url"),
            dcc.Store(id="scan-store", storage_type="session"),
            dcc.Store(id="xai-store", storage_type="session"),
            dcc.Store(id="sim-store", storage_type="session"),
            dcc.Store(id="chat-store", storage_type="session", data=[]),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("PFAS", className="brand-mark"),
                            html.Div(
                                [
                                    html.Div("Risk Intelligence", className="brand-title"),
                                    html.Div("Geospatial research console", className="brand-subtitle"),
                                ],
                                className="brand-copy",
                            ),
                        ],
                        className="brand",
                    ),
                    html.Nav([nav_link(label, href, icon) for label, href, icon in NAV_ITEMS], className="top-nav"),
                    html.Div(
                        [
                            html.Span(className=f"status-dot {'online' if predictor else 'offline'}"),
                            html.Span(status_text),
                        ],
                        className="system-status",
                    ),
                ],
                className="glass-topbar",
            ),
            html.Main(page_container, className="page-stage"),
        ],
        className="app-shell",
    )


app.layout = app_shell


@app.callback(
    Output({"type": "nav-link", "href": "/"}, "className"),
    Output({"type": "nav-link", "href": "/scanner"}, "className"),
    Output({"type": "nav-link", "href": "/simulation"}, "className"),
    Output({"type": "nav-link", "href": "/analysis"}, "className"),
    Output({"type": "nav-link", "href": "/explorer"}, "className"),
    Output({"type": "nav-link", "href": "/ai-assistant"}, "className"),
    Input("url", "pathname"),
)
def mark_active_nav(pathname: str | None):
    path = pathname or "/"
    return [f"nav-pill {'active' if href == path else ''}".strip() for _, href, _ in NAV_ITEMS]


if __name__ == "__main__":
    app.run(debug=False)
