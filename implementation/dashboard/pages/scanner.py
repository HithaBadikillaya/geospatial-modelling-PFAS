from __future__ import annotations

import dash
from dash import Input, Output, State, dcc, html, no_update

from dash_common import (
    ACCENT,
    CHART_MUTED,
    gauge_figure,
    get_backend,
    glass_card,
    metric,
    offline_geocode,
    page_header,
    section_title,
    serialize_scan_result,
    serialize_xai_result,
)

dash.register_page(__name__, path="/scanner", name="Scanner", title="PFAS Scanner")

SUBSTANCES = ["GENERAL (Total)", "PFOS", "PFOA", "PFHXS", "PFNA", "PFDA", "PFHPA", "PFBS"]
MEDIA = ["Surface Water", "Groundwater", "Drinking Water", "Soil", "Sediment", "Wastewater"]


def field(label: str, child) -> html.Div:
    return html.Div([html.Label(label), child], className="form-field")


def layout():
    return html.Div(
        [
            page_header(
                "Scanner",
                "Site risk scanner",
                "Search or enter coordinates, select PFAS parameters, and update only the prediction panel with Dash callbacks.",
            ),
            html.Div(
                [
                    glass_card(
                        [
                            section_title("Location and parameters"),
                            field("Address search", dcc.Input(id="scanner-address", placeholder="e.g. Brussels", type="text")),
                            html.Div(
                                [
                                    html.Button("Find location", id="scanner-geocode", className="secondary-button"),
                                    html.Button("Calculate risk", id="scanner-run", className="primary-button"),
                                ],
                                className="button-row",
                            ),
                            dcc.Loading(
                                html.Div(id="scanner-geo-status", className="status-note"),
                                type="dot",
                                color=ACCENT,
                            ),
                            html.Div(
                                [
                                    field("Latitude", dcc.Input(id="scanner-lat", type="number", value=51.5, step=0.0001)),
                                    field("Longitude", dcc.Input(id="scanner-lon", type="number", value=-0.12, step=0.0001)),
                                    field("PFAS compound", dcc.Dropdown(id="scanner-substance", value="GENERAL (Total)", options=SUBSTANCES, clearable=False)),
                                    field("Media type", dcc.Dropdown(id="scanner-media", value="Surface Water", options=MEDIA, clearable=False)),
                                ],
                                className="form-grid",
                            ),
                            html.Div(
                                [
                                    html.Label("ASSESSMENT YEAR"),
                                    dcc.Slider(
                                        id="scanner-year",
                                        min=2001,
                                        max=2030,
                                        step=1,
                                        value=2024,
                                        marks={2001: "2001", 2024: "2024", 2030: "2030"},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                    ),
                                ],
                                className="assessment-area",
                            ),
                            dcc.Loading(
                                html.Div(id="scanner-run-status", className="status-note"),
                                type="dot",
                                color=ACCENT,
                            ),
                        ],
                        "span-4",
                    ),
                    html.Div(
                        dcc.Loading(
                            html.Div(id="scanner-result"),
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


@dash.callback(
    Output("scanner-lat", "value"),
    Output("scanner-lon", "value"),
    Output("scanner-geo-status", "children"),
    Input("scanner-geocode", "n_clicks"),
    State("scanner-address", "value"),
    prevent_initial_call=True,
)
def geocode_address(_clicks, address):
    if not address:
        return no_update, no_update, "Enter an address first."

    def offline_fallback():
        offline_match = offline_geocode(str(address))
        if offline_match:
            lat, lon, label = offline_match
            return round(lat, 5), round(lon, 5), f"Offline location match: {label}"
        return no_update, no_update, "Location search is unavailable. Enter coordinates manually, or search by country name while offline."

    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError
        import time

        geolocator = Nominatim(user_agent="pfas_risk_dash", timeout=10)
        loc = None
        for attempt in range(3):
            try:
                loc = geolocator.geocode(address, exactly_one=True)
                break
            except GeocoderTimedOut:
                time.sleep(1 * (attempt + 1))

        if not loc:
            return offline_fallback()
        return round(loc.latitude, 5), round(loc.longitude, 5), loc.address[:90]
    except GeocoderServiceError:
        return offline_fallback()
    except Exception:
        return offline_fallback()


@dash.callback(
    Output("scan-store", "data"),
    Output("xai-store", "data"),
    Output("scanner-run-status", "children"),
    Input("scanner-run", "n_clicks"),
    State("scanner-lat", "value"),
    State("scanner-lon", "value"),
    State("scanner-substance", "value"),
    State("scanner-year", "value"),
    State("scanner-media", "value"),
    prevent_initial_call=True,
)
def run_scan(_clicks, lat, lon, substance, year, media):
    predictor, _, xai = get_backend()
    if not predictor:
        return no_update, no_update, "Model engine is offline. Run the training pipeline first."
    try:
        clean_substance = str(substance).split(" ")[0]
        result = predictor.predict(float(lat), float(lon), substance=clean_substance, year=int(year), media_type=str(media))
        scan_payload = serialize_scan_result({**result, "lat": float(lat), "lon": float(lon), "year": int(year), "media_type": str(media)})
        xai_payload = None
        if xai:
            X_feat, _, _ = predictor.build_feature_frame(float(lat), float(lon), clean_substance, int(year), str(media))
            xai_result = xai.explain(
                X_feat,
                result["exceedance_prob"],
                result["predicted_value_ngl"],
                clean_substance,
                result["dist_to_nearest_sample_km"],
            )
            xai_payload = serialize_xai_result(xai_result)
        return scan_payload, xai_payload, "full-page reload."
    except Exception as exc:
        return no_update, no_update, f"Analysis failed: {exc}"


@dash.callback(
    Output("scanner-result", "children"),
    Input("scan-store", "data"),
    Input("xai-store", "data"),
)
def render_scan_result(scan, xai_payload):
    if not scan:
        return glass_card(
            [
                section_title("Prediction output"),
                html.P(
                    "Run a site scan to generate a risk gauge, local telemetry, and an explanation summary.",
                    className="status-note",
                ),
            ],
            "span-12",
        )

    def numeric_value(key: str, default: float = 0.0) -> float:
        value = scan.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if value == value and abs(value) != float("inf") else default

    probability = numeric_value("exceedance_prob")
    predicted_value = numeric_value("predicted_value_ngl")
    nearest_sample = numeric_value("dist_to_nearest_sample_km", 120.0)
    airport_proximity = numeric_value("dist_to_airport_km", 45.0)
    color = "#2E8B57" if probability < 0.35 else "#F28C62" if probability < 0.65 else "#D96B34"
    headline = xai_payload.get("headline") if xai_payload else scan.get("confidence_note", "")
    return html.Div(
        [
            glass_card(
                [section_title("Exceedance probability"), dcc.Graph(figure=gauge_figure(probability * 100, "Exceedance Probability", color), config={"displayModeBar": False})],
                "span-5",
            ),
            glass_card(
                [section_title("Narrative summary"), html.P(headline, className="status-note")],
                "span-7",
            ),
            html.Div(
                [
                    metric("Risk probability", f"{probability*100:.1f}%", scan.get("confidence_level", "")),
                    metric(
                        "Est. concentration",
                        f"{predicted_value:.1f} ng/L",
                        f"90% CI: {scan.get('conc_lower_ngl', 0.0):.1f}–{scan.get('conc_upper_ngl', 0.0):.1f}",
                    ),
                    metric("Nearest sample", f"{nearest_sample:.0f} km", "Training distance"),
                    metric("Airport proximity", f"{airport_proximity:.0f} km", "AFFF proxy"),
                ],
                className="metric-grid-4 span-12",
            ),
        ],
        className="dashboard-grid",
    )

