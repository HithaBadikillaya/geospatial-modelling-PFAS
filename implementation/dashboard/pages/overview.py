from __future__ import annotations

import dash
from dash import dcc, html

from dash_common import (
    compound_donut_figure,
    glass_card,
    load_summary,
    metric,
    overview_map_html,
    page_header,
    section_title,
    source_mix_figure,
)

dash.register_page(__name__, path="/", name="Overview", title="PFAS Overview")


def layout():
    summary = load_summary()
    if not summary:
        return html.Div(
            [
                page_header("Overview", "No dataset loaded", "The PFAS golden dataset was not found."),
                glass_card("Run the data pipeline first, then refresh this dashboard."),
            ]
        )

    return html.Div(
        [
            page_header(
                "Overview",
                "Contamination intelligence",
                "A spatial command center for PFAS monitoring coverage, source mix, and compound composition.",
            ),
            html.Div(
                [
                    # Row 1: 4 Key Performance Indicators across the full 12-column width
                    html.Div(
                        [
                            metric("Total records", f"{summary['rows']:,}", "Curated PFAS measurements"),
                            metric("Countries", f"{summary['countries']:,}", "Geographic coverage"),
                            metric("Compounds", f"{summary['substances']:,}", "Tracked PFAS families"),
                            metric(
                                "Exceedance rate",
                                f"{summary['exceedance']:.1f}%" if summary["exceedance"] is not None else "—",
                                "Above screening threshold",
                            ),
                        ],
                        className="metric-grid-4 span-12",
                    ),
                    # Row 2: Sampling Density Map (8 columns) + Source Mix Chart (4 columns)
                    glass_card(
                        [section_title("Sampling density"), html.Iframe(srcDoc=overview_map_html(), className="map-frame")],
                        "span-8",
                    ),
                    glass_card(
                        [section_title("Records by source"), dcc.Graph(figure=source_mix_figure(), config={"displayModeBar": False})],
                        "span-4",
                    ),
                    # Row 3: Compound breakdown (6 columns) + Interpretation notes (6 columns)
                    glass_card(
                        [section_title("Compound breakdown"), dcc.Graph(figure=compound_donut_figure(), config={"displayModeBar": False})],
                        "span-6",
                    ),
                    glass_card(
                        [
                            section_title("Interpretation"),
                            html.P(
                                "Hotter map regions indicate denser historical evidence, not a regulatory determination. "
                                "Use Scanner for a site-specific model prediction and Analysis for factor attribution.",
                                className="status-note",
                            ),
                            html.Div(
                                [
                                    html.Span("screening", className="pill"),
                                    html.Span("non-regulatory", className="pill"),
                                    html.Span("cached map", className="pill"),
                                    html.Span("ensemble ML", className="pill"),
                                ],
                                className="pill-list",
                            ),
                        ],
                        "span-6",
                    ),
                ],
                className="dashboard-grid",
            ),
        ]
    )
