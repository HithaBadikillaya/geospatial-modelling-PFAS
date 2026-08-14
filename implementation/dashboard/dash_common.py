from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
from dash import dcc, html
from folium.plugins import FastMarkerCluster, HeatMap

ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_PATH = ROOT / "dataset" / "pfas_golden.parquet"
HOTSPOT_PATH = ROOT / "outputs" / "spatial" / "pfas_hotspots.geojson"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pfas-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "implementation"))

try:
    from api import PFASPredictor
    from simulation import SCENARIO_PRESETS, SimulationEngine
    from xai import ExplanationResult, XAIEngine

    MODELS_AVAILABLE = True
    LOAD_ERROR = ""
except Exception as exc:  # pragma: no cover - surfaced in UI
    PFASPredictor = None  # type: ignore[assignment]
    SimulationEngine = None  # type: ignore[assignment]
    XAIEngine = None  # type: ignore[assignment]
    ExplanationResult = None  # type: ignore[assignment]
    SCENARIO_PRESETS = {}
    MODELS_AVAILABLE = False
    LOAD_ERROR = str(exc)

ACCENT = "#F28C62"
ACCENT_SOFT = "#F6B08F"
SUCCESS = "#2E8B57"
SUCCESS_SOFT = "rgba(46, 139, 87, 0.14)"
TEXT = "#191715"
MUTED = "#6E6760"
CHART_MUTED = "#A19A92"
CHART_SEQUENCE = ["#F28C62", "#2E8B57", "#D96B34", "#1E5E3A", "#F6B08F", "#52B788"]

XAI_SUGGESTED_QUESTIONS = [
    "Why is the risk high?",
    "What is the biggest factor?",
    "Is this safe to drink?",
    "What is PFOS?",
    "How accurate is this model?",
    "How can risk be reduced?",
    "What are all the factors?",
    "What are the health effects?",
    "What are the regulatory thresholds?",
]


@lru_cache(maxsize=1)
def load_summary() -> dict[str, Any] | None:
    if not GOLDEN_PATH.exists():
        return None
    schema = pq.read_schema(GOLDEN_PATH).names
    cols = [c for c in ["country", "substance", "year", "above_100_ng_l", "source_system", "lat", "lon"] if c in schema]
    df = pd.read_parquet(GOLDEN_PATH, columns=cols)
    year_s = pd.to_numeric(df.get("year", pd.Series(dtype=float)), errors="coerce")
    exc_s = pd.to_numeric(df.get("above_100_ng_l", pd.Series(dtype=float)), errors="coerce")
    return {
        "rows": len(df),
        "countries": int(df["country"].replace("Unknown", np.nan).nunique()) if "country" in df else 0,
        "substances": int(df["substance"].replace("Unknown", np.nan).nunique()) if "substance" in df else 0,
        "year_min": int(year_s.min()) if year_s.notna().any() else None,
        "year_max": int(year_s.max()) if year_s.notna().any() else None,
        "exceedance": float(exc_s.mean() * 100) if exc_s.notna().any() else None,
        "source_mix": df["source_system"].value_counts() if "source_system" in df else pd.Series(dtype=int),
        "top_substances": df["substance"].value_counts().head(7) if "substance" in df else pd.Series(dtype=int),
        "map_points": (
            df.dropna(subset=["lat", "lon"])[["lat", "lon"]]
            .sample(min(3000, len(df)), random_state=42)
            .values.tolist()
            if "lat" in df.columns and "lon" in df.columns
            else []
        ),
    }


@lru_cache(maxsize=1)
def load_trend_data() -> pd.DataFrame:
    if not GOLDEN_PATH.exists():
        return pd.DataFrame()
    schema = pq.read_schema(GOLDEN_PATH).names
    cols = [c for c in ["year", "substance", "log_value", "above_100_ng_l"] if c in schema]
    df = pd.read_parquet(GOLDEN_PATH, columns=cols)
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    return df.dropna(subset=["year"])


@lru_cache(maxsize=1)
def load_hotspots() -> gpd.GeoDataFrame | None:
    if not HOTSPOT_PATH.exists():
        return None
    gdf = gpd.read_file(HOTSPOT_PATH)
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x
    return gdf


@lru_cache(maxsize=1)
def get_backend():
    predictor, simulator, xai = None, None, None
    try:
        if PFASPredictor is not None:
            predictor = PFASPredictor()
    except Exception as exc:
        log.warning(f"Predictor init exception: {exc}")

    try:
        if SimulationEngine is not None:
            simulator = SimulationEngine()
    except Exception as exc:
        log.warning(f"Simulator init exception: {exc}")

    try:
        if XAIEngine is not None and predictor is not None:
            xai = XAIEngine(predictor.clf)
    except Exception as exc:
        log.warning(f"XAI init exception: {exc}")

    return predictor, simulator, xai


@lru_cache(maxsize=1)
def overview_map_html() -> str:
    summary = load_summary()
    hotspots = load_hotspots()
    fmap = folium.Map(location=[51.0, 10.0], zoom_start=4, tiles="CartoDB positron")
    if hotspots is not None and not hotspots.empty:
        heat_data = [[r["lat"], r["lon"], max(float(r["gi_zscore"]), 0)] for _, r in hotspots.iterrows()]
        HeatMap(
            heat_data,
            radius=18,
            blur=22,
            gradient={"0.2": "#D9C7B8", "0.5": "#C8A68D", "0.8": ACCENT, "1.0": ACCENT_SOFT},
        ).add_to(fmap)
    if summary and summary["map_points"]:
        FastMarkerCluster(summary["map_points"]).add_to(fmap)
    return fmap.get_root().render()


def base_figure_layout(height: int = 360) -> dict[str, Any]:
    return {
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 36, "r": 20, "t": 20, "b": 30},
        "font": {"family": "Inter, system-ui, sans-serif", "size": 11, "color": MUTED},
        "colorway": CHART_SEQUENCE,
        "xaxis": {"gridcolor": "rgba(25,23,21,0.08)", "linecolor": "rgba(25,23,21,0.12)", "zeroline": False, "automargin": True},
        "yaxis": {"gridcolor": "rgba(25,23,21,0.08)", "linecolor": "rgba(25,23,21,0.12)", "zeroline": False, "automargin": True},
    }


def empty_figure(message: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(base_figure_layout(height))
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"color": MUTED})
    return fig


def gauge_figure(value: float, title: str = "Risk Score", color: str = ACCENT) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 13, "color": MUTED}},
            number={"font": {"size": 34, "color": TEXT}, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "rgba(25,23,21,0.16)", "tickfont": {"color": MUTED}},
                "bar": {"color": color, "thickness": 0.24},
                "bgcolor": "rgba(255,255,255,0.18)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(46,139,87,0.08)"},
                    {"range": [35, 65], "color": "rgba(242,140,98,0.10)"},
                    {"range": [65, 100], "color": "rgba(217,107,52,0.12)"},
                ],
            },
        )
    )
    fig.update_layout(base_figure_layout(260), margin={"l": 8, "r": 8, "t": 36, "b": 24})
    return fig


def metric(label: str, value: str, sub: str | None = None) -> html.Div:
    return html.Div(
        [html.Div(label, className="metric-label"), html.Div(value, className="metric-value"), html.Div(sub or "", className="metric-sub")],
        className="metric-card",
    )


def page_header(eyebrow: str, title: str, subtitle: str) -> html.Div:
    return html.Div(
        [html.Div(eyebrow, className="eyebrow"), html.H1(title), html.P(subtitle)],
        className="page-hero",
    )


def glass_card(children, className: str = "") -> html.Div:
    return html.Div(children, className=f"glass-card {className}".strip())


def section_title(text: str) -> html.Div:
    return html.Div(text, className="section-title")


def serialize_scan_result(result: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(result, default=float))


def serialize_xai_result(result: Any) -> dict[str, Any]:
    return asdict(result)


def restore_xai_context(xai: Any, payload: dict[str, Any] | None) -> None:
    if not xai or not payload or ExplanationResult is None:
        return
    try:
        xai._context = ExplanationResult(**payload)
    except Exception:
        return


def serialize_sim_result(result: Any) -> dict[str, Any]:
    return asdict(result)


def source_mix_figure() -> go.Figure:
    summary = load_summary()
    if not summary or summary["source_mix"].empty:
        return empty_figure("No source mix data available", 260)
    df = summary["source_mix"].reset_index()
    df.columns = ["Source", "Records"]
    fig = px.bar(df, x="Records", y="Source", orientation="h", color_discrete_sequence=[ACCENT])
    fig.update_layout(base_figure_layout(260), showlegend=False, yaxis={"autorange": "reversed"})
    return fig


def compound_donut_figure() -> go.Figure:
    summary = load_summary()
    if not summary or summary["top_substances"].empty:
        return empty_figure("No compound data available", 260)
    fig = px.pie(
        values=summary["top_substances"].values,
        names=summary["top_substances"].index,
        hole=0.58,
        color_discrete_sequence=CHART_SEQUENCE,
    )
    fig.update_traces(textinfo="percent", marker={"line": {"color": "rgba(255,255,255,0.55)", "width": 1}})
    fig.update_layout(base_figure_layout(260), showlegend=True)
    return fig
