#!/usr/bin/env python3
"""
implementation/evaluate_extended.py
=====================================
Extended Model Diagnostics — plots beyond confusion matrix and accuracy.

Produces
--------
outputs/figures/eval_shap_global.png          — global SHAP feature importance bar
outputs/figures/eval_shap_beeswarm.png        — SHAP beeswarm (summary plot)
outputs/figures/eval_regressor_scatter.png    — predicted vs actual concentration
outputs/figures/eval_residuals.png            — residual distribution + QQ plot
outputs/figures/eval_spatial_error.png        — error-by-distance-to-training-data

Run
---
    python implementation/evaluate_extended.py
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
FIG_DIR    = ROOT / "outputs" / "figures"
GOLDEN     = ROOT / "dataset" / "pfas_golden.parquet"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = json.loads((MODELS_DIR / "feature_schema.json").read_text())
TARGET_CLF   = "above_100_ng_l"
TARGET_REG   = "log_value"
GROUP_COL    = "spatial_block_id"

FEATURE_LABELS = {
    "substance_ord":             "PFAS Compound Type",
    "is_long_chain":             "Long-Chain Compound",
    "carbon_chain_length":       "Carbon Chain Length",
    "is_sulfonyl":               "Sulfonate Group",
    "is_aquatic":                "Aquatic Sampling",
    "is_soil_based":             "Soil/Sediment Sampling",
    "is_wastewater":             "Wastewater Source",
    "year_normalized":           "Measurement Year",
    "is_post_2018":              "Post-2018 (EU Restrictions)",
    "month":                     "Month",
    "spatial_density_50km":      "Nearby Measurements (50 km)",
    "mean_log_value_50km":       "Mean PFAS Level Nearby",
    "nearest_training_point_km": "Distance to Nearest Data",
    "dist_to_airport_km":        "Distance to Airport",
}

PALETTE = {
    "bg": "#0f172a", "panel": "#1e293b", "border": "#334155",
    "text": "#e2e8f0", "muted": "#94a3b8",
    "green": "#10b981", "red": "#ef4444", "yellow": "#f59e0b",
    "blue": "#6366f1", "cyan": "#06b6d4",
}


def _style(fig):
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in fig.axes:
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text"])
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE["border"])


# ---------------------------------------------------------------------------
# 1. Global SHAP feature importance
# ---------------------------------------------------------------------------

def plot_shap_global(sample_n=2000):
    print("Computing global SHAP importance …")
    df = pd.read_parquet(GOLDEN).dropna(subset=[TARGET_CLF])
    X  = df[FEATURE_COLS].fillna(-1).values
    sample_idx = np.random.default_rng(42).choice(len(X), min(sample_n, len(X)), replace=False)
    X_sample   = X[sample_idx]

    with open(MODELS_DIR / "lgbm_calibrated.pkl", "rb") as f:
        model = pickle.load(f)

    base = model.calibrated_classifiers_[0].estimator
    while hasattr(base, "estimator"):
        base = base.estimator

    explainer   = shap.TreeExplainer(base)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    mean_abs = np.abs(shap_vals).mean(axis=0)
    labels   = [FEATURE_LABELS.get(f, f.replace("_", " ").title()) for f in FEATURE_COLS]
    order    = np.argsort(mean_abs)

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(10, 7))
    _style(fig)
    colors = [PALETTE["green"] if v > np.median(mean_abs) else PALETTE["blue"]
              for v in mean_abs[order]]
    bars = ax.barh([labels[i] for i in order], mean_abs[order],
                   color=colors, edgecolor=PALETTE["border"], linewidth=0.6)
    for bar, v in zip(bars, mean_abs[order]):
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8, color=PALETTE["muted"])
    ax.set_xlabel("Mean |SHAP Value| (impact on model output)", fontsize=10)
    ax.set_title("Global Feature Importance — SHAP\n(mean absolute SHAP across all samples)",
                 fontsize=12, color=PALETTE["text"], pad=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.15, color=PALETTE["border"])
    plt.tight_layout()
    out = FIG_DIR / "eval_shap_global.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"Saved → {out}")

    # Return for beeswarm
    return shap_vals, X_sample, labels


# ---------------------------------------------------------------------------
# 2. SHAP Beeswarm (summary plot)
# ---------------------------------------------------------------------------

def plot_shap_beeswarm(shap_vals, X_sample, labels):
    print("Rendering SHAP beeswarm …")
    plt.rcParams["figure.facecolor"] = PALETTE["bg"]
    plt.rcParams["axes.facecolor"]   = PALETTE["panel"]
    plt.rcParams["text.color"]       = PALETTE["text"]
    plt.rcParams["axes.labelcolor"]  = PALETTE["text"]
    plt.rcParams["xtick.color"]      = PALETTE["text"]
    plt.rcParams["ytick.color"]      = PALETTE["text"]

    fig, ax = plt.subplots(figsize=(11, 7))
    shap.summary_plot(
        shap_vals, X_sample,
        feature_names=labels,
        show=False, plot_size=None,
        color_bar_label="Feature value",
        plot_type="dot",
    )
    ax = plt.gca()
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["text"])
    fig = plt.gcf()
    fig.patch.set_facecolor(PALETTE["bg"])
    plt.title("SHAP Beeswarm — Impact Direction per Feature",
              fontsize=12, color=PALETTE["text"], pad=10, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "eval_shap_beeswarm.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close("all")
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# 3. Regressor — predicted vs actual scatter
# ---------------------------------------------------------------------------

def plot_regressor_scatter(sample_n=4000):
    print("Evaluating regressor …")
    df = pd.read_parquet(GOLDEN).dropna(subset=[TARGET_REG])
    X  = df[FEATURE_COLS].fillna(-1).values
    y  = df[TARGET_REG].values
    groups = df[GROUP_COL].values if GROUP_COL in df.columns else np.arange(len(df))

    with open(MODELS_DIR / "lgbm_regressor.pkl", "rb") as f:
        reg = pickle.load(f)

    gkf = GroupKFold(n_splits=3)
    y_true_all, y_pred_all = [], []
    for tr_idx, val_idx in gkf.split(X, y, groups):
        preds = reg.predict(X[val_idx])
        y_true_all.extend(y[val_idx])
        y_pred_all.extend(preds)

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)

    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)

    # Sample for scatter
    rng  = np.random.default_rng(0)
    idx  = rng.choice(len(y_true), min(sample_n, len(y_true)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _style(fig)

    # Scatter
    ax = axes[0]
    sc = ax.scatter(y_true[idx], y_pred[idx],
                    alpha=0.35, s=12, c=PALETTE["cyan"], edgecolors="none")
    lo = min(y_true.min(), y_pred.min()) - 0.2
    hi = max(y_true.max(), y_pred.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "--", color=PALETTE["yellow"], lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual log(PFAS + 1)", fontsize=10)
    ax.set_ylabel("Predicted log(PFAS + 1)", fontsize=10)
    ax.set_title("Predicted vs Actual — Concentration Regressor", fontsize=11,
                 color=PALETTE["text"], pad=10, fontweight="bold")
    ax.legend(fontsize=8, facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"])
    ax.text(0.05, 0.95,
            f"R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}",
            transform=ax.transAxes, fontsize=9, va="top",
            color=PALETTE["muted"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["panel"],
                      edgecolor=PALETTE["border"], alpha=0.9))
    ax.grid(True, alpha=0.12, color=PALETTE["border"])

    # Residuals histogram
    ax2   = axes[1]
    resid = y_pred - y_true
    ax2.hist(resid, bins=60, color=PALETTE["blue"], alpha=0.75, edgecolor="none",
             density=True)
    xs  = np.linspace(resid.min(), resid.max(), 200)
    mu, sigma = stats.norm.fit(resid)
    ax2.plot(xs, stats.norm.pdf(xs, mu, sigma), color=PALETTE["green"], lw=2,
             label=f"Normal fit (μ={mu:.3f}, σ={sigma:.3f})")
    ax2.axvline(0, color=PALETTE["yellow"], lw=1.2, linestyle="--", label="Zero error")
    ax2.set_xlabel("Residual (predicted − actual)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title("Residual Distribution", fontsize=11, color=PALETTE["text"],
                  pad=10, fontweight="bold")
    ax2.legend(fontsize=8, facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
               labelcolor=PALETTE["text"])
    ax2.grid(True, alpha=0.12, color=PALETTE["border"])

    fig.suptitle("LightGBM Concentration Regressor — Diagnostics",
                 fontsize=13, color=PALETTE["text"], y=1.02, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "eval_regressor_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# 4. Spatial error analysis — classifier error vs distance to training data
# ---------------------------------------------------------------------------

def plot_spatial_error():
    print("Computing spatial error vs distance …")
    df = pd.read_parquet(GOLDEN).dropna(subset=[TARGET_CLF, "nearest_training_point_km"])

    X      = df[FEATURE_COLS].fillna(-1).values
    y      = df[TARGET_CLF].values.astype(int)
    dist   = df["nearest_training_point_km"].values
    groups = df[GROUP_COL].values if GROUP_COL in df.columns else np.arange(len(df))

    with open(MODELS_DIR / "lgbm_calibrated.pkl", "rb") as f:
        model = pickle.load(f)

    probs = model.predict_proba(X)[:, 1]
    error = np.abs(probs - y)  # absolute error per sample

    # Bin by distance
    bins   = [0, 10, 25, 50, 100, 200, 500, 2000, np.inf]
    labels = ["0-10", "10-25", "25-50", "50-100", "100-200", "200-500", "500-2000", ">2000"]
    bucket = pd.cut(dist, bins=bins, labels=labels)
    summary = pd.DataFrame({"error": error, "bucket": bucket}).groupby("bucket")["error"]
    mean_err   = summary.mean()
    count_each = summary.count()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _style(fig)

    ax = axes[0]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(mean_err)))
    bars = ax.bar(mean_err.index, mean_err.values, color=colors,
                  edgecolor=PALETTE["border"], linewidth=0.6)
    for bar, v in zip(bars, mean_err.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=8, color=PALETTE["muted"])
    ax.set_xlabel("Distance to Nearest Training Point (km)", fontsize=10)
    ax.set_ylabel("Mean Absolute Error (|prob − label|)", fontsize=10)
    ax.set_title("Classifier Error vs Spatial Coverage\n(how error grows far from training data)",
                 fontsize=11, color=PALETTE["text"], pad=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.15, color=PALETTE["border"])

    ax2 = axes[1]
    ax2.bar(count_each.index, count_each.values, color=PALETTE["blue"],
            edgecolor=PALETTE["border"], linewidth=0.6, alpha=0.8)
    ax2.set_xlabel("Distance to Nearest Training Point (km)", fontsize=10)
    ax2.set_ylabel("Number of Samples", fontsize=10)
    ax2.set_title("Data Coverage by Distance Band",
                  fontsize=11, color=PALETTE["text"], pad=10, fontweight="bold")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", alpha=0.15, color=PALETTE["border"])

    fig.suptitle("Spatial Error Analysis — Trust Degrades with Distance",
                 fontsize=13, color=PALETTE["text"], y=1.02, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "eval_spatial_error.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    shap_vals, X_sample, labels = plot_shap_global()
    plot_shap_beeswarm(shap_vals, X_sample, labels)
    plot_regressor_scatter()
    plot_spatial_error()
    print("\nAll extended diagnostic figures saved to outputs/figures/")
