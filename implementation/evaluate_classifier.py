#!/usr/bin/env python3
"""
implementation/evaluate_classifier.py
======================================
Confusion Matrix + ROC Curve — uses the production trained model directly.
Evaluates on a 20% stratified spatial hold-out (fast, no re-training).

Outputs → outputs/figures/eval_confusion_matrix.png
          outputs/figures/eval_roc_curve.png
"""

import json
import pickle
import warnings
import sys
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from sklearn.model_selection import GroupShuffleSplit

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
FIG_DIR    = ROOT / "outputs" / "figures"
GOLDEN     = ROOT / "dataset" / "pfas_golden.parquet"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = json.loads((MODELS_DIR / "feature_schema.json").read_text())
TARGET    = "above_100_ng_l"
GROUP_COL = "spatial_block_id"

PALETTE = {
    "bg": "#0f172a", "panel": "#1e293b", "border": "#334155",
    "text": "#e2e8f0", "muted": "#94a3b8",
    "green": "#10b981", "red": "#ef4444", "yellow": "#f59e0b",
    "blue": "#6366f1",
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


def load_eval_data():
    print("Loading dataset …")
    df = pd.read_parquet(GOLDEN).dropna(subset=[TARGET])
    X      = df[FEATURE_COLS].fillna(-1)           # keep as DataFrame → feature names preserved
    y      = df[TARGET].values.astype(int)
    groups = df[GROUP_COL].values if GROUP_COL in df.columns else np.arange(len(df))

    # Spatial hold-out: 20% of spatial blocks go to test
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, test_idx = next(splitter.split(X, y, groups))

    print(f"  Test set: {len(test_idx):,} samples  "
          f"(positive rate: {y[test_idx].mean()*100:.1f}%)")

    with open(MODELS_DIR / "lgbm_calibrated.pkl", "rb") as f:
        model = pickle.load(f)

    X_test = X.iloc[test_idx]
    y_test = y[test_idx]
    probs  = model.predict_proba(X_test)[:, 1]
    return y_test, probs


def plot_confusion_matrix(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _style(fig)

    # Left — raw counts (Using LogNorm for better visual balance across imbalanced classes)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Below 100 ng/L\n(Safe)", "Above 100 ng/L\n(Contaminated)"],
    )
    # Filter out zeros for LogNorm to avoid math errors if one cell is empty
    norm = LogNorm(vmin=max(1, cm.min()), vmax=cm.max())
    disp.plot(ax=axes[0], colorbar=False, cmap="Greens", values_format="d", im_kw={"norm": norm})
    axes[0].set_title(f"Confusion Matrix (Log Scale, threshold={threshold})", pad=12, fontsize=12)

    # Right — normalised (recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    disp2   = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=["Below 100 ng/L\n(Safe)", "Above 100 ng/L\n(Contaminated)"],
    )
    disp2.plot(ax=axes[1], colorbar=False, cmap="Blues", values_format=".1%")
    axes[1].set_title("Normalised (recall per class)", pad=12, fontsize=12)

    for ax in axes:
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text"])
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    fig.text(0.5, 0.01,
             f"TP={tp:,}   FP={fp:,}   FN={fn:,}   TN={tn:,}   "
             f"Precision={precision:.3f}   Recall={recall:.3f}   F1={f1:.3f}",
             ha="center", fontsize=9, color=PALETTE["muted"])
    fig.suptitle("PFAS Exceedance Classifier — Confusion Matrix",
                 fontsize=14, color=PALETTE["text"], fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "eval_confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"✓ Saved → {out}")


def plot_roc(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold (Youden's J = sensitivity + specificity − 1)
    J = tpr - fpr
    opt_idx   = np.argmax(J)
    opt_thresh = thresholds[opt_idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    _style(fig)

    ax.plot(fpr, tpr, color=PALETTE["green"], lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.10, color=PALETTE["green"])
    ax.plot([0, 1], [0, 1], ":", color=PALETTE["muted"], lw=1.2, label="Random classifier")
    ax.scatter(fpr[opt_idx], tpr[opt_idx], s=120,
               color=PALETTE["yellow"], zorder=10, edgecolors=PALETTE["bg"],
               label=f"Optimal threshold = {opt_thresh:.2f}\n"
                     f"(TPR={tpr[opt_idx]:.3f}, FPR={fpr[opt_idx]:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve — PFAS Exceedance Classifier",
                 fontsize=13, color=PALETTE["text"], pad=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right",
              facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"])
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.06])
    ax.grid(True, alpha=0.15, color=PALETTE["border"])

    plt.tight_layout()
    out = FIG_DIR / "eval_roc_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"✓ Saved → {out}")
    print(f"  ROC-AUC = {roc_auc:.4f}   Optimal threshold = {opt_thresh:.3f}")


if __name__ == "__main__":
    y_true, y_prob = load_eval_data()
    plot_confusion_matrix(y_true, y_prob)
    plot_roc(y_true, y_prob)
    print("\nDone — figures saved to outputs/figures/")
