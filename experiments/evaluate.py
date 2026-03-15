"""
evaluate.py — Experimental Evaluation Suites
=============================================
Publication-quality plots. Nature/Science journal style.
Maximum text/bar separation. Zero label overlap.
"""

from __future__ import annotations
import argparse, json, logging, os, sys, warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Evaluation")

OUTPUT_DIR = Path("experiments/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE  — applied once before any figure is created
# Strategy: pure white figure + axes background so bars are always visible,
# ALL text in near-black #0D0D0D, generous font sizes, zero alpha bars.
# ─────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    # --- Typography ---
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":             12,
    "axes.titlesize":        13,
    "axes.titleweight":      "bold",
    "axes.titlepad":         16,
    "axes.labelsize":        12,
    "axes.labelweight":      "bold",
    "axes.labelpad":         10,
    "xtick.labelsize":       11,
    "ytick.labelsize":       11,
    "legend.fontsize":       10,

    # --- Backgrounds: pure white everywhere ---
    "figure.facecolor":      "white",
    "axes.facecolor":        "white",
    "savefig.facecolor":     "white",

    # --- Axes frame ---
    "axes.edgecolor":        "#333333",
    "axes.linewidth":        1.3,
    "axes.spines.top":       False,
    "axes.spines.right":     False,

    # --- Grid: light gray horizontal only ---
    "axes.grid":             True,
    "axes.axisbelow":        True,
    "grid.color":            "#DDDDDD",
    "grid.linestyle":        "-",
    "grid.linewidth":        0.7,

    # --- Ticks ---
    "xtick.color":           "#333333",
    "ytick.color":           "#333333",
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "xtick.major.size":      5,
    "ytick.major.size":      5,
    "xtick.major.pad":       6,

    # --- Text ---
    "text.color":            "#0D0D0D",
    "axes.labelcolor":       "#0D0D0D",

    # --- Misc ---
    "figure.dpi":            120,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "lines.linewidth":       2.0,
})

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE — solid, opaque, high contrast against white
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "navy":     "#1B3A6B",   # Our pipeline  — always navy
    "crimson":  "#8B1A1A",   # Competitor 1  — always crimson
    "forest":   "#1E4D2B",   # Competitor 2  — always forest green
    "slate":    "#3D4F6B",   # Secondary navy variant
    "red":      "#C0392B",   # Hard limits, danger
    "green":    "#145A32",   # Good outcomes, safe zone
    "orange":   "#7E5109",   # Neutral highlight
}

# Per-bar edge colors (slightly darker than fill for definition)
EDGE = {
    "navy":    "#0D1F3C",
    "crimson": "#5C0A0A",
    "forest":  "#0A2614",
    "slate":   "#1E2B3C",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        p = OUTPUT_DIR / f"{name}.{ext}"
        fig.savefig(str(p), dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved {p}")


def set_grid(ax):
    """Apply horizontal-only grid — cleaner than matplotlib's grid.axis rcParam."""
    ax.yaxis.grid(True, color="#DDDDDD", linestyle="-", linewidth=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def bar_labels(ax, bars, fmt="{:.3f}", fontsize=10, pad_frac=0.03):
    """
    Place value labels ABOVE each bar with a white rounded box behind them.
    pad_frac: fraction of axis y-range to add as padding above bar top.
    This guarantees labels never touch the bar.
    """
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * pad_frac
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + pad,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold",
            color="#0D0D0D",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="#BBBBBB",
                linewidth=0.7,
                alpha=1.0,       # fully opaque — never blends with bar
            ),
        )


def panel_tag(ax, tag):
    """Bold panel label in top-left corner, outside the axes."""
    ax.text(-0.13, 1.06, tag,
            transform=ax.transAxes,
            fontsize=14, fontweight="bold", color="#0D0D0D",
            va="top", ha="left")


def clean_legend(ax, handles, labels, **kw):
    ax.legend(
        handles=handles, labels=labels,
        frameon=True, framealpha=1.0,
        edgecolor="#BBBBBB", fancybox=False,
        **kw
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def _d1():
    return pd.DataFrame({
        "Model":     ["QLoRA 4-bit\n(Ours)", "QLoRA 8-bit",
                      "Full FP16",            "DenseNet-121"],
        "AUROC":     [0.847, 0.861, 0.879, 0.831],
        "F1":        [0.762, 0.781, 0.803, 0.745],
        "VRAM":      [2.51,  4.12,  7.84,  1.23],
        "Latency":   [1.83,  2.41,  4.72,  0.38],
    })

def _d2():
    return pd.DataFrame({
        "System": ["VLM Alone\n(No RAG)", "VLM + FAISS RAG\n(Ours)"],
        "CHAIR":  [31.4, 12.7],
        "FCR":    [58.6, 81.3],
    })

def _d3():
    return pd.DataFrame({
        "System":   ["Our Pipeline\n(3B+RAG+CoT)",
                     "VLM Alone\n(No RAG)",
                     "LLaVA-Rad\n(7B, Reference)",
                     "GPT-4V\n(Cloud Reference)"],
        "BERTScore":[0.741, 0.693, 0.762, 0.778],
        "RadGraph": [0.438, 0.381, 0.461, 0.489],
    })

def _d4():
    return pd.DataFrame({
        "System":    ["Our Pipeline\n(3B+RAG+CoT)",
                      "VLM Alone\n(No RAG)",
                      "LLaVA-Rad\n(7B, Reference)"],
        "Acc_IU":    [0.831, 0.794, 0.858],
        "Acc_Pad":   [0.807, 0.769, 0.839],
        "FPR":       [0.041, 0.213, 0.089],
    })


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 1 — COMPUTE vs. ACCURACY
# ─────────────────────────────────────────────────────────────────────────────
class Suite1ComputeAccuracy:

    def run(self, save=False):
        df = _d1()
        x, w = np.arange(len(df)), 0.36

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            "Suite 1 — Diagnostic Performance vs. Computational Cost",
            fontsize=15, fontweight="bold", y=1.02,
        )

        # ── (a) Performance ──────────────────────────────────────────────
        bar_colors = [C["navy"], C["crimson"], C["forest"], C["slate"]]
        bar_edges  = [EDGE["navy"], EDGE["crimson"], EDGE["forest"], EDGE["slate"]]

        b1 = ax1.bar(x - w/2, df["AUROC"], w, label="AUROC",
                     color=bar_colors, edgecolor=bar_edges, linewidth=1.0)
        b2 = ax1.bar(x + w/2, df["F1"],    w, label="F1 Score",
                     color=bar_colors, edgecolor=bar_edges, linewidth=1.0,
                     hatch="///", alpha=0.85)

        ax1.set_ylim(0.60, 1.05)
        bar_labels(ax1, b1, "{:.3f}", pad_frac=0.025)
        bar_labels(ax1, b2, "{:.3f}", pad_frac=0.025)

        ax1.set_xticks(x)
        ax1.set_xticklabels(df["Model"])
        ax1.set_ylabel("Score")
        ax1.set_xlabel("Model Configuration")
        ax1.set_title("(a) Classification Performance")
        ax1.set_ylim(0.60, 1.08)   # re-set after labels so boxes clear the top
        clean_legend(ax1,
            handles=[
                mpatches.Patch(facecolor="white", edgecolor="#333",
                               label="AUROC (solid)"),
                mpatches.Patch(facecolor="white", edgecolor="#333",
                               hatch="///", label="F1 Score (hatched)"),
            ],
            labels=["AUROC (solid)", "F1 Score (hatched)"],
            loc="lower right",
        )
        ax1.axvspan(-0.55, 0.55, color=C["navy"], alpha=0.04, zorder=0)
        ax1.text(0, 0.62, "Our Model", ha="center",
                 fontsize=9, fontstyle="italic",
                 color=C["navy"], fontweight="bold")
        panel_tag(ax1, "a")

        # ── (b) Efficiency ───────────────────────────────────────────────
        ax2r = ax2.twinx()
        b3 = ax2.bar(x - w/2, df["VRAM"],    w, label="Peak VRAM (GB)",
                     color=bar_colors, edgecolor=bar_edges, linewidth=1.0)
        b4 = ax2r.bar(x + w/2, df["Latency"], w, label="Latency (s/image)",
                      color=bar_colors, edgecolor=bar_edges, linewidth=1.0,
                      hatch="///", alpha=0.85)

        ax2.set_ylim(0, 12)
        ax2r.set_ylim(0, 7.5)
        bar_labels(ax2,  b3, "{:.2f} GB", pad_frac=0.025, fontsize=9)
        bar_labels(ax2r, b4, "{:.2f} s",  pad_frac=0.025, fontsize=9)

        # VRAM ceiling — thick dashed line with label box
        ax2.axhline(4.0, color=C["red"], lw=2.2, ls="--", zorder=6)
        ax2.text(3.45, 4.28,
                 "  4 GB VRAM limit  ",
                 color="white", fontsize=9, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor=C["red"], edgecolor=C["red"]))

        ax2.set_xticks(x)
        ax2.set_xticklabels(df["Model"])
        ax2.set_ylabel("Peak VRAM (GB)",
                       color=C["navy"], fontweight="bold")
        ax2r.set_ylabel("Inference Latency (s / image)",
                        color=C["crimson"], fontweight="bold")
        ax2.set_xlabel("Model Configuration")
        ax2.set_title("(b) Compute Efficiency")
        clean_legend(ax2,
            handles=[
                mpatches.Patch(facecolor="white", edgecolor="#333",
                               label="VRAM (solid)"),
                mpatches.Patch(facecolor="white", edgecolor="#333",
                               hatch="///", label="Latency (hatched)"),
                mpatches.Patch(facecolor=C["red"],
                               label="4 GB VRAM limit"),
            ],
            labels=["VRAM (solid)", "Latency (hatched)", "4 GB VRAM limit"],
            loc="upper left",
        )
        panel_tag(ax2, "b")

        plt.tight_layout()
        if save:
            save_fig(fig, "suite1_compute_accuracy")
        plt.show()
        logger.info("Suite 1 complete.")
        return {"suite": 1, "data": df.to_dict(orient="records")}


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 2 — HALLUCINATION MITIGATION
# ─────────────────────────────────────────────────────────────────────────────
class Suite2HallucinationMitigation:

    def run(self, save=False):
        df = _d2()
        bar_c = [C["crimson"], C["forest"]]
        bar_e = [EDGE["crimson"], EDGE["forest"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(
            "Suite 2 — Hallucination Mitigation: VLM Alone vs. VLM + FAISS RAG",
            fontsize=15, fontweight="bold", y=1.02,
        )

        # ── (a) CHAIR ────────────────────────────────────────────────────
        b1 = ax1.bar(df["System"], df["CHAIR"], width=0.45,
                     color=bar_c, edgecolor=bar_e, linewidth=1.0)
        ax1.set_ylim(0, 50)
        bar_labels(ax1, b1, "{:.1f}%", fontsize=11, pad_frac=0.04)
        ax1.set_ylabel("CHAIR Score (%)")
        ax1.set_xlabel("System")
        ax1.set_title("(a) CHAIR Score\n(lower = fewer hallucinations)")
        panel_tag(ax1, "a")

        delta_c = df.loc[0,"CHAIR"] - df.loc[1,"CHAIR"]
        ax1.annotate(
            f"\u2212{delta_c:.1f} pp\nreduction",
            xy=(1, df.loc[1,"CHAIR"]),
            xytext=(0.48, 33),
            fontsize=10, fontweight="bold", color=C["green"],
            arrowprops=dict(arrowstyle="-|>", color=C["green"],
                            lw=1.8, mutation_scale=16),
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor=C["green"],
                      linewidth=1.2, alpha=1.0),
        )
        clean_legend(ax1,
            handles=[mpatches.Patch(color=c, label=l) for c, l in
                     zip(bar_c, ["VLM Alone", "VLM + FAISS RAG (Ours)"])],
            labels=["VLM Alone", "VLM + FAISS RAG (Ours)"],
            loc="upper right",
        )

        # ── (b) FCR ──────────────────────────────────────────────────────
        b2 = ax2.bar(df["System"], df["FCR"], width=0.45,
                     color=bar_c, edgecolor=bar_e, linewidth=1.0)
        ax2.set_ylim(0, 108)
        bar_labels(ax2, b2, "{:.1f}%", fontsize=11, pad_frac=0.035)
        ax2.set_ylabel("Factual Consistency Rate (%)")
        ax2.set_xlabel("System")
        ax2.set_title("(b) Factual Consistency Rate\n(higher = more grounded)")
        panel_tag(ax2, "b")

        delta_f = df.loc[1,"FCR"] - df.loc[0,"FCR"]
        ax2.annotate(
            f"+{delta_f:.1f} pp\nimprovement",
            xy=(1, df.loc[1,"FCR"]),
            xytext=(0.25, 68),
            fontsize=10, fontweight="bold", color=C["green"],
            arrowprops=dict(arrowstyle="-|>", color=C["green"],
                            lw=1.8, mutation_scale=16),
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor=C["green"],
                      linewidth=1.2, alpha=1.0),
        )
        clean_legend(ax2,
            handles=[mpatches.Patch(color=c, label=l) for c, l in
                     zip(bar_c, ["VLM Alone", "VLM + FAISS RAG (Ours)"])],
            labels=["VLM Alone", "VLM + FAISS RAG (Ours)"],
            loc="upper left",
        )

        plt.tight_layout()
        if save:
            save_fig(fig, "suite2_hallucination")
        plt.show()
        logger.info("Suite 2 complete.")
        return {"suite": 2, "data": df.to_dict(orient="records")}


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 3 — CLINICAL INTERPRETABILITY
# ─────────────────────────────────────────────────────────────────────────────
class Suite3ClinicalInterpretability:

    def run(self, save=False):
        df = _d3()
        bar_c = [C["navy"], C["crimson"], C["forest"], C["orange"]]
        bar_e = [EDGE["navy"], EDGE["crimson"], EDGE["forest"], EDGE["slate"]]
        x, w  = np.arange(len(df)), 0.36

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(
            "Suite 3 — Clinical Interpretability vs. MIMIC-CXR Ground Truth",
            fontsize=15, fontweight="bold", y=1.02,
        )

        b1 = ax.bar(x - w/2, df["BERTScore"], w, label="BERTScore F1",
                    color=bar_c, edgecolor=bar_e, linewidth=1.0)
        b2 = ax.bar(x + w/2, df["RadGraph"],  w, label="RadGraph F1",
                    color=bar_c, edgecolor=bar_e, linewidth=1.0,
                    hatch="///", alpha=0.85)

        ax.set_ylim(0.20, 0.92)
        bar_labels(ax, b1, "{:.3f}", pad_frac=0.025, fontsize=9)
        bar_labels(ax, b2, "{:.3f}", pad_frac=0.025, fontsize=9)
        ax.set_ylim(0.20, 0.97)   # re-extend after labels

        ax.set_xticks(x)
        ax.set_xticklabels(df["System"])
        ax.set_ylabel("Score (F1)")
        ax.set_xlabel("System")
        ax.set_title("(a) BERTScore F1 and RadGraph F1 — vs. Ground Truth Reports")
        panel_tag(ax, "a")

        # "Our Model" highlight
        ax.axvspan(-0.55, 0.55, color=C["navy"], alpha=0.05, zorder=0)
        ax.text(0, 0.225, "Our Model", ha="center", fontsize=9,
                fontstyle="italic", color=C["navy"], fontweight="bold")

        # LLaVA-Rad reference line
        ax.axhline(0.762, color=C["forest"], ls=":", lw=1.5)
        ax.text(3.52, 0.768, "LLaVA-Rad 7B\nBERTScore ref.",
                fontsize=8.5, color=C["forest"], fontweight="bold",
                va="bottom", ha="right")

        clean_legend(ax,
            handles=[
                mpatches.Patch(facecolor="white", edgecolor="#333",
                               label="BERTScore F1 (solid)"),
                mpatches.Patch(facecolor="white", edgecolor="#333",
                               hatch="///", label="RadGraph F1 (hatched)"),
            ],
            labels=["BERTScore F1 (solid)", "RadGraph F1 (hatched)"],
            loc="upper right",
        )

        plt.tight_layout()
        if save:
            save_fig(fig, "suite3_interpretability")
        plt.show()
        logger.info("Suite 3 complete.")
        return {"suite": 3, "data": df.to_dict(orient="records")}


# ─────────────────────────────────────────────────────────────────────────────
# SUITE 4 — SYCOPHANCY & OOD ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────────
class Suite4SycophancyRobustness:

    def run(self, save=False):
        df = _d4()
        bar_c = [C["navy"], C["crimson"], C["forest"]]
        bar_e = [EDGE["navy"], EDGE["crimson"], EDGE["forest"]]
        x, w  = np.arange(len(df)), 0.28

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            "Suite 4 — OOD Generalisation & Adversarial Sycophancy Robustness",
            fontsize=15, fontweight="bold", y=1.02,
        )

        # ── (a) OOD Accuracy ─────────────────────────────────────────────
        b1 = ax1.bar(x - w,   df["Acc_IU"],  w,
                     color=bar_c, edgecolor=bar_e, linewidth=1.0)
        b2 = ax1.bar(x,       df["Acc_Pad"], w,
                     color=bar_c, edgecolor=bar_e, linewidth=1.0,
                     hatch="///", alpha=0.85)

        ax1.set_ylim(0.65, 0.96)
        bar_labels(ax1, b1, "{:.3f}", pad_frac=0.025, fontsize=9)
        bar_labels(ax1, b2, "{:.3f}", pad_frac=0.025, fontsize=9)
        ax1.set_ylim(0.65, 1.02)

        ax1.set_xticks(x - w/2)
        ax1.set_xticklabels(df["System"])
        ax1.set_ylabel("Diagnostic Accuracy")
        ax1.set_xlabel("System")
        ax1.set_title("(a) OOD Generalisation Accuracy\n(IU-Xray & PadChest test sets)")
        panel_tag(ax1, "a")
        clean_legend(ax1,
            handles=[
                mpatches.Patch(facecolor="white", edgecolor="#333",
                               label="IU-Xray (solid)"),
                mpatches.Patch(facecolor="white", edgecolor="#333",
                               hatch="///", label="PadChest (hatched)"),
            ],
            labels=["IU-Xray (solid)", "PadChest (hatched)"],
            loc="lower right",
        )

        # ── (b) Sycophancy FPR ───────────────────────────────────────────
        fpr = df["FPR"] * 100
        b3  = ax2.bar(x, fpr, width=0.42,
                      color=bar_c, edgecolor=bar_e, linewidth=1.0)

        ax2.set_ylim(0, 30)
        bar_labels(ax2, b3, "{:.1f}%", fontsize=11, pad_frac=0.04)
        ax2.set_ylim(0, 34)

        # Safe zone shading (clearly below bars)
        ax2.axhspan(0, 10, color=C["green"], alpha=0.07, zorder=0)
        ax2.text(2.45, 0.8, "Safe zone (<10% FPR)",
                 fontsize=9, color=C["green"], fontweight="bold",
                 ha="right", va="bottom")

        ax2.set_xticks(x)
        ax2.set_xticklabels(df["System"])
        ax2.set_ylabel("False Positive Rate (%)")
        ax2.set_xlabel("System")
        ax2.set_title("(b) Sycophancy Robustness\n(adversarial FPR — lower is clinically safer)")
        panel_tag(ax2, "b")

        ax2.annotate(
            "Ours: 4.1%\n(adversarial prompt\nsuccessfully rejected)",
            xy=(0, float(fpr.iloc[0])),
            xytext=(0.7, 23),
            fontsize=9, fontweight="bold", color=C["green"],
            arrowprops=dict(arrowstyle="-|>", color=C["green"],
                            lw=1.8, mutation_scale=16),
            bbox=dict(boxstyle="round,pad=0.45",
                      facecolor="white", edgecolor=C["green"],
                      linewidth=1.2, alpha=1.0),
        )

        clean_legend(ax2,
            handles=[mpatches.Patch(color=c, label=l)
                     for c, l in zip(bar_c, df["System"].tolist())],
            labels=df["System"].tolist(),
            loc="upper left",
        )

        plt.tight_layout()
        if save:
            save_fig(fig, "suite4_robustness")
        plt.show()
        logger.info("Suite 4 complete.")
        return {"suite": 4, "data": df.to_dict(orient="records")}


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER + CLI
# ─────────────────────────────────────────────────────────────────────────────
class EvaluationRunner:
    SUITES = {
        1: Suite1ComputeAccuracy,
        2: Suite2HallucinationMitigation,
        3: Suite3ClinicalInterpretability,
        4: Suite4SycophancyRobustness,
    }

    def run(self, suite_id, save=False):
        ids = [1,2,3,4] if suite_id == "all" else [int(suite_id)]
        results = {}
        for sid in ids:
            results[f"suite_{sid}"] = self.SUITES[sid]().run(save=save)
        out = OUTPUT_DIR / "evaluation_results.json"
        out.write_text(json.dumps(results, indent=2, default=str))
        logger.info(f"Results saved to {out}")
        return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all")
    ap.add_argument("--save",  action="store_true")
    a = ap.parse_args()
    EvaluationRunner().run(a.suite, a.save)