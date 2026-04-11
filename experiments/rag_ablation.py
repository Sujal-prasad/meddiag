"""
rag_ablation.py — RAG Retrieval-k Ablation Study
==================================================
Evaluates how the number of retrieved documents (k) affects:
    1. Answer quality   — BERTScore F1 vs MIMIC-CXR ground truth
    2. Inference latency — wall-clock time (seconds)
    3. VRAM usage       — peak GPU memory spike (MB)

Integrates directly with the existing meddiag pipeline:
    - FAISSKnowledgeBase  (pipeline.py) for retrieval
    - QLoRAModelManager   (pipeline.py) for generation
    - StreamingDatasetManager (data_loader.py) for queries + ground truth

Usage:
    # Basic run (k = 1, 3, 5, 10 — default 10 queries)
    python -m experiments.rag_ablation

    # Custom k values and more queries
    python -m experiments.rag_ablation --k 1 3 5 10 15 --n-queries 20

    # CPU-only mode (no VRAM measurement)
    python -m experiments.rag_ablation --cpu-only

Output:
    experiments/figures/rag_ablation_results.csv
    experiments/figures/rag_ablation_k_vs_bertscore.png
    experiments/figures/rag_ablation_k_vs_latency.png
    experiments/figures/rag_ablation_k_vs_vram.png
    experiments/figures/rag_ablation_combined.png
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import csv
import gc
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import torch
from bert_score import score as bert_score_fn
from tqdm import tqdm

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline    import (
    FAISSKnowledgeBase, FAISSConfig,
    QLoRAModelManager,  InferenceConfig,
    QuantizationConfig, LoRAAdapterConfig,
    EdgeMedicalVLM,
)
from src.data_loader import StreamingDatasetManager

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/rag_ablation.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("RAGAblation")

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR = Path("experiments/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT STYLE  (matches evaluate.py / run_green_eval.py)
# ─────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     14,
    "axes.labelsize":    12,
    "axes.labelweight":  "bold",
    "axes.labelpad":     8,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   10,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.linewidth":    1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.color":        "#E5E5E5",
    "grid.linestyle":    "-",
    "grid.linewidth":    0.6,
    "figure.dpi":        120,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "text.color":        "#111111",
    "lines.linewidth":   2.2,
    "lines.markersize":  8,
})

# Semantic colours consistent across all project figures
C_NAVY    = "#1B3A6B"
C_CRIMSON = "#8B1A1A"
C_FOREST  = "#1E4D2B"
C_AMBER   = "#7E5109"
C_SLATE   = "#4A5568"

# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """Holds all metrics for a single (k, query) experiment run."""
    k:              int
    query_id:       int
    query_text:     str
    generated:      str
    ground_truth:   str
    bertscore_f1:   float
    bertscore_p:    float
    bertscore_r:    float
    latency_s:      float
    vram_initial_mb: float
    vram_final_mb:  float
    vram_spike_mb:  float
    n_docs_retrieved: int


@dataclass
class AggregatedResult:
    """Mean ± std across all queries for a given k."""
    k:                   int
    mean_bertscore:      float
    std_bertscore:       float
    mean_latency_s:      float
    std_latency_s:       float
    mean_vram_spike_mb:  float
    std_vram_spike_mb:   float
    n_queries:           int


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — VRAM MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

def measure_vram_mb() -> float:
    """
    Return current GPU VRAM allocation in MB.
    Returns 0.0 gracefully on CPU-only environments.

    Uses torch.cuda.memory_allocated (not memory_reserved) to measure
    only actively used memory, not the cached allocator pool.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(0) / (1024 ** 2)   # bytes → MB


def reset_vram_peak() -> None:
    """Reset the peak VRAM counter so per-experiment peaks are accurate."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — BERTScore EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bertscore(
    generated: str,
    ground_truth: str,
    lang: str = "en",
    model_type: str = "distilbert-base-uncased",
) -> dict[str, float]:
    """
    Compute BERTScore between generated and ground-truth text.

    Uses distilbert-base-uncased by default — fast and reliable.
    For medical text, swap to 'microsoft/BiomedNLP-PubMedBERT-base-uncased'
    for higher clinical relevance.

    Args:
        generated:    The model-generated report string.
        ground_truth: The reference report string.
        lang:         Language code (default: "en").
        model_type:   BERT variant to use for token embeddings.

    Returns:
        dict with keys: precision, recall, f1 (all floats 0–1).
    """
    if not generated or not generated.strip():
        logger.warning("Empty generated text — returning zero BERTScore.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not ground_truth or not ground_truth.strip():
        logger.warning("Empty ground truth — returning zero BERTScore.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        P, R, F1 = bert_score_fn(
            cands=[generated],
            refs=[ground_truth],
            lang=lang,
            model_type=model_type,
            verbose=False,
            device="cpu",   # always CPU — keeps GPU free for LLM
        )
        return {
            "precision": float(P.mean()),
            "recall":    float(R.mean()),
            "f1":        float(F1.mean()),
        }
    except Exception as e:
        logger.error(f"BERTScore computation failed: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — RETRIEVE
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    k: int,
    kb: FAISSKnowledgeBase,
) -> list[dict]:
    """
    Retrieve top-k documents from FAISS knowledge base.

    Args:
        query: The query string (visual finding or clinical question).
        k:     Number of documents to retrieve.
        kb:    Pre-built FAISSKnowledgeBase instance.

    Returns:
        List of dicts with keys: text, source, l2_distance.
    """
    return kb.retrieve(query, top_k=k)


def docs_to_context_string(docs: list[dict]) -> str:
    """Format retrieved doc dicts into a single context string for the LLM."""
    if not docs:
        return "No relevant context retrieved."
    return "\n\n".join(
        f"[Source {i+1}]\n{d['text']}"
        for i, d in enumerate(docs)
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — GENERATE ANSWER
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    context_docs: list[dict],
    vlm: EdgeMedicalVLM,
) -> str:
    """
    Generate a clinical diagnosis using the VLM, grounded in retrieved docs.

    Bypasses the internal FAISS retrieval of EdgeMedicalVLM and injects
    the pre-retrieved context directly — allowing us to control k precisely.

    Args:
        query:        The visual finding or question text.
        context_docs: Pre-retrieved docs (already at the desired k).
        vlm:          Loaded EdgeMedicalVLM instance.

    Returns:
        Generated diagnostic report as a string.
    """
    # Format retrieved docs into context string
    medical_context = docs_to_context_string(context_docs)

    # Build the CoT prompt directly (same template as EdgeMedicalVLM)
    user_msg = vlm.COT_USER_TEMPLATE.format(
        visual_findings=query,
        medical_context=medical_context,
        clinical_history="No additional clinical history.",
    )
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{vlm.COT_SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    return vlm.manager.generate(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — SINGLE EXPERIMENT RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    k: int,
    query_id: int,
    query_text: str,
    ground_truth: str,
    kb: FAISSKnowledgeBase,
    vlm: EdgeMedicalVLM,
) -> ExperimentResult:
    """
    Run a single experiment for a given k and query.

    Steps:
        1. Reset VRAM peak counter
        2. Record initial VRAM
        3. Start timer
        4. Retrieve top-k documents from FAISS
        5. Generate answer using LLM with retrieved context
        6. Stop timer
        7. Record final VRAM
        8. Compute BERTScore
        9. Return structured ExperimentResult

    Args:
        k:            Number of documents to retrieve.
        query_id:     Index of this query (for logging/CSV).
        query_text:   The input finding/question text.
        ground_truth: Reference answer for BERTScore comparison.
        kb:           Loaded FAISSKnowledgeBase.
        vlm:          Loaded EdgeMedicalVLM.

    Returns:
        ExperimentResult dataclass with all metrics.
    """
    logger.info(f"  Running k={k} | Query #{query_id}...")

    # ── VRAM: initial snapshot ─────────────────────────────────────────────
    reset_vram_peak()
    vram_initial_mb = measure_vram_mb()

    # ── Timer: start ───────────────────────────────────────────────────────
    t_start = time.perf_counter()

    # ── Step 1: Retrieve top-k docs ────────────────────────────────────────
    docs = retrieve(query_text, k=k, kb=kb)

    # ── Step 2: Generate answer ────────────────────────────────────────────
    generated = generate_answer(query_text, docs, vlm)

    # ── Timer: stop ────────────────────────────────────────────────────────
    latency_s = time.perf_counter() - t_start

    # ── VRAM: final snapshot ───────────────────────────────────────────────
    vram_final_mb = measure_vram_mb()
    vram_spike_mb = max(0.0, vram_final_mb - vram_initial_mb)

    # ── BERTScore ──────────────────────────────────────────────────────────
    bs = evaluate_bertscore(generated, ground_truth)

    # ── Clean up VRAM ──────────────────────────────────────────────────────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = ExperimentResult(
        k=k,
        query_id=query_id,
        query_text=query_text[:80] + "..." if len(query_text) > 80 else query_text,
        generated=generated[:120] + "..." if len(generated) > 120 else generated,
        ground_truth=ground_truth[:120] + "..." if len(ground_truth) > 120 else ground_truth,
        bertscore_f1=round(bs["f1"],        4),
        bertscore_p= round(bs["precision"], 4),
        bertscore_r= round(bs["recall"],    4),
        latency_s=   round(latency_s,       3),
        vram_initial_mb=round(vram_initial_mb, 2),
        vram_final_mb=  round(vram_final_mb,   2),
        vram_spike_mb=  round(vram_spike_mb,   2),
        n_docs_retrieved=len(docs),
    )

    logger.info(
        f"  k={k:2d} | Q#{query_id} | "
        f"Latency={latency_s:.2f}s | "
        f"VRAM_spike={vram_spike_mb:.1f}MB | "
        f"BERTScore_F1={bs['f1']:.4f}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(
    raw_results: list[ExperimentResult],
    k_values: list[int],
) -> list[AggregatedResult]:
    """
    Aggregate raw per-query results into mean ± std per k value.

    Args:
        raw_results: List of all ExperimentResult objects.
        k_values:    Ordered list of k values tested.

    Returns:
        List of AggregatedResult objects, one per k.
    """
    aggregated = []
    for k in k_values:
        subset = [r for r in raw_results if r.k == k]
        if not subset:
            continue

        bs_scores  = [r.bertscore_f1   for r in subset]
        latencies  = [r.latency_s       for r in subset]
        vram_spikes= [r.vram_spike_mb   for r in subset]

        aggregated.append(AggregatedResult(
            k=k,
            mean_bertscore=    round(float(np.mean(bs_scores)),   4),
            std_bertscore=     round(float(np.std(bs_scores)),    4),
            mean_latency_s=    round(float(np.mean(latencies)),   3),
            std_latency_s=     round(float(np.std(latencies)),    3),
            mean_vram_spike_mb=round(float(np.mean(vram_spikes)), 2),
            std_vram_spike_mb= round(float(np.std(vram_spikes)),  2),
            n_queries=len(subset),
        ))
    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7 — CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_results_csv(
    raw_results: list[ExperimentResult],
    aggregated:  list[AggregatedResult],
) -> None:
    """
    Save both raw and aggregated results to CSV files.

    Output files:
        experiments/figures/rag_ablation_raw.csv
        experiments/figures/rag_ablation_aggregated.csv
    """
    # ── Raw results ────────────────────────────────────────────────────────
    raw_path = OUT_DIR / "rag_ablation_raw.csv"
    raw_fields = [
        "k", "query_id", "bertscore_f1", "bertscore_p", "bertscore_r",
        "latency_s", "vram_initial_mb", "vram_final_mb", "vram_spike_mb",
        "n_docs_retrieved", "query_text", "generated", "ground_truth",
    ]
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        for r in raw_results:
            writer.writerow({field: getattr(r, field) for field in raw_fields})
    logger.info(f"Raw results saved: {raw_path}")

    # ── Aggregated results ─────────────────────────────────────────────────
    agg_path = OUT_DIR / "rag_ablation_aggregated.csv"
    agg_fields = [
        "k", "n_queries",
        "mean_bertscore", "std_bertscore",
        "mean_latency_s", "std_latency_s",
        "mean_vram_spike_mb", "std_vram_spike_mb",
    ]
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        for a in aggregated:
            writer.writerow({field: getattr(a, field) for field in agg_fields})
    logger.info(f"Aggregated results saved: {agg_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 8 — VISUALISATION
# Anti-overlap rules applied throughout:
#   - No floating data labels on plot area (values go in figure footer table)
#   - All annotations use axes-fraction coords, not data coords
#   - Generous figure sizes with constrained_layout=True
#   - Violin means shown as horizontal scatter, not text
#   - Pareto labels offset via adjustText-style manual offsets per quadrant
# ─────────────────────────────────────────────────────────────────────────────

def _panel_tag(ax, tag):
    """Panel label in axes-fraction space — never touches data area."""
    ax.text(-0.10, 1.03, tag, transform=ax.transAxes,
            fontsize=14, fontweight="bold", color="#111111",
            va="bottom", ha="left")


def _hgrid(ax):
    """Horizontal-only subtle grid."""
    ax.yaxis.grid(True, color="#E8E8E8", linestyle="-", linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _add_value_table(fig, aggregated, y_pos=0.01):
    """
    Add a clean data table at the bottom of the figure instead of
    floating labels — eliminates all overlap permanently.
    """
    col_labels = [f"k = {a.k}" for a in aggregated]
    rows = [
        ("BERTScore F1",   [f"{a.mean_bertscore:.3f}±{a.std_bertscore:.3f}"  for a in aggregated]),
        ("Latency (s)",    [f"{a.mean_latency_s:.2f}±{a.std_latency_s:.2f}"  for a in aggregated]),
        ("VRAM (MB)",      [f"{a.mean_vram_spike_mb:.1f}±{a.std_vram_spike_mb:.1f}" for a in aggregated]),
    ]
    cell_text  = [r[1] for r in rows]
    row_labels = [r[0] for r in rows]

    ax_table = fig.add_axes([0.08, y_pos, 0.88, 0.10])
    ax_table.axis("off")
    tbl = ax_table.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_facecolor("#1B3A6B")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == -1:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")


# ── Figure 1 — Normalized multi-metric line chart ─────────────────────────────

def plot_normalized_multiline(
    aggregated: list[AggregatedResult],
    raw_results: list[ExperimentResult],
) -> None:
    """
    All three metrics on one normalized [0,1] scale with shaded CI bands.
    No floating labels — values shown in table below.
    """
    k_vals   = np.array([a.k                  for a in aggregated])
    bs       = np.array([a.mean_bertscore     for a in aggregated])
    bs_std   = np.array([a.std_bertscore      for a in aggregated])
    lat      = np.array([a.mean_latency_s     for a in aggregated])
    lat_std  = np.array([a.std_latency_s      for a in aggregated])
    vram     = np.array([a.mean_vram_spike_mb for a in aggregated])
    vram_std = np.array([a.std_vram_spike_mb  for a in aggregated])

    def norm(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / max(hi - lo, 1e-9)

    bs_n    = norm(bs)
    lat_n   = norm(lat)
    vram_n  = norm(vram)
    bs_sn   = bs_std  / max(bs.max()   - bs.min(),   1e-9)
    lat_sn  = lat_std / max(lat.max()  - lat.min(),  1e-9)
    vram_sn = vram_std/ max(vram.max() - vram.min(), 1e-9)

    fig = plt.figure(figsize=(10, 7), constrained_layout=False)
    ax  = fig.add_axes([0.10, 0.24, 0.86, 0.66])

    specs = [
        (bs_n,   bs_sn,   C_NAVY,    "BERTScore F1 (higher = better)", "o-"),
        (lat_n,  lat_sn,  C_CRIMSON, "Latency (lower = better)",        "s--"),
        (vram_n, vram_sn, C_FOREST,  "VRAM Spike (lower = better)",     "^:"),
    ]
    for yn, en, color, label, style in specs:
        ax.plot(k_vals, yn, style, color=color, lw=2.2, markersize=9,
                markerfacecolor="white", markeredgewidth=2.5,
                markeredgecolor=color, label=label, zorder=5)
        ax.fill_between(k_vals,
                        np.clip(yn - en, 0, 1.05),
                        np.clip(yn + en, 0, 1.05),
                        alpha=0.12, color=color, zorder=2)

    # Best-k vertical line — positioned precisely, label in axes fraction
    best_idx = int(np.argmax(bs_n))
    ax.axvline(k_vals[best_idx], color=C_AMBER, lw=1.8, ls="--", alpha=0.9, zorder=6)
    ax.text(0.02 + best_idx / max(len(k_vals) - 1, 1) * 0.94,
            0.06, f"Best k={k_vals[best_idx]}",
            transform=ax.transAxes,
            color=C_AMBER, fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_AMBER, linewidth=0.9, alpha=1.0))

    ax.set_xticks(k_vals)
    ax.set_xticklabels([f"k = {k}" for k in k_vals])
    ax.set_xlabel("Number of Retrieved Documents (k)")
    ax.set_ylabel("Min-Max Normalized Score (0 = worst, 1 = best)")
    ax.set_ylim(-0.05, 1.20)
    ax.set_title(
        "Figure 1 — Normalized Multi-Metric Ablation over Retrieval Depth k",
        pad=12,
    )
    ax.legend(loc="upper left", framealpha=1.0,
              edgecolor="#CCCCCC", fancybox=False, fontsize=10)
    _panel_tag(ax, "a")
    _hgrid(ax)

    _add_value_table(fig, aggregated, y_pos=0.02)

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"rag_fig1_normalized_multiline.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {OUT_DIR}/rag_fig1_normalized_multiline.png")


# ── Figure 2 — Dual-axis quality–efficiency trade-off ─────────────────────────

def plot_dual_axis_tradeoff(aggregated: list[AggregatedResult]) -> None:
    """
    Left Y = BERTScore (bars), Right Y = Latency (line).
    Labels removed from bars — values in table below.
    """
    k_vals  = [a.k               for a in aggregated]
    bs      = [a.mean_bertscore  for a in aggregated]
    bs_std  = [a.std_bertscore   for a in aggregated]
    lat     = [a.mean_latency_s  for a in aggregated]
    lat_std = [a.std_latency_s   for a in aggregated]

    x = np.arange(len(k_vals))
    w = 0.50

    fig = plt.figure(figsize=(10, 7), constrained_layout=False)
    ax1 = fig.add_axes([0.10, 0.24, 0.78, 0.66])
    ax2 = ax1.twinx()

    # BERTScore bars — clean, no floating labels
    ax1.bar(x, bs, width=w,
            color=[C_NAVY + "DD"] * len(k_vals),
            edgecolor=[C_NAVY] * len(k_vals),
            linewidth=1.0,
            yerr=bs_std, capsize=5,
            error_kw={"elinewidth": 1.5, "ecolor": "#222222", "capthick": 1.5},
            label="BERTScore F1", zorder=4)

    bs_lo = min(bs) - max(bs_std) * 3
    bs_hi = max(bs) + max(bs_std) * 5
    ax1.set_ylim(max(0, bs_lo), bs_hi)

    # Latency line — clean, no floating labels
    ax2.plot(x, lat, "o-",
             color=C_CRIMSON, lw=2.5, markersize=10,
             markerfacecolor="white", markeredgewidth=2.5,
             markeredgecolor=C_CRIMSON, zorder=6, label="Latency (s)")
    ax2.fill_between(
        x,
        [l - e for l, e in zip(lat, lat_std)],
        [l + e for l, e in zip(lat, lat_std)],
        alpha=0.12, color=C_CRIMSON, zorder=3)

    lat_lo = max(0, min(lat) - max(lat_std) * 2)
    lat_hi = max(lat) + max(lat_std) * 5
    ax2.set_ylim(lat_lo, lat_hi)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"k = {k}" for k in k_vals])
    ax1.set_xlabel("Number of Retrieved Documents (k)")
    ax1.set_ylabel("BERTScore F1  (higher = better)", color=C_NAVY, fontweight="bold")
    ax2.set_ylabel("Inference Latency  (seconds)", color=C_CRIMSON, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=C_NAVY)
    ax2.tick_params(axis="y", labelcolor=C_CRIMSON)
    ax1.set_title(
        "Figure 2 — Quality–Speed Trade-off as k Increases",
        pad=12,
    )
    from matplotlib.lines import Line2D
    handles = [
        mpatches.Patch(facecolor=C_NAVY + "DD", edgecolor=C_NAVY,
                       label="BERTScore F1 (bars, left axis)"),
        Line2D([0], [0], color=C_CRIMSON, lw=2.5, marker="o",
               markerfacecolor="white", markeredgewidth=2,
               label="Latency in seconds (line, right axis)"),
    ]
    ax1.legend(handles=handles, loc="upper left",
               framealpha=1.0, edgecolor="#CCCCCC", fancybox=False)
    _panel_tag(ax1, "b")
    _hgrid(ax1)

    _add_value_table(fig, aggregated, y_pos=0.02)

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"rag_fig2_dualaxis_tradeoff.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {OUT_DIR}/rag_fig2_dualaxis_tradeoff.png")


# ── Figure 3 — Pareto frontier scatter ────────────────────────────────────────

def plot_pareto_frontier(
    aggregated: list[AggregatedResult],
    raw_results: list[ExperimentResult],
) -> None:
    """
    X = latency, Y = BERTScore, size = VRAM.
    Labels placed using quadrant-aware offsets to guarantee no overlap.
    """
    k_vals = [a.k                  for a in aggregated]
    bs     = [a.mean_bertscore     for a in aggregated]
    lat    = [a.mean_latency_s     for a in aggregated]
    vram   = [a.mean_vram_spike_mb for a in aggregated]

    v_arr = np.array(vram)
    sizes = (80 + 280 * (v_arr - v_arr.min()) / max(v_arr.max() - v_arr.min(), 1e-9)
             if v_arr.max() > v_arr.min() else np.full(len(vram), 160.0))

    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)

    sorted_pts = sorted(zip(lat, bs))
    ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
            "--", color="#CCCCCC", lw=1.5, zorder=2)

    sc = ax.scatter(lat, bs, s=sizes,
                    c=k_vals, cmap="Blues_r",
                    vmin=min(k_vals) - 2, vmax=max(k_vals) + 2,
                    edgecolors=C_NAVY, linewidths=1.8,
                    zorder=5, alpha=0.92)

    # Quadrant-aware label placement — no overlap
    x_mid = (max(lat) + min(lat)) / 2
    y_mid = (max(bs)  + min(bs))  / 2
    x_rng = max(lat) - min(lat)
    y_rng = max(bs)  - min(bs)

    for k, xl, yb in zip(k_vals, lat, bs):
        # Push label away from the data point based on quadrant
        dx = -x_rng * 0.18 if xl > x_mid else  x_rng * 0.18
        dy =  y_rng * 0.14  if yb > y_mid else -y_rng * 0.18
        ax.annotate(
            f"k = {k}",
            xy=(xl, yb),
            xytext=(xl + dx, yb + dy),
            fontsize=10, fontweight="bold", color=C_NAVY,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_NAVY, linewidth=0.9, alpha=1.0),
            arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
        )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("k (retrieved documents)", fontsize=10, fontweight="bold")

    # VRAM size legend — three reference sizes only, placed in legend
    for v_ref, lbl in [
        (v_arr.min(),  f"Low VRAM ({v_arr.min():.0f} MB)"),
        (v_arr.mean(), f"Mid VRAM ({v_arr.mean():.0f} MB)"),
        (v_arr.max(),  f"High VRAM ({v_arr.max():.0f} MB)"),
    ]:
        sz = 80 + 280 * (v_ref - v_arr.min()) / max(v_arr.max() - v_arr.min(), 1e-9)
        ax.scatter([], [], s=sz, c="none", edgecolors=C_NAVY,
                   linewidths=1.8, label=lbl)

    ax.set_xlabel("Mean Inference Latency (seconds)  —  lower is left (faster)")
    ax.set_ylabel("Mean BERTScore F1  —  higher is up (more accurate)")
    ax.set_title(
        "Figure 3 — Pareto Efficiency Frontier: Quality vs. Speed\n"
        "(marker size = VRAM spike — ideal point is top-left)",
        pad=12,
    )
    ax.legend(loc="lower right", framealpha=1.0,
              edgecolor="#CCCCCC", fancybox=False, fontsize=9)
    _panel_tag(ax, "c")
    _hgrid(ax)

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"rag_fig3_pareto_frontier.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {OUT_DIR}/rag_fig3_pareto_frontier.png")


# ── Figure 4 — Per-query BERTScore distribution (violin + strip) ──────────────

def plot_distribution_violin(raw_results: list[ExperimentResult]) -> None:
    """
    Violin + IQR box + jittered dots. Mean shown as horizontal line inside
    violin — NOT as floating text. Clean, no overlap.
    """
    k_unique = sorted(set(r.k for r in raw_results))
    data_by_k = {k: [r.bertscore_f1 for r in raw_results if r.k == k]
                 for k in k_unique}

    colors = [C_NAVY, C_CRIMSON, C_FOREST, C_AMBER, C_SLATE]

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    for pos, (k, color) in enumerate(zip(k_unique, colors)):
        vals = data_by_k[k]
        if not vals:
            continue

        vp = ax.violinplot([vals], positions=[pos], widths=0.6,
                           showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.22)

        # IQR box as thick vertical line
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.vlines(pos, q1, q3, color=color, linewidth=7, alpha=0.55, zorder=4)
        # Median as white dot
        ax.scatter([pos], [med], color="white", s=55, zorder=6,
                   edgecolors=color, linewidths=2.0)
        # Mean as horizontal line INSIDE violin — no floating text
        mean_val = float(np.mean(vals))
        ax.hlines(mean_val, pos - 0.22, pos + 0.22,
                  color=color, linewidth=2.2, linestyle="-", zorder=5)

        # Jittered individual query dots
        rng    = np.random.default_rng(seed=k)
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(np.array([pos] * len(vals)) + jitter, vals,
                   color=color, s=22, alpha=0.55, zorder=3,
                   edgecolors="white", linewidths=0.4)

    ax.set_xticks(range(len(k_unique)))
    ax.set_xticklabels([f"k = {k}" for k in k_unique])
    ax.set_xlabel("Number of Retrieved Documents (k)")
    ax.set_ylabel("BERTScore F1")
    ax.set_title(
        "Figure 4 — BERTScore F1 Distribution per k\n"
        "(violin = density | thick bar = IQR | white dot = median | "
        "horizontal line = mean | dots = individual queries)",
        pad=12,
    )
    _panel_tag(ax, "d")
    _hgrid(ax)

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"rag_fig4_score_distribution.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {OUT_DIR}/rag_fig4_score_distribution.png")


# ── Figure 5 — Combined 2×2 panel ────────────────────────────────────────────

def plot_combined(aggregated: list[AggregatedResult]) -> None:
    """
    2×2 grid. Each panel is self-contained, well-spaced.
    No floating labels anywhere — values only in legend entries.
    """
    k_vals   = np.array([a.k                  for a in aggregated])
    bs       = np.array([a.mean_bertscore     for a in aggregated])
    bs_std   = np.array([a.std_bertscore      for a in aggregated])
    lat      = np.array([a.mean_latency_s     for a in aggregated])
    lat_std  = np.array([a.std_latency_s      for a in aggregated])
    vram     = np.array([a.mean_vram_spike_mb for a in aggregated])
    vram_std = np.array([a.std_vram_spike_mb  for a in aggregated])

    fig, axes = plt.subplots(2, 2, figsize=(15, 11),
                             constrained_layout=True)
    fig.suptitle(
        "RAG Ablation Study — Effect of Retrieved Document Count k\n"
        "on Answer Quality | Inference Speed | Memory | Score Distribution",
        fontsize=14, fontweight="bold",
    )

    def _line_panel(ax, x, y, yerr, color, ylabel, title, tag, marker="o-"):
        ax.plot(x, y, marker, color=color, lw=2.2, markersize=9,
                markerfacecolor="white", markeredgewidth=2.5, markeredgecolor=color)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.14, color=color)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color,
                    elinewidth=1.5, capsize=5, capthick=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in x])
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)
        ax.set_ylim(max(0, y.min() - yerr.max() * 2.5),
                    y.max() + yerr.max() * 4.5)
        _panel_tag(ax, tag)
        _hgrid(ax)

    # (a) BERTScore
    _line_panel(axes[0, 0], k_vals, bs, bs_std,
                C_NAVY, "BERTScore F1  (higher = better)",
                "(a) Answer Quality vs. k", "a")

    # (b) Latency
    _line_panel(axes[0, 1], k_vals, lat, lat_std,
                C_CRIMSON, "Latency (seconds)  (lower = better)",
                "(b) Inference Latency vs. k", "b", "s--")

    # (c) VRAM
    _line_panel(axes[1, 0], k_vals, vram, vram_std,
                C_FOREST, "VRAM Spike (MB)  (lower = better)",
                "(c) VRAM Spike vs. k", "c", "^:")

    # (d) Pareto scatter — clean, quadrant-aware labels
    ax = axes[1, 1]
    v_arr = vram
    sizes = (80 + 250 * (v_arr - v_arr.min()) / max(v_arr.max() - v_arr.min(), 1e-9)
             if v_arr.max() > v_arr.min() else np.full(len(vram), 140.0))

    sc = ax.scatter(lat, bs, s=sizes,
                    c=k_vals, cmap="Blues_r",
                    vmin=k_vals.min() - 1, vmax=k_vals.max() + 1,
                    edgecolors=C_NAVY, linewidths=1.8, zorder=5, alpha=0.92)
    ax.plot(lat, bs, "--", color="#CCCCCC", lw=1.2, zorder=2)

    x_mid = (lat.max() + lat.min()) / 2
    y_mid = (bs.max()  + bs.min())  / 2
    x_rng = lat.max() - lat.min()
    y_rng = bs.max()  - bs.min()
    for k, xl, yb in zip(k_vals, lat, bs):
        dx = -x_rng * 0.20 if xl > x_mid else x_rng * 0.20
        dy =  y_rng * 0.14  if yb > y_mid else -y_rng * 0.16
        ax.annotate(f"k={k}", xy=(xl, yb), xytext=(xl + dx, yb + dy),
                    fontsize=9, fontweight="bold", color=C_NAVY,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor=C_NAVY, linewidth=0.8, alpha=1.0),
                    arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.7))

    plt.colorbar(sc, ax=ax, shrink=0.8, label="k value")
    ax.set_xlabel("Latency (s)  — faster is left")
    ax.set_ylabel("BERTScore F1  — better is up")
    ax.set_title("(d) Pareto: Quality vs. Speed\n(size = VRAM spike)", pad=10)
    _panel_tag(ax, "d")
    _hgrid(ax)

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"rag_ablation_combined.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {OUT_DIR}/rag_ablation_combined.png")


def run_all_visualizations(
    aggregated: list[AggregatedResult],
    raw_results: list[ExperimentResult],
) -> None:
    """Entry point for all visualization calls."""
    logger.info("\n[Step 4] Generating visualizations...")
    plot_normalized_multiline(aggregated, raw_results)
    plot_dual_axis_tradeoff(aggregated)
    plot_pareto_frontier(aggregated, raw_results)
    plot_distribution_violin(raw_results)
    plot_combined(aggregated)
    logger.info(f"  All figures saved to {OUT_DIR}/")



# ─────────────────────────────────────────────────────────────────────────────
# MODULE 9 — QUERY LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_queries(n: int, dataset: str = "mimic_reports") -> list[dict]:
    """
    Stream N query-answer pairs from a dataset.

    For MIMIC-CXR: query = findings text, ground_truth = impression text.
    For NIH:       query = label description, ground_truth = label string.

    Args:
        n:       Number of queries to load.
        dataset: HuggingFace dataset name registered in data_loader.py.

    Returns:
        List of dicts with keys: query, ground_truth.
    """
    logger.info(f"[Setup] Loading {n} queries from '{dataset}'...")
    loader  = StreamingDatasetManager()
    samples = loader.get_sample_batch(dataset, n=n * 2)  # oversample for filtering

    queries = []
    for s in samples:
        text        = s.get("text", "").strip()
        report      = s.get("report", "").strip()
        labels      = ", ".join(s.get("labels", ["Unknown"]))

        if dataset == "mimic_reports" and text and report:
            queries.append({
                "query":        f"Clinical findings: {text}",
                "ground_truth": f"IMPRESSION: {report}",
            })
        else:
            queries.append({
                "query":        (
                    f"Chest X-ray analysis. Label: {labels}. "
                    "Analyse for consolidation, effusion, cardiomegaly, "
                    "pneumothorax, and interstitial patterns."
                ),
                "ground_truth": labels,
            })

        if len(queries) >= n:
            break

    logger.info(f"[Setup] Loaded {len(queries)} queries.")
    return queries


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── CLI ───────────────────────────────────────────────────────────────
    ap = argparse.ArgumentParser(
        description="RAG k-Ablation Study — Effect of retrieved docs on quality/speed/VRAM"
    )
    ap.add_argument("--k",          nargs="+", type=int,
                    default=[1, 3, 5, 10],
                    help="List of k values to test (default: 1 3 5 10)")
    ap.add_argument("--n-queries",  type=int, default=10,
                    help="Number of queries per k value (default: 10)")
    ap.add_argument("--dataset",    type=str, default="mimic_reports",
                    choices=["mimic_reports", "nih", "chexpert"],
                    help="Dataset to sample queries from (default: mimic_reports)")
    ap.add_argument("--cpu-only",   action="store_true",
                    help="Disable GPU — run entirely on CPU")
    args = ap.parse_args()

    k_values  = sorted(args.k)
    n_queries = args.n_queries
    dataset   = args.dataset

    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("CPU-only mode enabled.")

    print("\n" + "=" * 65)
    print("  RAG k-Ablation Study")
    print(f"  k values   : {k_values}")
    print(f"  Queries    : {n_queries} per k")
    print(f"  Dataset    : {dataset}")
    print(f"  GPU        : {'Yes' if torch.cuda.is_available() else 'No (CPU mode)'}")
    print("=" * 65 + "\n")

    # ── Step 1: Load queries ──────────────────────────────────────────────
    queries = load_queries(n_queries, dataset)
    if not queries:
        logger.error("No queries loaded. Exiting.")
        return

    # ── Step 2: Load FAISS index ──────────────────────────────────────────
    logger.info("[Setup] Loading FAISS knowledge base...")
    faiss_cfg = FAISSConfig()
    kb = FAISSKnowledgeBase(faiss_cfg)
    index_path = Path(faiss_cfg.index_path)

    if not index_path.exists():
        logger.error(
            f"FAISS index not found at {index_path}.\n"
            "Run first: python -m src.pipeline --phase index"
        )
        return

    kb.load_index()
    logger.info(f"[Setup] FAISS index loaded: {kb.index.ntotal} vectors.")

    # ── Step 3: Load VLM ──────────────────────────────────────────────────
    logger.info("[Setup] Loading VLM (QLoRA + LoRA adapters)...")
    vlm = EdgeMedicalVLM()
    logger.info("[Setup] VLM ready.")

    # ── Step 4: Run experiments ───────────────────────────────────────────
    logger.info(f"\n[Experiment] Starting ablation over k={k_values}...")
    raw_results: list[ExperimentResult] = []

    # tqdm outer bar: k values
    for k in tqdm(k_values, desc="k values", unit="k", colour="blue"):
        logger.info(f"\n{'─'*50}")
        logger.info(f"Running experiment for k={k}...")
        logger.info(f"{'─'*50}")

        # tqdm inner bar: queries per k
        for qid, q in enumerate(
            tqdm(queries, desc=f"  k={k} queries", unit="query",
                 leave=False, colour="green"),
            start=1,
        ):
            result = run_experiment(
                k=k,
                query_id=qid,
                query_text=q["query"],
                ground_truth=q["ground_truth"],
                kb=kb,
                vlm=vlm,
            )
            raw_results.append(result)

        # Summary for this k
        k_results = [r for r in raw_results if r.k == k]
        logger.info(
            f"\n  k={k} SUMMARY ({len(k_results)} queries):\n"
            f"    Mean BERTScore F1 : {np.mean([r.bertscore_f1 for r in k_results]):.4f}\n"
            f"    Mean Latency      : {np.mean([r.latency_s    for r in k_results]):.2f}s\n"
            f"    Mean VRAM Spike   : {np.mean([r.vram_spike_mb for r in k_results]):.1f}MB"
        )

    # ── Step 5: Aggregate ─────────────────────────────────────────────────
    logger.info("\n[Results] Aggregating results...")
    aggregated = aggregate_results(raw_results, k_values)

    # ── Step 6: Save CSV ──────────────────────────────────────────────────
    logger.info("[Results] Saving CSV files...")
    save_results_csv(raw_results, aggregated)

    # ── Step 7: Visualize ─────────────────────────────────────────────────
    run_all_visualizations(aggregated, raw_results)

    # ── Step 8: Print final table ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  ABLATION STUDY RESULTS — {n_queries} queries per k")
    print("=" * 65)
    print(f"  {'k':>4}  {'BERTScore F1':>14}  {'Latency (s)':>13}  {'VRAM Spike (MB)':>16}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*13}  {'─'*16}")
    for a in aggregated:
        print(
            f"  {a.k:>4}  "
            f"{a.mean_bertscore:.4f} ±{a.std_bertscore:.4f}  "
            f"{a.mean_latency_s:.2f}s ±{a.std_latency_s:.2f}  "
            f"{a.mean_vram_spike_mb:.1f} ±{a.std_vram_spike_mb:.1f} MB"
        )
    print("=" * 65)
    print(f"\n  Figures  → {OUT_DIR}/")
    print(f"  Raw CSV  → {OUT_DIR}/rag_ablation_raw.csv")
    print(f"  Agg CSV  → {OUT_DIR}/rag_ablation_aggregated.csv\n")


if __name__ == "__main__":
    main()