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
        7. Record peak VRAM
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

    # ── VRAM: final snapshot and peak tracking ──────────────────────────────
    vram_final_mb = measure_vram_mb()
    
    # Get the actual peak memory hit during generation
    if torch.cuda.is_available():
        vram_peak_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    else:
        vram_peak_mb = 0.0
        
    vram_spike_mb = max(0.0, vram_peak_mb - vram_initial_mb)

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
# ─────────────────────────────────────────────────────────────────────────────

def _add_value_labels(ax, x, y, fmt="{:.3f}", pad=0.01, fontsize=9):
    """Add value labels above each data point with a white background box."""
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.04
    for xi, yi in zip(x, y):
        ax.text(
            xi, yi + offset, fmt.format(yi),
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", color="#111111",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#CCCCCC", linewidth=0.7, alpha=1.0),
        )


def plot_individual(aggregated: list[AggregatedResult]) -> None:
    """Save three individual plots: k vs BERTScore, Latency, VRAM."""
    k_vals = [a.k               for a in aggregated]
    bs     = [a.mean_bertscore  for a in aggregated]
    bs_std = [a.std_bertscore   for a in aggregated]
    lat    = [a.mean_latency_s  for a in aggregated]
    lat_std= [a.std_latency_s   for a in aggregated]
    vram   = [a.mean_vram_spike_mb   for a in aggregated]
    vram_std=[a.std_vram_spike_mb    for a in aggregated]

    configs = [
        {
            "y": bs, "yerr": bs_std,
            "color": C_NAVY,
            "ylabel": "BERTScore F1 (↑ better)",
            "title": "RAG Ablation — k vs. BERTScore F1\n"
                     "(higher = more clinically accurate generation)",
            "fname": "rag_ablation_k_vs_bertscore",
            "fmt": "{:.3f}",
        },
        {
            "y": lat, "yerr": lat_std,
            "color": C_CRIMSON,
            "ylabel": "Inference Latency (seconds) (↓ better)",
            "title": "RAG Ablation — k vs. Inference Latency\n"
                     "(lower = faster generation)",
            "fname": "rag_ablation_k_vs_latency",
            "fmt": "{:.2f}s",
        },
        {
            "y": vram, "yerr": vram_std,
            "color": C_FOREST,
            "ylabel": "VRAM Spike (MB) (↓ better)",
            "title": "RAG Ablation — k vs. VRAM Spike\n"
                     "(lower = more memory-efficient)",
            "fname": "rag_ablation_k_vs_vram",
            "fmt": "{:.1f}MB",
        },
    ]

    for cfg in configs:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Line + shaded error band
        ax.plot(k_vals, cfg["y"], "o-",
                color=cfg["color"], lw=2.2, markersize=9,
                markerfacecolor="white", markeredgewidth=2.5,
                markeredgecolor=cfg["color"], zorder=5)
        ax.fill_between(
            k_vals,
            [y - e for y, e in zip(cfg["y"], cfg["yerr"])],
            [y + e for y, e in zip(cfg["y"], cfg["yerr"])],
            alpha=0.15, color=cfg["color"], zorder=2,
        )

        # Error bar caps
        ax.errorbar(k_vals, cfg["y"], yerr=cfg["yerr"],
                    fmt="none", ecolor=cfg["color"],
                    elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)

        ax.set_ylim(
            max(0, min(cfg["y"]) - max(cfg["yerr"]) * 2),
            max(cfg["y"]) + max(cfg["yerr"]) * 3.5,
        )
        _add_value_labels(ax, k_vals, cfg["y"], fmt=cfg["fmt"])

        ax.set_xticks(k_vals)
        ax.set_xticklabels([f"k={k}" for k in k_vals])
        ax.set_xlabel("Number of Retrieved Documents (k)")
        ax.set_ylabel(cfg["ylabel"])
        ax.set_title(cfg["title"], pad=14)

        fig.patch.set_facecolor("white")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"{cfg['fname']}.{ext}",
                        dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"  Saved: {OUT_DIR / cfg['fname']}.png")


def plot_combined(aggregated: list[AggregatedResult]) -> None:
    """
    Save one combined 3-panel publication figure with all three metrics.
    """
    k_vals  = [a.k                  for a in aggregated]
    bs      = [a.mean_bertscore     for a in aggregated]
    bs_std  = [a.std_bertscore      for a in aggregated]
    lat     = [a.mean_latency_s     for a in aggregated]
    lat_std = [a.std_latency_s      for a in aggregated]
    vram    = [a.mean_vram_spike_mb for a in aggregated]
    vram_std= [a.std_vram_spike_mb  for a in aggregated]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        "RAG Ablation Study — Effect of Retrieved Document Count (k)\n"
        "on Clinical Report Quality, Inference Speed, and Memory Usage",
        fontsize=14, fontweight="bold", y=1.02,
    )

    panels = [
        (axes[0], bs,   bs_std,   C_NAVY,
         "BERTScore F1", "(a) Answer Quality (↑ better)", "{:.3f}"),
        (axes[1], lat,  lat_std,  C_CRIMSON,
         "Latency (s)",  "(b) Inference Latency (↓ better)", "{:.2f}s"),
        (axes[2], vram, vram_std, C_FOREST,
         "VRAM Spike (MB)", "(c) VRAM Spike (↓ better)", "{:.1f}MB"),
    ]

    for ax, y, yerr, color, ylabel, title, fmt in panels:
        ax.plot(k_vals, y, "o-", color=color, lw=2.2, markersize=9,
                markerfacecolor="white", markeredgewidth=2.5,
                markeredgecolor=color, zorder=5)
        ax.fill_between(
            k_vals,
            [yi - e for yi, e in zip(y, yerr)],
            [yi + e for yi, e in zip(y, yerr)],
            alpha=0.15, color=color, zorder=2,
        )
        ax.errorbar(k_vals, y, yerr=yerr,
                    fmt="none", ecolor=color,
                    elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)

        ax.set_ylim(
            max(0, min(y) - max(yerr) * 2),
            max(y) + max(yerr) * 4,
        )
        _add_value_labels(ax, k_vals, y, fmt=fmt, fontsize=8)

        ax.set_xticks(k_vals)
        ax.set_xticklabels([f"k={k}" for k in k_vals], fontsize=10)
        ax.set_xlabel("Retrieved Documents (k)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=12)
        ax.yaxis.grid(True, color="#DDDDDD", ls="-", lw=0.7)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"rag_ablation_combined.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {OUT_DIR}/rag_ablation_combined.png")


def run_all_visualizations(aggregated: list[AggregatedResult]) -> None:
    """Entry point for all visualization calls."""
    logger.info("\n[Step 4] Generating visualizations...")
    plot_individual(aggregated)
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
    run_all_visualizations(aggregated)

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