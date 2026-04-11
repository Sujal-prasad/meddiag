"""
run_green_eval.py — GREEN Score Evaluation Pipeline
====================================================
Evaluates the VLM+RAG+CoT pipeline using three local Ollama LLM judges
following the GREEN (Generative Radiology Report Evaluation and Error
Notation) scoring framework.

Prerequisites:
    1. Install Ollama:  https://ollama.ai
    2. Pull models:
         ollama pull mistral
         ollama pull qwen2.5:3b
         ollama pull gemma2
    3. Make sure Ollama is running:
         ollama serve

Usage:
    python -m green_eval.run_green_eval
    python -m green_eval.run_green_eval --n 10 --dataset mimic_reports
    python -m green_eval.run_green_eval --n 15 --skip-generate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import requests
import seaborn as sns

# ── Project path ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import StreamingDatasetManager
from src.pipeline    import EdgeMedicalVLM, FAISSConfig

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("GREEN")

# ── Paths ─────────────────────────────────────────────────────────────────────
EVAL_DIR   = Path(__file__).parent
GEN_DIR    = EVAL_DIR / "reports" / "generated"
REF_DIR    = EVAL_DIR / "reports" / "reference"
RESULT_DIR = EVAL_DIR / "results"
FIG_DIR    = RESULT_DIR / "figures"

for d in [GEN_DIR, REF_DIR, RESULT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
JUDGES = [
    {"name": "Mistral-7B",   "model": "mistral"},
    {"name": "Qwen2.5-3B",   "model": "qwen2.5:3b"},
    {"name": "Gemma2-9B",    "model": "gemma2"},
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — GREEN SCORE SYSTEM PROMPT
# Based on: Ostmeier et al. (2024) GREEN: Generative Radiology Report
# Evaluation and Error Notation. arXiv:2405.03595
# ─────────────────────────────────────────────────────────────────────────────

GREEN_SYSTEM_PROMPT = """You are an expert radiologist evaluating an AI-generated chest X-ray radiology report.

Your task is to score the GENERATED REPORT against the REFERENCE REPORT using the GREEN scoring framework.

GREEN EVALUATION CRITERIA:
==========================
Score each of the following error categories (count occurrences):

1. CLINICALLY_SIGNIFICANT_ERRORS (CSE) — Errors that could cause patient harm:
   - Wrong diagnosis (e.g., reporting NORMAL when clearly ABNORMAL)
   - Missing critical finding (e.g., missed pneumothorax, large effusion)
   - Incorrect anatomical location of a finding
   - Wrong severity assessment (mild vs severe)

2. CLINICALLY_INSIGNIFICANT_ERRORS (CIE) — Errors that are minor:
   - Slightly different wording for same finding
   - Missing minor/incidental finding
   - Style or phrasing differences
   - Extra normal findings mentioned

3. MATCHED_FINDINGS (MF) — Correct findings present in both reports

4. MISSING_FINDINGS (MSF) — Findings in reference NOT mentioned in generated

SCORING FORMULA:
  GREEN = (MF - CSE - 0.5*CIE) / max(total_reference_findings, 1)
  Clipped to range [0, 1]

OUTPUT FORMAT — You MUST respond with ONLY valid JSON, nothing else:
{
  "clinically_significant_errors": <integer>,
  "clinically_insignificant_errors": <integer>,
  "matched_findings": <integer>,
  "missing_findings": <integer>,
  "total_reference_findings": <integer>,
  "green_score": <float between 0 and 1>,
  "reasoning": "<one sentence explaining your score>",
  "critical_errors_found": ["<list any CSE found, empty list if none>"],
  "overall_quality": "<EXCELLENT|GOOD|ACCEPTABLE|POOR>"
}

IMPORTANT:
- Be strict but fair. Focus on clinical significance, not wording.
- A score of 1.0 = perfect match, 0.0 = completely wrong or harmful.
- Respond ONLY with the JSON object. No preamble, no explanation outside JSON."""


GREEN_USER_TEMPLATE = """REFERENCE REPORT (ground truth from radiologist):
{reference}

GENERATED REPORT (from AI system being evaluated):
{generated}

Evaluate the generated report against the reference using GREEN criteria.
Return ONLY valid JSON."""


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — GENERATE REPORTS FROM VLM PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_reports(n: int = 10, dataset: str = "mimic_reports") -> list[dict]:
    """
    Stream N samples from the dataset, run through VLM+RAG+CoT pipeline,
    save generated and reference reports as JSON files.

    Returns list of report pairs.
    """
    logger.info(f"[Step 1] Generating reports for {n} samples from '{dataset}'")
    logger.info("Loading VLM pipeline (this takes ~30 seconds)...")

    vlm = EdgeMedicalVLM()

    logger.info(f"Streaming {n} samples from {dataset}...")
    loader  = StreamingDatasetManager()
    samples = loader.get_sample_batch(dataset, n=n)

    # Filter to samples that have actual text content
    valid = [
        s for s in samples
        if s.get("text") or s.get("report") or s.get("image_pil")
    ][:n]

    if not valid:
        raise RuntimeError(f"No valid samples from {dataset}.")

    logger.info(f"Got {len(valid)} valid samples. Generating reports...")

    report_pairs = []
    for i, sample in enumerate(valid):
        idx = f"{i+1:03d}"
        logger.info(f"  [{i+1}/{len(valid)}] Processing sample {idx}...")

        # ── Build visual finding ───────────────────────────────────────────
        findings_text = sample.get("text", "").strip()
        labels        = ", ".join(sample.get("labels", ["Unknown"]))

        if findings_text:
            visual_finding = f"Clinical findings from radiology report: {findings_text}"
        else:
            visual_finding = (
                f"Chest X-ray analysis. Annotated label: {labels}. "
                f"Analyse for consolidation, pleural effusion, cardiomegaly, "
                f"pneumothorax, and interstitial patterns."
            )

        # ── Reference report ───────────────────────────────────────────────
        impression     = sample.get("report", "").strip()
        reference_text = (
            f"FINDINGS: {findings_text}\nIMPRESSION: {impression}"
            if impression else
            f"LABEL: {labels}\nNo detailed report available."
        )

        # ── Generate with pipeline ─────────────────────────────────────────
        try:
            t0        = time.perf_counter()
            generated = vlm.generate_diagnosis(
                visual_findings=visual_finding,
                clinical_history="No additional clinical history.",
            )
            latency = time.perf_counter() - t0
            logger.info(f"    Generated in {latency:.1f}s")
        except Exception as e:
            logger.warning(f"    Generation failed: {e}")
            generated = "ERROR: Generation failed."

        # ── Save to JSON files ─────────────────────────────────────────────
        gen_record = {
            "id":       idx,
            "dataset":  dataset,
            "labels":   sample.get("labels", []),
            "report":   generated,
            "latency_s": round(latency, 2) if 'latency' in dir() else 0,
        }
        ref_record = {
            "id":      idx,
            "dataset": dataset,
            "labels":  sample.get("labels", []),
            "report":  reference_text,
        }

        (GEN_DIR / f"generated_{idx}.json").write_text(
            json.dumps(gen_record, indent=2)
        )
        (REF_DIR / f"reference_{idx}.json").write_text(
            json.dumps(ref_record, indent=2)
        )

        report_pairs.append({"id": idx, "generated": generated,
                              "reference": reference_text,
                              "labels": sample.get("labels", [])})

    logger.info(f"[Step 1] Done. {len(report_pairs)} report pairs saved.")
    return report_pairs


def load_saved_reports() -> list[dict]:
    """Load previously generated report pairs from disk."""
    pairs = []
    gen_files = sorted(GEN_DIR.glob("generated_*.json"))
    for gf in gen_files:
        idx = gf.stem.split("_")[1]
        rf  = REF_DIR / f"reference_{idx}.json"
        if rf.exists():
            gen = json.loads(gf.read_text())
            ref = json.loads(rf.read_text())
            pairs.append({
                "id":        idx,
                "generated": gen["report"],
                "reference": ref["report"],
                "labels":    gen.get("labels", []),
            })
    logger.info(f"Loaded {len(pairs)} saved report pairs.")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LLM JUDGING via Ollama
# ─────────────────────────────────────────────────────────────────────────────

def check_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def trim_text(text: str, max_chars: int = 800) -> str:
    """Trim text to max_chars to avoid context overflow in smaller models."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated for brevity]"


def query_judge(judge: dict, generated: str, reference: str,
                max_retries: int = 2) -> dict:
    """
    Send a report pair to one Ollama judge and parse the GREEN score response.
    Trims prompts and retries on failure.
    """
    # Trim to prevent context overflow — main cause of 500 errors
    generated_trim = trim_text(generated, max_chars=600)
    reference_trim = trim_text(reference, max_chars=600)

    prompt = GREEN_USER_TEMPLATE.format(
        reference=reference_trim,
        generated=generated_trim,
    )

    payload = {
        "model":  judge["model"],
        "prompt": prompt,
        "system": GREEN_SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature":   0.1,
            "top_p":         0.9,
            "num_predict":   400,
            "num_ctx":       2048,   # explicit context window limit
        },
    }

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"    {judge['name']} — attempt {attempt}/{max_retries}")
            r = requests.post(OLLAMA_URL, json=payload, timeout=180)
            r.raise_for_status()
            raw = r.json().get("response", "").strip()

            # Extract JSON from response
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]

            result = json.loads(raw)

            # Compute GREEN score if missing
            if "green_score" not in result:
                mf  = result.get("matched_findings", 0)
                cse = result.get("clinically_significant_errors", 0)
                cie = result.get("clinically_insignificant_errors", 0)
                trf = max(result.get("total_reference_findings", 1), 1)
                result["green_score"] = max(0.0, min(1.0,
                    (mf - cse - 0.5 * cie) / trf
                ))

            result["judge"]  = judge["name"]
            result["model"]  = judge["model"]
            result["status"] = "ok"
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"    {judge['name']} invalid JSON on attempt {attempt}: {e}")
            if attempt == max_retries:
                return {"judge": judge["name"], "model": judge["model"],
                        "status": "parse_error", "green_score": None,
                        "reasoning": "Could not parse judge response."}
            time.sleep(3)

        except requests.exceptions.HTTPError as e:
            logger.warning(f"    {judge['name']} HTTP {e.response.status_code} "
                           f"on attempt {attempt} — waiting 10s before retry...")
            if attempt == max_retries:
                return {"judge": judge["name"], "model": judge["model"],
                        "status": "error", "green_score": None,
                        "reasoning": str(e)}
            time.sleep(10)   # give Ollama time to free memory

        except Exception as e:
            logger.warning(f"    {judge['name']} failed on attempt {attempt}: {e}")
            if attempt == max_retries:
                return {"judge": judge["name"], "model": judge["model"],
                        "status": "error", "green_score": None,
                        "reasoning": str(e)}
            time.sleep(5)

    return {"judge": judge["name"], "model": judge["model"],
            "status": "error", "green_score": None,
            "reasoning": "All retries exhausted."}


def judge_all_reports(report_pairs: list[dict]) -> list[dict]:
    """
    For each report pair, query all 3 judges in parallel.
    Returns the full results list.
    """
    if not check_ollama():
        raise RuntimeError(
            "Ollama is not running! Start it with: ollama serve\n"
            "Then pull models:\n"
            "  ollama pull mistral\n"
            "  ollama pull qwen2.5:3b\n"
            "  ollama pull gemma2"
        )

    logger.info(f"\n[Step 2] Judging {len(report_pairs)} reports with "
                f"{len(JUDGES)} LLM judges...")

    all_results = []

    for i, pair in enumerate(report_pairs):
        logger.info(f"  Report {pair['id']} ({i+1}/{len(report_pairs)}) | "
                    f"Labels: {pair['labels']}")
        report_result = {
            "id":        pair["id"],
            "labels":    pair["labels"],
            "generated": pair["generated"][:300] + "...",
            "reference": pair["reference"][:300] + "...",
            "judge_scores": [],
        }

        # Run judges SEQUENTIALLY — prevents Ollama OOM / 500 errors
        # Each model must fully unload before the next one loads
        for judge in JUDGES:
            logger.info(f"    Querying {judge['name']}...")
            result = query_judge(judge, pair["generated"], pair["reference"])
            score  = result.get("green_score")
            status = result.get("status", "unknown")
            if score is not None:
                logger.info(f"    {judge['name']:15s} → GREEN={score:.3f}")
            else:
                logger.warning(f"    {judge['name']:15s} → FAILED ({status})")
            report_result["judge_scores"].append(result)
            time.sleep(3)   # let Ollama fully unload before next model

        # Aggregate
        valid_scores = [
            r["green_score"]
            for r in report_result["judge_scores"]
            if r.get("green_score") is not None
        ]
        report_result["aggregated_green"] = (
            round(float(np.mean(valid_scores)), 4)
            if valid_scores else None
        )
        report_result["score_std"] = (
            round(float(np.std(valid_scores)), 4)
            if len(valid_scores) > 1 else 0.0
        )

        logger.info(f"    AGGREGATED GREEN = {report_result['aggregated_green']:.4f} "
                    f"(±{report_result['score_std']:.4f})")

        all_results.append(report_result)

    # Save results
    out = RESULT_DIR / "scores.json"
    out.write_text(json.dumps(all_results, indent=2))
    logger.info(f"\n[Step 2] Results saved to {out}")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — AGGREGATION & STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistics(results: list[dict]) -> dict:
    """Compute summary statistics across all report evaluations."""
    agg_scores = [r["aggregated_green"] for r in results
                  if r.get("aggregated_green") is not None]

    judge_scores = {j["name"]: [] for j in JUDGES}
    for r in results:
        for js in r.get("judge_scores", []):
            if js.get("green_score") is not None:
                judge_scores[js["judge"]].append(js["green_score"])

    stats = {
        "n_reports":        len(results),
        "mean_green":       round(float(np.mean(agg_scores)), 4),
        "std_green":        round(float(np.std(agg_scores)), 4),
        "median_green":     round(float(np.median(agg_scores)), 4),
        "min_green":        round(float(np.min(agg_scores)), 4),
        "max_green":        round(float(np.max(agg_scores)), 4),
        "per_judge": {
            name: {
                "mean": round(float(np.mean(scores)), 4),
                "std":  round(float(np.std(scores)), 4),
            }
            for name, scores in judge_scores.items() if scores
        },
    }

    logger.info("\n[Step 3] SUMMARY STATISTICS")
    logger.info(f"  Reports evaluated : {stats['n_reports']}")
    logger.info(f"  Mean GREEN score  : {stats['mean_green']:.4f}")
    logger.info(f"  Std GREEN score   : {stats['std_green']:.4f}")
    logger.info(f"  Range             : [{stats['min_green']:.4f}, {stats['max_green']:.4f}]")
    for name, s in stats["per_judge"].items():
        logger.info(f"  {name:15s}  : {s['mean']:.4f} ± {s['std']:.4f}")

    (RESULT_DIR / "summary_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

# Design system — consistent with evaluate.py
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.titlepad":    14,
    "axes.labelsize":   12,
    "axes.labelweight": "bold",
    "axes.labelpad":    8,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#333333",
    "axes.linewidth":   1.2,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "axes.axisbelow":   True,
    "grid.color":       "#E5E5E5",
    "grid.linestyle":   "-",
    "grid.linewidth":   0.6,
    "figure.dpi":       120,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "text.color":       "#111111",
})

C_NAVY    = "#1B3A6B"
C_CRIMSON = "#8B1A1A"
C_FOREST  = "#1E4D2B"
C_AMBER   = "#7E5109"
JUDGE_COLORS = {
    "Mistral-7B":  C_NAVY,
    "Qwen2.5-3B":  C_CRIMSON,
    "Gemma2-9B":   C_FOREST,
}


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        p = FIG_DIR / f"{name}.{ext}"
        fig.savefig(str(p), dpi=300, bbox_inches="tight", facecolor="white")
    logger.info(f"  Saved: {FIG_DIR / name}.png")



def _hgrid(ax):
    ax.yaxis.grid(True, color="#E5E5E5", ls="-", lw=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def plot_aggregated_scores(results: list[dict]) -> None:
    """Bar chart — one bar per report, colored by quality tier."""
    ids    = [r["id"]                               for r in results]
    scores = [r.get("aggregated_green", 0) or 0     for r in results]
    stds   = [r.get("score_std", 0) or 0            for r in results]
    labels = [", ".join(r.get("labels", []))[:18]   for r in results]

    colors = [C_FOREST if s >= 0.7 else C_AMBER if s >= 0.4 else C_CRIMSON
              for s in scores]

    fig, ax = plt.subplots(figsize=(max(10, len(ids) * 0.9), 6.5),
                            constrained_layout=True)
    x = np.arange(len(ids))
    ax.bar(x, scores, color=colors, edgecolor="#111111",
           linewidth=0.8, width=0.62,
           yerr=stds, capsize=4,
           error_kw={"elinewidth": 1.5, "ecolor": "#555555"})

    ax.set_ylim(0, 1.28)
    # Labels above bars — axes-relative pad so they never touch error bars
    yrange = 1.28
    for xi, s, e in zip(x, scores, stds):
        ax.text(xi, s + e + yrange * 0.05, f"{s:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#0D0D0D",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#BBBBBB", linewidth=0.7, alpha=1.0))

    ax.axhline(float(np.mean(scores)), color=C_NAVY, ls="--", lw=1.8)
    ax.text(0.98, float(np.mean(scores))/1.28 + 0.02,
            f"Mean = {np.mean(scores):.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            color=C_NAVY, fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=C_NAVY, linewidth=0.8, alpha=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i}\n{l}" for i, l in zip(ids, labels)],
                       fontsize=8.5, rotation=20, ha="right")
    ax.set_ylabel("Aggregated GREEN Score (0 – 1)")
    ax.set_xlabel("Report ID  (label abbreviation)")
    ax.set_title("Figure 1 — Aggregated GREEN Scores per Report\n"
                 "(mean of 3 judges: Mistral-7B | Qwen2.5-3B | Gemma2-9B)", pad=12)
    ax.legend(handles=[
        mpatches.Patch(color=C_FOREST,  label="HIGH  (>= 0.70)"),
        mpatches.Patch(color=C_AMBER,   label="MED   (0.40 – 0.70)"),
        mpatches.Patch(color=C_CRIMSON, label="LOW   (< 0.40)"),
    ], loc="upper left", framealpha=1.0, edgecolor="#CCCCCC", fancybox=False)
    _hgrid(ax)

    save_fig(fig, "fig1_aggregated_green_scores")
    plt.show()


def plot_judge_heatmap(results: list[dict]) -> None:
    """Heatmap: rows = reports, cols = judges, cell = GREEN score."""
    judge_names = [j["name"] for j in JUDGES]
    report_ids  = [r["id"]   for r in results]

    matrix = np.full((len(results), len(JUDGES)), np.nan)
    for i, r in enumerate(results):
        for js in r.get("judge_scores", []):
            j_idx = next((k for k, j in enumerate(JUDGES)
                          if j["name"] == js["judge"]), None)
            if j_idx is not None and js.get("green_score") is not None:
                matrix[i, j_idx] = js["green_score"]

    fig, ax = plt.subplots(figsize=(7, max(5, len(results) * 0.58)),
                            constrained_layout=True)
    sns.heatmap(
        matrix, ax=ax,
        annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.6, linecolor="#EEEEEE",
        annot_kws={"size": 10, "weight": "bold", "color": "#111111"},
        cbar_kws={"label": "GREEN Score", "shrink": 0.75, "pad": 0.04},
        mask=np.isnan(matrix),
    )
    ax.set_xticklabels(judge_names, fontsize=10, fontweight="bold", rotation=0)
    ax.set_yticklabels([f"#{i}" for i in report_ids], fontsize=9, rotation=0)
    ax.set_xlabel("LLM Judge", labelpad=10)
    ax.set_ylabel("Report ID", labelpad=10)
    ax.set_title("Figure 2 — Per-Judge GREEN Scores\n"
                 "(rows = reports | cols = judges)", pad=12)
    fig.patch.set_facecolor("white")

    save_fig(fig, "fig2_judge_heatmap")
    plt.show()


def plot_judge_agreement(results: list[dict]) -> None:
    """3×3 agreement matrix — diagonal = histogram, off-diagonal = scatter."""
    judge_names = [j["name"] for j in JUDGES]
    scores_by_judge = {name: [] for name in judge_names}

    for r in results:
        row = {js["judge"]: js["green_score"]
               for js in r.get("judge_scores", [])
               if js.get("green_score") is not None}
        if len(row) == len(JUDGES):
            for name in judge_names:
                scores_by_judge[name].append(row[name])

    if not scores_by_judge[judge_names[0]]:
        logger.warning("Not enough data for agreement plot.")
        return

    n   = len(judge_names)
    fig, axes = plt.subplots(n, n, figsize=(11, 10),
                              gridspec_kw={"wspace": 0.38, "hspace": 0.38})
    fig.suptitle("Figure 3 — Inter-Judge Agreement Matrix",
                 fontsize=13, fontweight="bold", y=0.98)

    for i, name_i in enumerate(judge_names):
        for j, name_j in enumerate(judge_names):
            ax = axes[i, j]
            xi = np.array(scores_by_judge[name_i])
            xj = np.array(scores_by_judge[name_j])
            color = list(JUDGE_COLORS.values())[i]

            if i == j:
                ax.hist(xi, bins=min(8, len(xi)), color=color,
                        edgecolor="white", linewidth=0.8, alpha=0.9)
                ax.set_title(name_i, fontsize=10, fontweight="bold",
                             color=color, pad=6)
                ax.set_xlabel("Score", fontsize=8)
                ax.set_ylabel("Count", fontsize=8)
            else:
                ax.scatter(xj, xi, color=color, edgecolors="#333333",
                           linewidths=0.4, s=45, alpha=0.8, zorder=3)
                ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1.0)
                corr = float(np.corrcoef(xi, xj)[0, 1]) if len(xi) > 1 else 0.0
                # Correlation label — always top-left corner in axes space
                ax.text(0.05, 0.88, f"r = {corr:.2f}",
                        transform=ax.transAxes, fontsize=8.5,
                        fontweight="bold", color="#111111",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor="#CCCCCC", linewidth=0.7, alpha=1.0))
                ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
            if i == n - 1: ax.set_xlabel(name_j, fontsize=9)
            if j == 0:     ax.set_ylabel(name_i, fontsize=9)

    fig.patch.set_facecolor("white")
    save_fig(fig, "fig3_judge_agreement")
    plt.show()


def plot_error_breakdown(results: list[dict]) -> None:
    """Stacked bars: Matched Findings | CIE | CSE per report."""
    report_ids, cse_vals, cie_vals, mf_vals = [], [], [], []
    for r in results:
        valid = [js for js in r.get("judge_scores", []) if js.get("status") == "ok"]
        if not valid:
            continue
        report_ids.append(r["id"])
        cse_vals.append(float(np.mean([js.get("clinically_significant_errors",   0) for js in valid])))
        cie_vals.append(float(np.mean([js.get("clinically_insignificant_errors", 0) for js in valid])))
        mf_vals.append(float(np.mean([js.get("matched_findings",                 0) for js in valid])))

    if not report_ids:
        logger.warning("No error data to plot.")
        return

    x = np.arange(len(report_ids))
    w = 0.6
    fig, ax = plt.subplots(figsize=(max(10, len(report_ids) * 0.9), 6),
                            constrained_layout=True)
    ax.bar(x, mf_vals,  width=w, label="Matched Findings",
           color=C_FOREST,  edgecolor="#0A2614", linewidth=0.8)
    ax.bar(x, cie_vals, width=w, label="Clinically Insignificant Errors",
           color=C_AMBER,   edgecolor="#3D2000", linewidth=0.8, bottom=mf_vals)
    ax.bar(x, cse_vals, width=w, label="Clinically Significant Errors (CSE)",
           color=C_CRIMSON, edgecolor="#5C0A0A", linewidth=0.8,
           bottom=[m + c for m, c in zip(mf_vals, cie_vals)])

    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i}" for i in report_ids], fontsize=10)
    ax.set_ylabel("Average Count (across 3 judges)")
    ax.set_xlabel("Report ID")
    ax.set_title("Figure 4 — Error Category Breakdown per Report\n"
                 "(averaged across Mistral-7B | Qwen2.5-3B | Gemma2-9B)", pad=12)
    ax.legend(loc="upper right", framealpha=1.0,
              edgecolor="#CCCCCC", fancybox=False)
    _hgrid(ax)

    save_fig(fig, "fig4_error_breakdown")
    plt.show()


def plot_summary_radar(stats: dict) -> None:
    """Radar chart — one spoke per judge + aggregated system score."""
    judge_names    = list(stats["per_judge"].keys())
    means          = [stats["per_judge"][n]["mean"] for n in judge_names]
    judge_names_ext = judge_names + ["Aggregated\n(System)"]
    means_ext       = means       + [stats["mean_green"]]

    npts        = len(judge_names_ext)
    theta       = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    means_plot  = means_ext + [means_ext[0]]
    theta_plot  = np.concatenate([theta, [theta[0]]])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True),
                            constrained_layout=True)
    ax.plot(theta_plot, means_plot, "o-", color=C_NAVY, lw=2.5, markersize=9)
    ax.fill(theta_plot, means_plot, alpha=0.18, color=C_NAVY)

    for ring in (0.25, 0.5, 0.75, 1.0):
        ax.plot(theta_plot, [ring] * (npts + 1), "--",
                color="#CCCCCC", lw=0.7, zorder=0)
        # Ring labels placed at a fixed angular position — no overlap possible
        ax.text(np.pi / 8, ring + 0.03, f"{ring:.2f}",
                ha="center", va="bottom", fontsize=7.5, color="#888888")

    ax.set_xticks(theta)
    # Pad spoke labels outward so they clear the polygon
    ax.set_xticklabels(judge_names_ext, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", pad=16)
    ax.set_ylim(0, 1.20)
    ax.set_yticks([])
    ax.set_title("Figure 5 — Per-Judge GREEN Score Comparison\n"
                 "(outer ring = 1.0 = best)",
                 fontsize=13, fontweight="bold", pad=28)

    # Score labels pushed 0.15 units beyond data point — clear of polygon
    for t, m in zip(theta, means_ext):
        ax.text(t, m + 0.14, f"{m:.3f}",
                ha="center", va="center", fontsize=10,
                fontweight="bold", color=C_NAVY,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=C_NAVY, linewidth=0.9, alpha=1.0))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    save_fig(fig, "fig5_radar_judge_comparison")
    plt.show()


def run_all_visualizations(results: list[dict], stats: dict) -> None:
    logger.info("\n[Step 4] Generating visualizations...")
    plot_aggregated_scores(results)
    plot_judge_heatmap(results)
    plot_judge_agreement(results)
    plot_error_breakdown(results)
    plot_summary_radar(stats)
    logger.info(f"  All figures saved to {FIG_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="GREEN Score Evaluation Pipeline with 3 LLM Judges"
    )
    ap.add_argument("--n",              type=int,   default=10,
                    help="Number of X-ray reports to evaluate (default: 10)")
    ap.add_argument("--dataset",        type=str,   default="mimic_reports",
                    choices=["mimic_reports", "nih", "chexpert", "iu_xray"],
                    help="Dataset to sample from (default: mimic_reports)")
    ap.add_argument("--skip-generate",  action="store_true",
                    help="Skip generation step, use previously saved reports")
    ap.add_argument("--skip-judging",   action="store_true",
                    help="Skip judging, use previously saved scores.json")
    ap.add_argument("--viz-only",       action="store_true",
                    help="Only run visualizations from saved scores.json")
    args = ap.parse_args()

    print("\n" + "="*65)
    print("  GREEN Score Evaluation Pipeline")
    print("  VLM+RAG+CoT vs 3 Local LLM Judges")
    print("="*65 + "\n")

    # ── Step 1: Generate or load reports ─────────────────────────────────
    if args.viz_only or args.skip_generate:
        report_pairs = load_saved_reports()
    else:
        report_pairs = generate_reports(n=args.n, dataset=args.dataset)

    if not report_pairs:
        logger.error("No report pairs found. Run without --skip-generate first.")
        return

    # ── Step 2: Judge reports ─────────────────────────────────────────────
    scores_file = RESULT_DIR / "scores.json"
    if args.viz_only or args.skip_judging:
        if scores_file.exists():
            results = json.loads(scores_file.read_text())
            logger.info(f"Loaded saved scores from {scores_file}")
        else:
            logger.error("No scores.json found. Run without --skip-judging first.")
            return
    else:
        results = judge_all_reports(report_pairs)

    # ── Step 3: Statistics ────────────────────────────────────────────────
    stats = compute_statistics(results)

    # ── Step 4: Visualize ─────────────────────────────────────────────────
    run_all_visualizations(results, stats)

    print("\n" + "="*65)
    print(f"  FINAL RESULTS")
    print(f"  Reports evaluated : {stats['n_reports']}")
    print(f"  Mean GREEN score  : {stats['mean_green']:.4f}")
    print(f"  Std GREEN score   : {stats['std_green']:.4f}")
    print(f"  Figures saved to  : {FIG_DIR}/")
    print(f"  Scores saved to   : {RESULT_DIR}/scores.json")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()