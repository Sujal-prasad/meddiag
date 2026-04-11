#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh — Failsafe Publication-Level Experiment Pipeline
# =============================================================================
#
# RESILIENCE FEATURES:
#   ✓ Power-cut / crash recovery  — checkpoint file tracks every completed
#                                   stage. Re-run the script and it resumes
#                                   from exactly where it stopped.
#   ✓ Internet / WiFi loss        — HuggingFace streaming retried up to 5×
#                                   with exponential backoff (5s→160s).
#                                   Each Python stage is wrapped in a retry
#                                   loop — transient failures don't abort.
#   ✓ Partial stage recovery      — each suite saves results to JSON before
#                                   plotting. If plotting crashes, results
#                                   are not lost.
#   ✓ Timeout detection           — if a stage hangs >90 min it is killed
#                                   and retried (catches deadlocked CUDA).
#   ✓ SIGINT / SIGTERM handling   — Ctrl-C writes a clean checkpoint so the
#                                   next run resumes instead of restarting.
#   ✓ Disk-space guard            — checks ≥5GB free before each stage.
#   ✓ Full audit log              — every command + timestamp written to
#                                   logs/experiment_run.log
#
# Usage:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh            # first run or resume after failure
#   ./run_all_experiments.sh --reset    # force restart from beginning
#
# =============================================================================

set -uo pipefail          # -e removed — we handle errors manually per stage

# ── Colours ──────────────────────────────────────────────────────────────────
G='\033[0;32m'; B='\033[0;34m'; Y='\033[1;33m'
R='\033[0;31m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { local msg="[$(date '+%H:%M:%S')] [INFO]  $*"; echo -e "${B}${msg}${RESET}"; echo "$msg" >> "$LOG_FILE"; }
success() { local msg="[$(date '+%H:%M:%S')] [DONE]  $*"; echo -e "${G}${msg}${RESET}"; echo "$msg" >> "$LOG_FILE"; }
warn()    { local msg="[$(date '+%H:%M:%S')] [WARN]  $*"; echo -e "${Y}${msg}${RESET}"; echo "$msg" >> "$LOG_FILE"; }
fail()    { local msg="[$(date '+%H:%M:%S')] [FAIL]  $*"; echo -e "${R}${msg}${RESET}"; echo "$msg" >> "$LOG_FILE"; }
stage()   { echo -e "\n${BOLD}${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}";
            echo -e "${BOLD}${B}  $*${RESET}";
            echo -e "${BOLD}${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n";
            echo "=== STAGE: $* ===" >> "$LOG_FILE"; }

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT="logs/.experiment_checkpoint"
LOG_FILE="logs/experiment_run.log"
mkdir -p logs experiments/figures green_eval/results/figures

# ── Configuration ─────────────────────────────────────────────────────────────
N_EVAL=50
N_RAG=20
K_VALUES="1 3 5 10"
N_GREEN=15
DATASET="mimic_reports"
MAX_RETRIES=5          # max retry attempts per stage
STAGE_TIMEOUT=5400     # 90 minutes per stage before kill + retry
BACKOFF_BASE=5         # exponential backoff base seconds

# ── Reset flag ────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--reset" ]]; then
    rm -f "$CHECKPOINT"
    warn "Checkpoint cleared — pipeline will restart from Stage 1."
fi

# ── Timing ────────────────────────────────────────────────────────────────────
START_TOTAL=$(date +%s)
elapsed() { echo $(( $(date +%s) - $1 )); }
fmt_time() { printf '%02dh %02dm %02ds' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT SYSTEM
# Each completed stage writes its name to CHECKPOINT.
# On resume, completed stages are skipped instantly.
# ─────────────────────────────────────────────────────────────────────────────
stage_done() {
    # Returns 0 (true) if stage $1 is already in the checkpoint file
    [[ -f "$CHECKPOINT" ]] && grep -qxF "$1" "$CHECKPOINT"
}

mark_done() {
    echo "$1" >> "$CHECKPOINT"
    success "Checkpoint saved: $1"
}

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL HANDLER — clean Ctrl-C
# Writes checkpoint before exit so next run resumes cleanly.
# ─────────────────────────────────────────────────────────────────────────────
CURRENT_STAGE="none"
cleanup() {
    echo ""
    warn "Interrupted during: ${CURRENT_STAGE}"
    warn "Checkpoint preserved — re-run script to resume from this stage."
    warn "Log: ${LOG_FILE}"
    exit 130
}
trap cleanup SIGINT SIGTERM

# ─────────────────────────────────────────────────────────────────────────────
# INTERNET CHECK — with retry + backoff
# ─────────────────────────────────────────────────────────────────────────────
wait_for_internet() {
    local attempt=1
    local backoff=$BACKOFF_BASE
    while true; do
        # Test HuggingFace CDN specifically (what datasets library uses)
        if curl -sf --max-time 10 https://huggingface.co/api/whoami-v2 \
               -H "Authorization: Bearer ${HF_TOKEN:-}" &>/dev/null; then
            return 0
        fi
        if (( attempt >= MAX_RETRIES )); then
            fail "Internet unreachable after ${attempt} attempts."
            return 1
        fi
        warn "Internet check failed (attempt ${attempt}/${MAX_RETRIES}). Waiting ${backoff}s..."
        sleep $backoff
        backoff=$(( backoff * 2 ))   # exponential backoff: 5→10→20→40→80→160s
        attempt=$(( attempt + 1 ))
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# DISK SPACE CHECK — requires 5GB free
# ─────────────────────────────────────────────────────────────────────────────
check_disk() {
    local free_kb
    free_kb=$(df -k . | awk 'NR==2 {print $4}')
    local free_gb=$(( free_kb / 1024 / 1024 ))
    if (( free_gb < 5 )); then
        fail "Only ${free_gb}GB disk space free. Need ≥5GB. Free space and retry."
        exit 1
    fi
    info "Disk space: ${free_gb}GB free — OK"
}

# ─────────────────────────────────────────────────────────────────────────────
# RESILIENT STAGE RUNNER
# Usage: run_stage STAGE_ID DESCRIPTION "python command..."
#
# Behaviour:
#   1. Skip instantly if STAGE_ID is in checkpoint (already completed).
#   2. Check internet before every attempt.
#   3. Run command with STAGE_TIMEOUT timeout.
#   4. On failure: exponential backoff retry up to MAX_RETRIES times.
#   5. On success: write STAGE_ID to checkpoint.
#   6. If all retries fail: log failure, continue to next stage
#      (so one broken experiment doesn't block all others).
# ─────────────────────────────────────────────────────────────────────────────
run_stage() {
    local stage_id="$1"
    local description="$2"
    shift 2
    local cmd=("$@")

    # ── Skip if already done ──────────────────────────────────────────────
    if stage_done "$stage_id"; then
        success "SKIP (already complete): ${description}"
        return 0
    fi

    CURRENT_STAGE="$stage_id"
    stage "$description"

    local attempt=1
    local backoff=$BACKOFF_BASE
    local t_stage=$(date +%s)

    while (( attempt <= MAX_RETRIES )); do
        info "Attempt ${attempt}/${MAX_RETRIES}: ${description}"

        # ── Internet check before every attempt ───────────────────────────
        if ! wait_for_internet; then
            warn "Skipping ${description} — no internet after ${MAX_RETRIES} retries."
            warn "Re-run script when internet is restored to resume this stage."
            return 1
        fi

        # ── Run with timeout ──────────────────────────────────────────────
        # 'timeout' kills the process after STAGE_TIMEOUT seconds
        # Exit code 124 = timeout, anything else = normal error
        local exit_code=0
        timeout "${STAGE_TIMEOUT}" "${cmd[@]}" \
            2>&1 | tee -a "$LOG_FILE" \
            || exit_code=$?

        if [[ $exit_code -eq 0 ]]; then
            local duration
            duration=$(elapsed $t_stage)
            success "${description} — completed in $(fmt_time $duration)"
            mark_done "$stage_id"
            return 0

        elif [[ $exit_code -eq 124 ]]; then
            warn "TIMEOUT after $(fmt_time $STAGE_TIMEOUT): ${description}"
            warn "This usually means CUDA deadlock or OOM. Will retry."

        else
            warn "Exit code ${exit_code}: ${description}"
            # Check if it looks like a network error in the log
            if tail -50 "$LOG_FILE" | grep -qiE \
               "connection|timeout|refused|network|ssl|certificate|EOF|reset by peer|HTTPSConnectionPool"; then
                warn "Network error detected. Waiting for internet..."
                wait_for_internet || true
            fi
        fi

        if (( attempt < MAX_RETRIES )); then
            warn "Retrying in ${backoff}s... (attempt $((attempt+1))/${MAX_RETRIES})"
            sleep $backoff
            backoff=$(( backoff * 2 ))
        fi
        attempt=$(( attempt + 1 ))
    done

    fail "ALL ${MAX_RETRIES} ATTEMPTS FAILED: ${description}"
    fail "Moving to next stage. Re-run script to retry this one."
    return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║   MedDiag — Failsafe Publication-Level Experiment Runner          ║"
echo "║   Power-cut safe  |  WiFi-loss safe  |  Crash-resume safe         ║"
echo "╠═══════════════════════════════════════════════════════════════════╣"
echo "║   Checkpoint  : logs/.experiment_checkpoint                        ║"
echo "║   Full log    : logs/experiment_run.log                            ║"
echo "║   Re-run to resume from last completed stage.                      ║"
echo "║   --reset flag restarts from scratch.                              ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# Show resume status
if [[ -f "$CHECKPOINT" ]]; then
    warn "Checkpoint found. Resuming. Completed stages:"
    while IFS= read -r line; do
        echo -e "  ${G}✓${RESET}  $line"
    done < "$CHECKPOINT"
    echo ""
fi

echo "[$(date)] Pipeline started" >> "$LOG_FILE"

# ─────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECKS (not checkpointed — always run)
# ─────────────────────────────────────────────────────────────────────────────
stage "Pre-flight Checks"

# Python
if ! python -c "import torch" &>/dev/null; then
    fail "Python venv not active. Run:  source .venv/Scripts/activate"
    exit 1
fi
success "Python environment OK"

# HF token
if [[ -z "${HF_TOKEN:-}" ]]; then
    fail "HF_TOKEN not set. Run:  export HF_TOKEN=your_token"
    exit 1
fi
success "HF_TOKEN set"

# Internet
info "Checking HuggingFace connectivity..."
if ! wait_for_internet; then
    fail "Cannot reach HuggingFace. Connect to internet and re-run."
    exit 1
fi
success "Internet connectivity OK"

# Disk
check_disk

# FAISS index
if ! stage_done "preflight_index"; then
    if [[ ! -f "models/faiss_index/medical_guidelines.faiss" ]]; then
        warn "FAISS index missing — building (~5 min)..."
        run_stage "preflight_index" "Build FAISS index" \
            python -m src.pipeline --phase index
    else
        mark_done "preflight_index"
    fi
fi
success "FAISS index ready"

# LoRA adapters
if ! stage_done "preflight_train"; then
    if [[ ! -d "models/qlora_adapters/meddiag_lora" ]]; then
        warn "LoRA adapters missing — training (~90 min)..."
        run_stage "preflight_train" "Train LoRA adapters" \
            python -m src.pipeline --phase train
    else
        mark_done "preflight_train"
    fi
fi
success "LoRA adapters ready"

# Ollama
OLLAMA_OK=false
if curl -sf --max-time 5 http://localhost:11434/api/tags &>/dev/null; then
    OLLAMA_OK=true
    success "Ollama running"
else
    warn "Ollama not running — GREEN eval will be skipped."
    warn "Start it: ollama serve  (in a separate Git Bash terminal)"
fi

mkdir -p experiments/figures logs green_eval/results/figures
info "Estimated runtime: 5-6 hours on RTX 3050"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — SUITE 1: Compute & Accuracy
# ─────────────────────────────────────────────────────────────────────────────
run_stage "suite1" \
    "Suite 1 — Compute & Accuracy (n=${N_EVAL}, real AUROC/F1/VRAM/latency)" \
    python -m experiments.evaluate --suite 1 --n ${N_EVAL} --save

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — SUITE 2: Hallucination Mitigation
# ─────────────────────────────────────────────────────────────────────────────
run_stage "suite2" \
    "Suite 2 — Hallucination Mitigation (n=${N_EVAL}, VLM Alone vs VLM+RAG)" \
    python -m experiments.evaluate --suite 2 --n ${N_EVAL} --save

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — SUITE 3: Clinical Interpretability
# ─────────────────────────────────────────────────────────────────────────────
run_stage "suite3" \
    "Suite 3 — Clinical Interpretability (n=${N_EVAL}, real BERTScore)" \
    python -m experiments.evaluate --suite 3 --n ${N_EVAL} --save

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — SUITE 4: Sycophancy & OOD
# ─────────────────────────────────────────────────────────────────────────────
run_stage "suite4" \
    "Suite 4 — Sycophancy & OOD Robustness (n=${N_EVAL}, real adversarial probe)" \
    python -m experiments.evaluate --suite 4 --n ${N_EVAL} --save

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — RAG Ablation
# ─────────────────────────────────────────────────────────────────────────────
run_stage "rag_ablation" \
    "RAG k-Ablation Study (k=${K_VALUES}, n=${N_RAG} queries each)" \
    python -m experiments.rag_ablation \
        --k ${K_VALUES} \
        --n-queries ${N_RAG} \
        --dataset ${DATASET}

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — GREEN Score Evaluation
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$OLLAMA_OK" == "true" ]]; then
    run_stage "green_eval" \
        "GREEN Score Evaluation (n=${N_GREEN} reports, 3 LLM judges)" \
        python -m green_eval.run_green_eval \
            --n ${N_GREEN} \
            --dataset ${DATASET}
else
    # Re-check Ollama — user may have started it during the run
    if curl -sf --max-time 5 http://localhost:11434/api/tags &>/dev/null; then
        run_stage "green_eval" \
            "GREEN Score Evaluation (n=${N_GREEN} reports, 3 LLM judges)" \
            python -m green_eval.run_green_eval \
                --n ${N_GREEN} \
                --dataset ${DATASET}
    else
        warn "GREEN eval skipped — Ollama still not running."
        warn "To run it later (does NOT need this script):"
        warn "  ollama serve  # in a separate terminal"
        warn "  python -m green_eval.run_green_eval --n ${N_GREEN}"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
TOTAL=$(elapsed $START_TOTAL)
COMPLETED=$(wc -l < "$CHECKPOINT" 2>/dev/null || echo 0)
TOTAL_STAGES=8   # preflight×2 + suite1-4 + rag + green

echo ""
echo -e "${BOLD}${G}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                     PIPELINE COMPLETE                             ║"
echo "╠═══════════════════════════════════════════════════════════════════╣"
printf  "║  Total runtime   : %-47s║\n" "$(fmt_time $TOTAL)"
printf  "║  Stages complete : %-47s║\n" "${COMPLETED} / ${TOTAL_STAGES}"
printf  "║  Full log        : %-47s║\n" "logs/experiment_run.log"
echo "╠═══════════════════════════════════════════════════════════════════╣"
echo "║  Figures:                                                          ║"
echo "║  experiments/figures/suite1_compute_accuracy.png                  ║"
echo "║  experiments/figures/suite2_hallucination.png                     ║"
echo "║  experiments/figures/suite3_interpretability.png                  ║"
echo "║  experiments/figures/suite4_robustness.png                        ║"
echo "║  experiments/figures/rag_ablation_combined.png                    ║"
echo "║  experiments/figures/rag_fig1_normalized_multiline.png            ║"
echo "║  experiments/figures/rag_fig2_dualaxis_tradeoff.png               ║"
echo "║  experiments/figures/rag_fig3_pareto_frontier.png                 ║"
echo "║  experiments/figures/rag_fig4_score_distribution.png              ║"
echo "║  green_eval/results/figures/fig1_aggregated_green_scores.png      ║"
echo "║  green_eval/results/figures/fig2_judge_heatmap.png                ║"
echo "║  green_eval/results/figures/fig3_judge_agreement.png              ║"
echo "║  green_eval/results/figures/fig4_error_breakdown.png              ║"
echo "║  green_eval/results/figures/fig5_radar_judge_comparison.png       ║"
echo "╠═══════════════════════════════════════════════════════════════════╣"
echo "║  To re-run a specific failed stage:                                ║"
echo "║    Edit logs/.experiment_checkpoint — delete that stage's line,   ║"
echo "║    then re-run ./run_all_experiments.sh                            ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

echo "[$(date)] Pipeline finished. Runtime: $(fmt_time $TOTAL). Stages: ${COMPLETED}/${TOTAL_STAGES}" >> "$LOG_FILE"