#!/usr/bin/env bash
# =============================================================================
# setup.sh — Environment Bootstrap & Dataset Acquisition
# Project:  Compressed Medical Diagnostic Pipeline (QLoRA + FAISS RAG + CoT)
# Target:   Edge hardware with ≤4GB VRAM (e.g., NVIDIA RTX 3050 @ 55W TGP)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# This script will:
#   1. Verify system prerequisites (Python, CUDA, disk space)
#   2. Create the full project directory tree
#   3. Set up a Python virtual environment and install dependencies
#   4. Prompt for PhysioNet and Kaggle credentials
#   5. Download all five datasets used in the pipeline
# =============================================================================

set -euo pipefail   # Exit on error, undefined var, or pipe failure
IFS=$'\n\t'         # Safer word splitting

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers
# ─────────────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# 0. BANNER
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║        Medical Diagnostic Pipeline — Environment Setup           ║"
echo "║   QLoRA 3B + FAISS RAG + Chain-of-Thought  |  Edge ≤4GB VRAM   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ─────────────────────────────────────────────────────────────────────────────
# 1. PREREQUISITE CHECKS
# ─────────────────────────────────────────────────────────────────────────────
info "Checking prerequisites..."

# Python ≥ 3.10 required for match statements and modern type hints
if ! command -v python3 &>/dev/null; then
    error "Python 3 not found. Install Python ≥ 3.10 and retry."
fi
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 10 ]]; }; then
    error "Python ≥ 3.10 required (found $PY_VERSION)."
fi
success "Python $PY_VERSION detected."

# CUDA — optional but strongly recommended; warn if absent
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+')
    success "CUDA $CUDA_VERSION detected."
else
    warn "CUDA not found. Pipeline will fall back to CPU-only inference (very slow)."
fi

# Disk space: only ~2GB needed for models/FAISS index (no dataset downloads)
AVAILABLE_GB=$(df -BG . | awk 'NR==2 {gsub("G",""); print $4}')
if [[ "$AVAILABLE_GB" -lt 5 ]]; then
    warn "Only ${AVAILABLE_GB}GB free. Need at least 5GB for model adapters and FAISS index."
    read -rp "Continue anyway? [y/N] " CONT
    [[ "${CONT,,}" == "y" ]] || exit 0
else
    success "${AVAILABLE_GB}GB free — sufficient (datasets are streamed, not stored)."
fi

# wget and unzip are required for download & extraction
for cmd in wget unzip curl git; do
    command -v "$cmd" &>/dev/null || error "'$cmd' not installed. Run: sudo apt install $cmd"
done
success "All CLI tools present."

# ─────────────────────────────────────────────────────────────────────────────
# 2. DIRECTORY TREE
# ─────────────────────────────────────────────────────────────────────────────
info "Creating project directory tree..."

# All directories created idempotently with -p
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
declare -a DIRS=(
    "src"
    "data/raw/mimic_cxr"
    "data/raw/chexpert"
    "data/raw/nih_chestxray14"
    "data/raw/iu_xray"
    "data/raw/padchest"
    "data/processed"
    "data/embeddings"
    "models/qlora_adapters"
    "models/faiss_index"
    "notebooks"
    "scripts"
    "experiments"
    "docker"
    "logs"
)
for dir in "${DIRS[@]}"; do
    mkdir -p "${PROJECT_ROOT}/${dir}"
done
success "Directory tree ready."

# ─────────────────────────────────────────────────────────────────────────────
# 3. PYTHON VIRTUAL ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
VENV_DIR="${PROJECT_ROOT}/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment at .venv/ ..."
    python3 -m venv "$VENV_DIR"
else
    info "Virtual environment already exists — skipping creation."
fi

# Activate
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel --quiet
success "Virtual environment activated."

# Install dependencies from requirements.txt (generated in Step 5 of pipeline)
if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
    info "Installing Python dependencies (this may take several minutes)..."
    pip install -r "${PROJECT_ROOT}/requirements.txt" --quiet
    success "Dependencies installed."
else
    warn "requirements.txt not found. Run pip install manually after generation."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. CREDENTIAL COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 4. DATASET STRATEGY — STREAMING (Zero local storage)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Dataset Strategy: HuggingFace Streaming${RESET}"
echo "════════════════════════════════════════"
echo " No datasets will be downloaded to your laptop."
echo " All 5 datasets are fetched live from HuggingFace Hub"
echo " one sample at a time when you run the pipeline."
echo ""
echo " Datasets streamed (0 GB local storage):"
echo "   ✅ NIH ChestX-ray14       alkzar90/NIH-Chest-X-ray-dataset"
echo "   ✅ CheXpert-Small          khaled-alefari/chexpert-small"
echo "   ✅ IU-Xray                 Groakos/iu-xray"
echo "   ✅ MIMIC-CXR reports       medical-nlp/mimic-cxr-reports"
echo "   ✅ PadChest                cropped-padchest"
echo ""
info "No credentials required for streaming."
info "Just make sure you have an internet connection when running the pipeline."

# ─────────────────────────────────────────────────────────────────────────────
# 5. VERIFY STREAMING CONNECTIONS
# ─────────────────────────────────────────────────────────────────────────────
echo ""
info "Testing HuggingFace streaming connections..."
echo "(This fetches 1 sample per dataset to verify access — no data saved)"
echo ""

python3 - <<'PYEOF'
import sys
sys.path.insert(0, ".")

try:
    from src.data_loader import StreamingDatasetManager
    manager = StreamingDatasetManager()
    results = manager.test_all_connections()
    passed  = sum(results.values())
    total   = len(results)
    if passed == total:
        print(f"\n✅ All {total} datasets reachable via streaming.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n⚠️  {passed}/{total} datasets reachable.")
        print(f"   Failed: {', '.join(failed)}")
        print("   Check your internet connection and try again.")
except ImportError:
    print("⚠️  data_loader.py not found yet — run this after installing requirements.")
except Exception as e:
    print(f"⚠️  Connection test failed: {e}")
    print("   This is OK for now — verify internet before running the pipeline.")
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
# 7. HUGGING FACE MODEL PRE-CACHING
# ─────────────────────────────────────────────────────────────────────────────
info "Pre-caching Hugging Face models (requires internet + HF token for Llama 3.2)..."
echo "Set your HF token: export HUGGINGFACE_HUB_TOKEN=<your_token>"
echo "Then models will be cached to ~/.cache/huggingface/"
echo ""
echo "Models required:"
echo "  1. meta-llama/Llama-3.2-3B-Instruct  (needs HF access request)"
echo "  2. sentence-transformers/all-MiniLM-L6-v2  (public)"
echo ""
echo "  Run manually:"
echo "  python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')\""

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN}║  Setup complete! Next steps:             ║${RESET}"
echo -e "${BOLD}${GREEN}║  1. source .venv/bin/activate            ║${RESET}"
echo -e "${BOLD}${GREEN}║  2. python src/pipeline.py --phase train ║${RESET}"
echo -e "${BOLD}${GREEN}║  3. python src/pipeline.py --phase index ║${RESET}"
echo -e "${BOLD}${GREEN}║  4. python src/pipeline.py --phase infer ║${RESET}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════╝${RESET}"
