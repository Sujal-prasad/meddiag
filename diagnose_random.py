"""
diagnose_random.py — Fetch a random chest X-ray and generate a diagnosis
=========================================================================
Picks a random sample from any HuggingFace chest X-ray dataset,
displays the image, and runs the full pipeline (FAISS RAG + CoT) on it.

Usage:
    python diagnose_random.py
    python diagnose_random.py --dataset nih
    python diagnose_random.py --dataset mimic_reports --show
    python diagnose_random.py --dataset chexpert --seed 99
"""

import argparse
import os
import random
import sys
import tempfile
import textwrap
from pathlib import Path

# ── HuggingFace auth ──────────────────────────────────────────────────────────
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

from huggingface_hub import login
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token, add_to_git_credential=False)

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_loader import StreamingDatasetManager
from src.pipeline    import EdgeMedicalVLM, FAISSConfig

# ─────────────────────────────────────────────────────────────────────────────
# AVAILABLE DATASETS
# ─────────────────────────────────────────────────────────────────────────────
DATASETS = {
    "nih":           "NIH Chest X-ray (NORMAL / PNEUMONIA)",
    "chexpert":      "CheXpert Chest X-ray",
    "mimic_reports": "MIMIC-CXR (real clinical reports + images)",
    "iu_xray":       "IU-Xray (Indiana University, held-out test split)",
    "padchest":      "PadChest substitute (OOD validation split)",
}


def fetch_random_sample(dataset_name: str, seed: int, n_pool: int = 20) -> dict:
    """
    Stream n_pool samples, pick one at random.
    n_pool=20 gives enough variety without waiting too long.
    """
    print(f"\n[1/4] Streaming {n_pool} samples from '{dataset_name}'...")
    loader  = StreamingDatasetManager()
    samples = loader.get_sample_batch(dataset_name, n=n_pool)

    if not samples:
        raise RuntimeError(f"Could not stream any samples from {dataset_name}.")

    # Filter to samples that have an actual image
    image_samples = [s for s in samples if s.get("image_pil") is not None]
    if not image_samples:
        raise RuntimeError(
            f"No image samples found in {dataset_name}. "
            "Try --dataset nih or --dataset mimic_reports"
        )

    random.seed(seed)
    sample = random.choice(image_samples)
    print(f"    Picked sample with labels: {sample['labels']}")
    print(f"    Dataset: {sample['dataset']}")
    return sample


def show_image(pil_img, label: str) -> None:
    """Display the X-ray image using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("TkAgg")  # works on Windows

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(pil_img, cmap="gray")
        ax.set_title(f"Input X-ray — Label: {label}",
                     fontsize=12, fontweight="bold", pad=10)
        ax.axis("off")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(2)
        print("    X-ray displayed.")
    except Exception as e:
        print(f"    Could not display image ({e}). Continuing without display.")


def save_temp_image(pil_img) -> str:
    """Save PIL image to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pil_img.save(tmp.name)
    tmp.close()
    return tmp.name


def build_visual_finding(sample: dict) -> str:
    """
    Build a textual description of the X-ray for the pipeline.
    For MIMIC samples this uses the real clinical findings text.
    For image-only samples it builds a structured description.
    """
    dataset = sample["dataset"]
    labels  = sample["labels"]

    if dataset in ("mimic_reports", "mimic_rag"):
        # Use the actual clinical findings text
        findings = sample.get("text", "").strip()
        report   = sample.get("report", "").strip()
        if findings:
            return f"Clinical findings: {findings}"
        elif report:
            return f"Clinical impression: {report}"

    # For image-based datasets, build from labels
    label_str = ", ".join(labels) if labels else "unknown"
    return (
        f"Chest X-ray from {dataset.upper()} dataset. "
        f"Annotated label(s): {label_str}. "
        f"Analyse for consolidation, pleural effusion, cardiomegaly, "
        f"pneumothorax, infiltrates, and interstitial patterns. "
        f"Provide an objective radiological assessment."
    )


def print_report(report: str, sample: dict, dataset_name: str) -> None:
    """Pretty-print the diagnostic report."""
    width = 70
    print("\n" + "=" * width)
    print(" DIAGNOSTIC REPORT")
    print(f" Dataset : {DATASETS.get(dataset_name, dataset_name)}")
    print(f" Labels  : {', '.join(sample['labels'])}")
    print("=" * width)

    # Try to extract and format XML sections
    sections = [
        ("VISUAL_FINDINGS",    "VISUAL FINDINGS"),
        ("CLINICAL_EVIDENCE",  "CLINICAL EVIDENCE (RAG)"),
        ("DEDUCTIVE_REASONING","DEDUCTIVE REASONING"),
        ("FINAL_DIAGNOSIS",    "FINAL DIAGNOSIS"),
    ]

    found_any = False
    for tag, title in sections:
        start = report.find(f"<{tag}>")
        end   = report.find(f"</{tag}>")
        if start != -1 and end != -1:
            content = report[start + len(tag) + 2 : end].strip()
            print(f"\n{'─' * width}")
            print(f" {title}")
            print(f"{'─' * width}")
            # Wrap long lines
            for line in content.splitlines():
                if line.strip():
                    wrapped = textwrap.fill(line.strip(), width=width - 2)
                    print(f" {wrapped}")
            found_any = True

    if not found_any:
        # Model didn't follow the XML format — print raw output
        print("\n Raw model output:")
        print("─" * width)
        for line in report.splitlines():
            if line.strip():
                print(f" {textwrap.fill(line.strip(), width=width-2)}")

    print("\n" + "=" * width)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a random chest X-ray and generate a diagnosis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python diagnose_random.py                          # default (nih)
          python diagnose_random.py --dataset mimic_reports  # MIMIC-CXR
          python diagnose_random.py --dataset chexpert --show # show image
          python diagnose_random.py --dataset nih --seed 42  # reproducible
          python diagnose_random.py --list                   # list datasets
        """),
    )
    parser.add_argument(
        "--dataset", "-d",
        default="nih",
        choices=list(DATASETS.keys()),
        help="Which dataset to sample from (default: nih)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the X-ray image before diagnosis",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=20,
        help="How many samples to stream before picking one (default: 20)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets and exit",
    )
    parser.add_argument(
        "--history",
        type=str,
        default="No clinical history provided.",
        help="Optional clinical history to provide to the model",
    )
    args = parser.parse_args()

    # ── List datasets ─────────────────────────────────────────────────────
    if args.list:
        print("\nAvailable datasets:")
        for key, desc in DATASETS.items():
            print(f"  --dataset {key:<18} {desc}")
        return

    seed = args.seed if args.seed is not None else random.randint(0, 9999)
    print(f"\nRandom seed: {seed}  (use --seed {seed} to reproduce this result)")

    # ── Step 1: Fetch random sample ───────────────────────────────────────
    sample = fetch_random_sample(args.dataset, seed=seed, n_pool=args.pool)

    # ── Step 2: Show image ────────────────────────────────────────────────
    if args.show and sample.get("image_pil"):
        print("\n[2/4] Displaying X-ray image...")
        show_image(
            sample["image_pil"],
            label=", ".join(sample["labels"])
        )
    else:
        print("\n[2/4] Image display skipped. Use --show to display.")

    # ── Step 3: Load pipeline ─────────────────────────────────────────────
    print("\n[3/4] Loading pipeline (QLoRA model + FAISS index)...")
    print("      This takes ~30 seconds on first run...")

    vlm = EdgeMedicalVLM()

    # ── Step 4: Build finding + run inference ─────────────────────────────
    print("\n[4/4] Running diagnosis...")
    visual_finding = build_visual_finding(sample)
    print(f"      Finding: {visual_finding[:120]}...")

    report = vlm.generate_diagnosis(
        visual_findings=visual_finding,
        clinical_history=args.history,
    )

    # ── Print report ──────────────────────────────────────────────────────
    print_report(report, sample, args.dataset)
    print(f"\nDone. Seed was {seed} — use --seed {seed} to reproduce.\n")


if __name__ == "__main__":
    main()