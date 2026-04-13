"""
train_lora.py - Stage 2: LoRA Fine-tuning (BALANCED VERSION)
==============================================================
FIX from v1: NIH and CheXpert parquets are label-sorted (all NORMAL first).
Round-robin streaming saw only NORMAL samples for ~1200+ steps, causing
the model to learn "always say NORMAL". This version PRE-FETCHES balanced
pools for NIH and CheXpert before training begins.

Frozen: MobileViT, Projector (resampler)
Trainable: LoRA adapters on Llama q_proj + v_proj (~1.15M params)

Dataset sampling strategy:
    NIH:      pre-fetch 400 balanced samples (200 NORMAL + 200 PNEUMONIA), keep in RAM
    CheXpert: pre-fetch 400 balanced samples, keep in RAM
    MIMIC:    stream live (already naturally balanced)
    Per step: randomly pick dataset, then pick a random sample from it

Usage:
    python -m experiments.train_lora --max-steps 1500 --save-every 500

    python -m experiments.train_lora --max-steps 3000 --save-every 500 \\
        --resume-from models/qlora_adapters/meddiag_lora_vlm/checkpoint-1500.pt \\
        --resume-lora-dir models/qlora_adapters/meddiag_lora_vlm/lora-step-1500
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision_encoder    import VisualProjector
from src.multimodal_fusion import _get_llama_embed_tokens
from src.data_loader       import StreamingDatasetManager

os.makedirs("logs", exist_ok=True)
OUTPUT_DIR = "models/qlora_adapters/meddiag_lora_vlm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train_lora.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("Stage2")
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


SYSTEM_PROMPT = (
    "You are a radiology AI. Examine the chest X-ray and classify it as "
    "NORMAL or ABNORMAL based solely on visual findings."
)
USER_PROMPT = "Classify this chest X-ray:"

NORMAL_VARIANTS = [
    "NORMAL. No acute cardiopulmonary findings.",
    "NORMAL. Clear lung fields bilaterally.",
    "NORMAL. No acute intrathoracic abnormality.",
    "NORMAL. Unremarkable chest radiograph.",
]
ABNORMAL_VARIANTS = [
    "ABNORMAL. {desc}.",
    "ABNORMAL. Findings consistent with {desc}.",
    "ABNORMAL. Radiographic evidence of {desc}.",
    "ABNORMAL. {desc} identified.",
]

ABNORMAL_KEYWORDS = [
    "consolidation", "opacity", "opacities", "effusion", "edema",
    "pneumothorax", "atelectasis", "infiltrate", "congestion",
    "cardiomegaly", "enlarged", "metastas", "mass", "nodule",
    "fracture", "abnormal", "pneumonia", "hemorrhage", "severe",
    "worsen", "increased", "pulmonary vascular",
]
NORMAL_KEYWORDS = [
    "no acute", "unremarkable", "clear", "no focal",
    "no evidence of", "within normal limits", "no cardiopulmonary",
]


def classify_mimic_impression(impression: str) -> tuple[int, str]:
    text = impression.lower()
    for kw in NORMAL_KEYWORDS:
        if kw in text:
            return 0, ""
    for kw in ABNORMAL_KEYWORDS:
        if kw in text:
            first_sentence = impression.split(".")[0].strip()
            desc = first_sentence[:80] if first_sentence else kw
            return 1, desc
    return 0, ""


def build_target(dataset: str, labels: list[str], text: str, rng: random.Random):
    if dataset in ("nih", "chexpert"):
        label_str = " ".join(labels).lower()
        if "normal" in label_str and "pneumonia" not in label_str:
            label = 0
        elif "pneumonia" in label_str:
            label = 1
        else:
            return None, None
        if label == 0:
            target = rng.choice(NORMAL_VARIANTS)
        else:
            tpl = rng.choice(ABNORMAL_VARIANTS)
            target = tpl.format(desc="pneumonia")
        return label, target

    elif dataset == "mimic_reports":
        impression = text.strip()
        if len(impression) < 10:
            return None, None
        label, desc = classify_mimic_impression(impression)
        if label == 0:
            target = rng.choice(NORMAL_VARIANTS)
        else:
            if not desc:
                desc = "abnormal findings"
            tpl = rng.choice(ABNORMAL_VARIANTS)
            target = tpl.format(desc=desc)
        return label, target
    return None, None


def prefetch_balanced(loader, dataset_name: str, n_per_class: int):
    """Pre-fetch N_per_class NORMAL + N_per_class ABNORMAL from a label-sorted dataset."""
    logger.info(f"Pre-fetching {dataset_name}: {n_per_class} NORMAL + {n_per_class} ABNORMAL...")
    normal, abnormal = [], []
    scanned = 0
    for sample in loader.stream(dataset_name, max_samples=None):
        scanned += 1
        if sample.get("image_pil") is None:
            continue
        labels_lower = [l.lower() for l in sample.get("labels", [])]
        is_normal = any("normal" in l and "pneumonia" not in l for l in labels_lower)
        is_pneum  = any("pneumonia" in l for l in labels_lower)

        if is_normal and len(normal) < n_per_class:
            normal.append(sample)
        elif is_pneum and len(abnormal) < n_per_class:
            abnormal.append(sample)

        if len(normal) >= n_per_class and len(abnormal) >= n_per_class:
            break
        if scanned % 500 == 0:
            logger.info(f"  [{dataset_name}] scanned {scanned} | N={len(normal)} | A={len(abnormal)}")

    logger.info(f"  [{dataset_name}] DONE: {len(normal)} NORMAL + {len(abnormal)} ABNORMAL (scanned {scanned})")
    combined = normal + abnormal
    random.Random(42).shuffle(combined)
    return combined


def build_training_batch(llama_model, tokenizer, projector,
                         pil_image, target_text, device):
    embed_tokens = _get_llama_embed_tokens(llama_model)
    embed_dtype  = next(embed_tokens.parameters()).dtype

    seg_a = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{USER_PROMPT}"
    )
    seg_b = (
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{target_text}<|eot_id|>"
    )

    tok_a = tokenizer(seg_a, return_tensors="pt", add_special_tokens=False).to(device)
    tok_b = tokenizer(seg_b, return_tensors="pt", truncation=True,
                      max_length=80, add_special_tokens=False).to(device)

    with torch.no_grad():
        emb_a = embed_tokens(tok_a["input_ids"]).to(embed_dtype)
        emb_b = embed_tokens(tok_b["input_ids"]).to(embed_dtype)
        pixel_values = projector.preprocess([pil_image]).to(device)
        visual_embeds = projector(pixel_values).to(embed_dtype)

    inputs_embeds = torch.cat([emb_a, visual_embeds, emb_b], dim=1)
    Ta, Tv, Tb = emb_a.shape[1], visual_embeds.shape[1], emb_b.shape[1]
    labels = torch.full((1, Ta + Tv + Tb), -100, dtype=torch.long, device=device)
    target_only = tokenizer(f"{target_text}<|eot_id|>", return_tensors="pt",
                            add_special_tokens=False)
    target_len = target_only["input_ids"].shape[1]
    labels[0, -target_len:] = tok_b["input_ids"][0, -target_len:]
    attention_mask = torch.ones((1, inputs_embeds.shape[1]), dtype=torch.long, device=device)
    return inputs_embeds, attention_mask, labels


def save_checkpoint(llama, optimizer, scheduler, step, path, save_lora_dir):
    llama.save_pretrained(save_lora_dir)
    state = {"step": step, "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "lora_dir": save_lora_dir}
    torch.save(state, path)
    logger.info(f"[CKPT] step={step} -> {path}")


def load_checkpoint(path, optimizer, scheduler):
    state = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    logger.info(f"[RESUME] continuing from step {state['step']}")
    return state["step"]


def mixed_sample_iterator(nih_pool, chex_pool, loader, rng):
    """Yield samples by randomly picking from NIH pool / CheXpert pool / MIMIC stream."""
    nih_idx = 0
    chex_idx = 0
    mimic_stream = loader.stream("mimic_reports", max_samples=None)

    while True:
        choice = rng.choice(["nih", "chexpert", "mimic_reports"])
        try:
            if choice == "nih":
                sample = nih_pool[nih_idx % len(nih_pool)]
                nih_idx += 1
            elif choice == "chexpert":
                sample = chex_pool[chex_idx % len(chex_pool)]
                chex_idx += 1
            else:
                sample = next(mimic_stream)
            sample_copy = dict(sample)
            sample_copy["_dataset"] = choice
            yield sample_copy
        except StopIteration:
            # MIMIC ran out, restart
            mimic_stream = loader.stream("mimic_reports", max_samples=None)
        except Exception as e:
            logger.warning(f"Sample iteration error: {e}")
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps",   type=int, default=3000)
    ap.add_argument("--save-every",  type=int, default=500)
    ap.add_argument("--log-every",   type=int, default=20)
    ap.add_argument("--grad-accum",  type=int, default=8)
    ap.add_argument("--lr",          type=float, default=2e-4)
    ap.add_argument("--warmup",      type=int, default=50)
    ap.add_argument("--resume-from", type=str, default=None)
    ap.add_argument("--resume-lora-dir", type=str, default=None)
    ap.add_argument("--total-steps-for-scheduler", type=int, default=3000)
    ap.add_argument("--projector-path", type=str,
                    default="models/visual_projector/projector.safetensors")
    ap.add_argument("--prefetch-per-class", type=int, default=200,
                    help="Balanced samples per class for NIH and CheXpert (200 = 400 each dataset)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(42)
    logger.info(f"Device: {device} | LR: {args.lr} | Max steps: {args.max_steps}")

    # ── PRE-FETCH balanced NIH and CheXpert pools ─────────────────────
    loader = StreamingDatasetManager()
    nih_pool  = prefetch_balanced(loader, "nih",      args.prefetch_per_class)
    chex_pool = prefetch_balanced(loader, "chexpert", args.prefetch_per_class)
    logger.info(f"Pools ready: NIH={len(nih_pool)} | CheXpert={len(chex_pool)}")

    # ── Load Llama + LoRA ──────────────────────────────────────────────
    logger.info("Loading Llama-3.2-3B in 4-bit with LoRA...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        quantization_config=bnb, device_map="auto", attn_implementation="sdpa",
    )
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if args.resume_from and args.resume_lora_dir:
        from peft import PeftModel
        logger.info(f"Loading LoRA from {args.resume_lora_dir}")
        llama = PeftModel.from_pretrained(base, args.resume_lora_dir, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.10, bias="none",
            task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"],
        )
        llama = get_peft_model(base, lora_cfg)
    llama.print_trainable_parameters()

    # ── Frozen projector ───────────────────────────────────────────────
    logger.info(f"Loading Stage 1 projector from {args.projector_path}")
    projector = VisualProjector(llama_hidden=3072, n_visual_tokens=8).to(device)
    state = torch.load(args.projector_path, map_location="cpu", weights_only=True)
    projector.resampler.load_state_dict(state)
    projector.eval()
    for p in projector.parameters():
        p.requires_grad = False
    logger.info("Projector FROZEN.")

    optimizer = torch.optim.AdamW(
        [p for p in llama.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup,
        num_training_steps=args.total_steps_for_scheduler,
    )

    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(args.resume_from, optimizer, scheduler)

    csv_path = Path("logs/stage2_loss.csv")
    csv_exists = csv_path.exists()
    csv_f = open(csv_path, "a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    if not csv_exists:
        csv_w.writerow(["step", "loss", "lr", "dataset", "label", "elapsed_min"])

    # ── Training loop ──────────────────────────────────────────────────
    sample_iter = mixed_sample_iterator(nih_pool, chex_pool, loader, rng)
    step  = start_step
    n_acc = 0
    losses = []
    labels_seen = {"0": 0, "1": 0}  # running counter of NORMAL vs ABNORMAL
    t0 = time.perf_counter()
    optimizer.zero_grad()

    logger.info(f"Training from step {start_step} to {args.max_steps}...")

    while step < args.max_steps:
        try:
            sample = next(sample_iter)
            img = sample.get("image_pil")
            if img is None:
                continue
            dataset = sample["_dataset"]
            labels  = sample.get("labels", [])
            text    = (sample.get("text") or sample.get("report") or "").strip()

            label, target = build_target(dataset, labels, text, rng)
            if target is None:
                continue

            try:
                inputs_embeds, attn_mask, label_ids = build_training_batch(
                    llama, tokenizer, projector, img, target, device,
                )
                out = llama(inputs_embeds=inputs_embeds,
                            attention_mask=attn_mask, labels=label_ids)
                raw_loss = out.loss
                loss = raw_loss / args.grad_accum
                loss.backward()
                n_acc += 1

                if n_acc % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in llama.parameters() if p.requires_grad], max_norm=1.0,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1
                    losses.append(raw_loss.item())
                    labels_seen[str(label)] += 1

                    if step % args.log_every == 0:
                        avg = sum(losses[-args.log_every:]) / min(len(losses), args.log_every)
                        lr_now = scheduler.get_last_lr()[0]
                        elapsed = (time.perf_counter() - t0) / 60
                        eta = elapsed * (args.max_steps - step) / max(step - start_step, 1)
                        ratio_abn = labels_seen["1"] / max(step - start_step, 1) * 100
                        logger.info(
                            f"Step {step}/{args.max_steps} | loss={avg:.4f} | "
                            f"lr={lr_now:.2e} | ds={dataset} | lbl={label} | "
                            f"ABN%={ratio_abn:.0f} | elapsed={elapsed:.1f}min | ETA={eta:.1f}min"
                        )
                        csv_w.writerow([step, round(avg,4), f"{lr_now:.2e}",
                                        dataset, label, round(elapsed,2)])
                        csv_f.flush()

                    if step % args.save_every == 0:
                        lora_subdir = f"{OUTPUT_DIR}/lora-step-{step}"
                        save_checkpoint(llama, optimizer, scheduler, step,
                                        f"{OUTPUT_DIR}/checkpoint-{step}.pt", lora_subdir)

                del inputs_embeds, attn_mask, label_ids, out, loss
                gc.collect()
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM at step {step}, skipping")
                optimizer.zero_grad()
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Sample skipped: {type(e).__name__}: {e}")

        except Exception as e:
            logger.warning(f"Iterator error: {e}. Rebuilding iterator...")
            time.sleep(3)
            sample_iter = mixed_sample_iterator(nih_pool, chex_pool, loader, rng)

    # ── Final save ────────────────────────────────────────────────────
    final_lora = f"{OUTPUT_DIR}/lora-final"
    save_checkpoint(llama, optimizer, scheduler, step,
                    f"{OUTPUT_DIR}/checkpoint-{step}.pt", final_lora)
    llama.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Final LoRA saved to {OUTPUT_DIR}")

    logger.info(f"\nLabel distribution seen during training:")
    logger.info(f"  NORMAL   = {labels_seen['0']}")
    logger.info(f"  ABNORMAL = {labels_seen['1']}")
    logger.info(f"  Ratio    = {labels_seen['1']/(labels_seen['0']+labels_seen['1'])*100:.1f}% ABNORMAL")

    csv_f.close()
    total_min = (time.perf_counter() - t0) / 60
    logger.info(f"Done. Trained {step - start_step} steps in {total_min:.1f} min.")


if __name__ == "__main__":
    main()