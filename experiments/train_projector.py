"""
train_projector.py - Stage 1: Projector Pretraining
=====================================================
Freezes Llama + MobileViT. Trains ONLY the Perceiver resampler (~6M params)
to produce visual tokens that Llama can use to generate MIMIC-CXR findings.

Why this stage exists:
    After random initialisation, the projector outputs 8 tokens of noise.
    Llama ignores them. Stage 1 aligns the projector's output distribution
    to Llama's embedding space using paired (image, findings) data.

Training objective:
    Given (image, findings_text), maximise P(findings | visual_tokens).
    Standard next-token cross-entropy loss on the findings tokens only
    (visual + system prompt tokens are masked with -100).

Gradient flow:
    loss -> Llama (frozen, no grad) -> inputs_embeds
         -> embeds_a (frozen text embed, no grad)
         -> visual_embeds (GRAD FLOWS HERE through projector)
         -> embeds_b (frozen text embed, no grad)
    Only resampler params update.

Resumability:
    Checkpoints saved every --save-every steps.
    --resume-from <path> loads projector + optimizer + scheduler + step count.
    Loss curve CSV append-mode so interrupted runs don't lose history.

Usage:
    # First run: 1000 steps, checkpoint at 250/500/750/1000
    python -m experiments.train_projector --max-steps 1000 --save-every 250

    # Inspect loss, then continue for 1000 more
    python -m experiments.train_projector --max-steps 2000 \\
        --resume-from models/visual_projector/checkpoint-1000.pt

    # Final run to 5000
    python -m experiments.train_projector --max-steps 5000 \\
        --resume-from models/visual_projector/checkpoint-2000.pt
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision_encoder     import VisualProjector
from src.multimodal_fusion  import _get_llama_embed_tokens
from src.data_loader        import StreamingDatasetManager

os.makedirs("logs", exist_ok=True)
os.makedirs("models/visual_projector", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train_projector.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("Stage1")

# Suppress HF noise
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


SYSTEM_PROMPT = (
    "You are a radiology AI. Describe the findings visible in this chest X-ray "
    "in the style of a MIMIC-CXR radiology report."
)
USER_PREFIX = "Findings:"


def build_training_batch(
    llama_model, tokenizer, projector,
    pil_image: Image.Image, findings_text: str,
    device: str, max_target_tokens: int = 180,
):
    """
    Build inputs_embeds + labels for one training example.

    Layout:
        [BOS + system + user_header]  <- visual tokens -->  [findings + eot]
         ^^^^^^^^^^^^^ frozen ^^^^^^^^^   ^^^^^^^^^^^^^       ^^^ labels ^^^

    Labels mask everything except the findings tokens (so loss is only
    computed on "predict the findings text").
    """
    embed_tokens = _get_llama_embed_tokens(llama_model)
    embed_dtype  = next(embed_tokens.parameters()).dtype

    # Segment A: system prompt + user turn header
    seg_a = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{USER_PREFIX}"
    )
    # Segment B: the target findings + end-of-turn
    seg_b = f" {findings_text}<|eot_id|>"

    tok_a = tokenizer(seg_a, return_tensors="pt",
                      add_special_tokens=False).to(device)
    tok_b = tokenizer(seg_b, return_tensors="pt",
                      truncation=True, max_length=max_target_tokens,
                      add_special_tokens=False).to(device)

    # Embed text through frozen embed_tokens (no grad)
    with torch.no_grad():
        emb_a = embed_tokens(tok_a["input_ids"]).to(embed_dtype)   # (1, Ta, D)
        emb_b = embed_tokens(tok_b["input_ids"]).to(embed_dtype)   # (1, Tb, D)

    # Visual tokens — GRAD FLOWS HERE
    pixel_values = projector.preprocess([pil_image]).to(device)
    visual_embeds = projector(pixel_values).to(embed_dtype)        # (1, 8, D)

    # Concatenate: [A | visual | B]
    inputs_embeds = torch.cat([emb_a, visual_embeds, emb_b], dim=1)

    # Labels: -100 for A + visual, real token ids for B
    Ta = emb_a.shape[1]
    Tv = visual_embeds.shape[1]
    Tb = emb_b.shape[1]
    labels = torch.full(
        (1, Ta + Tv + Tb), -100, dtype=torch.long, device=device,
    )
    labels[0, Ta + Tv:] = tok_b["input_ids"][0]

    attention_mask = torch.ones(
        (1, inputs_embeds.shape[1]), dtype=torch.long, device=device,
    )
    return inputs_embeds, attention_mask, labels


def save_checkpoint(projector, optimizer, scheduler, step, path):
    state = {
        "step": step,
        "resampler": {k: v.cpu() for k, v in projector.resampler.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(state, path)
    logger.info(f"[CKPT] step={step} -> {path}")


def load_checkpoint(path, projector, optimizer, scheduler, device):
    logger.info(f"Resuming from {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    projector.resampler.load_state_dict(state["resampler"])
    projector.to(device)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    logger.info(f"[RESUME] continuing from step {state['step']}")
    return state["step"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps",   type=int, default=1000)
    ap.add_argument("--save-every",  type=int, default=250)
    ap.add_argument("--log-every",   type=int, default=10)
    ap.add_argument("--grad-accum",  type=int, default=8)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--warmup",      type=int, default=50)
    ap.add_argument("--resume-from", type=str, default=None)
    ap.add_argument("--total-steps-for-scheduler", type=int, default=5000,
                    help="Cosine scheduler total horizon (keep constant across resumes)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Load Llama 4-bit (frozen) ──────────────────────────────────────
    logger.info("Loading Llama-3.2-3B in 4-bit (frozen)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    llama = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        quantization_config=bnb, device_map="auto",
        attn_implementation="sdpa",
    )
    for p in llama.parameters():
        p.requires_grad = False
    llama.eval()
    logger.info(f"Llama VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    # ── Load projector ─────────────────────────────────────────────────
    logger.info("Loading VisualProjector (MobileViT frozen, resampler trainable)...")
    projector = VisualProjector(llama_hidden=3072, n_visual_tokens=8).to(device)
    projector.encoder.eval()
    projector.resampler.train()
    logger.info(f"Trainable params: {projector.num_trainable():,}")

    # ── Optimizer + scheduler ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        projector.trainable_parameters(),
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=args.total_steps_for_scheduler,
    )

    # ── Resume if requested ────────────────────────────────────────────
    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(
            args.resume_from, projector, optimizer, scheduler, device,
        )

    # ── Loss CSV (append mode for resumes) ─────────────────────────────
    csv_path = Path("logs/stage1_loss.csv")
    csv_exists = csv_path.exists()
    csv_f = open(csv_path, "a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    if not csv_exists:
        csv_w.writerow(["step", "loss", "lr", "elapsed_min"])

    # ── Training loop ──────────────────────────────────────────────────
    loader = StreamingDatasetManager()
    step   = start_step
    n_acc  = 0
    losses = []
    t0     = time.perf_counter()
    optimizer.zero_grad()

    logger.info(f"Training from step {start_step} to {args.max_steps}...")

    while step < args.max_steps:
        try:
            stream = loader.stream("mimic_reports", max_samples=None)
            for sample in stream:
                if step >= args.max_steps:
                    break

                img  = sample.get("image_pil")
                text = (sample.get("text") or "").strip()
                if img is None or len(text) < 20:
                    continue

                try:
                    inputs_embeds, attn_mask, labels = build_training_batch(
                        llama, tokenizer, projector, img, text, device,
                    )

                    # Forward: frozen Llama, grad flows back through visual_embeds
                    out = llama(
                        inputs_embeds  = inputs_embeds,
                        attention_mask = attn_mask,
                        labels         = labels,
                    )
                    raw_loss = out.loss
                    loss = raw_loss / args.grad_accum
                    loss.backward()
                    n_acc += 1

                    if n_acc % args.grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(
                            projector.trainable_parameters(), max_norm=1.0,
                        )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        step += 1
                        losses.append(raw_loss.item())

                        if step % args.log_every == 0:
                            avg_loss = sum(losses[-args.log_every:]) / min(len(losses), args.log_every)
                            lr_now   = scheduler.get_last_lr()[0]
                            elapsed  = (time.perf_counter() - t0) / 60
                            eta_min  = elapsed * (args.max_steps - step) / max(step - start_step, 1)
                            logger.info(
                                f"Step {step}/{args.max_steps} | "
                                f"loss={avg_loss:.4f} | lr={lr_now:.2e} | "
                                f"elapsed={elapsed:.1f}min | ETA={eta_min:.1f}min"
                            )
                            csv_w.writerow([step, round(avg_loss, 4),
                                            f"{lr_now:.2e}", round(elapsed, 2)])
                            csv_f.flush()

                        if step % args.save_every == 0:
                            save_checkpoint(
                                projector, optimizer, scheduler, step,
                                f"models/visual_projector/checkpoint-{step}.pt",
                            )

                    del inputs_embeds, attn_mask, labels, out, loss
                    gc.collect()
                    torch.cuda.empty_cache()

                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM at step {step} — skipping sample")
                    optimizer.zero_grad()
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    logger.warning(f"Sample skipped at step {step}: {type(e).__name__}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Stream disconnected: {e}. Reconnecting in 5s...")
            time.sleep(5)
            continue

    # ── Final save ─────────────────────────────────────────────────────
    save_checkpoint(
        projector, optimizer, scheduler, step,
        f"models/visual_projector/checkpoint-{step}.pt",
    )
    projector_final = Path("models/visual_projector/projector.safetensors")
    state = {k: v.cpu() for k, v in projector.resampler.state_dict().items()}
    torch.save(state, projector_final)
    logger.info(f"Final projector saved to {projector_final}")

    csv_f.close()
    total_min = (time.perf_counter() - t0) / 60
    logger.info(f"Done. Trained {step - start_step} steps in {total_min:.1f} min.")


if __name__ == "__main__":
    main()