"""
multimodal_fusion.py — Visual + Text Embedding Fusion for Llama-3.2
====================================================================
Core operation: take an image and a text prompt, produce the combined
`inputs_embeds` and `attention_mask` that Llama.generate() expects.

Why we need this:
    Llama normally does: input_ids -> embed_tokens -> hidden states
    We want:            [visual_embeds | text_embeds] -> hidden states

    There is no input_ids that represents "visual token" — visual tokens
    come from the projector, not the vocabulary. So we must:
      1. Tokenize text normally -> text_input_ids
      2. Run text_input_ids through Llama's embed_tokens -> text_embeds (B, T, 3072)
      3. Run image through projector -> visual_embeds (B, 8, 3072)
      4. Concatenate along sequence dim -> (B, 8+T, 3072)
      5. Build attention_mask of shape (B, 8+T) — all ones (no padding in single-sample case)
      6. Pass inputs_embeds + attention_mask to model.generate()

    Llama accepts `inputs_embeds` as an alternative to `input_ids`.
    When both are passed, `inputs_embeds` wins.

Dtype handling:
    Llama runs in bf16 (4-bit quantized weights dequantize to bf16 at compute time).
    Projector runs in fp32 by default.
    We cast visual_embeds to bf16 BEFORE concat so the fused tensor is bf16 throughout.

Chat template handling:
    Llama-3.2-Instruct expects a specific chat format with <|begin_of_text|>,
    <|start_header_id|>, etc. We build that format as text, tokenize it,
    embed it, and then splice visual tokens in AFTER the system prompt and
    BEFORE the user content. This mimics how LLaVA-1.5 does it.

    Layout:
        [BOS] [system_header] system_prompt [eot]
        [user_header] [VISUAL_TOKENS] user_prompt [eot]
        [assistant_header]

Usage:
    from src.multimodal_fusion import build_multimodal_inputs

    inputs_embeds, attention_mask = build_multimodal_inputs(
        llama_model    = vlm.manager.model,
        tokenizer      = vlm.manager.tokenizer,
        projector      = vlm.visual_projector,
        pil_image      = xray_pil,
        system_prompt  = "You are a radiology AI.",
        user_prompt    = "Analyze this chest X-ray.",
        device         = "cuda",
    )
    output_ids = llama_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=256,
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger("MultimodalFusion")


def _get_llama_embed_tokens(model) -> torch.nn.Module:
    """
    Retrieve Llama's embed_tokens module regardless of PEFT wrapping.

    Possible paths:
      - Raw model:              model.model.embed_tokens
      - PEFT-wrapped:           model.base_model.model.model.embed_tokens
      - PEFT + 4bit double wrap: same as above
    """
    # Try PEFT wrapper first
    if hasattr(model, "base_model"):
        try:
            return model.base_model.model.model.embed_tokens
        except AttributeError:
            pass
    # Try direct access
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    raise RuntimeError(
        "Could not locate embed_tokens on model. "
        "Inspect model structure with: print(model)"
    )


def build_multimodal_inputs(
    llama_model,
    tokenizer,
    projector,
    pil_image: Optional[Image.Image],
    system_prompt: str,
    user_prompt: str,
    device: str = "cuda",
    max_text_length: int = 768,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct `inputs_embeds` and `attention_mask` for Llama-3.2-Instruct
    with a visual token block spliced into the user turn.

    If `pil_image` is None → text-only fallback (no visual tokens, standard path).

    Returns:
        inputs_embeds:  (1, seq_len, hidden_size) bfloat16 on `device`
        attention_mask: (1, seq_len) int64 on `device`
    """
    embed_tokens = _get_llama_embed_tokens(llama_model)
    embed_dtype  = next(embed_tokens.parameters()).dtype   # usually bf16

    # ── Build the three text segments separately so we can splice in between ──
    # Segment A: [BOS] system block + start of user turn header
    # Segment B: user prompt content + [eot] + assistant header
    # Visual tokens go between A and B.
    segment_a = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
    )
    segment_b = (
        f"{user_prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    # ── Text-only fallback: no image provided ──
    if pil_image is None:
        full_text = segment_a + segment_b
        tok = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_text_length,
            add_special_tokens=False,   # we've already included <|begin_of_text|>
        ).to(device)
        with torch.no_grad():
            text_embeds = embed_tokens(tok["input_ids"]).to(embed_dtype)
        return text_embeds, tok["attention_mask"]

    # ── Multimodal path ──
    tok_a = tokenizer(
        segment_a, return_tensors="pt",
        truncation=True, max_length=max_text_length,
        add_special_tokens=False,
    ).to(device)
    tok_b = tokenizer(
        segment_b, return_tensors="pt",
        truncation=True, max_length=max_text_length,
        add_special_tokens=False,
    ).to(device)

    # Embed text segments through Llama's embed_tokens
    with torch.no_grad():
        embeds_a = embed_tokens(tok_a["input_ids"]).to(embed_dtype)   # (1, Ta, 3072)
        embeds_b = embed_tokens(tok_b["input_ids"]).to(embed_dtype)   # (1, Tb, 3072)

    # Process image through projector
    # projector.preprocess is on CPU, projector.forward is on GPU
    pixel_values = projector.preprocess([pil_image]).to(device)
    visual_embeds = projector(pixel_values).to(embed_dtype)           # (1, 8, 3072)

    # Concatenate: [A | visual | B]
    inputs_embeds = torch.cat([embeds_a, visual_embeds, embeds_b], dim=1)

    # Attention mask: all ones (no padding in single-sample inference)
    seq_len = inputs_embeds.shape[1]
    attention_mask = torch.ones(
        (1, seq_len), dtype=torch.long, device=device
    )

    return inputs_embeds, attention_mask


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────
# Run: python -m src.multimodal_fusion
#
# This test:
#   1. Loads Llama-3.2-3B in 4-bit
#   2. Loads VisualProjector (untrained)
#   3. Feeds a fake grayscale image + text through build_multimodal_inputs
#   4. Calls model.generate() with inputs_embeds
#   5. Prints the generated text
#
# Expected: generation produces SOMETHING coherent (even if off-topic, since
# the projector is untrained). If it crashes or produces pure gibberish
# (repeated tokens, special tokens), the fusion math is wrong.

if __name__ == "__main__":
    import sys
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("Multimodal Fusion Smoke Test")
    print("=" * 60)

    # Import our projector
    from src.vision_encoder import VisualProjector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load projector
    print("\n[1/4] Loading VisualProjector...")
    projector = VisualProjector(llama_hidden=3072, n_visual_tokens=8).to(device)
    projector.eval()
    print(f"      Projector params (trainable): {projector.num_trainable():,}")

    # 2. Load Llama in 4-bit
    print("\n[2/4] Loading Llama-3.2-3B-Instruct in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    llama = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="sdpa",
    )
    llama.eval()
    print("      Llama loaded.")

    # 3. Build fake grayscale X-ray
    fake_xray = Image.fromarray(
        (np.random.rand(512, 512) * 255).astype(np.uint8), mode="L"
    )
    print(f"\n[3/4] Built fake X-ray: {fake_xray.size}, mode={fake_xray.mode}")

    # 4. Fusion + generate
    print("\n[4/4] Building multimodal inputs and generating...")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(0)

    system_prompt = "You are a precise radiology AI assistant."
    user_prompt   = (
        "Analyze the chest X-ray above. "
        "State whether it appears NORMAL or ABNORMAL in one sentence."
    )

    inputs_embeds, attention_mask = build_multimodal_inputs(
        llama_model   = llama,
        tokenizer     = tokenizer,
        projector     = projector,
        pil_image     = fake_xray,
        system_prompt = system_prompt,
        user_prompt   = user_prompt,
        device        = device,
    )
    print(f"      inputs_embeds shape:   {tuple(inputs_embeds.shape)}")
    print(f"      inputs_embeds dtype:   {inputs_embeds.dtype}")
    print(f"      attention_mask shape:  {tuple(attention_mask.shape)}")

    with torch.inference_mode():
        output_ids = llama.generate(
            inputs_embeds  = inputs_embeds,
            attention_mask = attention_mask,
            max_new_tokens = 80,
            do_sample      = False,
            repetition_penalty = 1.15,
            pad_token_id   = tokenizer.eos_token_id,
        )

    # NOTE: when inputs_embeds is used, generate() returns ONLY new tokens
    # (no prompt prefix to strip, unlike with input_ids).
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("GENERATED OUTPUT:")
    print("=" * 60)
    print(generated)
    print("=" * 60)

    if device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"\nPeak VRAM: {peak_gb:.2f} GB")

    print("\n[OK] Fusion works end-to-end.")
    print("Note: output will be low-quality because projector is untrained.")
    print("      We just need coherent English, not correct diagnosis.")