"""
vision_encoder.py — MobileViT + Perceiver Projector for Edge VLM
==================================================================
Lightweight vision stack for the medical diagnostic VLM.

Design:
    Image (3, 224, 224)
        |
        v
    MobileViT-XXS  (frozen, ~1.3M params, ~5MB weights)
        |
        v
    Feature map (B, 320, 7, 7) -> flatten -> (B, 49, 320)
        |
        v
    PerceiverResampler (TRAINABLE, ~1M params)
        8 learned query tokens, cross-attend to the 49 spatial tokens
        |
        v
    Visual tokens (B, 8, 3072)   <-- matches Llama-3.2-3B hidden_size
        |
        v
    Prepended to Llama text embeddings in pipeline.py

VRAM budget on RTX 5060 (8GB):
    MobileViT-XXS inference : ~80MB
    Projector forward       : ~15MB
    Activations (train)     : ~200MB
    Total overhead          : ~300MB on top of Llama (well under 8GB)

Why MobileViT-XXS and not full ViT:
    ViT-B/16 at 224 = 86M params and ~350MB VRAM. MobileViT-XXS achieves
    comparable ImageNet top-1 (69%) at 1.3M params. For chest X-rays
    (single channel replicated to 3ch, low-frequency features) the extra
    capacity of ViT is wasted — MobileViT's CNN stem captures edges and
    densities efficiently.

Why 8 visual tokens and not 1:
    One pooled token collapses all spatial info — model cannot tell
    left lung from right. 49 raw tokens (7x7) use too much context
    length. Perceiver resampler to 8 is the LLaVA-1.5 / BLIP-2 sweet
    spot — preserves spatial signal at ~3% of Llama's 1024 ctx budget.

Usage:
    from src.vision_encoder import VisionEncoder, VisualProjector

    encoder   = VisionEncoder().cuda().eval()
    projector = VisualProjector(n_visual_tokens=8, llama_hidden=3072).cuda()

    # pixel_values: (B, 3, 224, 224) float tensor, ImageNet-normalised
    with torch.no_grad():
        features = encoder(pixel_values)          # (B, 49, 320)
    visual_tokens = projector(features)           # (B, 8, 3072)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, MobileViTModel

logger = logging.getLogger("VisionEncoder")


# ─────────────────────────────────────────────────────────────────────────────
# VISION ENCODER — frozen MobileViT-XXS
# ─────────────────────────────────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    """
    Wraps HuggingFace MobileViT-XXS. Frozen during all training stages —
    we only train the projector on top of its features.

    Output: spatial feature tokens (B, 49, 320) suitable for cross-attention.
    """

    MODEL_ID = "apple/mobilevit-xx-small"
    HIDDEN_SIZE = 320   # MobileViT-XXS last stage channel count

    def __init__(self, freeze: bool = True):
        super().__init__()
        logger.info(f"Loading {self.MODEL_ID} (frozen={freeze})...")

        # AutoImageProcessor handles: resize→256, center_crop→256, normalize.
        # We override the processor to use 224 to match standard medical preprocessing.
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.processor.size = {"shortest_edge": 224}
        self.processor.crop_size = {"height": 224, "width": 224}

        # Load in fp32 then cast — MobileViT is small enough to stay fp32 easily.
        # Do NOT use 4-bit quantization here; it hurts small-model accuracy badly.
        self.backbone = MobileViTModel.from_pretrained(self.MODEL_ID)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self._frozen = freeze

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Convert PIL images to a normalised (B, 3, 224, 224) tensor.
        Use this on CPU before moving to GPU.

        MIMIC-CXR images are single-channel grayscale. We replicate to 3 channels
        so MobileViT's ImageNet-trained stem works. This is the standard approach
        used by LLaVA-Rad and CheXagent.
        """
        # Ensure RGB (replicate grayscale to 3 channels)
        rgb = [img.convert("RGB") for img in images]
        out = self.processor(images=rgb, return_tensors="pt")
        return out["pixel_values"]  # (B, 3, 224, 224)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224) normalised tensor on same device as model.

        Returns:
            spatial_tokens: (B, 49, 320) feature tokens — 7x7 grid flattened.
        """
        # Always no_grad when frozen — saves activation memory during training.
        ctx = torch.no_grad() if self._frozen else torch.enable_grad()
        with ctx:
            out = self.backbone(pixel_values=pixel_values)

        # last_hidden_state: (B, 320, 7, 7) — spatial feature map
        feat = out.last_hidden_state                  # (B, C, H, W)
        B, C, H, W = feat.shape
        assert C == self.HIDDEN_SIZE, f"Expected {self.HIDDEN_SIZE} channels, got {C}"

        # Flatten spatial dims to token sequence: (B, 49, 320)
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        return tokens

    def train(self, mode: bool = True):
        """Override: stay in eval mode even when parent .train() is called."""
        super().train(mode)
        if self._frozen:
            self.backbone.eval()
        return self


# ─────────────────────────────────────────────────────────────────────────────
# PERCEIVER RESAMPLER — trainable projector
# ─────────────────────────────────────────────────────────────────────────────

class PerceiverResampler(nn.Module):
    """
    Compresses variable-length visual features into a fixed number of learned
    query tokens via cross-attention. Works in a smaller inner_dim (768) to
    keep params manageable, then projects up to llama_hidden (3072) at the end.

    With defaults: ~3-4M trainable params (vs 152M if we worked in 3072 throughout).
    """

    def __init__(
        self,
        input_dim:    int = 320,     # MobileViT output dim
        llama_hidden: int = 3072,    # Llama-3.2-3B hidden_size
        n_queries:    int = 8,
        n_heads:      int = 8,
        n_layers:     int = 1,
        ffn_mult:     int = 1,
        inner_dim:    int = 768,     # small working space
    ):
        super().__init__()
        self.n_queries    = n_queries
        self.llama_hidden = llama_hidden
        self.inner_dim    = inner_dim

        # 320 -> 768: project MobileViT features into the small working space
        self.input_proj = nn.Linear(input_dim, inner_dim)

        # Learned query tokens in inner_dim
        self.queries = nn.Parameter(torch.randn(n_queries, inner_dim) * 0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_q":  nn.LayerNorm(inner_dim),
                "norm_kv": nn.LayerNorm(inner_dim),
                "attn":    nn.MultiheadAttention(
                    embed_dim=inner_dim,
                    num_heads=n_heads,
                    batch_first=True,
                ),
                "norm_ff": nn.LayerNorm(inner_dim),
                "ffn":     nn.Sequential(
                    nn.Linear(inner_dim, inner_dim * ffn_mult),
                    nn.GELU(),
                    nn.Linear(inner_dim * ffn_mult, inner_dim),
                ),
            })
            for _ in range(n_layers)
        ])

        # 768 -> 3072: final projection into Llama's embedding space
        self.output_proj = nn.Linear(inner_dim, llama_hidden)
        self.final_norm  = nn.LayerNorm(llama_hidden)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, N_in, input_dim)  e.g. (B, 49, 320)

        Returns:
            visual_tokens: (B, n_queries, llama_hidden)  e.g. (B, 8, 3072)
        """
        B = features.shape[0]
        kv = self.input_proj(features)                          # (B, 49, 768)
        q  = self.queries.unsqueeze(0).expand(B, -1, -1)        # (B, 8, 768)

        for layer in self.layers:
            q_norm  = layer["norm_q"](q)
            kv_norm = layer["norm_kv"](kv)
            attn_out, _ = layer["attn"](q_norm, kv_norm, kv_norm, need_weights=False)
            q = q + attn_out
            q = q + layer["ffn"](layer["norm_ff"](q))

        q = self.output_proj(q)                                 # (B, 8, 3072)
        return self.final_norm(q)

# ─────────────────────────────────────────────────────────────────────────────
# COMBINED: VisualProjector — what pipeline.py imports
# ─────────────────────────────────────────────────────────────────────────────

class VisualProjector(nn.Module):
    """
    End-to-end image -> visual tokens module.
    Wraps VisionEncoder (frozen) + PerceiverResampler (trainable).

    pipeline.py only needs to instantiate this and call .forward(pixel_values).
    """

    def __init__(
        self,
        llama_hidden: int = 3072,
        n_visual_tokens: int = 8,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder   = VisionEncoder(freeze=freeze_encoder)
        self.resampler = PerceiverResampler(
            input_dim    = self.encoder.HIDDEN_SIZE,
            llama_hidden = llama_hidden,
            n_queries    = n_visual_tokens,
        )
        self.n_visual_tokens = n_visual_tokens

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224) on GPU.

        Returns:
            visual_tokens: (B, n_visual_tokens, llama_hidden)
        """
        features = self.encoder(pixel_values)       # (B, 49, 320)
        tokens   = self.resampler(features)         # (B, 8, 3072)
        return tokens

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """Pass-through to encoder's PIL preprocessing."""
        return self.encoder.preprocess(images)

    def trainable_parameters(self):
        """Only resampler params are trainable (encoder is frozen)."""
        return self.resampler.parameters()

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST — run directly to verify everything loads and runs
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("VisualProjector smoke test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Build the projector
    proj = VisualProjector(llama_hidden=3072, n_visual_tokens=8).to(device)
    print(f"Trainable params: {proj.num_trainable():,}")
    total = sum(p.numel() for p in proj.parameters())
    print(f"Total params:     {total:,}")
    print(f"Trainable %:      {proj.num_trainable() / total * 100:.2f}%")

    # 2. Make a fake PIL grayscale image (mimics an X-ray)
    import numpy as np
    fake = Image.fromarray(
        (np.random.rand(512, 512) * 255).astype(np.uint8), mode="L"
    )
    print(f"\nTest image: {fake.size}, mode={fake.mode}")

    # 3. Preprocess
    pixel_values = proj.preprocess([fake, fake]).to(device)   # batch of 2
    print(f"Preprocessed shape: {tuple(pixel_values.shape)}")

    # 4. Forward
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(0)
    with torch.no_grad():
        out = proj(pixel_values)
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Output dtype: {out.dtype}")
    print(f"Output norm (per token): {out.norm(dim=-1).mean().item():.3f}")

    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(0) / 1024**2
        print(f"\nPeak VRAM during forward: {peak_mb:.1f} MB")

    print("\n[OK] Vision encoder works. Ready for pipeline.py integration.")