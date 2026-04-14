from src.vision_encoder import VisualProjector
import torch

proj = VisualProjector(llama_hidden=3072, n_visual_tokens=8)

print("=== ENCODER (should all be frozen) ===")
enc_train = sum(p.numel() for p in proj.encoder.parameters() if p.requires_grad)
enc_total = sum(p.numel() for p in proj.encoder.parameters())
print(f"Encoder trainable: {enc_train:,} / {enc_total:,}")

print("\n=== RESAMPLER (should all be trainable) ===")
res_train = sum(p.numel() for p in proj.resampler.parameters() if p.requires_grad)
res_total = sum(p.numel() for p in proj.resampler.parameters())
print(f"Resampler trainable: {res_train:,} / {res_total:,}")

print("\n=== TOP-LEVEL (proj.parameters()) ===")
all_train = sum(p.numel() for p in proj.parameters() if p.requires_grad)
all_total = sum(p.numel() for p in proj.parameters())
print(f"All trainable: {all_train:,} / {all_total:,}")

print(f"\nproj.num_trainable() returns: {proj.num_trainable():,}")