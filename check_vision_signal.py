"""Check if model has vision signal despite greedy-decode bias toward ABNORMAL."""
import sys, os, torch
sys.path.insert(0, os.getcwd())

from src.pipeline import EdgeMedicalVLM, InferenceConfig
from src.multimodal_fusion import build_multimodal_inputs
from src.data_loader import StreamingDatasetManager

cfg = InferenceConfig()
cfg.adapter_path = "models/qlora_adapters/meddiag_lora_vlm"
vlm = EdgeMedicalVLM(infer_cfg=cfg)

# Find token ids for "NORMAL" and "ABNORMAL" first tokens
tok = vlm.manager.tokenizer
normal_ids   = tok.encode("NORMAL",   add_special_tokens=False)
abnormal_ids = tok.encode("ABNORMAL", add_special_tokens=False)
print(f"NORMAL token ids:   {normal_ids}")
print(f"ABNORMAL token ids: {abnormal_ids}")
# They likely share a prefix — we need the FIRST differentiating token
# In Llama tokenizer: "NORMAL" = [38868], "ABNORMAL" = [1905, 34536] usually
# Or "AB" is a common prefix. Inspect and pick the distinguishing id.

loader = StreamingDatasetManager()
samples = []
for s in loader.stream("nih", max_samples=None):
    if s.get("image_pil") is None: continue
    labels = [l.lower() for l in s.get("labels", [])]
    if any("normal" in l and "pneumonia" not in l for l in labels) and len([x for x in samples if x[1]=="NORMAL"]) < 5:
        samples.append((s, "NORMAL"))
    elif any("pneumonia" in l for l in labels) and len([x for x in samples if x[1]=="PNEUMONIA"]) < 5:
        samples.append((s, "PNEUMONIA"))
    if len(samples) >= 10: break

print(f"\nGot {len(samples)} samples")

# For each sample, compute P(ABNORMAL) / P(NORMAL) at first decision token
for i, (sample, gt) in enumerate(samples):
    inputs_embeds, attn_mask = build_multimodal_inputs(
        llama_model=vlm.manager.model,
        tokenizer=tok,
        projector=vlm.visual_projector,
        pil_image=sample["image_pil"],
        system_prompt=vlm.CLASSIFY_SYSTEM_PROMPT,
        user_prompt=vlm.CLASSIFY_USER_PROMPT,
        device=str(vlm.manager.model.device),
    )
    with torch.inference_mode():
        out = vlm.manager.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
    # Logits at last position = next-token distribution
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    p_abnormal = probs[abnormal_ids[0]].item()
    p_normal   = probs[normal_ids[0]].item()
    print(f"[{i+1}] GT={gt:10s}  P(AB)={p_abnormal:.4f}  P(N)={p_normal:.4f}  ratio={p_abnormal/max(p_normal,1e-9):.2f}")