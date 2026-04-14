"""Quick sanity check: does the trained VLM produce NORMAL/ABNORMAL for real X-rays?"""
import sys, os
sys.path.insert(0, os.getcwd())

from src.pipeline import EdgeMedicalVLM, InferenceConfig
from src.data_loader import StreamingDatasetManager

# Update InferenceConfig to use the new VLM LoRA path
cfg = InferenceConfig()
cfg.adapter_path = "models/qlora_adapters/meddiag_lora_vlm"

vlm = EdgeMedicalVLM(infer_cfg=cfg)

loader = StreamingDatasetManager()
print("\n" + "="*60)
print("TESTING 6 SAMPLES: 3 NORMAL, 3 PNEUMONIA")
print("="*60)

# Collect 3 of each
normals, pneumonias = [], []
for s in loader.stream("nih", max_samples=None):
    if s.get("image_pil") is None: continue
    labels = [l.lower() for l in s.get("labels", [])]
    if any("normal" in l and "pneumonia" not in l for l in labels) and len(normals) < 3:
        normals.append(s)
    elif any("pneumonia" in l for l in labels) and len(pneumonias) < 3:
        pneumonias.append(s)
    if len(normals) >= 3 and len(pneumonias) >= 3: break

for i, sample in enumerate(normals + pneumonias):
    gt = "NORMAL" if sample in normals else "PNEUMONIA"
    print(f"\n[{i+1}/6] Ground truth: {gt}")
    report = vlm.generate_diagnosis(
        image=sample["image_pil"],
        findings_query="",
    )
    # Show first 200 chars of output
    print(f"Model: {report[:200]}")
    # Quick classification extract
    rl = report.lower()
    pred = "ABNORMAL" if "abnormal" in rl[:100] else ("NORMAL" if "normal" in rl[:100] else "UNCLEAR")
    correct = (gt == "NORMAL" and pred == "NORMAL") or (gt == "PNEUMONIA" and pred == "ABNORMAL")
    print(f"Pred: {pred}  {'✓' if correct else '✗'}")