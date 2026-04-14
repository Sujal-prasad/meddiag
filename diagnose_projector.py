"""Compare loss with trained projector vs zeroed-out projector."""
import torch, sys, os
sys.path.insert(0, os.getcwd())
from src.vision_encoder import VisualProjector
from src.multimodal_fusion import _get_llama_embed_tokens
from src.data_loader import StreamingDatasetManager
from experiments.train_projector import build_training_batch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = "cuda"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok.pad_token = tok.eos_token
llama = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=bnb, device_map="auto", attn_implementation="sdpa")
llama.eval()

proj = VisualProjector(llama_hidden=3072, n_visual_tokens=8).to(device)
state = torch.load("models/visual_projector/projector.safetensors",
                   map_location="cpu", weights_only=True)
proj.resampler.load_state_dict(state)
proj.eval()

loader = StreamingDatasetManager()
samples = []
for s in loader.stream("mimic_reports", max_samples=10):
    if s.get("image_pil") and len((s.get("text") or "").strip()) > 50:
        samples.append(s)
    if len(samples) == 5:
        break

trained_losses, zero_losses = [], []
for s in samples:
    # Trained projector
    ie, am, lb = build_training_batch(llama, tok, proj, s["image_pil"], s["text"], device)
    with torch.no_grad():
        trained_losses.append(llama(inputs_embeds=ie, attention_mask=am, labels=lb).loss.item())

    # Zero visual tokens (simulates no-vision baseline)
    class ZeroProj:
        def preprocess(self, imgs): return proj.preprocess(imgs)
        def __call__(self, pv):
            return torch.zeros(pv.shape[0], 8, 3072, device=device, dtype=torch.float32)
    ie, am, lb = build_training_batch(llama, tok, ZeroProj(), s["image_pil"], s["text"], device)
    with torch.no_grad():
        zero_losses.append(llama(inputs_embeds=ie, attention_mask=am, labels=lb).loss.item())

import statistics as st
print(f"\nTrained projector avg loss:  {st.mean(trained_losses):.4f}")
print(f"Zero-visual-tokens avg loss: {st.mean(zero_losses):.4f}")
print(f"Improvement from vision:     {st.mean(zero_losses) - st.mean(trained_losses):.4f}")
print(f"\nPer-sample:")
for i, (t, z) in enumerate(zip(trained_losses, zero_losses)):
    print(f"  Sample {i+1}: trained={t:.3f} | zero={z:.3f} | delta={z-t:+.3f}")