"""Show what TARGET STRINGS (not just labels) were produced during training."""
import sys, os, random
sys.path.insert(0, os.getcwd())
from src.data_loader import StreamingDatasetManager
from experiments.train_lora import build_target, prefetch_balanced, mixed_sample_iterator

rng = random.Random(42)
loader = StreamingDatasetManager()

nih_pool  = prefetch_balanced(loader, "nih",      200)
chex_pool = prefetch_balanced(loader, "chexpert", 200)

sample_iter = mixed_sample_iterator(nih_pool, chex_pool, loader, rng)

target_counts = {"NORMAL": 0, "ABNORMAL": 0}
target_strings = {"NORMAL": [], "ABNORMAL": []}
n = 0

for sample in sample_iter:
    if n >= 300: break
    n += 1
    ds = sample["_dataset"]
    labels = sample.get("labels", [])
    text = (sample.get("text") or sample.get("report") or "").strip()
    label, target = build_target(ds, labels, text, rng)
    if target is None: continue
    key = "ABNORMAL" if target.startswith("ABNORMAL") else "NORMAL"
    target_counts[key] += 1
    if len(target_strings[key]) < 10:
        target_strings[key].append(f"[{ds}] {target[:100]}")

print(f"\n=== After {n} samples ===")
print(f"Targets starting with NORMAL:   {target_counts['NORMAL']}")
print(f"Targets starting with ABNORMAL: {target_counts['ABNORMAL']}")
print(f"Ratio: {target_counts['ABNORMAL']/(sum(target_counts.values()))*100:.1f}% ABNORMAL")

print("\n=== Sample NORMAL targets ===")
for t in target_strings["NORMAL"][:10]:
    print(f"  {t}")
print("\n=== Sample ABNORMAL targets ===")
for t in target_strings["ABNORMAL"][:10]:
    print(f"  {t}")