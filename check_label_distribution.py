"""Check what labels build_target produces for the first 150 samples."""
import sys, os, random
sys.path.insert(0, os.getcwd())
from src.data_loader import StreamingDatasetManager
from experiments.train_lora import build_target, round_robin_stream

rng = random.Random(42)
loader = StreamingDatasetManager()
datasets = ["nih", "chexpert", "mimic_reports"]
stream = round_robin_stream(loader, datasets, rng)

counts = {ds: {"normal": 0, "abnormal": 0, "skipped": 0} for ds in datasets}
n_checked = 0
MAX_CHECK = 150

for sample in stream:
    if n_checked >= MAX_CHECK:
        break
    n_checked += 1
    ds = sample["_dataset"]
    labels = sample.get("labels", [])
    text = (sample.get("text") or sample.get("report") or "").strip()

    label, target = build_target(ds, labels, text, rng)
    if label is None:
        counts[ds]["skipped"] += 1
        if n_checked <= 20:
            print(f"  SKIPPED [{ds}] labels={labels[:1]} text_len={len(text)}")
    elif label == 0:
        counts[ds]["normal"] += 1
    else:
        counts[ds]["abnormal"] += 1

print(f"\n=== After {n_checked} samples ===")
for ds in datasets:
    c = counts[ds]
    total = c["normal"] + c["abnormal"] + c["skipped"]
    print(f"{ds:20s}: NORMAL={c['normal']:3d} | ABNORMAL={c['abnormal']:3d} | "
          f"SKIPPED={c['skipped']:3d} | (total {total})")

total_n = sum(c["normal"] for c in counts.values())
total_a = sum(c["abnormal"] for c in counts.values())
print(f"\nGRAND TOTAL: NORMAL={total_n} | ABNORMAL={total_a}")
if total_a == 0:
    print("\n[!] ZERO abnormal samples generated. Training cannot learn.")
elif total_a < total_n / 4:
    print(f"\n[!] Severe imbalance: only {total_a/(total_a+total_n)*100:.1f}% ABNORMAL.")