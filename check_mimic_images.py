from src.data_loader import StreamingDatasetManager
from PIL import Image
import time

loader = StreamingDatasetManager()
print("Streaming 10 MIMIC-CXR samples with images + findings text...")
t0 = time.perf_counter()
count_with_both = 0
count_total = 0

for i, sample in enumerate(loader.stream("mimic_reports", max_samples=10)):
    count_total += 1
    has_image = sample.get("image_pil") is not None
    has_text  = bool(sample.get("text", "").strip())
    print(f"  [{i+1}/10] image={has_image} | text_len={len(sample.get('text', ''))} | labels={sample.get('labels', [])[:2]}")
    if has_image and has_text:
        count_with_both += 1
        if i == 0:
            img = sample["image_pil"]
            print(f"       first image: {img.size}, mode={img.mode}")
            print(f"       first text (80 chars): {sample['text'][:80]}...")

elapsed = time.perf_counter() - t0
print(f"\nResult: {count_with_both}/{count_total} samples have BOTH image + text")
print(f"Stream rate: {count_total/elapsed:.1f} samples/sec")
print(f"Estimated time for 5000 samples: {5000/max(count_total/elapsed, 0.1)/60:.1f} minutes")