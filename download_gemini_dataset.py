"""Download bitmind/nano-banana via streaming and save images to multi_dataset."""
import os
import random
from datasets import load_dataset
from PIL import Image

DEST_TRAIN = 'multi_dataset/train/5_gemini'
DEST_TEST  = 'multi_dataset/test/5_gemini'
os.makedirs(DEST_TRAIN, exist_ok=True)
os.makedirs(DEST_TEST,  exist_ok=True)

print("[1/3] Streaming bitmind/nano-banana from Hugging Face...")
ds = load_dataset("bitmind/nano-banana", split="train", streaming=True)

all_images = []
for i, row in enumerate(ds):
    all_images.append(row["image"])
    if (i + 1) % 1000 == 0:
        print(f"  streamed {i + 1} images...")

n = len(all_images)
print(f"  Total images streamed: {n}")

print("[2/3] Splitting 80/20 and saving...")
indices = list(range(n))
random.seed(42)
random.shuffle(indices)
split = int(n * 0.8)
train_idx = set(indices[:split])

train_saved, test_saved = 0, 0
for i, img in enumerate(all_images):
    if not isinstance(img, Image.Image):
        continue
    img = img.convert("RGB")
    if i in train_idx:
        img.save(os.path.join(DEST_TRAIN, f"gemini_{i:05d}.jpg"), "JPEG", quality=95)
        train_saved += 1
    else:
        img.save(os.path.join(DEST_TEST, f"gemini_{i:05d}.jpg"), "JPEG", quality=95)
        test_saved += 1
    if (train_saved + test_saved) % 1000 == 0:
        print(f"  saved {train_saved + test_saved}/{n}")

print(f"\n[3/3] Summary:")
print(f"  Train: {train_saved} images in {DEST_TRAIN}")
print(f"  Test:  {test_saved} images in {DEST_TEST}")
print("Done!")
