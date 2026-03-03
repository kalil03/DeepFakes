"""Extract SDXL images from single parquet and Midjourney images from raw folder."""
import os
import io
import shutil
import random
import pyarrow.parquet as pq
from PIL import Image
from glob import glob

SDXL_PARQUET = 'sdxl_raw/data/train-00000-of-00001.parquet'
DEST_TRAIN_SDXL = 'multi_dataset/train/4_stable_diffusion'
DEST_TEST_SDXL  = 'multi_dataset/test/4_stable_diffusion'

MJ_DIR = 'midjourney_raw/images'
DEST_TRAIN_MJ = 'multi_dataset/train/3_midjourney'
DEST_TEST_MJ  = 'multi_dataset/test/3_midjourney'

os.makedirs(DEST_TRAIN_SDXL, exist_ok=True)
os.makedirs(DEST_TEST_SDXL, exist_ok=True)
os.makedirs(DEST_TRAIN_MJ, exist_ok=True)
os.makedirs(DEST_TEST_MJ, exist_ok=True)

# 1. SDXL
if os.path.exists(SDXL_PARQUET):
    print("Processing SDXL parquet...")
    table = pq.read_table(SDXL_PARQUET, columns=["image"])
    col = table.column("image")
    n = len(col)
    
    # We want max 10k SDXL images to keep it balanced
    target_n = min(n, 10000)
    indices = random.sample(range(n), target_n)
    indices_set = set(indices)
    
    train_c, test_c = 0, 0
    for i in range(n):
        if i not in indices_set: continue
        row = col[i].as_py()
        if not (isinstance(row, dict) and row.get("bytes")): continue
        try:
            img = Image.open(io.BytesIO(row["bytes"])).convert("RGB")
            # 80/20 train/test
            if random.random() < 0.2:
                img.save(os.path.join(DEST_TEST_SDXL, f"sdxl_{i:05d}.jpg"), "JPEG", quality=90)
                test_c += 1
            else:
                img.save(os.path.join(DEST_TRAIN_SDXL, f"sdxl_{i:05d}.jpg"), "JPEG", quality=90)
                train_c += 1
        except Exception as e:
            pass
    print(f"SDXL Extracted: Train {train_c} | Test {test_c}")

# 2. Midjourney
if os.path.exists(MJ_DIR):
    print("\nCopying Midjourney images...")
    mj_imgs = glob(os.path.join(MJ_DIR, '*.*'))
    random.shuffle(mj_imgs)
    train_c, test_c = 0, 0
    for img_path in mj_imgs:
        fname = os.path.basename(img_path)
        if random.random() < 0.2:
            shutil.copy2(img_path, os.path.join(DEST_TEST_MJ, fname))
            test_c += 1
        else:
            shutil.copy2(img_path, os.path.join(DEST_TRAIN_MJ, fname))
            train_c += 1
    print(f"Midjourney Copied: Train {train_c} | Test {test_c}")

print("\nDone parsing AI datasets.")
