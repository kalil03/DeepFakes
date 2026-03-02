from datasets import load_dataset
import os
from collections import defaultdict
import random

output_dir = "/home/kalilzera/Documentos/DeepFakes/new_dataset"
os.makedirs(os.path.join(output_dir, "real"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "fake"), exist_ok=True)

print("Downloading the validation split (9,000 images, ~680MB)...")
# Load the parquet files directly to bypass HF Dataset Builder split verification errors
ds = load_dataset(
    "parquet", 
    data_files={"validation": "hf://datasets/Rajarshi-Roy-research/Defactify_Image_Dataset/data/validation-*.parquet"},
    split="validation"
)

TARGET_REALS = 2500
TARGET_FAKES_PER_GEN = 500
TARGET_TOTAL_FAKES = 2500

reals_count = 0
fakes_counts = defaultdict(int)
total_fakes = 0
total_processed = 0

print(f"Downloaded split of {len(ds)} images. Extracting target samples...")

# Shuffle to get a random subset if needed, though sequential is fine for this test
indices = list(range(len(ds)))
random.seed(42)
random.shuffle(indices)

for idx in indices:
    item = ds[idx]
    total_processed += 1
    label_a = item.get("Label_A", -1)
    
    if label_a == 0 and reals_count < TARGET_REALS:
        path = os.path.join(output_dir, "real", f"coco_{total_processed}.jpg")
        item["Image"].convert("RGB").save(path)
        reals_count += 1
        
    elif label_a == 1 and total_fakes < TARGET_TOTAL_FAKES:
        generator = f"gen_{item.get('Label_B', 'unknown')}"
        if fakes_counts[generator] < TARGET_FAKES_PER_GEN:
            path = os.path.join(output_dir, "fake", f"{generator}_{total_processed}.jpg")
            item["Image"].convert("RGB").save(path)
            fakes_counts[generator] += 1
            total_fakes += 1
            
    if reals_count >= TARGET_REALS and total_fakes >= TARGET_TOTAL_FAKES:
        break

print(f"Finished! Processed {total_processed} items. Saved R:{reals_count}, F:{total_fakes} {dict(fakes_counts)}")
