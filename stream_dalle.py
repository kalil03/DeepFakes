"""Stream extract 10k DALL-E 3 images from a remote TAR file without downloading the whole archive."""
import os
import requests
import tarfile
import io
from PIL import Image

URL = "https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions/resolve/main/data/data-000000.tar"
DEST_TRAIN = 'multi_dataset/train/2_dalle'
DEST_TEST  = 'multi_dataset/test/2_dalle'

os.makedirs(DEST_TRAIN, exist_ok=True)
os.makedirs(DEST_TEST, exist_ok=True)

print(f"Connecting to {URL}...")
req = requests.get(URL, stream=True)
req.raise_for_status()

print("Streaming tar file (extracting 10,000 images)...")
try:
    # Use mode 'r|' for streaming without seek
    tar = tarfile.open(fileobj=req.raw, mode="r|*")
    
    count = 0
    train_c = 0
    test_c = 0
    
    for member in tar:
        if member.isfile() and (member.name.lower().endswith(".jpg") or member.name.lower().endswith(".png") or member.name.lower().endswith(".webp")):
            f = tar.extractfile(member)
            if f:
                try:
                    img_bytes = f.read()
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    
                    # 80/20 train/test split
                    if count % 5 == 0:
                        img.save(os.path.join(DEST_TEST, f"dalle_{count:05d}.jpg"), "JPEG", quality=90)
                        test_c += 1
                    else:
                        img.save(os.path.join(DEST_TRAIN, f"dalle_{count:05d}.jpg"), "JPEG", quality=90)
                        train_c += 1
                        
                    count += 1
                    
                    if count % 500 == 0:
                        print(f"Extracted {count}/10000 (Train: {train_c}, Test: {test_c})")
                        
                    if count >= 10000:
                        print("Reached 10,000 images! Stopping stream.")
                        break
                except Exception as e:
                    print(f"  skip file exception: {e}")
                    
except EOFError:
    print("Reached end of tar file earlier than expected.")
except Exception as e:
    print(f"Stream error: {e}")

try:
    tar.close()
    req.close()
except:
    pass

print(f"\nDone! Extracted total {count} DALL-E 3 images (Train: {train_c} | Test: {test_c})")
