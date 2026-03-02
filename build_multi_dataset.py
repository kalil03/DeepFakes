import os
import shutil
import glob
import random

BASE_DIR = '/home/kalilzera/Documentos/DeepFakes/archive/real_vs_fake/real-vs-fake'
NEW_DS_DIR = '/home/kalilzera/Documentos/DeepFakes/new_dataset'
MULTI_DIR = '/home/kalilzera/Documentos/DeepFakes/multi_dataset'

classes = ['0_real', '1_gan', '2_dalle', '3_midjourney', '4_stable_diffusion']

print("Setting up mult-class symlink structure...")
for split in ['train', 'test']:
    for c in classes:
        os.makedirs(os.path.join(MULTI_DIR, split, c), exist_ok=True)

# 1. Real Images
print("Linking real images...")
# Kaggle Reals (train + test)
for split in ['train', 'test']:
    src_dir = os.path.join(BASE_DIR, split, 'real')
    dst_dir = os.path.join(MULTI_DIR, split, '0_real')
    for f in os.listdir(src_dir):
        if not os.path.exists(os.path.join(dst_dir, f)):
            os.symlink(os.path.join(src_dir, f), os.path.join(dst_dir, f))

# Defactify Reals
new_reals = list(glob.glob(os.path.join(NEW_DS_DIR, 'real', '*.jpg')))
random.seed(42)
random.shuffle(new_reals)
split_idx = int(0.8 * len(new_reals))
for i, src in enumerate(new_reals):
    split = 'train' if i < split_idx else 'test'
    fname = os.path.basename(src)
    dst = os.path.join(MULTI_DIR, split, '0_real', fname)
    if not os.path.exists(dst): os.symlink(src, dst)

# 2. GANs (Kaggle Fakes)
print("Linking GANs...")
for split in ['train', 'test']:
    src_dir = os.path.join(BASE_DIR, split, 'fake')
    dst_dir = os.path.join(MULTI_DIR, split, '1_gan')
    for f in os.listdir(src_dir):
        if not os.path.exists(os.path.join(dst_dir, f)):
            os.symlink(os.path.join(src_dir, f), os.path.join(dst_dir, f))

# 3. Modern Fakes (Defactify)
print("Linking Modern Fakes...")
new_fakes = list(glob.glob(os.path.join(NEW_DS_DIR, 'fake', '*.jpg')))
random.shuffle(new_fakes)
for fd in new_fakes:
    fname = os.path.basename(fd)
    # the filenames are like gen_X_idx.jpg
    if fname.startswith("gen_4"): c = '2_dalle'
    elif fname.startswith("gen_5"): c = '3_midjourney'
    else: c = '4_stable_diffusion' # gen 1, 2, 3
    
    # 80/20 train/test split via random chance since already shuffled
    split = 'train' if random.random() < 0.8 else 'test'
    dst = os.path.join(MULTI_DIR, split, c, fname)
    if not os.path.exists(dst): os.symlink(fd, dst)

for split in ['train', 'test']:
    print(f"\n[{split}]")
    for c in classes:
        p = os.path.join(MULTI_DIR, split, c)
        print(f"  {c}: {len(os.listdir(p))} imgs")
