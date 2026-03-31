"""
Download the raw datasets used by build_multi_dataset.py.

This downloads two Kaggle datasets containing real and GAN-generated faces:
  - 140k Real and Fake Faces (StyleGAN)
  - Deepfake and Real Images (OpenForensics)

Requirements:
  - Kaggle CLI configured (~/.kaggle/kaggle.json)

Usage:
    python download_datasets.py
"""

import subprocess
from pathlib import Path

RAW_ROOT = Path("datasets") / "raw"
RAW_ROOT.mkdir(parents=True, exist_ok=True)


def run(cmd, cwd=None):
    print(f"\n[cmd] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Command failed: {e}")


def download_kaggle_dataset(dataset_id: str, target_dir: Path, unzip: bool = True):
    """Download a dataset from Kaggle using the official CLI."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[+] Kaggle :: {dataset_id} -> {target_dir}")

    args = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_id,
        "-p",
        str(target_dir),
    ]
    if unzip:
        args.append("--unzip")

    run(args)


def main():
    kaggle_root = RAW_ROOT / "kaggle"

    # 140k Real and Fake Faces (StyleGAN generated fakes)
    download_kaggle_dataset(
        "xhlulu/140k-real-and-fake-faces",
        kaggle_root / "140k-real-and-fake-faces",
    )

    # Deepfake and Real Images (OpenForensics — various GAN manipulations)
    download_kaggle_dataset(
        "manjilkarki/deepfake-and-real-images",
        kaggle_root / "deepfake-and-real-images",
    )

    print("\n[OK] Downloads complete.")
    print("Now run `python build_multi_dataset.py` to build the balanced dataset.")


if __name__ == "__main__":
    main()
