"""
Training script: DenseNet121 feature extraction → MLP classifier.

Usage:
    python train_mlp.py

    # AMD ROCm:
    USE_ROCM=1 python train_mlp.py
"""

import os
from pathlib import Path
import time
import copy
import warnings
from typing import List, Tuple

# ROCm environment (only set when actually using AMD GPU locally)
if os.environ.get("USE_ROCM", "0") == "1":
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("ROCM_PATH", "/opt/rocm")

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    log_loss,
)
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm.auto import tqdm
from PIL import Image, ImageFile

from model import DenseNetExtractor, train_transform, eval_transform, DEVICE

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Configuration ────────────────────────────────────────────────────
BASE_DIR = Path("multi_dataset")
BATCH_SIZE = 64
RANDOM_SEED = 42


# ── Dataset ──────────────────────────────────────────────────────────
class PathsDataset(Dataset):
    def __init__(self, paths: List[Path], labels: List[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ── Helpers ──────────────────────────────────────────────────────────
def build_label_encoder(slugs: List[str]) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(sorted(slugs))
    return le


def collect_split(
    split: str,
    label_encoder: LabelEncoder,
    img_exts=(".jpg", ".jpeg", ".png", ".webp"),
) -> Tuple[List[Path], List[int]]:
    paths: List[Path] = []
    labels: List[int] = []

    for slug in sorted(d.name for d in BASE_DIR.iterdir() if d.is_dir()):
        split_dir = BASE_DIR / slug / split
        if not split_dir.exists():
            continue
        for p in split_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in img_exts:
                paths.append(p)
                label = int(label_encoder.transform([slug])[0])
                labels.append(label)

    return paths, labels


def extract_features(
    extractor: DenseNetExtractor,
    paths: List[Path],
    labels: List[int],
    transform,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract DenseNet121 features from images using a pre-loaded extractor."""
    if not paths:
        raise RuntimeError(f"No images found for split '{split_name}'.")

    dataset = PathsDataset(paths, labels, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
    )

    feats, labs = [], []
    start = time.time()
    with torch.no_grad():
        for imgs, ys in tqdm(loader, desc=f"Extracting features [{split_name}]"):
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                f = extractor(imgs)
            feats.append(f.cpu().numpy())
            labs.append(ys.numpy())

    X = np.vstack(feats)
    y = np.concatenate(labs)
    elapsed = time.time() - start
    print(f"  [{split_name}] done: {X.shape} in {elapsed:.1f}s")
    return X, y


# ── Main ─────────────────────────────────────────────────────────────
def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if not BASE_DIR.exists():
        raise RuntimeError(
            f"Directory '{BASE_DIR}' not found. "
            "Make sure to run `python build_multi_dataset.py` first."
        )

    # Discover classes (slugs) from directories
    slugs = sorted(d.name for d in BASE_DIR.iterdir() if d.is_dir())
    if not slugs:
        raise RuntimeError(f"No classes found in {BASE_DIR}.")

    print(f"Device: {DEVICE}")
    print(f"Classes (slugs): {slugs}")

    label_encoder = build_label_encoder(slugs)

    # Collect path/label lists
    print("\n[collect] paths per split...")
    train_paths, train_labels = collect_split("train", label_encoder)
    val_paths, val_labels = collect_split("val", label_encoder)
    test_paths, test_labels = collect_split("test", label_encoder)

    print(f"  train: {len(train_paths)} images")
    print(f"  val:   {len(val_paths)} images")
    print(f"  test:  {len(test_paths)} images")

    # Create extractor ONCE (reused for all splits)
    print("\n[features] Loading DenseNet121 extractor...")
    extractor = DenseNetExtractor().to(DEVICE)
    extractor.eval()

    # Feature extraction
    print("[features] Extracting DenseNet121 features...")
    X_train, y_train = extract_features(
        extractor, train_paths, train_labels, train_transform, "train"
    )
    X_val, y_val = extract_features(
        extractor, val_paths, val_labels, eval_transform, "val"
    )
    X_test, y_test = extract_features(
        extractor, test_paths, test_labels, eval_transform, "test"
    )

    # Free GPU memory after feature extraction
    del extractor
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # StandardScaler (fit on train only)
    print("\n[scaler] Fitting StandardScaler on train set...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # MLP training (scikit-learn) with manual early stopping
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=1,
        warm_start=True,
        random_state=RANDOM_SEED,
    )

    best_loss = float("inf")
    best_weights = None
    best_intercepts = None
    patience = 10
    no_improve = 0
    max_epochs = 500

    print("\n[train] MLP with early stopping (val log-loss)...")

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        epoch_iter = tqdm(range(max_epochs), desc="Training MLP")
        for epoch in epoch_iter:
            mlp.fit(X_train_s, y_train)

            # Training loss (MLPClassifier exposes loss_ from last fit)
            train_loss = getattr(mlp, "loss_", np.nan)

            # Validation metrics
            val_probs = mlp.predict_proba(X_val_s)
            val_loss = log_loss(y_val, val_probs)
            val_pred = np.argmax(val_probs, axis=1)
            val_acc = accuracy_score(y_val, val_pred)

            epoch_iter.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_acc": f"{val_acc:.4f}",
                    "no_improve": f"{no_improve}/{patience}",
                }
            )

            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )

            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                no_improve = 0
                best_weights = copy.deepcopy(mlp.coefs_)
                best_intercepts = copy.deepcopy(mlp.intercepts_)
            else:
                no_improve += 1

            if no_improve >= patience:
                print(
                    f"\n[early stopping] Stopped at epoch {epoch}. "
                    "Restoring best weights (lowest val_loss)."
                )
                mlp.coefs_ = best_weights
                mlp.intercepts_ = best_intercepts
                break

    print(f"\n[train] Completed in {time.time() - t0:.1f}s")

    # Final evaluation on test set
    print("\n[eval] Test set evaluation...")
    y_pred = mlp.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall accuracy (test): {acc:.4%}")

    class_names = list(label_encoder.classes_)
    print("\nClassification report (test):")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4,
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    print("\nPer-class accuracy (test):")
    for name, a in zip(class_names, per_class_acc):
        print(f"  {name:16s}: {a:.4%}")

    # Normalized confusion matrix
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix — MLP ({acc:.2%})")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Save artifacts
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    joblib.dump(mlp, "model_mlp.pkl")
    print("\n[save] Artifacts saved: model_mlp.pkl, scaler.pkl, label_encoder.pkl")


if __name__ == "__main__":
    main()
