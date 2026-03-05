import os
from pathlib import Path
import time
from typing import List, Tuple

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["ROCM_PATH"] = "/opt/rocm"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121, DenseNet121_Weights

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    log_loss,
)
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm.auto import tqdm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------
# Configurações gerais
# ---------------------------------------------------------------------
BASE_DIR = Path("multi_dataset")
IMG_SIZE = 224
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42


class DenseNetExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = self.densenet.features

    def forward(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = F.adaptive_avg_pool2d(f, (1, 1))
        return torch.flatten(f, 1)


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


def get_transforms():
    # Apenas train recebe augmentação pesada
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3)],
                p=0.3,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_tf, eval_tf


def extract_features(
    paths: List[Path],
    labels: List[int],
    transform,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if not paths:
        raise RuntimeError(f"Nenhuma imagem encontrada para o split '{split_name}'.")

    dataset = PathsDataset(paths, labels, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = DenseNetExtractor().to(DEVICE)
    model.eval()

    feats, labs = [], []
    start = time.time()
    with torch.no_grad():
        for imgs, ys in tqdm(loader, desc=f"Extraindo features [{split_name}]"):
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast(
                "cuda", enabled=(DEVICE.type == "cuda")
            ):
                f = model(imgs)
            feats.append(f.cpu().numpy())
            labs.append(ys.numpy())
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    X = np.vstack(feats)
    y = np.concatenate(labs)
    print(f"  [{split_name}] done: {X.shape} em {time.time() - start:.1f}s")
    return X, y


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if not BASE_DIR.exists():
        raise RuntimeError(
            f"Diretório '{BASE_DIR}' não encontrado. "
            "Certifique-se de rodar antes `python build_multi_dataset.py`."
        )

    # Descobre as classes (slugs) a partir das pastas
    slugs = sorted(d.name for d in BASE_DIR.iterdir() if d.is_dir())
    if not slugs:
        raise RuntimeError(f"Nenhuma classe encontrada em {BASE_DIR}.")

    print(f"device: {DEVICE}")
    print(f"classes (slugs): {slugs}")

    label_encoder = build_label_encoder(slugs)

    # Carrega listas de paths/labels
    print("\n[coleta] paths por split...")
    train_paths, train_labels = collect_split("train", label_encoder)
    val_paths, val_labels = collect_split("val", label_encoder)
    test_paths, test_labels = collect_split("test", label_encoder)

    print(f"  train: {len(train_paths)} imagens")
    print(f"  val:   {len(val_paths)} imagens")
    print(f"  test:  {len(test_paths)} imagens")

    train_tf, eval_tf = get_transforms()

    # Pré-cálculo de features
    print("\n[features] extraindo features DenseNet121...")
    X_train, y_train = extract_features(train_paths, train_labels, train_tf, "train")
    X_val, y_val = extract_features(val_paths, val_labels, eval_tf, "val")
    X_test, y_test = extract_features(test_paths, test_labels, eval_tf, "test")

    # Normalização (StandardScaler)
    print("\n[scaler] ajustando StandardScaler com treino...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Treino do MLP (scikit-learn)
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

    print("\n[treino] MLP com early stopping (val log-loss)...")
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        epoch_iter = tqdm(range(max_epochs), desc="Treinando MLP")
        for epoch in epoch_iter:
            mlp.fit(X_train_s, y_train)

            # Loss de treino (MLPClassifier expõe loss_ do último fit)
            train_loss = getattr(mlp, "loss_", np.nan)

            # Métricas de validação
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
                    f"\n[early stopping] parada em epoch {epoch}. "
                    "Restaurando melhores pesos (menor val_loss)."
                )
                mlp.coefs_ = best_weights
                mlp.intercepts_ = best_intercepts
                break

    print(f"\n[treino] concluído em {time.time() - t0:.1f}s")

    # Avaliação final no conjunto de teste
    print("\n[avaliacao] conjunto de teste...")
    y_pred = mlp.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAcurácia geral (test): {acc:.4%}")

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
    # acurácia por classe = diag / total da classe
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    print("\nAcurácia por classe (test):")
    for name, a in zip(class_names, per_class_acc):
        print(f"  {name:16s}: {a:.4%}")

    # Matriz de confusão normalizada para visualização
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

    # Salvando artefatos
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    joblib.dump(mlp, "model_mlp.pkl")
    print("\n[save] Artefatos salvos: model_mlp.pkl, scaler.pkl, label_encoder.pkl")


if __name__ == "__main__":
    main()

