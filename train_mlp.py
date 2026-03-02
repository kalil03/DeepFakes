import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["ROCM_PATH"] = "/opt/rocm"

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR  = '/home/kalilzera/Documentos/DeepFakes/archive/real_vs_fake/real-vs-fake'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR  = os.path.join(BASE_DIR, 'test')
CACHE_DIR = 'features_cache'

BATCH_SIZE = 64
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def extract_features(folder, device, split_name, augment=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_X = os.path.join(CACHE_DIR, f'X_{split_name}.npy')
    cache_y = os.path.join(CACHE_DIR, f'y_{split_name}.npy')
    cache_c = os.path.join(CACHE_DIR, f'classes_{split_name}.npy')

    if os.path.exists(cache_X):
        print(f"[cache] {split_name}")
        X       = np.load(cache_X)
        y       = np.load(cache_y)
        classes = list(np.load(cache_c))
        return X, y, classes

    print(f"\n[extract] {split_name} — {folder}")
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if augment:
        ops.append(transforms.RandomHorizontalFlip(0.5))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    dataset = ImageFolder(folder, transforms.Compose(ops))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model   = DenseNetExtractor().to(device)
    model.eval()

    feats, labels = [], []
    start = time.time()
    with torch.no_grad():
        for i, (imgs, labs) in enumerate(loader):
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                f = model(imgs.to(device))
            feats.append(f.cpu().numpy())
            labels.append(labs.numpy())
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if i % 50 == 0:
                print(f"  {i * BATCH_SIZE} imgs")

    X       = np.vstack(feats)
    y       = np.concatenate(labels)
    classes = dataset.classes
    print(f"  done: {X.shape} in {time.time()-start:.1f}s")

    np.save(cache_X, X)
    np.save(cache_y, y)
    np.save(cache_c, classes)
    return X, y, classes


def main():
    print(f"device: {DEVICE}")

    X_train, y_train, classes = extract_features(TRAIN_DIR, DEVICE, 'train', augment=True)
    X_test,  y_test,  _       = extract_features(TEST_DIR,  DEVICE, 'test')

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        tol=1e-5,
        random_state=42,
        verbose=True
    )
    t0 = time.time()
    mlp.fit(X_train_s, y_train)
    print(f"trained in {time.time()-t0:.1f}s — {mlp.n_iter_} epochs")

    y_pred = mlp.predict(X_test_s)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\ntest accuracy: {acc:.4%}")
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Greens', xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix — MLP ({acc:.2%})")
    plt.tight_layout()
    plt.savefig("confusion_matrix_densenet.png")

    joblib.dump(scaler, 'scaler_densenet.pkl')
    joblib.dump(None,   'pca_densenet.pkl')
    joblib.dump(mlp,    'mlp_densenet.pkl')
    print("saved: scaler_densenet.pkl, pca_densenet.pkl, mlp_densenet.pkl")


if __name__ == "__main__":
    main()
