"""
Shared model definitions for DeepTrace.

Contains the DenseNet121 feature extractor and image transforms
used by both training (train_mlp.py) and inference (app.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights

# ── Constants ────────────────────────────────────────────────────────
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Feature Extractor ────────────────────────────────────────────────
class DenseNetExtractor(nn.Module):
    """DenseNet121 (ImageNet) headless feature extractor → 1024-d vector."""

    def __init__(self):
        super().__init__()
        self.densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = self.densenet.features

    def forward(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = F.adaptive_avg_pool2d(f, (1, 1))
        return torch.flatten(f, 1)


# ── Transforms ───────────────────────────────────────────────────────
eval_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

train_transform = transforms.Compose(
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
