#!/bin/bash
# ────────────────────────────────────────────────────────────────────
# Deploy to Hugging Face Space
#
# Usage:
#   ./deploy_hf.sh
#
# Prerequisites:
#   - huggingface-cli logged in (huggingface-cli login)
#   - Git LFS installed (git lfs install)
# ────────────────────────────────────────────────────────────────────
set -euo pipefail

SPACE_REPO="https://huggingface.co/spaces/kalilzera/DeepFakes"
DEPLOY_DIR="/tmp/hf_deploy_deeptrace"

echo "=== DeepTrace → Hugging Face Space deploy ==="

# Clean previous deploy dir
rm -rf "$DEPLOY_DIR"

# Clone the HF Space repo
echo "[1/6] Cloning HF Space..."
git clone "$SPACE_REPO" "$DEPLOY_DIR"
cd "$DEPLOY_DIR"

# Enable LFS
git lfs install

# Wipe everything except .git
echo "[2/6] Cleaning old files..."
find . -maxdepth 1 ! -name '.git' ! -name '.' -exec rm -rf {} +

# Copy the HF-specific config
echo "[3/6] Copying deploy files..."
SRC="/home/kalilzera/DeepFakes"

# HF Space metadata
cp "$SRC/huggingface/README.md" ./README.md
cp "$SRC/huggingface/Dockerfile" ./Dockerfile
cp "$SRC/huggingface/requirements.txt" ./requirements.txt
cp "$SRC/.dockerignore" ./.dockerignore
cp "$SRC/.gitattributes" ./.gitattributes

# Backend Python files
cp "$SRC/app.py" ./app.py
cp "$SRC/model.py" ./model.py
cp "$SRC/sightengine.py" ./sightengine.py

# Model weights (LFS tracked)
cp "$SRC/model_mlp.pkl" ./model_mlp.pkl
cp "$SRC/scaler.pkl" ./scaler.pkl
cp "$SRC/label_encoder.pkl" ./label_encoder.pkl

# Frontend (full source — built inside Docker)
cp -r "$SRC/frontend" ./frontend
rm -rf ./frontend/node_modules ./frontend/dist

echo "[4/6] Files to deploy:"
find . -maxdepth 2 ! -path './.git/*' ! -name '.git' | sort

# Stage and commit
echo "[5/6] Committing..."
git add -A
git status --short
git commit -m "deploy: clean 2-class GAN deepfake detector with DenseNet+MLP" || echo "(no changes to commit)"

# Push
echo "[6/6] Pushing to Hugging Face..."
git push origin main

echo ""
echo "=== Deploy complete! ==="
echo "Space URL: https://huggingface.co/spaces/kalilzera/DeepFakes"
echo "Note: HF Spaces may take a few minutes to rebuild the Docker image."
