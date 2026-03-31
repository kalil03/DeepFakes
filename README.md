## DeepTrace — Face Deepfake Detector

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/kalilzera/DeepFakes)

> **[🔗 Demo no Hugging Face](https://huggingface.co/spaces/kalilzera/DeepFakes)**  
> *DeepTrace — Forensic deepfake detection for human faces.*

This project implements a **deepfake detector focused on human faces** (cropped faces, 224×224), with two classes:

- 👤 `human_real` → **Human (Real)**
- 🤖 `deepfake_gan` → **Deepfake (GAN)**

Internally it uses:

- **DenseNet121 (ImageNet)** as a feature extractor (1024 dimensions).
- **MLPClassifier (scikit‑learn)** 1024 → 512 → 256 → 128 → 2 classes.

The goal is **≥ 98% accuracy** on validation, distinguishing real faces from GAN-generated/manipulated faces (StyleGAN, OpenForensics).

---

## How to run locally

### 1. Clone and install dependencies

```bash
git clone https://github.com/kalil03/DeepFakes.git
cd DeepFakes

python3 -m venv venv_ia
source venv_ia/bin/activate  # or venv\Scripts\activate on Windows

python3 -m pip install -r requirements.txt
```

For the frontend (React):

```bash
cd frontend
npm install
cd ..
```

### 2. Configure Kaggle (face datasets)

The pipeline uses two public Kaggle datasets:

- [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (StyleGAN)
- [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) (OpenForensics)

1. Install the Kaggle CLI:

```bash
pip install kaggle
```

2. Create the file `~/.kaggle/kaggle.json`:

```json
{
  "username": "YOUR_USERNAME_HERE",
  "key": "YOUR_KAGGLE_API_KEY_HERE"
}
```

3. Set permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download the raw datasets

```bash
python3 download_datasets.py
```

This downloads and extracts:

- `datasets/raw/kaggle/140k-real-and-fake-faces/...`
- `datasets/raw/kaggle/deepfake-and-real-images/...`

### 4. Build the balanced 2-class dataset

```bash
python3 build_multi_dataset.py
```

This script:

- Wipes any existing `multi_dataset/`.
- Collects real and fake images from the downloaded sources.
- Deduplicates by filename across sources.
- Balances classes to the same size.
- Splits 80/10/10:

```text
multi_dataset/
  human_real/{train,val,test}/
  deepfake_gan/{train,val,test}/
```

### 5. Train the model (DenseNet + MLP)

#### NVIDIA (or CPU)

```bash
python3 train_mlp.py
```

#### AMD (ROCm)

```bash
USE_ROCM=1 python3 train_mlp.py
```

The script:

- Extracts DenseNet121 features from `multi_dataset/{train,val,test}`.
- Fits a `StandardScaler`.
- Trains a `MLPClassifier` with early stopping (patience=10).
- Prints accuracy and classification report on the test set.
- Generates `confusion_matrix.png`.
- Saves:

```text
model_mlp.pkl
scaler.pkl
label_encoder.pkl
```

### 6. Build the frontend

```bash
cd frontend
npm run build
cd ..
```

This generates `frontend/dist/`, served by Flask.

### 7. Start the Flask API + UI

With the virtual environment active and trained `.pkl` files:

```bash
python3 app.py
```

The API runs at `http://localhost:5000`.  
Open in your browser and:

1. Upload a **face/portrait image**.
2. Click **Analyze**.
3. See the result:
   - 👤 **Human (Real)** → "No manipulation detected."
   - 🤖 **Deepfake (GAN)** → "AI generation or manipulation detected."
   - Confidence bars for both classes.
   - Uncertainty banner if confidence `< 60%`.

---

## File Structure

- `model.py` — Shared DenseNetExtractor, transforms, and constants.
- `app.py` — Backend Flask (`/health`, `/predict`) + serves `frontend/dist`.
- `frontend/` — UI in React + TypeScript (Vite + Tailwind).
- `download_datasets.py` — Downloads Kaggle datasets (StyleGAN + OpenForensics).
- `build_multi_dataset.py` — Builds `multi_dataset/` balanced (2 classes).
- `train_mlp.py` — Extracts DenseNet121 features, trains MLP, saves `.pkl`.
- `sightengine.py` — Optional Sightengine API cross-check.
- `huggingface/` — Dockerfile and metadata for Hugging Face Spaces deploy.
- `requirements.txt` — Python dependencies.

> **Git LFS — model weights**
>
> The weight files (`model_mlp.pkl`, `scaler.pkl`, `label_encoder.pkl`) are stored via **Git LFS**.  
> Before pulling weights from a remote LFS repo:
>
> ```bash
> git lfs install
> git lfs pull
> ```
