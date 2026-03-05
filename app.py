import os
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["ROCM_PATH"] = "/opt/rocm"

from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

# Garantir que dependências instaladas em ./vendor estejam no PYTHONPATH
VENDOR_DIR = Path(__file__).parent / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights
from PIL import Image, UnidentifiedImageError
import numpy as np
import joblib
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from flask_cors import CORS

from sightengine import check_image as se_check_image, is_configured as se_is_configured

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, static_folder="frontend/dist", static_url_path="/")
CORS(app)


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


transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


feature_extractor = None
scaler = None
label_encoder = None
mlp_model = None
model_load_error = None


def _load_models():
    global feature_extractor, scaler, label_encoder, mlp_model, model_load_error
    try:
        feature_extractor = DenseNetExtractor().to(DEVICE)
        feature_extractor.eval()
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        mlp_model = joblib.load("model_mlp.pkl")
        model_load_error = None
        print("[models] Modelos carregados com sucesso.", flush=True)
    except Exception as e:
        model_load_error = str(e)
        print(f"[models] Falha ao carregar modelos: {e}", flush=True)
        feature_extractor = None
        scaler = None
        label_encoder = None
        mlp_model = None


_load_models()

EXECUTOR = ThreadPoolExecutor(max_workers=4)

SLUG_DISPLAY_MAP = {
    "human_real": "Human (Real)",
    "deepfake_gan": "Deepfake (GAN)",
    "dalle3": "DALL-E 3",
    "midjourney": "Midjourney v6",
    "stable_diffusion": "Stable Diffusion",
    "gemini_imagen": "Gemini / Imagen",
}


def _model_is_loaded() -> bool:
    return (
        feature_extractor is not None
        and scaler is not None
        and label_encoder is not None
        and mlp_model is not None
    )


def _display_name_from_slug(slug: str) -> str:
    return SLUG_DISPLAY_MAP.get(slug, slug)


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/health")
def health():
    if _model_is_loaded():
        class_slugs = list(label_encoder.classes_)
        classes = [_display_name_from_slug(s) for s in class_slugs]
    else:
        classes = []

    return jsonify(
        {
            "status": "ok",
            "model_loaded": _model_is_loaded(),
            "classes": classes,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not _model_is_loaded():
        return (
            jsonify(
                {
                    "error": "Model not loaded",
                    "detail": model_load_error
                    or "Required .pkl files are missing or failed to load.",
                }
            ),
            503,
        )

    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "no file provided"}), 400

    file = request.files["file"]

    try:
        # Lê bytes primeiro
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "empty file"}), 400

        # Carrega com PIL a partir de BytesIO
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            return jsonify(
                {"error": "Invalid image format or corrupted file"}
            ), 400

        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Sightengine em paralelo (se configurado)
        se_future = None
        if se_is_configured():
            se_future = EXECUTOR.submit(se_check_image, image_bytes)

        with torch.no_grad():
            features = feature_extractor(img_tensor).cpu().numpy()

        features_scaled = scaler.transform(features)

        probs = mlp_model.predict_proba(features_scaled)[0]
        pred_label = mlp_model.predict(features_scaled)[0]

        # Traduz índice -> slug -> display name
        slug = label_encoder.inverse_transform([pred_label])[0]
        display_pred = _display_name_from_slug(slug)

        # Probabilidades por classe (nome legível)
        prob_dict = {}
        for cls_idx, p in zip(mlp_model.classes_, probs):
            slug_i = label_encoder.inverse_transform([cls_idx])[0]
            name = _display_name_from_slug(slug_i)
            prob_dict[name] = float(p)

        confidence = float(max(probs))
        uncertain = confidence < 0.60

        response = {
            "predicted_class": display_pred,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "uncertain": uncertain,
        }

        if se_future is not None:
            try:
                se_result = se_future.result(timeout=10)
                if se_result is not None:
                    response["sightengine"] = se_result
            except Exception:
                # Falhas na API externa não devem quebrar a rota principal
                pass

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

