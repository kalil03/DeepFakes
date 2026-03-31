import os
from io import BytesIO

# ROCm environment (only set when actually using AMD GPU locally)
if os.environ.get("USE_ROCM", "0") == "1":
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("ROCM_PATH", "/opt/rocm")

from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

# Ensure vendored dependencies are on PYTHONPATH (local/deploy fallback)
VENDOR_DIR = Path(__file__).parent / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import torch
import numpy as np
import joblib
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify
from flask_cors import CORS

from model import DenseNetExtractor, eval_transform, DEVICE

# Optional Sightengine integration
try:
    import sightengine as se_module
except ImportError:
    se_module = None

# ── Flask app ────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend/dist", static_url_path="/")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit
CORS(app)

# ── Model loading ────────────────────────────────────────────────────
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
        print("[models] Models loaded successfully.", flush=True)
    except Exception as e:
        model_load_error = str(e)
        print(f"[models] Failed to load models: {e}", flush=True)
        feature_extractor = None
        scaler = None
        label_encoder = None
        mlp_model = None


_load_models()

SLUG_DISPLAY_MAP = {
    "human_real": "Human (Real)",
    "deepfake_gan": "Deepfake (GAN)",
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


# ── Routes ───────────────────────────────────────────────────────────
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
            "sightengine_configured": se_module is not None
            and se_module.is_configured(),
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
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]

    try:
        # Read bytes first
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "Empty file."}), 400

        # Load with PIL from BytesIO
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            return jsonify(
                {"error": "Invalid image format or corrupted file."}
            ), 400

        img_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            features = feature_extractor(img_tensor).cpu().numpy()

        features_scaled = scaler.transform(features)

        probs = mlp_model.predict_proba(features_scaled)[0]
        pred_label = mlp_model.predict(features_scaled)[0]

        # Translate index → slug → display name
        slug = label_encoder.inverse_transform([pred_label])[0]
        display_pred = _display_name_from_slug(slug)

        # Probabilities per class (human-readable name)
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

        # Optional: Sightengine cross-check (best-effort)
        if se_module is not None and se_module.is_configured():
            try:
                se_result = se_module.check_image(image_bytes)
                if se_result is not None:
                    response["sightengine"] = se_result
            except Exception:
                pass  # Sightengine is optional; never block predictions

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Error handler for oversized files ────────────────────────────────
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 10 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
