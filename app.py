import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["ROCM_PATH"] = "/opt/rocm"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights
from PIL import Image
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

IMG_SIZE = 224
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app      = Flask(__name__)


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


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    feature_extractor = DenseNetExtractor().to(DEVICE)
    feature_extractor.eval()
    scaler    = joblib.load('scaler_densenet.pkl')
    pca       = joblib.load('pca_densenet.pkl')
    mlp_model = joblib.load('mlp_densenet.pkl')
except Exception as e:
    print(f"failed to load models: {e}")
    feature_extractor = scaler = pca = mlp_model = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not mlp_model:
        return jsonify({'error': 'models not loaded'}), 500
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'no file provided'}), 400

    try:
        image      = Image.open(request.files['file'].stream).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            features = feature_extractor(img_tensor).cpu().numpy()

        features_scaled = scaler.transform(features)
        features_final  = pca.transform(features_scaled) if pca is not None else features_scaled

        prediction    = mlp_model.predict(features_final)[0]
        probabilities = mlp_model.predict_proba(features_final)[0]

        return jsonify({
            'prediction':   str(prediction),
            'confidence':   float(max(probabilities)),
            'probabilities': probabilities.tolist(),
            'classes':      mlp_model.classes_.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
