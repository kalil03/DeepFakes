import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["ROCM_PATH"] = "/opt/rocm"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
import cv2

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Caminho para o classificador de faces do OpenCV
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
app = Flask(__name__)
class EfficientNetExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetExtractor, self).__init__()
        # Carrega pesos pré-treinados (Transfer Learning)
        self.efficientnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.features = self.efficientnet.features
        self.avgpool = self.efficientnet.avgpool
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def get_feature_extractor(device):
    print(f"[MODELO] Carregando EfficientNet-V2-S...")
    model = EfficientNetExtractor()
    model = model.to(device)
    model.eval()
    return model

def make_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

# Carregar modelos globais
print("Carregando modelos treinados...")
try:
    feature_extractor = get_feature_extractor(DEVICE)
    scaler = joblib.load('scaler_efficientnet.pkl')
    pca = joblib.load('pca_efficientnet.pkl')
    stacking_model = joblib.load('stacking_efficientnet.pkl')
    print("Modelos carregados com sucesso!")
except Exception as e:
    print(f"ERRO CRÍTICO ao carregar modelos: {e}")
    print("Certifique-se de que os arquivos .pkl estão no diretório atual.")
    scaler = None
    pca = None
    stacking_model = None
    feature_extractor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not stacking_model:
        return jsonify({'error': 'Modelos não carregados corretamente.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado.'}), 400

    try:
        # 1. Carregar e Pré-processar Imagem
        image = Image.open(file.stream).convert('RGB')
        transform = make_transforms()
        img_tensor = transform(image).unsqueeze(0).to(DEVICE) # Batch size 1
        
        # 2. Extrair Features (DenseNet)
        with torch.no_grad():
            features = feature_extractor(img_tensor)
            features_cpu = features.cpu().numpy()
            
        # 3. Escalar e PCA
        features_scaled = scaler.transform(features_cpu)
        features_pca = pca.transform(features_scaled)
        
        # 4. Predição (Stacking)
        prediction = stacking_model.predict(features_pca)[0]
        probabilities = stacking_model.predict_proba(features_pca)[0]
        predicted_class = str(prediction)
        confidence = float(max(probabilities))
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.tolist(),
            'classes': stacking_model.classes_.tolist()
        })

    except Exception as e:
        print(f"Erro na predição: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
