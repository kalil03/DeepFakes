import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights
from PIL import Image
import numpy as np
import joblib
import gradio as gr

IMG_SIZE = 224
DEVICE   = torch.device("cpu")

# Hide ALL default Gradio chrome via CSS (NOT visible=False — that kills /upload endpoint)
CSS = """
/* Push native Gradio components off-screen, keep them functional */
#g-image, #g-output, #g-btn {
    position: fixed !important;
    top: -9999px !important;
    left: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    pointer-events: none !important;
    opacity: 0 !important;
}

body, .gradio-container {
    background: #0f172a !important;
    min-height: 100vh;
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
    overflow-x: hidden;
}
.gradio-container > .main { padding: 0 !important; }
.contain { padding: 0 !important; max-width: 100% !important; }
footer, .built-with, .eta-bar, .progress-bar { display: none !important; }
"""

CUSTOM_UI = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">

<style>
  *, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }

  #vr-app {
    font-family: 'Outfit', sans-serif;
    background: #0f172a;
    color: #f8fafc;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2.5rem 1.5rem 3rem;
    position: relative;
    overflow: hidden;
  }

  /* Animated blobs */
  .blob { position: fixed; border-radius: 50%; filter: blur(90px); opacity: .35; pointer-events: none; animation: blob-float 12s ease-in-out infinite; z-index: 0; }
  .blob-1 { width:480px; height:480px; background:#6366f1; top:-150px; left:-150px; }
  .blob-2 { width:340px; height:340px; background:#ec4899; bottom:-80px; right:-80px; animation-delay:3s; }
  .blob-3 { width:260px; height:260px; background:#8b5cf6; top:45%; left:45%; animation-delay:6s; }
  @keyframes blob-float { 0%,100%{transform:translate(0,0)} 50%{transform:translate(28px,-28px)} }

  /* Layout */
  #vr-content { width:100%; max-width:620px; z-index:1; display:flex; flex-direction:column; gap:1.8rem; }

  /* Header */
  .vr-header { text-align:center; }
  .vr-logo { display:inline-flex; align-items:center; gap:.5rem; color:#ec4899; font-size:.85rem; font-weight:700; text-transform:uppercase; letter-spacing:2px; margin-bottom:.6rem; }
  .vr-title { font-size:2.4rem; font-weight:900; background:linear-gradient(135deg,#fff 30%,#94a3b8); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; line-height:1.1; margin-bottom:.4rem; }
  .vr-sub { color:#64748b; font-weight:300; font-size:.95rem; }

  /* Glass card */
  .glass { background:rgba(30,41,59,.7); backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,.08); border-radius:24px; padding:1.8rem; box-shadow:0 25px 50px -12px rgba(0,0,0,.6); }

  /* Upload zone */
  #drop-zone { border:2px dashed rgba(255,255,255,.12); border-radius:18px; cursor:pointer; transition:border-color .25s, background .25s; position:relative; overflow:hidden; }
  #drop-zone:hover, #drop-zone.drag-over { border-color:#6366f1; background:rgba(99,102,241,.06); }
  #drop-zone.drag-over { border-color:#ec4899; background:rgba(236,72,153,.07); }

  .upload-placeholder { display:flex; flex-direction:column; align-items:center; gap:.9rem; padding:3rem 1.5rem; }
  .upload-icon { font-size:2.8rem; color:#6366f1; opacity:.85; }
  .upload-title { font-size:1.1rem; font-weight:600; }
  .upload-hint { color:#64748b; font-size:.88rem; }
  #file-input { display:none; }

  .select-btn { background:rgba(99,102,241,.15); color:#818cf8; border:1px solid rgba(99,102,241,.3); padding:.55rem 1.4rem; border-radius:10px; font-family:inherit; font-size:.9rem; font-weight:600; cursor:pointer; transition:all .2s; }
  .select-btn:hover { background:rgba(99,102,241,.25); color:#fff; }

  /* Preview */
  #preview-wrap { display:none; position:relative; }
  #preview-wrap img { width:100%; max-height:340px; object-fit:cover; border-radius:16px; display:block; }
  .remove-btn { position:absolute; top:10px; right:10px; background:rgba(0,0,0,.65); color:#fff; border:none; width:32px; height:32px; border-radius:50%; cursor:pointer; font-size:.85rem; display:flex; align-items:center; justify-content:center; transition:background .2s; }
  .remove-btn:hover { background:#ef4444; }

  /* Analyze button */
  #analyze-btn { width:100%; padding:1rem; border:none; border-radius:16px; font-family:inherit; font-size:1.05rem; font-weight:700; color:#fff; cursor:pointer; position:relative; overflow:hidden; transition:all .3s; background:linear-gradient(135deg,#6366f1,#ec4899); box-shadow:0 10px 25px -5px rgba(99,102,241,.35); }
  #analyze-btn:disabled { opacity:.4; cursor:not-allowed; transform:none !important; box-shadow:none !important; }
  #analyze-btn:not(:disabled):hover { transform:translateY(-2px); box-shadow:0 18px 30px -5px rgba(99,102,241,.45); }
  #analyze-btn::after { content:''; position:absolute; inset:0; background:linear-gradient(135deg,rgba(255,255,255,.15),transparent); }

  .btn-inner { display:flex; align-items:center; justify-content:center; gap:.7rem; position:relative; z-index:1; }
  .spinner { width:18px; height:18px; border:2.5px solid rgba(255,255,255,.35); border-top-color:#fff; border-radius:50%; animation:spin .75s linear infinite; display:none; }
  @keyframes spin { to { transform:rotate(360deg); } }

  /* Scanning overlay */
  #scan-overlay { display:none; position:absolute; inset:0; border-radius:16px; overflow:hidden; pointer-events:none; }
  .scan-line { position:absolute; left:0; right:0; height:3px; background:linear-gradient(90deg,transparent,#6366f1,#ec4899,transparent); top:0; animation:scan 1.4s ease-in-out infinite; }
  @keyframes scan { 0%{top:0;opacity:1} 90%{top:100%;opacity:1} 100%{top:100%;opacity:0} }
  .scan-grid { position:absolute; inset:0; background: repeating-linear-gradient(0deg, transparent, transparent 28px, rgba(99,102,241,.06) 28px, rgba(99,102,241,.06) 29px), repeating-linear-gradient(90deg, transparent, transparent 28px, rgba(99,102,241,.06) 28px, rgba(99,102,241,.06) 29px); }
  .scan-vignette { position:absolute; inset:0; background:radial-gradient(ellipse at center, transparent 40%, rgba(99,102,241,.25) 100%); }
  #scan-overlay.active { display:block; }

  /* Result card */
  #result-card { display:none; animation:fadeUp .5s ease forwards; }
  @keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }

  .result-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem; }
  .result-title { font-size:1rem; font-weight:600; color:#94a3b8; }
  .verdict-badge { padding:.4rem 1.2rem; border-radius:20px; font-size:.8rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; }
  .verdict-badge.real { background:rgba(16,185,129,.18); color:#10b981; border:1px solid rgba(16,185,129,.4); }
  .verdict-badge.fake { background:rgba(239,68,68,.18); color:#ef4444; border:1px solid rgba(239,68,68,.4); }

  /* Confidence meter */
  .conf-row { display:flex; justify-content:space-between; margin-bottom:.5rem; font-size:.85rem; color:#64748b; }
  .conf-pct { color:#f8fafc; font-weight:700; }
  .conf-track { width:100%; height:10px; background:rgba(255,255,255,.08); border-radius:5px; overflow:hidden; }
  .conf-fill { height:100%; border-radius:5px; background:linear-gradient(90deg,#6366f1,#ec4899); width:0%; transition:width 1.2s cubic-bezier(.4,0,.2,1); }

  /* Details grid */
  .details { display:grid; grid-template-columns:1fr 1fr 1fr; gap:.8rem; margin-top:1.5rem; padding-top:1.2rem; border-top:1px solid rgba(255,255,255,.07); }
  .detail-box { background:rgba(15,23,42,.5); border-radius:12px; padding:.8rem 1rem; }
  .detail-label { font-size:.72rem; color:#475569; text-transform:uppercase; letter-spacing:.5px; margin-bottom:.3rem; }
  .detail-value { font-size:.95rem; font-weight:600; color:#e2e8f0; }

  /* Footer */
  .vr-footer { color:#334155; font-size:.78rem; text-align:center; margin-top:1rem; }

  /* Responsive */
  @media(max-width:480px) {
    .vr-title { font-size:2rem; }
    .details { grid-template-columns:1fr 1fr; }
  }
</style>

<div id="vr-app">
  <div class="blob blob-1"></div>
  <div class="blob blob-2"></div>
  <div class="blob blob-3"></div>

  <div id="vr-content">

    <!-- Header -->
    <div class="vr-header">
      <div class="vr-logo"><i class="fa-solid fa-fingerprint"></i> Veritas AI</div>
      <div class="vr-title">DeepFake Detector</div>
      <div class="vr-sub">Advanced Neural Network Analysis for Image Authenticity</div>
    </div>

    <!-- Upload card -->
    <div class="glass">
      <div id="drop-zone" onclick="document.getElementById('file-input').click()">
        <!-- Placeholder -->
        <div class="upload-placeholder" id="placeholder">
          <i class="fa-solid fa-cloud-arrow-up upload-icon"></i>
          <div class="upload-title">Arraste sua imagem aqui</div>
          <div class="upload-hint">ou clique para selecionar</div>
          <button class="select-btn" onclick="event.stopPropagation();document.getElementById('file-input').click()">
            Selecionar Arquivo
          </button>
        </div>

        <!-- Preview -->
        <div id="preview-wrap">
          <img id="preview-img" src="" alt="preview">
          <button class="remove-btn" id="remove-btn" onclick="event.stopPropagation();removeImage()" title="Remover">
            <i class="fa-solid fa-xmark"></i>
          </button>
          <div id="scan-overlay">
            <div class="scan-grid"></div>
            <div class="scan-vignette"></div>
            <div class="scan-line"></div>
          </div>
        </div>

        <input type="file" id="file-input" accept="image/*">
      </div>
    </div>

    <!-- Analyze button -->
    <button id="analyze-btn" disabled onclick="analyze()">
      <div class="btn-inner">
        <div class="spinner" id="spinner"></div>
        <i class="fa-solid fa-magnifying-glass" id="btn-icon"></i>
        <span id="btn-text">Analisar Imagem</span>
      </div>
    </button>

    <!-- Result card -->
    <div class="glass" id="result-card">
      <div class="result-header">
        <span class="result-title">Resultado da Análise</span>
        <span class="verdict-badge" id="verdict">—</span>
      </div>

      <div class="conf-row">
        <span>Confiança da IA</span>
        <span class="conf-pct" id="conf-text">—</span>
      </div>
      <div class="conf-track"><div class="conf-fill" id="conf-fill"></div></div>

      <div class="details" id="details-container">
        <!-- Dynmically filled by JS -->
      </div>
    </div>

    <div class="vr-footer">Powered by PyTorch &amp; Scikit-Learn &nbsp;·&nbsp; DenseNet121 + MLP &nbsp;·&nbsp; 97.19% accuracy</div>
  </div>
</div>

<script type="module">
  import { Client } from "https://esm.sh/@gradio/client";

  let currentFile = null;
  let gradioClient = null;

  async function getClient() {
    if (!gradioClient) {
      gradioClient = await Client.connect(window.location.origin);
    }
    return gradioClient;
  }
  getClient().catch(console.error);

  function handleFiles(files) {
    if (!files || !files[0] || !files[0].type.startsWith('image/')) return;
    currentFile = files[0];
    document.getElementById('preview-img').src = URL.createObjectURL(currentFile);
    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('preview-wrap').style.display = 'block';
    document.getElementById('analyze-btn').disabled = false;
    document.getElementById('result-card').style.display = 'none';
  }

  window.removeImage = function() {
    currentFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('preview-img').src = '';
    document.getElementById('preview-wrap').style.display = 'none';
    document.getElementById('placeholder').style.display = 'flex';
    document.getElementById('analyze-btn').disabled = true;
    document.getElementById('result-card').style.display = 'none';
    document.getElementById('scan-overlay').classList.remove('active');
  };

  function setLoading(on) {
    document.getElementById('spinner').style.display   = on ? 'block' : 'none';
    document.getElementById('btn-icon').style.display  = on ? 'none'  : 'block';
    document.getElementById('btn-text').textContent    = on ? 'Analisando...' : 'Analisar Imagem';
    document.getElementById('analyze-btn').disabled    = on;
    document.getElementById('scan-overlay').classList.toggle('active', on);
  }

  window.analyze = async function() {
    if (!currentFile) return;
    setLoading(true);
    try {
      const client = await getClient();
      const result = await client.predict("/predict", [currentFile]);
      showResult(result.data[0]);
    } catch(err) {
      console.error(err);
      alert('Erro ao analisar: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  function showResult(data) {
    let parsed;
    try { parsed = typeof data === 'string' ? JSON.parse(data) : data; }
    catch(e) { console.error('parse error', data); return; }

    const { is_real, confidence, top_class, probabilities } = parsed;
    const confPct = (confidence * 100).toFixed(1) + '%';

    const verdict = document.getElementById('verdict');
    verdict.textContent  = is_real ? 'REAL' : 'FAKE';
    verdict.className    = 'verdict-badge ' + (is_real ? 'real' : 'fake');

    document.getElementById('conf-text').textContent = top_class + ' (' + confPct + ')';

    const detailsContainer = document.getElementById('details-container');
    detailsContainer.innerHTML = '';
    
    for (const [aiName, probVal] of Object.entries(probabilities)) {
      const pct = (probVal * 100).toFixed(1) + '%';
      detailsContainer.innerHTML += `
        <div class="detail-box">
          <div class="detail-label">${aiName}</div>
          <div class="detail-value">${pct}</div>
          <div style="margin-top:8px; height:6px; background:rgba(255,255,255,0.08); border-radius:3px; overflow:hidden;">
            <div style="width:${pct}; height:100%; background:linear-gradient(90deg,#6366f1,#ec4899); border-radius:3px; transition:width 1s ease;"></div>
          </div>
        </div>
      `;
    }

    const card = document.getElementById('result-card');
    card.style.display   = 'block';
    card.style.animation = 'none';
    void card.offsetWidth;
    card.style.animation = 'fadeUp .5s ease forwards';

    requestAnimationFrame(() => setTimeout(() => {
      document.getElementById('conf-fill').style.width = confPct;
    }, 100));

    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Drag & drop and file input wiring
  const zone = document.getElementById('drop-zone');
  ['dragenter','dragover'].forEach(e => zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.add('drag-over'); }));
  ['dragleave','drop'].forEach(e     => zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.remove('drag-over'); }));
  zone.addEventListener('drop', ev  => handleFiles(ev.dataTransfer.files));
  document.getElementById('file-input').addEventListener('change', ev => handleFiles(ev.target.files));
</script>

"""


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


def load_models():
    extractor = DenseNetExtractor().to(DEVICE)
    extractor.eval()
    scaler = joblib.load("scaler_densenet.pkl")
    pca    = joblib.load("pca_densenet.pkl")
    mlp    = joblib.load("mlp_densenet.pkl")
    print("Success: Loaded Production Model V4 (Restored 2026-03-03) - Accuracy: 97.19%", flush=True)
    return extractor, scaler, pca, mlp


extractor, scaler, pca, mlp_model = load_models()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict(image):
    import json, time
    if image is None:
        return json.dumps({"error": "no image"})

    img    = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = extractor(tensor).cpu().numpy()

    features_scaled = scaler.transform(features)
    features_final  = pca.transform(features_scaled) if pca is not None else features_scaled

    probs   = mlp_model.predict_proba(features_final)[0]
    
    class_map = {
        '0': 'Humano (Real)',
        '1': 'Deepfake (GAN)',
        '2': 'DALL-E 3 (ChatGPT)',
        '3': 'Midjourney v6',
        '4': 'Stable Diffusion',
        '5': 'Google Gemini'
    }
    
    probs_dict = {}
    for i, cls in enumerate(mlp_model.classes_):
        name = class_map.get(str(cls), str(cls))
        probs_dict[name] = float(probs[i])
        
    probs_dict = dict(sorted(probs_dict.items(), key=lambda item: item[1], reverse=True))
    
    top_class = list(probs_dict.keys())[0]
    is_real   = (top_class == 'Humano (Real)')
    confidence = float(max(probs))

    return json.dumps({
        "is_real":    is_real,
        "confidence": round(confidence, 4),
        "top_class":  top_class,
        "probabilities": probs_dict
    })


with gr.Blocks(css=CSS, title="DeepFake Detector | Veritas AI") as demo:
    # Keep components interactive (not visible=False) so /upload endpoint stays alive
    with gr.Row():
        img_in  = gr.Image(type="numpy", elem_id="g-image", interactive=True)
    json_out = gr.Textbox(elem_id="g-output")
    btn      = gr.Button("predict", elem_id="g-btn")
    btn.click(fn=predict, inputs=img_in, outputs=json_out, api_name="predict")

    gr.HTML(CUSTOM_UI)

if __name__ == "__main__":
    demo.launch()
