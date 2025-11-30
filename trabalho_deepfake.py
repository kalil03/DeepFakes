import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torchvision.models import densenet121, DenseNet121_Weights

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Caminhos do dataset
BASE_DIR = '/home/kalilzera/Documentos/DeepFakes/archive/real_vs_fake/real-vs-fake'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR  = os.path.join(BASE_DIR, 'test')

# Hiperparâmetros
BATCH_SIZE = 64
IMG_SIZE = 224        # Resolução de entrada da DenseNet
PCA_COMPONENTS = 300  # Redução dimensional
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# EXTRATOR
class DenseNetExtractor(nn.Module):
    # Envolve DenseNet121 removendo a última camada
    def __init__(self):
        super(DenseNetExtractor, self).__init__()
        self.densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = self.densenet.features  

    def forward(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = F.adaptive_avg_pool2d(f, (1, 1))  
        return torch.flatten(f, 1)

def get_feature_extractor(device):
    print("[MODELO] Carregando DenseNet121…")
    model = DenseNetExtractor().to(device)
    model.eval()
    return model

# Transformações de imagem 
def make_transforms(train=False):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])


# Extração de features otimizada com GPU
def extract_features_optimized(folder, device, train=False):
    print(f"\n[EXTRAÇÃO] Pasta: {folder}")
    transform = make_transforms(train=train)
    dataset = ImageFolder(folder, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_feature_extractor(device)
    feats_list, labels_list = [], []

    start = time.time()
    with torch.no_grad():
        for i, (imgs, labs) in enumerate(dataloader):
            imgs = imgs.to(device)

            # AMP quando em GPU (reduz uso de VRAM)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                feats = model(imgs)

            feats_list.append(feats.cpu().numpy())  # Move para CPU
            labels_list.append(labs.numpy())

            if device.type == "cuda":
                torch.cuda.empty_cache()
            # Batchs processados
            if i % 50 == 0:
                print(f"  -> {i * BATCH_SIZE} imagens processadas")

    X = np.vstack(feats_list)
    y = np.concatenate(labels_list)
    print(f" -> Concluído ({len(dataset)} imgs) | Shape: {X.shape} | Tempo: {time.time()-start:.1f}s")
    return X, y, dataset.classes

# Mostra qtd por classe
def check_class_distribution(y_train, y_test, class_names):
    print("\n[DISTRIBUIÇÃO DE CLASSES]")
    t_u, t_c = np.unique(y_train, return_counts=True)
    v_u, v_c = np.unique(y_test, return_counts=True)
    for i, name in enumerate(class_names):
        print(f"  {name}: {t_c[i]} train, {v_c[i]} test")

# Pool de modelos usados no Stacking
def build_pool_20():
    pool = []
    # KNNs
    for k in [1,3,5,7,9]:
        pool.append((f'knn_{k}', KNeighborsClassifier(n_neighbors=k)))
    # Árvores
    for d in [5,10,20,30,None]:
        pool.append((f'dt_{d}', DecisionTreeClassifier(max_depth=d, random_state=42)))
    # Random Forest
    pool.append(('rf_50', RandomForestClassifier(n_estimators=50, random_state=42)))
    pool.append(('rf_100', RandomForestClassifier(n_estimators=100, random_state=42)))
    pool.append(('rf_entr', RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=42)))
    pool.append(('rf_200', RandomForestClassifier(n_estimators=200, random_state=42)))
    # SGD 
    for alpha in [1e-4,1e-3,1e-2,1e-1,1.0,10.0]:
        pool.append((f'sgd_{alpha}', SGDClassifier(loss='log_loss', alpha=alpha, random_state=42)))
    return pool[:20]

# Treina + mede overfitting
def evaluate_with_overfitting_check(clf, name, X_train, y_train, X_test, y_test, class_names):
    print("\n" + "="*60)
    print(f"CLASSIFICADOR: {name}")
    print("="*60)

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred_train = clf.predict(X_train)
    y_pred_test  = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test,  y_pred_test)

    print(f" > Tempo Treino: {train_time:.1f}s")
    print(f" > Acc Treino:  {train_acc:.4%}")
    print(f" > Acc Teste:   {test_acc:.4%}")

    print("\nRelatório de classificação (TESTE):")
    print(classification_report(y_test, y_pred_test, target_names=class_names, digits=4))

    return test_acc, train_acc

def main():
    print("--- INICIANDO (DenseNet121) ---")
    print(f"Usando: {DEVICE}")

    # Extração 
    X_train, y_train, class_names = extract_features_optimized(TRAIN_DIR, DEVICE, train=True)
    X_test, y_test, _ = extract_features_optimized(TEST_DIR, DEVICE, train=False)

    check_class_distribution(y_train, y_test, class_names)

    # Padronização + PCA
    print("\n[PROCESSAMENTO] Scaler + IncrementalPCA")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    pca = IncrementalPCA(n_components=PCA_COMPONENTS, batch_size=1024)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p  = pca.transform(X_test_s)

    print(f" -> PCA Shape: {X_train_p.shape}")

    # Treino de modelos individuais
    results = {}

    r = evaluate_with_overfitting_check(
        KNeighborsClassifier(5), "KNN (k=5)", 
        X_train_p, y_train, X_test_p, y_test, class_names
    )
    results['KNN'] = r

    r = evaluate_with_overfitting_check(
        DecisionTreeClassifier(max_depth=20, random_state=42),
        "Decision Tree", 
        X_train_p, y_train, X_test_p, y_test, class_names
    )
    results['Decision Tree'] = r

    # Subset para SVM 
    idx = np.random.RandomState(42).choice(len(X_train_p), min(5000, len(X_train_p)), replace=False)
    r = evaluate_with_overfitting_check(
        SVC(kernel='rbf', random_state=42), 
        "SVM (Subset 5k)", 
        X_train_p[idx], y_train[idx], X_test_p, y_test, class_names
    )
    results['SVM'] = r

    r = evaluate_with_overfitting_check(
        MLPClassifier(
            hidden_layer_sizes=(256,128),
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ),
        "MLP (Early Stopping)", 
        X_train_p, y_train, X_test_p, y_test, class_names
    )
    results['MLP'] = r

    # Stacking Ensemble
    print("\n=== STACKING ENSEMBLE ===")
    pool = build_pool_20()
    meta = RandomForestClassifier(n_estimators=200, random_state=42)

    stack = StackingClassifier(
        estimators=pool,
        final_estimator=meta,
        stack_method='predict_proba',
        n_jobs=2,
        cv=10
    )

    t0 = time.time()
    stack.fit(X_train_p, y_train)
    tstack = time.time() - t0

    y_pred_train = stack.predict(X_train_p)
    y_pred_test  = stack.predict(X_test_p)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test  = accuracy_score(y_test,  y_pred_test)

    print(f"\nStacking Treinado em: {tstack:.1f}s")
    print(f"Acurácia Treino: {acc_train:.4%}")
    print(f"Acurácia Teste:  {acc_test:.4%}")

    print("\nRelatório Stacking:")
    print(classification_report(y_test, y_pred_test, target_names=class_names, digits=4))

    results['Stacking'] = (acc_test, acc_train)

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred_test, normalize='true')
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Stacking")
    plt.tight_layout()
    plt.savefig("confusion_matrix_densenet.png")

    # Resumo final
    print("\n============================")
    print("RESUMO FINAL")
    print("============================")
    for m, (tst, trn) in results.items():
        print(f"{m:15} | Teste: {tst:.4%} | Treino: {trn:.4%}")

    # Salvar modelos treinados
    joblib.dump(scaler, 'scaler_densenet.pkl')
    joblib.dump(pca, 'pca_densenet.pkl')
    joblib.dump(stack, 'stacking_densenet.pkl')

if __name__ == "__main__":
    main()
