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

BASE_DIR  = '/home/kalilzera/Documentos/DeepFakes/archive/real_vs_fake/real-vs-fake'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR  = os.path.join(BASE_DIR, 'test')

BATCH_SIZE     = 64
IMG_SIZE       = 224
PCA_COMPONENTS = 300
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def make_transforms(train=False):
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        ops.append(transforms.RandomHorizontalFlip(0.5))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(ops)


def extract_features(folder, device, train=False):
    print(f"\n[extract] {folder}")
    dataset    = ImageFolder(folder, make_transforms(train=train))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model      = DenseNetExtractor().to(device)
    model.eval()

    feats, labels = [], []
    start = time.time()
    with torch.no_grad():
        for i, (imgs, labs) in enumerate(dataloader):
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                f = model(imgs.to(device))
            feats.append(f.cpu().numpy())
            labels.append(labs.numpy())
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if i % 50 == 0:
                print(f"  {i * BATCH_SIZE} imgs")

    X = np.vstack(feats)
    y = np.concatenate(labels)
    print(f"  done: {X.shape} in {time.time()-start:.1f}s")
    return X, y, dataset.classes


def evaluate(clf, name, X_train, y_train, X_test, y_test, class_names):
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    t0 = time.time()
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc  = accuracy_score(y_test,  clf.predict(X_test))

    print(f"  train: {train_acc:.4%}  test: {test_acc:.4%}  time: {time.time()-t0:.1f}s")
    print(classification_report(y_test, clf.predict(X_test), target_names=class_names, digits=4))
    return clf, test_acc, train_acc


def build_stacking_pool():
    pool = []
    for k in [1, 3, 5, 7, 9]:
        pool.append((f'knn_{k}', KNeighborsClassifier(n_neighbors=k)))
    for d in [5, 10, 20, 30, None]:
        pool.append((f'dt_{d}', DecisionTreeClassifier(max_depth=d, random_state=42)))
    pool.append(('rf_50',   RandomForestClassifier(n_estimators=50,  random_state=42)))
    pool.append(('rf_100',  RandomForestClassifier(n_estimators=100, random_state=42)))
    pool.append(('rf_entr', RandomForestClassifier(n_estimators=50,  criterion='entropy', random_state=42)))
    pool.append(('rf_200',  RandomForestClassifier(n_estimators=200, random_state=42)))
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        pool.append((f'sgd_{alpha}', SGDClassifier(loss='log_loss', alpha=alpha, random_state=42)))
    return pool[:20]


def main():
    print(f"device: {DEVICE}")

    X_train, y_train, classes = extract_features(TRAIN_DIR, DEVICE, train=True)
    X_test,  y_test,  _       = extract_features(TEST_DIR,  DEVICE)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    pca       = IncrementalPCA(n_components=PCA_COMPONENTS, batch_size=1024)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p  = pca.transform(X_test_s)

    results = {}

    _, acc_t, acc_tr = evaluate(KNeighborsClassifier(5), "KNN k=5",
                                X_train_p, y_train, X_test_p, y_test, classes)
    results['KNN'] = (acc_t, acc_tr)

    _, acc_t, acc_tr = evaluate(DecisionTreeClassifier(max_depth=20, random_state=42), "Decision Tree",
                                X_train_p, y_train, X_test_p, y_test, classes)
    results['Decision Tree'] = (acc_t, acc_tr)

    idx = np.random.RandomState(42).choice(len(X_train_p), min(5000, len(X_train_p)), replace=False)
    _, acc_t, acc_tr = evaluate(SVC(kernel='rbf', random_state=42, probability=True), "SVM subset 5k",
                                X_train_p[idx], y_train[idx], X_test_p, y_test, classes)
    results['SVM'] = (acc_t, acc_tr)

    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                        early_stopping=True, validation_fraction=0.1, random_state=42)
    mlp_trained, acc_t, acc_tr = evaluate(mlp, "MLP", X_train_p, y_train, X_test_p, y_test, classes)
    results['MLP'] = (acc_t, acc_tr)

    stack = StackingClassifier(
        estimators=build_stacking_pool(),
        final_estimator=RandomForestClassifier(n_estimators=200, random_state=42),
        stack_method='predict_proba',
        n_jobs=2,
        cv=10
    )
    t0 = time.time()
    stack.fit(X_train_p, y_train)
    acc_t  = accuracy_score(y_test,  stack.predict(X_test_p))
    acc_tr = accuracy_score(y_train, stack.predict(X_train_p))
    print(f"\nStacking  train: {acc_tr:.4%}  test: {acc_t:.4%}  time: {time.time()-t0:.1f}s")
    print(classification_report(y_test, stack.predict(X_test_p), target_names=classes, digits=4))
    results['Stacking'] = (acc_t, acc_tr)

    cm = confusion_matrix(y_test, mlp_trained.predict(X_test_p), normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Greens', xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix — MLP")
    plt.tight_layout()
    plt.savefig("confusion_matrix_densenet.png")

    names = list(results.keys())
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, [results[n][1]*100 for n in names], w, label='Train', color='#6366f1')
    ax.bar(x + w/2, [results[n][0]*100 for n in names], w, label='Test',  color='#ec4899')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.set_ylim(50, 105)
    plt.tight_layout()
    plt.savefig("comparison_plot_densenet.png")

    print("\nRESULTS")
    for m, (tst, trn) in results.items():
        print(f"  {m:20} test: {tst:.4%}  train: {trn:.4%}")

    joblib.dump(scaler,      'scaler_densenet.pkl')
    joblib.dump(pca,         'pca_densenet.pkl')
    joblib.dump(mlp_trained, 'mlp_densenet.pkl')


if __name__ == "__main__":
    main()
