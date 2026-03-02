# DeepFake Detection with Hybrid Architecture

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/kalil03/DeepFake-Detector)

> **[🔗 Testar a Demo ao Vivo](https://huggingface.co/spaces/kalil03/DeepFake-Detector)** — Faça upload de uma imagem direto no navegador, sem instalar nada.

Este projeto implementa um sistema robusto para detecção de DeepFakes utilizando uma abordagem híbrida que combina **Deep Learning** para extração de características e **Machine Learning Clássico** para a classificação final.

<img src="confusion_matrix_densenet.png" alt="Confusion Matrix" width="600">

## Arquitetura do Modelo

O sistema é composto por dois estágios principais:

1.  **Extração de Características (Feature Extraction)**:
    *   Utiliza a rede **DenseNet121** pré-treinada no ImageNet.
    *   Removemos a camada de classificação original e utilizamos a saída da camada de *Global Average Pooling*.
    *   Isso gera um vetor de características rico e denso para cada imagem.

2.  **Classificação (MLP)**:
    *   As características extraídas são normalizadas com **StandardScaler**.
    *   Um **MLP (Multi-Layer Perceptron)** com arquitetura `(512, 256, 128)`, *early stopping* e *adaptive learning rate* realiza a classificação final diretamente sobre as 1024 features.

### Modelos Analisados

Durante o desenvolvimento, avaliamos múltiplos classificadores sobre as features extraídas pela DenseNet121:

| Modelo | Acurácia Teste | Observação |
|---|---|---|
| Decision Tree | 68.02% | Severo overfitting |
| KNN (k=5) | 76.99% | Bom recall para fake, fraco para real |
| SVM (RBF, subset 5k) | 89.18% | Limitado pelo subset de treino |
| **MLP (256,128) + PCA=300** | **95.58%** | Boa baseline |
| **MLP (512,256,128) + PCA=500** | **96.32%** | Melhoria com mais componentes |
| **MLP (512,256,128) sem PCA** ✅ | **97.19%** | **Melhor resultado** |
| Stacking Ensemble (20 modelos) | < MLP | Custoso e inferior ao MLP |

O **MLP sem PCA** foi o grande vencedor: remover a redução dimensional preservou informação discriminativa que estava sendo descartada. Escolhido como modelo de produção pelo equilíbrio entre acurácia, tamanho e velocidade de inferência.

<img src="comparison_plot_densenet.png" alt="Model Comparison" width="600">

## Como Rodar o Projeto

### Pré-requisitos

*   Python 3.8+
*   GPU recomendada (NVIDIA ou AMD com ROCm)

### Instalação

1.  Clone o repositório:
    ```bash
    git clone https://github.com/kalil03/DeepFakes.git
    cd DeepFakes
    ```

2.  Crie e ative um ambiente virtual (recomendado):
    ```bash
    python3 -m venv venv_ia
    source venv_ia/bin/activate  # Linux/Mac
    # venv\Scripts\activate   # Windows
    ```

3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

### Usando a Interface Web

O projeto conta com uma interface web em Flask para testar imagens em tempo real.

1.  Inicie o servidor:
    ```bash
    python3 app.py
    ```
2.  Acesse **http://localhost:5000** no seu navegador.
3.  Faça upload de uma imagem para verificar se é **Real** ou **Fake**.

> **Nota para usuários AMD**: O arquivo `app.py` contém configurações específicas para GPUs AMD (`HSA_OVERRIDE_GFX_VERSION`). Se você usa NVIDIA ou CPU, pode remover essas linhas se necessário.

### Treinamento 

Para retreinar os modelos do zero, você precisará do dataset **140k Real and Fake Faces**.

1.  Baixe o dataset no Kaggle: [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
2.  Extraia o conteúdo para `archive/`. O script espera encontrar as imagens em `archive/real_vs_fake/real-vs-fake`.
3.  Execute o script de treinamento:
    ```bash
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python3 trabalho_deepfake.py
    ```
    *Este processo pode levar várias horas.*

## Estrutura de Arquivos

*   `app.py`: Aplicação Flask para inferência web.
*   `trabalho_deepfake.py`: Script completo de treinamento, validação e geração de gráficos.
*   `requirements.txt`: Lista de dependências do projeto.
*   `*.pkl`: Modelos serializados (Scaler, PCA, MLP).
*   `templates/` & `static/`: Arquivos de frontend.
*   `huggingface/`: Arquivos para deploy da demo no Hugging Face Spaces.

---
Desenvolvido como trabalho prático de Inteligência Artificial.
