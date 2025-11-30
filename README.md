# DeepFake Detection with Hybrid Architecture

Este projeto implementa um sistema robusto para detecção de DeepFakes utilizando uma abordagem híbrida que combina **Deep Learning** para extração de características e **Machine Learning Clássico** para a classificação final.

<img src="confusion_matrix_densenet.png" alt="Confusion Matrix" width="600">

## Arquitetura do Modelo

O sistema é composto por dois estágios principais:

1.  **Extração de Características (Feature Extraction)**:
    *   Utiliza a rede **DenseNet121** pré-treinada no ImageNet.
    *   Removemos a camada de classificação original e utilizamos a saída da camada de *Global Average Pooling*.
    *   Isso gera um vetor de características rico e denso para cada imagem.

2.  **Classificação (Stacking Ensemble)**:
    *   As características extraídas passam por uma redução de dimensionalidade via **PCA** (Principal Component Analysis).
    *   Um **Stacking Classifier** combina as previsões de múltiplos modelos fortes:
        *   MLP (Multi-Layer Perceptron)
        *   SVM (Support Vector Machine)
        *   Random Forest
    *   O meta-classificador (**Random Forest**) toma a decisão final baseada nas saídas desses modelos.



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

O projeto conta com uma interface web simples em Flask para testar imagens em tempo real.

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
*   `*.pkl`: Modelos serializados (Scaler, PCA, Stacking Model).
*   `templates/` & `static/`: Arquivos de frontend.

---
Desenvolvido como trabalho prático de Inteligência Artificial.
