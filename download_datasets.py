import os
import subprocess
from pathlib import Path

"""
Script de download dos datasets brutos usados no projeto.

Este script NÃO organiza as imagens em classes finais — isso é feito
por `build_multi_dataset.py`. Aqui apenas garantimos que todos os
datasets necessários estejam baixados em `datasets/raw/`.

Requisitos:
- Kaggle CLI configurado (arquivo ~/.kaggle/kaggle.json)
- Biblioteca `huggingface_hub` instalada (`pip install huggingface_hub`)

Uso recomendado:
    python download_datasets.py

Se algum download falhar (ex.: falta de credenciais), o script exibirá
um aviso, mas continuará tentando os demais.
"""

RAW_ROOT = Path("datasets") / "raw"
RAW_ROOT.mkdir(parents=True, exist_ok=True)


def run(cmd, cwd=None):
    print(f"\n[cmd] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Falha ao executar comando: {e}")


def download_kaggle_dataset(dataset_id: str, target_dir: Path, unzip: bool = True):
    """
    Baixa um dataset do Kaggle usando a CLI oficial.

    Exemplo de dataset_id:
      - 'prasoonkottarathil/flickr-faces-hq-dataset-ffhq'
      - 'xhlulu/140k-real-and-fake-faces'
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[+] Kaggle :: {dataset_id} -> {target_dir}")

    args = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_id,
        "-p",
        str(target_dir),
    ]
    if unzip:
        args.append("--unzip")

    run(args)


def download_hf_dataset(repo_id: str, target_dir: Path, allow_patterns=None):
    """
    Baixa um dataset ou repositório de imagens do Hugging Face Hub.

    Exemplo de repo_id:
      - 'defactify/defactify'
      - 'phantom9999/midjourney-text-to-image'
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "[WARN] huggingface_hub não instalado. "
            "Instale com `pip install huggingface_hub` para baixar datasets do HF."
        )
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[+] HuggingFace :: {repo_id} -> {target_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=allow_patterns,
        )
    except Exception as e:
        print(f"[WARN] Falha ao baixar {repo_id} do Hugging Face: {e}")


def main():
    # ------------------------------------------------------------------
    # Kaggle datasets
    # ------------------------------------------------------------------
    kaggle_root = RAW_ROOT / "kaggle"

    # 👤 Human (Real)
    download_kaggle_dataset(
        "prasoonkottarathil/flickr-faces-hq-dataset-ffhq",
        kaggle_root / "flickr-faces-hq-dataset-ffhq",
    )
    download_kaggle_dataset(
        "xhlulu/140k-real-and-fake-faces",
        kaggle_root / "140k-real-and-fake-faces",
    )

    # 🤖 Deepfake/GAN
    download_kaggle_dataset(
        "manjilkarki/deepfake-and-real-images",
        kaggle_root / "deepfake-and-real-images",
    )

    # ✨ Gemini / Imagen
    download_kaggle_dataset(
        "iamkdblue/google-imagen-generated-dataset",
        kaggle_root / "google-imagen-generated-dataset",
    )

    # ------------------------------------------------------------------
    # Hugging Face datasets
    # ------------------------------------------------------------------
    hf_root = RAW_ROOT / "huggingface"

    # 🎨 DALL-E / IA genéricas
    download_hf_dataset(
        "defactify/defactify",
        hf_root / "defactify",
        allow_patterns=["*.jpg", "*.jpeg", "*.png", "*.webp"],
    )

    # Tentativas de datasets adicionais relacionados a DALL-E
    # (podem ou não existir; uso melhor esforço)
    download_hf_dataset(
        "dalle3-dataset",
        hf_root / "dalle3-dataset",
        allow_patterns=["*.jpg", "*.jpeg", "*.png", "*.webp"],
    )
    download_hf_dataset(
        "poloclub/diffusiondb",
        hf_root / "diffusiondb",
        allow_patterns=["*.jpg", "*.jpeg", "*.png", "*.webp"],
    )

    # 🌌 Midjourney
    download_hf_dataset(
        "phantom9999/midjourney-text-to-image",
        hf_root / "midjourney-text-to-image",
        allow_patterns=["*.jpg", "*.jpeg", "*.png", "*.webp"],
    )

    # 🖌️ Stable Diffusion
    # Observação: 'stabilityai/stable-diffusion-xl' é um modelo, não
    # um dataset de imagens; ainda assim tentamos baixar qualquer
    # amostra de imagem eventualmente presente.
    download_hf_dataset(
        "stabilityai/stable-diffusion-xl",
        hf_root / "stable-diffusion-xl",
        allow_patterns=["*.jpg", "*.jpeg", "*.png", "*.webp"],
    )

    # ✨ Gemini / Imagen - melhor esforço para encontrar imagens
    # (você pode ajustar estes IDs para datasets mais adequados).
    download_hf_dataset(
        "google-imagen",
        hf_root / "google-imagen",
        allow_patterns=["*.jpg", "*.jpeg", "*.png", "*.webp"],
    )
    download_hf_dataset(
        "gemini-generated-images",
        hf_root / "gemini-generated-images",
        allow_patterns=["*.jpg", "*.jpeg", "*.png", "*.webp"],
    )

    print("\n[OK] Download finalizado (com melhor esforço).")
    print(
        "Agora rode `python build_multi_dataset.py` para organizar as "
        "imagens no formato multi-classe."
    )


if __name__ == "__main__":
    main()

