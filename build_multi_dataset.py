import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

"""
Constrói o dataset multi-classe a partir dos downloads brutos.

Pré-requisito:
    python download_datasets.py

Estrutura final:
    multi_dataset/
      human_real/{train,val,test}/
      deepfake_gan/{train,val,test}/
      dalle3/{train,val,test}/
      midjourney/{train,val,test}/
      stable_diffusion/{train,val,test}/
      gemini_imagen/{train,val,test}/

Regras:
- Mínimo de 8.000 imagens por classe (melhor esforço).
- Split 80% / 10% / 10%.
- Paths relativos ao repositório (symlinks quando possível).
"""

RANDOM_SEED = 42
MIN_PER_CLASS = 8000
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

RAW_ROOT = Path("datasets") / "raw"
MULTI_ROOT = Path("multi_dataset")


def _gather_files(patterns: List[Path]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        matched = list(pat.rglob("*"))
        for p in matched:
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p)
    # remover duplicados preservando ordem
    seen = set()
    unique_files = []
    for p in files:
        if p not in seen:
            seen.add(p)
            unique_files.append(p)
    return unique_files


def _gather_defactify_by_prefix(base_dir: Path, prefixes: Tuple[str, ...]) -> List[Path]:
    """
    Coleta imagens do Defactify pelo prefixo do filename (gen_1*, gen_4*, etc).
    """
    files: List[Path] = []
    if not base_dir.exists():
        return files
    for p in base_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        name = p.name
        if any(name.startswith(pref) for pref in prefixes):
            files.append(p)
    return files


def _symlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            return
        os.symlink(os.path.relpath(src, dst.parent), dst)
    except (OSError, NotImplementedError):
        # fallback para cópia em sistemas sem suporte a symlink
        import shutil

        if not dst.exists():
            shutil.copy2(src, dst)


def build_class_filelists() -> Dict[str, List[Path]]:
    """
    Coleta listas de arquivos para cada slug de classe, combinando
    múltiplas fontes (Kaggle + Hugging Face).
    """
    kaggle_root = RAW_ROOT / "kaggle"
    hf_root = RAW_ROOT / "huggingface"

    filelists: Dict[str, List[Path]] = {}

    # 👤 Human (Real)
    human_patterns = [
        # 140k real/fake - pasta real/
        kaggle_root
        / "140k-real-and-fake-faces",
        # FFHQ (flickr faces)
        kaggle_root / "flickr-faces-hq-dataset-ffhq",
    ]
    human_files = _gather_files(human_patterns)
    filelists["human_real"] = human_files

    # 🤖 Deepfake / GAN
    deepfake_patterns = [
        kaggle_root / "140k-real-and-fake-faces",
        kaggle_root / "deepfake-and-real-images",
    ]
    deepfake_files = _gather_files(deepfake_patterns)
    filelists["deepfake_gan"] = deepfake_files

    # 🎨 DALL-E 3 (principalmente Defactify gen_4*)
    defactify_dir = hf_root / "defactify"
    dalle_files = _gather_defactify_by_prefix(defactify_dir, ("gen_4",))

    # complementos de IA genérica, usados como reforço de amostras DALL-E-like
    dalle_files += _gather_files(
        [
            hf_root / "dalle3-dataset",
            hf_root / "diffusiondb",
        ]
    )
    filelists["dalle3"] = dalle_files

    # 🌌 Midjourney v6 (Defactify gen_5* + dataset dedicado)
    mid_files = _gather_defactify_by_prefix(defactify_dir, ("gen_5",))
    mid_files += _gather_files(
        [
            hf_root / "midjourney-text-to-image",
        ]
    )
    filelists["midjourney"] = mid_files

    # 🖌️ Stable Diffusion (Defactify gen_1/2/3* + melhor esforço)
    sd_files = _gather_defactify_by_prefix(defactify_dir, ("gen_1", "gen_2", "gen_3"))
    sd_files += _gather_files(
        [
            hf_root / "stable-diffusion-xl",
        ]
    )
    filelists["stable_diffusion"] = sd_files

    # ✨ Gemini / Imagen (Kaggle + HF melhor esforço)
    gemini_files = _gather_files(
        [
            kaggle_root / "google-imagen-generated-dataset",
            hf_root / "google-imagen",
            hf_root / "gemini-generated-images",
        ]
    )
    filelists["gemini_imagen"] = gemini_files

    return filelists


def split_and_save(
    files: List[Path],
    slug: str,
    min_per_class: int = MIN_PER_CLASS,
) -> Dict[str, int]:
    """
    Faz split 80/10/10 e escreve symlinks/cópias em multi_dataset/slug/...
    Retorna contagens por split.
    """
    random.seed(RANDOM_SEED)
    files = list(dict.fromkeys(files))  # remove duplicados preservando ordem
    random.shuffle(files)

    if not files:
        print(f"[WARN] Nenhuma imagem encontrada para classe '{slug}'.")
        return {"train": 0, "val": 0, "test": 0}

    if len(files) < min_per_class:
        print(
            f"[WARN] Classe '{slug}' abaixo do mínimo desejado: "
            f"{len(files)} < {min_per_class}"
        )

    # Opcional: limitar para evitar datasets gigantescos em testes locais
    # Aqui usamos tudo; se quiser limitar, ajuste aqui.
    n = len(files)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    for f in train_files:
        dst = MULTI_ROOT / slug / "train" / f.name
        _symlink_or_copy(f, dst)
    for f in val_files:
        dst = MULTI_ROOT / slug / "val" / f.name
        _symlink_or_copy(f, dst)
    for f in test_files:
        dst = MULTI_ROOT / slug / "test" / f.name
        _symlink_or_copy(f, dst)

    return {
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }


def main():
    MULTI_ROOT.mkdir(parents=True, exist_ok=True)

    print("[INFO] Construindo listas de arquivos por classe...")
    filelists = build_class_filelists()

    all_counts: Dict[str, Dict[str, int]] = {}
    total_per_class: Dict[str, int] = {}

    for slug, files in filelists.items():
        print(f"\n[CLASS] {slug}")
        print(f"  Arquivos encontrados (bruto): {len(files)}")
        counts = split_and_save(files, slug)
        all_counts[slug] = counts
        total_per_class[slug] = sum(counts.values())
        print(
            f"  => train={counts['train']}  val={counts['val']}  "
            f"test={counts['test']}  (total={total_per_class[slug]})"
        )

    # ------------------------------------------------------------------
    # Relatório final por classe/split
    # ------------------------------------------------------------------
    print("\n==================== RESUMO FINAL ====================")
    print("Classe\t\tTrain\tVal\tTest\tTotal")
    for slug, counts in all_counts.items():
        total = total_per_class[slug]
        print(
            f"{slug:16s}\t{counts['train']:6d}\t{counts['val']:6d}\t"
            f"{counts['test']:6d}\t{total:6d}"
        )

    # ------------------------------------------------------------------
    # Verificação de balanceamento entre classes
    # ------------------------------------------------------------------
    if total_per_class:
        max_size = max(total_per_class.values())
        print("\n[INFO] Verificando balanceamento de classes...")
        for slug, total in total_per_class.items():
            if total < 0.8 * max_size:
                print(
                    f"[WARN] Classe '{slug}' desbalanceada: {total} "
                    f"(< 80% de {max_size})"
                )
        print("\n[OK] Construção do multi_dataset concluída.")
    else:
        print(
            "\n[ERRO] Nenhuma classe foi construída. "
            "Verifique se `download_datasets.py` foi executado com sucesso "
            "e se os caminhos/IDs dos datasets estão corretos."
        )


if __name__ == "__main__":
    main()

