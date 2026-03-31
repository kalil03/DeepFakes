import os
import random
import shutil
from pathlib import Path
from typing import Dict, List

"""
Constrói o dataset multi-classe (2 classes) a partir dos downloads
brutos já existentes.

Escopo atual:
    - human_real
    - deepfake_gan

Fontes:
  human_real:
    - datasets/raw/kaggle/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/train/real/
    - datasets/raw/kaggle/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/valid/real/
    - datasets/raw/kaggle/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/test/real/
    - datasets/raw/kaggle/deepfake-and-real-images/Dataset/Train/Real/
    - datasets/raw/kaggle/deepfake-and-real-images/Dataset/Validation/Real/
    - datasets/raw/kaggle/deepfake-and-real-images/Dataset/Test/Real/

  deepfake_gan:
    - datasets/raw/kaggle/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/train/fake/
    - datasets/raw/kaggle/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/valid/fake/
    - datasets/raw/kaggle/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/test/fake/
    - datasets/raw/kaggle/deepfake-and-real-images/Dataset/Train/Fake/
    - datasets/raw/kaggle/deepfake-and-real-images/Dataset/Validation/Fake/
    - datasets/raw/kaggle/deepfake-and-real-images/Dataset/Test/Fake/

Regras:
- Deduplicar por filename entre fontes.
- Balancear classes: limitar ambas ao tamanho da menor.
- Embaralhar com seed=42.
- Split 80% / 10% / 10%.
- Recriar `multi_dataset/` do zero.
- Criar symlinks (fallback para cópia quando necessário).
"""

RANDOM_SEED = 42
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

RAW_ROOT = Path("datasets") / "raw"
KAGGLE_ROOT = RAW_ROOT / "kaggle"
MULTI_ROOT = Path("multi_dataset")


def _symlink_or_copy(src: Path, dst: Path) -> None:
  """
  Cria um symlink relativo; se não for possível, copia o arquivo.
  """
  dst.parent.mkdir(parents=True, exist_ok=True)
  try:
      if dst.exists():
          return
      os.symlink(os.path.relpath(src, dst.parent), dst)
  except (OSError, NotImplementedError):
      if not dst.exists():
          shutil.copy2(src, dst)


def _gather_from_dirs(dirs: List[Path]) -> List[Path]:
  """
  Coleta todos os arquivos de imagem em múltiplos diretórios,
  deduplicando por filename (basename) entre fontes.
  """
  files: List[Path] = []
  name_to_path: Dict[str, Path] = {}

  for d in dirs:
      if not d.exists():
          continue
      for p in d.rglob("*"):
          if p.is_file() and p.suffix.lower() in IMG_EXTS:
              name = p.name
              if name not in name_to_path:
                  name_to_path[name] = p

  files = list(name_to_path.values())
  return files


def build_balanced_filelists() -> Dict[str, List[Path]]:
  """
  Monta listas de arquivos para human_real e deepfake_gan,
  aplicando deduplicação e balanceamento.
  """
  sources = {
      "human_real": [
          KAGGLE_ROOT
          / "140k-real-and-fake-faces"
          / "real_vs_fake"
          / "real-vs-fake"
          / split
          / "real"
          for split in ("train", "valid", "test")
      ]
      + [
          KAGGLE_ROOT / "deepfake-and-real-images" / "Dataset" / split / "Real"
          for split in ("Train", "Validation", "Test")
      ],
      "deepfake_gan": [
          KAGGLE_ROOT
          / "140k-real-and-fake-faces"
          / "real_vs_fake"
          / "real-vs-fake"
          / split
          / "fake"
          for split in ("train", "valid", "test")
      ]
      + [
          KAGGLE_ROOT / "deepfake-and-real-images" / "Dataset" / split / "Fake"
          for split in ("Train", "Validation", "Test")
      ],
  }

  all_lists: Dict[str, List[Path]] = {}

  for slug, dirs in sources.items():
      files = _gather_from_dirs(dirs)
      all_lists[slug] = files
      print(f"[{slug}] arquivos únicos (por filename): {len(files)}")

  # Balanceamento: limitar ambas ao tamanho da menor
  available_counts = {slug: len(files) for slug, files in all_lists.items()}
  min_count = min(available_counts.values())
  print(f"\n[balance] menor classe tem {min_count} imagens; limitando ambas a isso.")

  random.seed(RANDOM_SEED)
  balanced: Dict[str, List[Path]] = {}
  for slug, files in all_lists.items():
      files_shuffled = list(files)
      random.shuffle(files_shuffled)
      balanced[slug] = files_shuffled[:min_count]
      print(f"[{slug}] balanceado para {len(balanced[slug])} imagens.")

  return balanced


def split_and_save(slug: str, files: List[Path]) -> Dict[str, int]:
  """
  Faz split 80/10/10 e escreve symlinks/cópias em multi_dataset/slug/...
  """
  random.seed(RANDOM_SEED)
  files = list(files)
  random.shuffle(files)

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

  return {"train": len(train_files), "val": len(val_files), "test": len(test_files)}


def main() -> None:
  # Limpa multi_dataset/ para remover dados sintéticos antigos
  if MULTI_ROOT.exists():
      print(f"[wipe] Removendo diretório existente: {MULTI_ROOT}")
      shutil.rmtree(MULTI_ROOT)
  MULTI_ROOT.mkdir(parents=True, exist_ok=True)

  print("[INFO] Construindo listas balanceadas para 2 classes (human_real, deepfake_gan)...")
  filelists = build_balanced_filelists()

  all_counts: Dict[str, Dict[str, int]] = {}

  for slug, files in filelists.items():
      print(f"\n[CLASS] {slug}")
      print(f"  Arquivos balanceados: {len(files)}")
      counts = split_and_save(slug, files)
      all_counts[slug] = counts
      total = sum(counts.values())
      print(
          f"  => train={counts['train']}  val={counts['val']}  "
          f"test={counts['test']}  (total={total})"
      )

  print("\n==================== RESUMO FINAL ====================")
  print("Classe\t\tTrain\tVal\tTest\tTotal")
  for slug, counts in all_counts.items():
      total = counts["train"] + counts["val"] + counts["test"]
      print(
          f"{slug:16s}\t{counts['train']:6d}\t{counts['val']:6d}\t"
          f"{counts['test']:6d}\t{total:6d}"
      )

  print("\n[OK] multi_dataset reconstruído com 2 classes balanceadas.")


if __name__ == "__main__":
  main()

