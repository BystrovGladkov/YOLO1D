"""Подключение датасета ECRv1: проверка структуры и создание манифеста `dataset.yaml`.

Сигналы не преобразуются: `Dataset` читает `.npy` и исходную разметку, предобработка выполняется
в трансформах на лету. Скрипт только находит (или скачивает) датасет, проверяет пары
сигнал/разметка по сплитам и пишет манифест.

Использование:
    python data/prepare_ecr.py                                   # локальная копия по умолчанию
    python data/prepare_ecr.py --source E:/rails/public_dataset/dataset --out data/ecr_v1.yaml
    python data/prepare_ecr.py --source hf://Nasochec/ECRv1 --cache-dir H:/dataset/ecr_v1_hf

Источник: https://huggingface.co/datasets/Nasochec/ECRv1
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.datasets.ecr import DEFAULT_LOCAL_ROOT, get_ecr_dataset, validate_layout, write_manifest  # noqa: E402

DEFAULT_OUT = REPO / "data" / "ecr_v1.yaml"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=str(DEFAULT_LOCAL_ROOT),
                   help="локальная папка или hf://Nasochec/ECRv1 (по умолчанию %(default)s)")
    p.add_argument("--cache-dir", default=None, help="куда скачивать копию с Hugging Face")
    p.add_argument("--revision", default=None, help="ветка/тег/коммит репозитория Hugging Face")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="путь манифеста (по умолчанию %(default)s)")
    p.add_argument("--preprocess-cache", default=None,
                   help="папка кэша предобработки (по умолчанию cache/ рядом с манифестом)")
    args = p.parse_args(argv)

    root = get_ecr_dataset(args.source, cache_dir=args.cache_dir, revision=args.revision)
    stats = validate_layout(root)
    for split, s in stats.items():
        print(f"{split:5s}: {s['pairs']} пар сигнал/разметка")
    out = write_manifest(root, args.out, preprocess_cache=args.preprocess_cache)
    print(f"манифест: {out}")


if __name__ == "__main__":
    main()
