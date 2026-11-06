"""Загрузка записей шумов MIT-BIH Noise Stress Test Database (NSTDB) для аугментации N5.

Скачиваются только три записи шумов (ЭКГ-записи стресс-теста не нужны):
    bw — дрейф изолинии (baseline wander),
    ma — мышечный артефакт (muscle / EMG),
    em — артефакт движения электродов (electrode motion).
Каждая — 2 канала, 360 Гц, 650 000 отсчётов (30 мин). Файлы кладутся в `data/raw/nstdb/`
и сверяются по SHA-256 со списком `SHA256SUMS.txt` PhysioNet; затем проверяется чтение через `wfdb`.

Использование:
    python data/download_nstdb.py                # скачать (если нужно) и проверить
    python data/download_nstdb.py --check-only
    python data/download_nstdb.py --force

Источник: https://physionet.org/content/nstdb/1.0.0/ (лицензия ODC-By 1.0).
Цитировать: Moody G. B., Muldrow W. E., Mark R. G. A noise stress test for arrhythmia detectors.
Computers in Cardiology 11:381–384, 1984; и PhysioNet (Goldberger et al., 2000).
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import wfdb

BASE_URL = "https://physionet.org/files/nstdb/1.0.0/"
DEFAULT_DIR = Path(__file__).resolve().parent / "raw" / "nstdb"
RECORDS = ["bw", "ma", "em"]
FS = 360
CHANNELS = 2
LENGTH = 650000


def expected_sums() -> dict:
    """Имя файла → SHA-256 из SHA256SUMS.txt PhysioNet (только нужные файлы)."""
    text = urllib.request.urlopen(BASE_URL + "SHA256SUMS.txt", timeout=60).read().decode("utf-8")
    wanted = {f"{r}.{ext}" for r in RECORDS for ext in ("dat", "hea")}
    sums = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] in wanted:
            sums[parts[1]] = parts[0]
    missing = wanted - set(sums)
    if missing:
        raise RuntimeError(f"в SHA256SUMS.txt нет {sorted(missing)}")
    return sums


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root: Path, sums: dict) -> list:
    """Контрольные суммы и чтение записей. Возвращает список ошибок."""
    errors = []
    for name, digest in sums.items():
        path = root / name
        if not path.is_file():
            errors.append(f"нет файла {name}")
        elif sha256(path) != digest:
            errors.append(f"неверная контрольная сумма: {name}")
    if errors:
        return errors
    for record in RECORDS:
        rec = wfdb.rdrecord(str(root / record))
        if rec.fs != FS or rec.p_signal.shape != (LENGTH, CHANNELS):
            errors.append(f"{record}: fs={rec.fs}, форма {rec.p_signal.shape}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    sums = expected_sums()
    if not args.check_only:
        for name, digest in sums.items():
            path = root / name
            if path.is_file() and sha256(path) == digest and not args.force:
                continue
            print(f"Скачивание {name}")
            urllib.request.urlretrieve(BASE_URL + name, path)
    errors = check(root, sums)
    if errors:
        print("Найдено проблем:", *errors, sep="\n  - ")
        return 1
    print(f"OK: записи шумов {', '.join(RECORDS)} — {CHANNELS} канала × {LENGTH} отсчётов @ {FS} Гц. Данные: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
