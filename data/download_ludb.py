"""Загрузка и проверка датасета LUDB (Lobachevsky University Electrocardiography Database).

Скачивает архив LUDB v1.0.1 с PhysioNet, распаковывает его в `data/raw/ludb/`,
сверяет SHA-256 всех файлов со списком `SHA256SUMS.txt` из архива и проверяет
структуру данных средствами `wfdb`:

- 200 записей из файла `RECORDS`;
- у каждой записи 12 отведений (i, ii, iii, avr, avl, avf, v1–v6), 500 Гц, 5000 отсчётов;
- для каждого отведения есть читаемый файл разметки с символами `(`, `)`, `p`, `N`, `t`.

Повторный запуск ничего не скачивает, если данные уже на месте и проходят проверку.

Использование:
    python data/download_ludb.py                # скачать (если нужно) и проверить
    python data/download_ludb.py --check-only   # только проверить уже скачанные данные
    python data/download_ludb.py --force        # скачать заново

Источник: https://physionet.org/content/ludb/1.0.1/ (лицензия ODC-By 1.0).
Подробнее о датасете — docs/research/R1_ludb.md.
"""
import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import wfdb

URL = ("https://physionet.org/static/published-projects/ludb/"
       "lobachevsky-university-electrocardiography-database-1.0.1.zip")
DEFAULT_DIR = Path(__file__).resolve().parent / "raw" / "ludb"

NUM_RECORDS = 200
LEADS = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]
FS = 500
SIGNAL_LENGTH = 5000
ANNOTATION_SYMBOLS = {"(", ")", "p", "N", "t"}


def download(url: str, destination: Path) -> None:
    """Скачивает файл по `url` в `destination`, показывая прогресс (раз в мегабайт)."""
    last_mb = [-1]

    def progress(blocks, block_size, total):
        done = min(blocks * block_size, total) if total > 0 else blocks * block_size
        if done // 2**20 != last_mb[0] or done == total:
            last_mb[0] = done // 2**20
            print(f"\r  {done / 2**20:5.1f} / {total / 2**20:.1f} МБ", end="", flush=True)

    print(f"Скачивание {url}")
    urllib.request.urlretrieve(url, destination, reporthook=progress)
    print()


def extract(archive: Path, target: Path) -> None:
    """Распаковывает архив PhysioNet в `target`, убирая верхнюю папку архива."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        # в архиве всё лежит в одной папке вида lobachevsky-university-...-1.0.1/
        roots = [p for p in Path(tmp).iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"Неожиданная структура архива: {[p.name for p in roots]}")
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(roots[0]), str(target))


def verify_checksums(root: Path) -> list:
    """Сверяет SHA-256 файлов со списком `SHA256SUMS.txt`. Возвращает список ошибок."""
    errors = []
    sums_file = root / "SHA256SUMS.txt"
    if not sums_file.is_file():
        return [f"нет файла {sums_file.name}"]
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        path = root / name.strip()
        if not path.is_file():
            errors.append(f"нет файла {name}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"неверная контрольная сумма: {name}")
    return errors


def check_integrity(root: Path) -> list:
    """Проверяет записи и разметку средствами `wfdb`. Возвращает список ошибок."""
    errors = []
    records_file = root / "RECORDS"
    if not records_file.is_file():
        return ["нет файла RECORDS"]
    records = [r for r in records_file.read_text().split() if r]
    if len(records) != NUM_RECORDS:
        errors.append(f"в RECORDS {len(records)} записей вместо {NUM_RECORDS}")
    if not (root / "ludb.csv").is_file():
        errors.append("нет файла ludb.csv с метаданными")

    for name in records:
        path = str(root / name)
        try:
            header = wfdb.rdheader(path)
        except Exception as exc:  # повреждённый или отсутствующий заголовок
            errors.append(f"{name}: не читается заголовок ({exc})")
            continue
        if header.fs != FS or header.sig_len != SIGNAL_LENGTH or header.sig_name != LEADS:
            errors.append(f"{name}: fs={header.fs}, длина={header.sig_len}, отведения={header.sig_name}")
        signal = wfdb.rdrecord(path).p_signal
        if signal.shape != (SIGNAL_LENGTH, len(LEADS)):
            errors.append(f"{name}: форма сигнала {signal.shape}")
        for lead in LEADS:
            try:
                ann = wfdb.rdann(path, lead)
            except Exception as exc:
                errors.append(f"{name}.{lead}: не читается разметка ({exc})")
                continue
            if len(ann.sample) == 0:
                errors.append(f"{name}.{lead}: пустая разметка")
            elif not set(ann.symbol) <= ANNOTATION_SYMBOLS:
                errors.append(f"{name}.{lead}: неизвестные символы {set(ann.symbol) - ANNOTATION_SYMBOLS}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="куда положить датасет")
    parser.add_argument("--check-only", action="store_true", help="только проверить скачанные данные")
    parser.add_argument("--force", action="store_true", help="скачать заново, даже если данные есть")
    args = parser.parse_args()
    root = args.dir.resolve()

    have_data = (root / "RECORDS").is_file()
    if args.check_only and not have_data:
        print(f"Данных нет в {root}")
        return 1
    if args.force or not have_data:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "ludb.zip"
            download(URL, archive)
            extract(archive, root)
        print(f"Распаковано в {root}")

    print("Проверка контрольных сумм...")
    errors = verify_checksums(root)
    print("Проверка записей и разметки...")
    errors += check_integrity(root)
    if errors:
        print(f"Найдено проблем: {len(errors)}")
        for e in errors[:50]:
            print("  -", e)
        return 1
    print(f"OK: {NUM_RECORDS} записей × {len(LEADS)} отведений × {SIGNAL_LENGTH} отсчётов @ {FS} Гц, "
          f"разметка всех отведений на месте. Данные: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
