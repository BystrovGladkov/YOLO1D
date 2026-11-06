"""Чтение датасета LUDB: сигналы, поотведенческая разметка P/QRS/T, метаданные.

Разметка LUDB хранится отдельно для каждого из 12 отведений. Каждая волна — тройка
символов WFDB: `(` (onset), пик `p` | `N` | `t` (P | QRS | T), `)` (offset).
В ~0.3 % волн тройка нарушена (см. docs/research/R1_ludb.md §2.4). Парсер
`parse_annotation` разбирает такие случаи явно и возвращает их как `AnnotationIssue`,
чтобы следующий этап (D3) сам решал, что делать с отведением.

Использование:
    from yolo_1d.utils.datasets.ludb import load_record, iter_records
    rec = load_record("data/raw/ludb", "1")
    rec.signal            # np.ndarray (12, 5000), мВ
    rec.waves["ii"]       # список Wave отведения II
    rec.zone("ii")        # (первый onset, последний offset) — размеченная зона
"""
import csv
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import wfdb

FS = 500
SIGNAL_LENGTH = 5000
LEADS = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]
# символ пика WFDB -> имя класса; порядок классов задаёт их индексы для детекции
PEAK_SYMBOLS = {"p": "P", "N": "QRS", "t": "T"}
CLASS_NAMES = ["P", "QRS", "T"]
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}
# пик может стоять раньше onset не более чем на столько отсчётов, чтобы считать это опиской
MAX_PEAK_BEFORE_ONSET = 2

# Виды нарушений разметки. Волны с kept=True исправлены и есть в списке волн,
# с kept=False — отброшены (по ним нельзя построить отрезок [onset, offset]).
ISSUE_PEAK_BEFORE_ONSET = "peak_before_onset"   # пик раньше onset на <= 2 отсчёта -> onset := пик
ISSUE_MERGED_DUPLICATES = "merged_duplicates"   # k вложенных копий одной волны -> медианы границ
ISSUE_MISSING_ONSET = "missing_onset"           # пик и offset без onset (чаще всего QRS при ЭКС)
ISSUE_MISSING_OFFSET = "missing_offset"         # onset и пик без offset
ISSUE_NO_PEAK = "no_peak"                       # пара скобок без пика: класс неизвестен
ISSUE_UNPARSED = "unparsed"                     # символ вне любой распознанной конструкции


@dataclass(frozen=True)
class Wave:
    """Размеченная волна одного отведения; позиции — в отсчётах (500 Гц)."""
    cls: str
    onset: int
    peak: int
    offset: int

    @property
    def cls_id(self) -> int:
        return CLASS_IDS[self.cls]


@dataclass(frozen=True)
class AnnotationIssue:
    """Нарушение структуры разметки в отведении."""
    kind: str
    sample: int                 # позиция первого символа конструкции
    cls: Optional[str] = None   # класс, если известен
    kept: bool = False          # волна исправлена и оставлена в списке


@dataclass
class LudbRecord:
    """Одна запись LUDB."""
    name: str
    signal: np.ndarray                                  # (12, 5000) float32, мВ
    waves: Dict[str, List[Wave]]                        # отведение -> волны по возрастанию onset
    issues: Dict[str, List[AnnotationIssue]]            # отведение -> нарушения разметки
    meta: Dict[str, List[str]] = field(default_factory=dict)  # строка ludb.csv: поле -> список значений

    def zone(self, lead: str) -> Optional[Tuple[int, int]]:
        """Размеченная зона отведения: (первый onset, последний offset); None, если волн нет."""
        w = self.waves[lead]
        return (w[0].onset, w[-1].offset) if w else None

    def dropped(self, lead: str) -> List[AnnotationIssue]:
        """Нарушения, из-за которых волна отброшена: в сигнале есть волна без метки.

        Отведение с такими нарушениями внутри размеченной зоны нельзя использовать для
        обучения как есть — сеть будет учиться считать реальную волну фоном (решается в D3).
        """
        return [issue for issue in self.issues[lead] if not issue.kept]

    def common_zone(self) -> Optional[Tuple[int, int]]:
        """Пересечение размеченных зон всех отведений (зона для 12-канального режима M)."""
        zones = [self.zone(lead) for lead in LEADS]
        if any(z is None for z in zones):
            return None
        start, end = max(z[0] for z in zones), min(z[1] for z in zones)
        return (start, end) if start < end else None


def parse_annotation(samples, symbols) -> Tuple[List[Wave], List[AnnotationIssue]]:
    """Разбирает последовательность символов разметки одного отведения.

    Распознаёт нормальные тройки `( пик )` и известные дефекты LUDB:
    пик раньше onset (исправляется), вложенные копии одной волны `((( ppp )))`
    (сливаются по медианам), волны без onset/offset/пика (отбрасываются).

    Returns
    -------
    waves
        Корректные и исправленные волны, отсортированные по onset.
    issues
        Все обнаруженные нарушения.
    """
    x = [int(v) for v in samples]
    s = list(symbols)
    n = len(s)
    waves, issues = [], []

    def run(start: int, predicate) -> int:
        """Длина серии подряд идущих символов, удовлетворяющих predicate."""
        k = 0
        while start + k < n and predicate(s[start + k]):
            k += 1
        return k

    i = 0
    while i < n:
        sym = s[i]
        if sym == "(":
            opens = run(i, lambda c: c == "(")
            if opens > 1:
                # вложенные копии: k '(' , k одинаковых пиков, k ')'
                peaks = run(i + opens, lambda c: c in PEAK_SYMBOLS)
                closes = run(i + opens + peaks, lambda c: c == ")")
                same_cls = len(set(s[i + opens:i + opens + peaks])) == 1
                if opens == peaks == closes and same_cls:
                    k = opens
                    cls = PEAK_SYMBOLS[s[i + k]]
                    onset = int(median(x[i:i + k]))
                    peak = int(median(x[i + k:i + 2 * k]))
                    offset = int(median(x[i + 2 * k:i + 3 * k]))
                    waves.append(Wave(cls, onset, peak, offset))
                    issues.append(AnnotationIssue(ISSUE_MERGED_DUPLICATES, x[i], cls, kept=True))
                    i += 3 * k
                else:
                    issues.append(AnnotationIssue(ISSUE_UNPARSED, x[i]))
                    i += 1
            elif i + 1 < n and s[i + 1] in PEAK_SYMBOLS:
                cls = PEAK_SYMBOLS[s[i + 1]]
                if i + 2 < n and s[i + 2] == ")":
                    waves.append(Wave(cls, x[i], x[i + 1], x[i + 2]))
                    i += 3
                else:
                    issues.append(AnnotationIssue(ISSUE_MISSING_OFFSET, x[i], cls))
                    i += 2
            elif i + 1 < n and s[i + 1] == ")":
                issues.append(AnnotationIssue(ISSUE_NO_PEAK, x[i]))
                i += 2
            else:
                issues.append(AnnotationIssue(ISSUE_UNPARSED, x[i]))
                i += 1
        elif sym in PEAK_SYMBOLS:
            cls = PEAK_SYMBOLS[sym]
            if (i + 2 < n and s[i + 1] == "(" and s[i + 2] == ")"
                    and x[i + 1] - x[i] <= MAX_PEAK_BEFORE_ONSET):
                # пик на 1–2 отсчёта раньше onset — описка: onset := пик
                waves.append(Wave(cls, x[i], x[i], x[i + 2]))
                issues.append(AnnotationIssue(ISSUE_PEAK_BEFORE_ONSET, x[i], cls, kept=True))
                i += 3
            elif i + 1 < n and s[i + 1] == ")":
                issues.append(AnnotationIssue(ISSUE_MISSING_ONSET, x[i], cls))
                i += 2
            else:
                issues.append(AnnotationIssue(ISSUE_UNPARSED, x[i], cls))
                i += 1
        else:
            issues.append(AnnotationIssue(ISSUE_UNPARSED, x[i]))
            i += 1

    waves.sort(key=lambda w: w.onset)
    return waves, issues


def load_metadata(root) -> Dict[str, Dict[str, List[str]]]:
    """Читает `ludb.csv`: ID записи -> {поле: список значений}.

    Ячейки ludb.csv многострочные (несколько диагнозов через перевод строки),
    поэтому каждое значение — список строк без пустых.
    """
    path = Path(root) / "ludb.csv"
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    meta = {}
    for row in rows:
        meta[row["ID"].strip()] = {key.strip(): [v.strip() for v in (value or "").splitlines() if v.strip()]
                                   for key, value in row.items() if key != "ID"}
    return meta


def rhythm_group(meta: Dict[str, List[str]]) -> str:
    """Группа ритма для стратификации разбиений (R1 §4.4): 'paced', 'af_flutter' или 'sinus'."""
    if meta.get("Cardiac pacing"):
        return "paced"
    rhythms = " ".join(meta.get("Rhythms", [])).lower()
    if "fibrillation" in rhythms or "flutter" in rhythms:
        return "af_flutter"
    return "sinus"


def record_names(root) -> List[str]:
    """Имена записей ('1'…'200') в порядке файла RECORDS."""
    lines = (Path(root) / "RECORDS").read_text().split()
    return [Path(line).name for line in lines if line]


def load_record(root, name: str, metadata: Optional[Dict[str, Dict[str, List[str]]]] = None) -> LudbRecord:
    """Читает одну запись: сигнал (12, 5000) в мВ, разметку всех отведений и метаданные."""
    path = str(Path(root) / "data" / name)
    record = wfdb.rdrecord(path)
    if record.sig_name != LEADS or record.fs != FS:
        raise ValueError(f"{name}: неожиданные отведения {record.sig_name} или частота {record.fs}")
    signal = record.p_signal.T.astype(np.float32)
    waves, issues = {}, {}
    for lead in LEADS:
        ann = wfdb.rdann(path, lead)
        waves[lead], issues[lead] = parse_annotation(ann.sample, ann.symbol)
    if metadata is None:
        metadata = load_metadata(root)
    return LudbRecord(name, signal, waves, issues, metadata.get(name, {}))


def iter_records(root) -> Iterator[LudbRecord]:
    """Последовательно читает все 200 записей."""
    metadata = load_metadata(root)
    for name in record_names(root):
        yield load_record(root, name, metadata)
