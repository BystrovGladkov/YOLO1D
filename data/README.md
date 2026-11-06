# data/

Датасеты и всё, что с ними связано. Сырые данные в git не хранятся (`data/raw/` в `.gitignore`), их нужно скачать скриптом.

## LUDB

```bash
python data/download_ludb.py               # скачать в data/raw/ludb/ и проверить
python data/download_ludb.py --check-only  # проверить уже скачанное
```

Скрипт скачивает LUDB v1.0.1 с PhysioNet (13.8 МБ) и проверяет:
- SHA-256 всех файлов по `SHA256SUMS.txt`;
- 200 записей × 12 отведений × 5000 отсчётов @ 500 Гц;
- наличие и читаемость разметки каждого отведения.

Структура `data/raw/ludb/`: `RECORDS`, `ludb.csv` (метаданные: ритм, диагнозы, ЭКС), `data/<N>.hea|.dat` (сигнал) и `data/<N>.<lead>` (разметка отведения; `lead` ∈ i, ii, iii, avr, avl, avf, v1–v6).

Чтение в коде — модуль `yolo_1d.utils.datasets.ludb` (`load_record`, `iter_records`): сигнал (12, 5000) в мВ, волны P/QRS/T по отведениям, дефекты разметки, метаданные из `ludb.csv`.

Отчёт о данных (сверка с публикацией, статистика, рисунки) → `data/reports/`:

```bash
python data/ludb_report.py
```

### Подготовка для обучения

```bash
python data/prepare_ludb.py    # -> data/processed/ludb/ и data/ludb.yaml
```

Пример = одно отведение одной записи, вход сети 1 × 5000 (паддинг до 5120 в трансформациях), разметка — собственная разметка этого отведения. 12-канальный режим с агрегированной разметкой не используется.

Что делает скрипт (подробнее — docstring `data/prepare_ludb.py`, `src/yolo_1d/utils/ludb_preprocess.py`):
1. Z-score по всему отведению (полосовой фильтр — опция `--bandpass`, по умолчанию выключен).
2. Сигнал сохраняется **без маски**; границы размеченной зоны (первый onset, последний offset) записываются в `manifest.csv` (`annot_start`, `annot_end`). При загрузке `Dataset` (ключ `zones` в yaml) обнуляет сигнал вне `[annot_start − δ₁, annot_end + δ₂]`: на обучении δ₁ ∈ [0, 12], δ₂ ∈ [0, 24] отсчётов случайно, на val/test фиксированно 6 и 12. Диапазоны выбраны по интервалам P→QRS и QRS→T, чтобы не открыть соседнюю неразмеченную волну. Случайный запас нужен, чтобы граница маски не служила сети подсказкой. Неразмеченные волны на краях в сеть не попадают; форма входа одинакова на обучении и валидации.
3. Разметка: строки `class center width`, нормированные на 5000; left = onset, right = offset; P = 0, QRS = 1, T = 2.
4. 24 отведения, где внутри зоны есть волна без метки (дефект разметки, см. отчёт D2), исключаются из всех выборок (`--keep-defective` — оставить).

Выход `data/processed/ludb/`: `signals/<запись>_<отведение>.pt`, `labels/<запись>_<отведение>.txt`, `manifest.csv` (все 2400 отведений: зона, число волн, группа ритма, `usable`, причина исключения), `preprocessing.json` (параметры). Конфиг `data/ludb.yaml` содержит классы и пути.

### Разбиения (`data/splits/`, хранятся в git)

```bash
python data/make_splits.py     # после prepare_ludb.py; детерминировано (--seed 2026)
```

5-fold кросс-валидация **по записям** (все отведения записи — в одном сплите), стратифицированная по группе ритма (sinus / af_flutter / paced). В каждом тесте 40 записей: 35 sinus, 3 af_flutter, 2 paced. Каждая запись ровно один раз попадает в test, поэтому итоговые метрики считаются по объединению тестов всех фолдов. Из обучающей части фолда ~10 % записей по группам уходит в val (16 записей).

- `folds.csv` — запись → группа ритма → тестовый фолд;
- `fold_<k>/{train,val,test}.txt` — ID примеров (`<запись>_<отведение>`);
- `fold_<k>.yaml` — конфиг `Dataset` фолда k (`--dataset data/splits/fold_<k>.yaml`);
- `summary.json` — размеры сплитов и проверки.

Формат конфига `Dataset` (обратно совместим со старыми конфигами): необязательный ключ `path` задаёт базовую папку относительных путей (относительно yaml-файла); `<split>_list` — список ID примеров из папки `<split>_data`. Кэш меток для списка пишется рядом с ним (`<список>.labels.cache`, в git не попадает) и пересоздаётся при изменении разметки.

Лицензия: Open Data Commons Attribution v1.0.

### Режим MC: 12 отведений на вход
```bash
python data/prepare_ludb_mc.py   # после prepare_ludb.py → data/processed/ludb_mc/
python data/make_splits_mc.py    # после make_splits.py  → data/splits/fold_<k>/{train,val,test}_mc.txt, fold_<k>_mc.yaml
```
- Пример — запись: `signals/<запись>.pt` (12, 5000), Z-score по отведению той же `prepare_lead`, что в режиме S (канал k = файл отведения S бит в бит).
- Разметка `labels/<запись>.txt`: `class lead center width` (lead — индекс 0..11 в порядке I, II, III, aVR, aVL, aVF, V1–V6).
- `leads.csv`: зона разметки и пригодность каждого отведения. 24 непригодных отведения (7 записей) остаются во входе, но исключаются из лосса и метрик (`valid` в `MultiLeadDataset`).
- Фолды — те же записи, что в режиме S (списки выводятся из `fold_<k>/*.txt`); тест — 200 записей, 2376 отведений, как в S.
- Загрузка: `yolo_1d.utils.datasets.multilead_dataset.MultiLeadDataset` (`mode: multi_lead`), маска краёв — по зоне каждого отведения с независимым δ.

## MIT-BIH Noise Stress Test Database

```bash
python data/download_nstdb.py   # -> data/raw/nstdb/{bw,ma,em}.{dat,hea}, сверка SHA-256
```

Скачиваются только три записи шумов: `bw` (дрейф изолинии), `ma` (мышечный артефакт), `em` (артефакт движения электродов). Каждая — 2 канала, 360 Гц, 30 мин. Лицензия ODC-By 1.0. Цитировать: Moody G. B., Muldrow W. E., Mark R. G., *A noise stress test for arrhythmia detectors*, Computers in Cardiology 11:381–384, 1984. Цитировать: Kalyakulina A. I. et al., *LUDB: A New Open-Access Validation Tool for Electrocardiogram Delineation Algorithms*, IEEE Access 8, 2020, и PhysioNet (doi:10.13026/eegm-h675). 
