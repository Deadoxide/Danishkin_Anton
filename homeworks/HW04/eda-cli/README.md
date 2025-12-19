# HW04 – eda_cli: мини-EDA + HTTP-сервис качества датасетов (FastAPI)

Проект на базе решения HW03: CLI-приложение для базового анализа CSV + HTTP API поверх EDA-ядра.
Используется в рамках Семинаров 03–04 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- uv: https://docs.astral.sh/uv/

## Инициализация проекта

В корне проекта (внутри `homeworks/HW04/eda-cli`):

    uv sync

Эта команда:
- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект в окружение (включая CLI-команду `eda-cli`).

## Запуск CLI

### Краткий обзор

    uv run eda-cli overview data/example.csv

Параметры:
- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

    uv run eda-cli report data/example.csv --out-dir reports

Полезные опции отчёта (HW03):
- `--title` — заголовок отчёта;
- `--min-missing-share` — порог доли пропусков, чтобы выделить «проблемные» колонки;
- `--top-k-categories` — сколько top-значений сохранять для категориальных колонок;
- `--max-cat-columns` — сколько категориальных колонок анализировать;
- `--max-hist-columns` — сколько числовых колонок включать в гистограммы;
- `--high-cardinality-unique` и `--high-cardinality-share` — пороги эвристики высокой кардинальности.

В результате в каталоге `reports/` появятся:
- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

    uv run pytest -q

## HTTP API (HW04)

### Запуск сервиса

Из корня проекта:

    uv run uvicorn eda_cli.api:app --reload --port 8000

Swagger UI:
- http://127.0.0.1:8000/docs

### Эндпоинты

- `GET /health` — проверка доступности сервиса.
- `POST /quality` — оценка качества по JSON summary (структура как в `QualityRequest` в `eda_cli.api`).
- `POST /quality-from-csv` — загрузка CSV-файла и расчёт качества (использует EDA-ядро: `summarize_dataset`, `missing_table`, `compute_quality_flags`).
- `POST /quality-flags-from-csv` — дополнительный эндпоинт HW04: возвращает полный набор `flags`, включая эвристики HW03:
  `has_constant_columns`, `has_high_cardinality_categoricals`, а также `high_cardinality_unique`, `high_cardinality_share`.

Параметры для CSV-эндпоинтов (query params):
- `high_cardinality_unique` (int)
- `high_cardinality_share` (float, 0..1)

### Примеры запросов (PowerShell)

    # health
    curl.exe -X GET "http://127.0.0.1:8000/health"

    # quality-from-csv
    curl.exe -X POST "http://127.0.0.1:8000/quality-from-csv" -F "file=@data/example.csv"

    # quality-flags-from-csv (доп. эндпоинт HW04)
    curl.exe -X POST "http://127.0.0.1:8000/quality-flags-from-csv" -F "file=@data/example.csv"

    # пример с настройкой порогов высокой кардинальности (PowerShell line continuation: `)
    curl.exe -X POST "http://127.0.0.1:8000/quality-flags-from-csv?high_cardinality_unique=2&high_cardinality_share=0.1" `
      -F "file=@data/example.csv"
