from __future__ import annotations

import time
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import ColumnSummary, DatasetSummary, compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(title="eda-cli quality service", version="0.1.0")


# ---------- Pydantic models (JSON endpoint /quality) ----------

class ColumnSummaryIn(BaseModel):
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any] = Field(default_factory=list)
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None


class DatasetSummaryIn(BaseModel):
    n_rows: int
    n_cols: int
    columns: List[ColumnSummaryIn]


class QualityRequest(BaseModel):
    summary: DatasetSummaryIn
    high_cardinality_unique: int = 50
    high_cardinality_share: float = 0.5


class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float
    latency_ms: int
    flags: Dict[str, Any]


# ---------- helpers ----------

def _summary_in_to_core(summary_in: DatasetSummaryIn) -> DatasetSummary:
    cols = [
        ColumnSummary(
            name=c.name,
            dtype=c.dtype,
            non_null=c.non_null,
            missing=c.missing,
            missing_share=c.missing_share,
            unique=c.unique,
            example_values=c.example_values,
            is_numeric=c.is_numeric,
            min=c.min,
            max=c.max,
            mean=c.mean,
            std=c.std,
        )
        for c in summary_in.columns
    ]
    return DatasetSummary(n_rows=summary_in.n_rows, n_cols=summary_in.n_cols, columns=cols)


def _missing_df_from_summary(summary: DatasetSummary) -> pd.DataFrame:
    # Формируем missing_df совместимый с compute_quality_flags(summary, missing_df)
    data = {
        c.name: {"missing_count": c.missing, "missing_share": c.missing_share}
        for c in summary.columns
    }
    if not data:
        return pd.DataFrame(columns=["missing_count", "missing_share"])
    return pd.DataFrame.from_dict(data, orient="index").sort_values("missing_share", ascending=False)


def _ok_for_model(flags: Dict[str, Any]) -> bool:
    # Простая логика для ok_for_model (достаточно для ДЗ)
    return bool(flags.get("quality_score", 0.0) >= 0.5) and not bool(flags.get("too_many_missing", False))


def _read_csv_upload(file: UploadFile) -> pd.DataFrame:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не задан")

    try:
        raw = file.file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Пустой CSV")
        df = pd.read_csv(BytesIO(raw))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV прочитан, но таблица пустая")
    return df


# ---------- endpoints ----------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/quality", response_model=QualityResponse)
def quality(req: QualityRequest) -> QualityResponse:
    t0 = time.perf_counter()

    if req.high_cardinality_unique < 1:
        raise HTTPException(status_code=400, detail="high_cardinality_unique должен быть >= 1")
    if not (0.0 <= req.high_cardinality_share <= 1.0):
        raise HTTPException(status_code=400, detail="high_cardinality_share должен быть в диапазоне [0..1]")

    summary = _summary_in_to_core(req.summary)
    missing_df = _missing_df_from_summary(summary)

    flags = compute_quality_flags(
        summary,
        missing_df,
        high_cardinality_unique=req.high_cardinality_unique,
        high_cardinality_share=req.high_cardinality_share,
    )

    latency_ms = int((time.perf_counter() - t0) * 1000)
    return QualityResponse(
        ok_for_model=_ok_for_model(flags),
        quality_score=float(flags.get("quality_score", 0.0)),
        latency_ms=latency_ms,
        flags=flags,
    )


@app.post("/quality-from-csv", response_model=QualityResponse)
def quality_from_csv(
    file: UploadFile = File(...),
    high_cardinality_unique: int = 50,
    high_cardinality_share: float = 0.5,
) -> QualityResponse:
    t0 = time.perf_counter()

    if high_cardinality_unique < 1:
        raise HTTPException(status_code=400, detail="high_cardinality_unique должен быть >= 1")
    if not (0.0 <= high_cardinality_share <= 1.0):
        raise HTTPException(status_code=400, detail="high_cardinality_share должен быть в диапазоне [0..1]")

    df = _read_csv_upload(file)

    summary = summarize_dataset(df)
    miss = missing_table(df)
    flags = compute_quality_flags(
        summary,
        miss,
        high_cardinality_unique=high_cardinality_unique,
        high_cardinality_share=high_cardinality_share,
    )

    latency_ms = int((time.perf_counter() - t0) * 1000)
    return QualityResponse(
        ok_for_model=_ok_for_model(flags),
        quality_score=float(flags.get("quality_score", 0.0)),
        latency_ms=latency_ms,
        flags=flags,
    )


# ---- ОБЯЗАТЕЛЬНЫЙ ДОП. ЭНДПОИНТ HW04 (использует эвристики HW03) ----
@app.post("/quality-flags-from-csv")
def quality_flags_from_csv(
    file: UploadFile = File(...),
    high_cardinality_unique: int = 50,
    high_cardinality_share: float = 0.5,
) -> Dict[str, Any]:
    """
    Возвращает полный набор флагов качества, включая эвристики HW03:
    has_constant_columns, has_high_cardinality_categoricals, high_cardinality_unique, high_cardinality_share, ...
    """
    t0 = time.perf_counter()

    if high_cardinality_unique < 1:
        raise HTTPException(status_code=400, detail="high_cardinality_unique должен быть >= 1")
    if not (0.0 <= high_cardinality_share <= 1.0):
        raise HTTPException(status_code=400, detail="high_cardinality_share должен быть в диапазоне [0..1]")

    df = _read_csv_upload(file)

    summary = summarize_dataset(df)
    miss = missing_table(df)
    flags = compute_quality_flags(
        summary,
        miss,
        high_cardinality_unique=high_cardinality_unique,
        high_cardinality_share=high_cardinality_share,
    )

    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {"latency_ms": latency_ms, "flags": flags}
