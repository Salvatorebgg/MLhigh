from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import aiofiles
from fastapi import UploadFile

from app.config import UPLOADS_DIR, EXAMPLES_DIR, MAX_UPLOAD_MB

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm"}

SEPARATORS = [",", "\t", ";", "|"]
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1")
MISSING_MARKERS = {"", "NA", "N/A", "NULL", "null", "None", "none", "NaN", "nan", "未记录", "缺失"}


async def save_upload(file: UploadFile) -> dict:
    """Save uploaded file and return metadata."""
    original_name = Path(file.filename or "upload.csv").name
    ext = Path(original_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式: {ext}。支持: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    upload_id = uuid.uuid4().hex[:12]
    safe_name = f"{upload_id}_{original_name}"
    dest = UPLOADS_DIR / safe_name
    max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
    total_bytes = 0
    try:
        async with aiofiles.open(dest, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise ValueError(f"文件超过 {MAX_UPLOAD_MB} MB 上传上限")
                await f.write(chunk)
    except Exception:
        if dest.exists():
            dest.unlink()
        raise
    return {"upload_id": upload_id, "path": str(dest), "filename": original_name, "ext": ext}


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a parsed dataframe using the same compact ingestion rules as the QC workbench."""
    df = df.dropna(how="all").copy()
    df.columns = [str(c).strip() if str(c).strip() else f"Unnamed_{i + 1}" for i, c in enumerate(df.columns)]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        df[col] = df[col].replace(list(MISSING_MARKERS), pd.NA)
    if df.empty:
        raise ValueError("文件为空或没有有效数据行。")
    return df


def read_csv_smart(filepath: str) -> pd.DataFrame:
    """Read delimited text with encoding and separator inference."""
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            df = pd.read_csv(filepath, sep=None, engine="python", encoding=encoding)
            if df.shape[1] >= 1:
                return normalize_dataframe(df)
        except Exception as err:
            last_error = err

    # Conservative fallback for uncommon separators.
    for encoding in CSV_ENCODINGS:
        try:
            with open(filepath, "r", encoding=encoding, errors="replace") as f:
                first_line = f.readline()
            best = max(SEPARATORS, key=lambda sep: first_line.count(sep))
            df = pd.read_csv(filepath, sep=best, encoding=encoding)
            return normalize_dataframe(df)
        except Exception as err:
            last_error = err
    raise ValueError(f"CSV 文件读取失败: {last_error}")


def read_file(filepath: str, filename: str, sheet_name: str | None = None) -> pd.DataFrame:
    """Read data file into DataFrame, auto-detecting format."""
    ext = Path(filename).suffix.lower()
    try:
        if ext in (".xlsx", ".xls", ".xlsm"):
            xl = pd.ExcelFile(filepath)
            if sheet_name and sheet_name in xl.sheet_names:
                return normalize_dataframe(pd.read_excel(filepath, sheet_name=sheet_name))
            return normalize_dataframe(pd.read_excel(filepath, sheet_name=0))
        elif ext in (".csv", ".tsv", ".txt", ""):
            return read_csv_smart(filepath)
        else:
            raise ValueError(f"无法读取文件格式: {ext}")
    except Exception as e:
        raise ValueError(f"文件读取失败: {e}")


def get_sheet_names(filepath: str, filename: str) -> list[str]:
    """Return sheet names for Excel files."""
    ext = Path(filename).suffix.lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        xl = pd.ExcelFile(filepath)
        return xl.sheet_names
    return []


def get_example_datasets() -> list[dict]:
    """List available example datasets."""
    examples = []
    for f in sorted(EXAMPLES_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(f, nrows=0)
            examples.append({
                "name": f.stem,
                "filename": f.name,
                "columns": list(df.columns),
                "col_count": len(df.columns),
            })
        except Exception:
            pass
    return examples
