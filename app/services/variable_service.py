from __future__ import annotations

import pandas as pd
import numpy as np

CONTINUOUS_CANDIDATE_DTYPES = {"int64", "float64", "int32", "float32", "Int64", "Float64"}
CATEGORICAL_MAX_UNIQUE_RATIO = 0.1
CATEGORICAL_MAX_UNIQUE = 30
BINARY_UNIQUE = 2
DATE_PATTERNS = ["date", "time", "日期", "时间", "year", "month", "day", "dt_"]
ID_PATTERNS = ["id", "ID", "编号", "序号", "patient", "subject", "record"]
REGION_PATTERNS = ["province", "prov", "省", "省份", "country", "国家", "region", "地区", "city", "城市", "area"]
GROUP_PATTERNS = ["group", "组", "treatment", "arm", "cohort", "处理", "分组", "组别"]
OUTCOME_PATTERNS = ["outcome", "结局", "event", "death", "survival", "mortality", "事件", "死亡", "生存"]


def classify_variables(df: pd.DataFrame) -> dict[str, list[str]]:
    """Auto-classify variables by type."""
    # Convert StringDtype columns to object for compatibility
    for col in df.columns:
        if isinstance(df[col].dtype, pd.StringDtype) or str(df[col].dtype) == "string":
            df[col] = df[col].astype(object)

    result = {
        "continuous": [],
        "categorical": [],
        "ordinal_categorical": [],
        "date": [],
        "time": [],
        "id": [],
        "region": [],
        "binary": [],
        "group": [],
        "outcome_candidate": [],
    }
    n = len(df)

    for col in df.columns:
        col_lower = col.lower().replace("_", "").replace(" ", "")
        series = df[col]
        dtype = str(series.dtype)
        nunique = series.nunique(dropna=True)

        # ID detection
        if any(p in col_lower for p in ID_PATTERNS) or (nunique == n and nunique > 50):
            result["id"].append(col)
            continue

        # Date detection
        if any(p in col_lower for p in DATE_PATTERNS):
            if pd.api.types.is_datetime64_any_dtype(series):
                result["date"].append(col)
                continue
            try:
                s = pd.to_datetime(series, errors="coerce")
                if s.notna().sum() > n * 0.7:
                    result["date"].append(col)
                    continue
            except Exception:
                pass

        # Region detection
        if any(p in col_lower for p in REGION_PATTERNS):
            result["region"].append(col)
            continue

        # Group detection
        if any(p in col_lower for p in GROUP_PATTERNS):
            result["group"].append(col)
            if nunique <= BINARY_UNIQUE:
                result["binary"].append(col)
            continue

        # Outcome detection
        if any(p in col_lower for p in OUTCOME_PATTERNS):
            result["outcome_candidate"].append(col)
            if nunique <= BINARY_UNIQUE:
                result["binary"].append(col)
            continue

        # Numeric vs categorical
        if dtype in CONTINUOUS_CANDIDATE_DTYPES or np.issubdtype(series.dtype, np.number):
            if nunique <= BINARY_UNIQUE:
                result["binary"].append(col)
            elif nunique <= CATEGORICAL_MAX_UNIQUE and nunique / max(n, 1) < CATEGORICAL_MAX_UNIQUE_RATIO:
                result["categorical"].append(col)
            else:
                result["continuous"].append(col)
        else:
            if nunique <= BINARY_UNIQUE:
                result["binary"].append(col)
            elif nunique <= CATEGORICAL_MAX_UNIQUE:
                result["categorical"].append(col)
            else:
                result["categorical"].append(col)

    return result


def summarize_dataset(df: pd.DataFrame, var_types: dict | None = None) -> dict:
    """Generate dataset summary statistics."""
    n = len(df)
    missing = df.isnull().sum().sum()
    total_cells = n * len(df.columns)
    if var_types is None:
        var_types = classify_variables(df)
    return {
        "sample_size": n,
        "variable_count": len(df.columns),
        "missing_percent": round(missing / max(total_cells, 1) * 100, 2),
        "continuous_count": len(var_types["continuous"]),
        "categorical_count": len(var_types["categorical"]),
        "date_count": len(var_types["date"]),
        "binary_count": len(var_types["binary"]),
        "region_count": len(var_types["region"]),
    }
