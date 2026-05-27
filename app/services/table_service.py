from __future__ import annotations

import pandas as pd
import numpy as np
from app.services.stats_service import calc_group_comparison, _test_normality, format_p_value


def build_baseline_table(
    df: pd.DataFrame,
    group_var: str,
    variables: list[str] | None = None,
    decimal_places: int = 2,
    p_digits: int = 3,
) -> dict:
    """Generate standard Table 1 (baseline characteristics by group)."""
    if variables is None:
        variables = [c for c in df.columns if c != group_var]

    groups = sorted(df[group_var].dropna().unique().tolist())
    rows = []

    for var in variables:
        if var == group_var:
            continue
        series = df[var].dropna()
        is_num = pd.api.types.is_numeric_dtype(series)

        row = {"Variable": var, "Type": "continuous" if is_num else "categorical"}

        if is_num:
            is_normal = _test_normality(series)
            if is_normal:
                row["Total"] = f"{series.mean():.{decimal_places}f} ± {series.std():.{decimal_places}f}"
            else:
                q1, med, q3 = series.quantile([0.25, 0.5, 0.75])
                row["Total"] = f"{med:.{decimal_places}f} ({q1:.{decimal_places}f}, {q3:.{decimal_places}f})"

            for g in groups:
                g_series = df.loc[df[group_var] == g, var].dropna()
                if len(g_series) > 0:
                    if is_normal:
                        row[str(g)] = f"{g_series.mean():.{decimal_places}f} ± {g_series.std():.{decimal_places}f}"
                    else:
                        q1g, medg, q3g = g_series.quantile([0.25, 0.5, 0.75])
                        row[str(g)] = f"{medg:.{decimal_places}f} ({q1g:.{decimal_places}f}, {q3g:.{decimal_places}f})"
                else:
                    row[str(g)] = "—"
        else:
            counts = series.value_counts()
            pcts = series.value_counts(normalize=True) * 100
            top_cats = counts.head(10)
            parts = []
            for cat, cnt in top_cats.items():
                pct = pcts.get(cat, 0)
                parts.append(f"{cat}: {cnt} ({pct:.1f}%)")
            row["Total"] = "; ".join(parts)

            for g in groups:
                g_series = df.loc[df[group_var] == g, var].dropna()
                if len(g_series) > 0:
                    g_counts = g_series.value_counts()
                    g_pcts = g_series.value_counts(normalize=True) * 100
                    g_parts = []
                    for cat, cnt in g_counts.head(10).items():
                        g_parts.append(f"{cat}: {cnt} ({g_pcts.get(cat, 0):.1f}%)")
                    row[str(g)] = "; ".join(g_parts)
                else:
                    row[str(g)] = "—"

        comp = calc_group_comparison(df, var, group_var)
        row["P value"] = format_p_value(comp["p_value"], p_digits)
        row["Method"] = comp["method"]
        rows.append(row)

    cols = ["Variable", "Type", "Total"] + [str(g) for g in groups] + ["P value", "Method"]
    return {"columns": cols, "rows": rows, "groups": groups, "group_var": group_var, "n_total": len(df)}


def build_descriptive_table(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    decimal_places: int = 2,
) -> dict:
    """Generate overall descriptive statistics table."""
    if variables is None:
        variables = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    rows = []
    for var in variables:
        series = df[var].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            continue
        is_normal = _test_normality(series)
        rows.append({
            "Variable": var,
            "N": len(series),
            "Missing": int(df[var].isnull().sum()),
            "Mean": round(float(series.mean()), decimal_places),
            "SD": round(float(series.std()), decimal_places),
            "Median": round(float(series.median()), decimal_places),
            "Q1": round(float(series.quantile(0.25)), decimal_places),
            "Q3": round(float(series.quantile(0.75)), decimal_places),
            "Min": round(float(series.min()), decimal_places),
            "Max": round(float(series.max()), decimal_places),
            "Distribution": "Normal" if is_normal else "Skewed",
            "Skewness": round(float(series.skew()), decimal_places),
        })

    cols = ["Variable", "N", "Missing", "Mean", "SD", "Median", "Q1", "Q3", "Min", "Max", "Distribution", "Skewness"]
    return {"columns": cols, "rows": rows, "n_total": len(df)}


def build_missing_table(df: pd.DataFrame) -> dict:
    """Generate missing value statistics table."""
    rows = []
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        complete = len(df) - missing
        rows.append({
            "Variable": col,
            "Complete (n)": complete,
            "Missing (n)": missing,
            "Complete (%)": round(complete / len(df) * 100, 2),
            "Missing (%)": round(missing / len(df) * 100, 2),
            "Type": str(df[col].dtype),
        })
    rows.sort(key=lambda r: r["Missing (n)"], reverse=True)
    cols = ["Variable", "Complete (n)", "Missing (n)", "Complete (%)", "Missing (%)", "Type"]
    return {"columns": cols, "rows": rows, "n_total": len(df)}


def build_model_result_table(coef_df: pd.DataFrame, title: str = "模型结果") -> dict:
    """Standardized model result table."""
    return {
        "title": title,
        "headers": list(coef_df.columns),
        "rows": coef_df.to_dict(orient="records"),
    }


def build_metrics_table(metrics: list[dict], title: str = "性能指标") -> dict:
    """Standardized metrics table."""
    return {
        "title": title,
        "headers": ["指标", "值"],
        "rows": metrics,
    }
