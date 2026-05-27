from __future__ import annotations

import io
import zipfile
import json
from pathlib import Path

import pandas as pd
from app.config import OUTPUTS_DIR


def export_to_csv(data: list[dict] | pd.DataFrame, filename: str) -> Path:
    """Export data to CSV file."""
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    dest = OUTPUTS_DIR / f"{filename}.csv"
    df.to_csv(dest, index=False, encoding="utf-8-sig")
    return dest


def export_to_excel(data: list[dict] | pd.DataFrame, filename: str, sheet_name: str = "Sheet1") -> Path:
    """Export data to Excel file."""
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    dest = OUTPUTS_DIR / f"{filename}.xlsx"
    with pd.ExcelWriter(dest, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return dest


def export_to_html_table(df: pd.DataFrame | list[dict], title: str = "Table") -> str:
    """Export table to HTML format with three-line table style."""
    if isinstance(df, list):
        df = pd.DataFrame(df)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: 'Noto Sans SC', 'Microsoft YaHei', sans-serif; padding: 24px; }}
.three-line-table {{ border-collapse: collapse; width: 100%; margin: 0 auto; }}
.three-line-table th, .three-line-table td {{ padding: 8px 12px; text-align: left; font-size: 13px; }}
.three-line-table thead {{ border-top: 2px solid #000; border-bottom: 1px solid #000; }}
.three-line-table tr:last-child {{ border-bottom: 2px solid #000; }}
.three-line-table th {{ font-weight: 800; background: #f9fafb; }}
</style>
</head>
<body>
<h2>{title}</h2>
{table_to_html(df, classes="three-line-table")}
</body>
</html>"""
    return html


def table_to_html(df: pd.DataFrame, classes: str = "") -> str:
    """Convert DataFrame to HTML table string."""
    html = f'<table class="{classes}">\n<thead>\n<tr>\n'
    for col in df.columns:
        html += f"<th>{col}</th>\n"
    html += "</tr>\n</thead>\n<tbody>\n"
    for _, row in df.iterrows():
        html += "<tr>\n"
        for val in row:
            html += f"<td>{val}</td>\n"
        html += "</tr>\n"
    html += "</tbody>\n</table>"
    return html


def export_chart_config(chart_data: dict, filename: str) -> Path:
    """Save chart configuration as JSON."""
    dest = OUTPUTS_DIR / f"{filename}_config.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(chart_data, f, ensure_ascii=False, indent=2)
    return dest


def make_zip(files: list[Path], zip_name: str) -> Path:
    """Create a ZIP archive from a list of files."""
    dest = OUTPUTS_DIR / f"{zip_name}.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in files:
            if fp.exists():
                zf.write(fp, fp.name)
    return dest


def export_method_results(method_id: str, results: dict) -> dict[str, str]:
    """Export all results from a method analysis."""
    exports = {}
    for i, table in enumerate(results.get("tables", [])):
        fname = f"{method_id}_table_{i+1}"
        exports[f"table_{i+1}_csv"] = f"/api/export/file/{fname}.csv"
        export_to_csv(table.get("rows", []), fname)
    return exports
