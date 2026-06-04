from __future__ import annotations

import json
import uuid
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import STATIC_DIR, EXAMPLES_DIR, UPLOADS_DIR, OUTPUTS_DIR
from app.schemas import AnalyzeRequest, MethodRequest, TableRequest, ExportRequest
from app.services.io_service import (
    read_file, get_sheet_names, save_upload, get_example_datasets,
)
from app.services.variable_service import classify_variables, summarize_dataset
from app.services.table_service import build_baseline_table, build_descriptive_table, build_missing_table
from app.services.chart_service import make_descriptive_charts
from app.services.stats_service import STATS_ROUTER
from app.services.ml_service import ML_ROUTER
from app.services.sample_service import EXAMPLE_MAKERS
from app.services.report_service import generate_discussion, generate_method_report
from app.services.export_service import (
    export_to_csv, export_to_excel, export_to_html_table, export_chart_config,
)

for d in [STATIC_DIR, EXAMPLES_DIR, UPLOADS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Generate example CSV files if missing
for name, maker_fn in EXAMPLE_MAKERS.items():
    dest = EXAMPLES_DIR / f"{name}.csv"
    if not dest.exists():
        try:
            maker_fn().to_csv(dest, index=False, encoding="utf-8-sig")
        except Exception:
            pass

app = FastAPI(
    title="MLhigh — Clinical Advanced Statistics & ML Platform",
    version="1.0.0",
    description="One-click platform for clinical advanced statistics and machine learning analysis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Method Catalog ──────────────────────────────────────────

METHOD_CATALOG = {
    "gee": {
        "id": "gee", "name": "广义估计方程 (GEE)", "icon": "GEE",
        "category": "advanced_stats",
        "description": "纵向重复测量数据的广义估计方程分析",
        "example_dataset": "gee_example",
        "params": [
            {"key": "outcome_var", "label": "结局变量", "type": "select", "default": "sbp"},
            {"key": "time_var", "label": "时间变量", "type": "select", "default": "time"},
            {"key": "group_var", "label": "分组变量", "type": "select", "default": "arm"},
            {"key": "subject_var", "label": "受试者ID", "type": "select", "default": "subject_id"},
            {"key": "family", "label": "分布族", "type": "select", "default": "gaussian",
             "options": ["gaussian", "binomial", "poisson"]},
        ],
    },
    "propensity_score": {
        "id": "propensity_score", "name": "倾向性评分匹配 (PSM)", "icon": "PSM",
        "category": "advanced_stats",
        "description": "基于倾向性评分的匹配分析，减少选择偏倚",
        "example_dataset": "propensity_score_example",
        "params": [
            {"key": "treatment_var", "label": "处理变量", "type": "select", "default": "treatment"},
            {"key": "outcome_var", "label": "结局变量", "type": "select", "default": "outcome_continuous"},
        ],
    },
    "sensitivity_analysis": {
        "id": "sensitivity_analysis", "name": "稳健性与敏感性分析", "icon": "SEN",
        "category": "advanced_stats",
        "description": "评估结果对未测量混杂因素的敏感性",
        "example_dataset": "sensitivity_analysis_example",
        "params": [
            {"key": "treatment_var", "label": "处理变量", "type": "select", "default": "arm"},
            {"key": "outcome_var", "label": "结局变量", "type": "select", "default": "followup_score1"},
            {"key": "baseline_var", "label": "基线变量", "type": "select", "default": "baseline_score1"},
        ],
    },
    "counterfactual": {
        "id": "counterfactual", "name": "反事实推断与因果推断", "icon": "CFI",
        "category": "advanced_stats",
        "description": "基于反事实框架的因果效应估计 (ATE/ATT)",
        "example_dataset": "counterfactual_example",
        "params": [
            {"key": "treatment_var", "label": "处理变量", "type": "select", "default": "treatment"},
            {"key": "outcome_var", "label": "结局变量", "type": "select", "default": "outcome_continuous"},
        ],
    },
    "survival_advanced": {
        "id": "survival_advanced", "name": "复杂生存与疾病进程建模", "icon": "SUR",
        "category": "advanced_stats",
        "description": "KM曲线、Cox回归、时依ROC、竞争风险分析",
        "example_dataset": "survival_advanced_example",
        "params": [
            {"key": "time_var", "label": "时间变量", "type": "select", "default": "time_os"},
            {"key": "event_var", "label": "事件变量", "type": "select", "default": "event_os"},
            {"key": "group_var", "label": "分组变量", "type": "select", "default": "stage"},
        ],
    },
    "markov_model": {
        "id": "markov_model", "name": "马尔可夫模型", "icon": "MKV",
        "category": "advanced_stats",
        "description": "疾病状态转移概率矩阵与 Markov 轨迹",
        "example_dataset": "markov_model_example",
        "params": [
            {"key": "state_var", "label": "状态变量", "type": "select", "default": "state"},
            {"key": "time_var", "label": "时间变量", "type": "select", "default": "month"},
        ],
    },
    "bayesian": {
        "id": "bayesian", "name": "贝叶斯统计分析", "icon": "BAY",
        "category": "advanced_stats",
        "description": "后验分布估计、贝叶斯因子、MCMC诊断",
        "example_dataset": "bayesian_example",
        "params": [
            {"key": "outcome_var", "label": "结局变量", "type": "select", "default": "das28_change"},
            {"key": "group_var", "label": "分组变量", "type": "select", "default": "arm"},
        ],
    },
    "latin_square": {
        "id": "latin_square", "name": "拉丁方裂项方差分析", "icon": "LSD",
        "category": "advanced_stats",
        "description": "拉丁方交叉设计的方差分析",
        "example_dataset": "latin_square_example",
        "params": [
            {"key": "response_var", "label": "响应变量", "type": "select", "default": "response"},
            {"key": "period_var", "label": "时期变量", "type": "select", "default": "period"},
            {"key": "formulation_var", "label": "处理变量", "type": "select", "default": "formulation_code"},
            {"key": "subject_var", "label": "受试者ID", "type": "select", "default": "subject_id"},
        ],
    },
    "meta_analysis": {
        "id": "meta_analysis", "name": "荟萃分析", "icon": "MTA",
        "category": "advanced_stats",
        "description": "多中心研究的固定/随机效应荟萃分析",
        "example_dataset": "meta_analysis_example",
        "params": [
            {"key": "study_var", "label": "研究变量", "type": "select", "default": "study"},
            {"key": "effect_var", "label": "效应量", "type": "select", "default": "effect_size"},
            {"key": "se_var", "label": "标准误", "type": "select", "default": "standard_error"},
        ],
    },
    "mediation": {
        "id": "mediation", "name": "中介效应分析", "icon": "MED",
        "category": "advanced_stats",
        "description": "Bootstrap中介效应分解与路径分析",
        "example_dataset": "mediation_example",
        "params": [
            {"key": "x_var", "label": "自变量 (X)", "type": "select", "default": "stress_score"},
            {"key": "m_var", "label": "中介变量 (M)", "type": "select", "default": "crp"},
            {"key": "y_var", "label": "因变量 (Y)", "type": "select", "default": "cvd_risk_score"},
        ],
    },
    "mixed_effects": {
        "id": "mixed_effects", "name": "混合效应模型", "icon": "MEM",
        "category": "advanced_stats",
        "description": "多中心纵向数据的混合效应模型",
        "example_dataset": "mixed_effects_example",
        "params": [
            {"key": "outcome_var", "label": "结局变量", "type": "select", "default": "sbp"},
            {"key": "time_var", "label": "时间变量", "type": "select", "default": "time"},
            {"key": "group_var", "label": "分组变量", "type": "select", "default": "arm"},
            {"key": "random_var", "label": "随机效应", "type": "select", "default": "center"},
        ],
    },
    "nhanes_analysis": {
        "id": "nhanes_analysis", "name": "复杂抽样分析", "icon": "NHN",
        "category": "advanced_stats",
        "description": "加权描述统计与 survey 回归",
        "example_dataset": "nhanes_analysis_example",
        "params": [
            {"key": "weight_var", "label": "权重变量", "type": "select", "default": "survey_weight"},
            {"key": "strata_var", "label": "分层变量", "type": "select", "default": "strata"},
            {"key": "outcome_var", "label": "结局变量", "type": "select", "default": "sbp"},
        ],
    },
    "ldsc": {
        "id": "ldsc", "name": "LDSC 共病分析", "icon": "LDS",
        "category": "advanced_stats",
        "description": "遗传力估计 (h²) 与多性状遗传相关性 (rg) 共病分析",
        "example_dataset": "ldsc_example",
        "params": [
            {"key": "group_var", "label": "分组变量（可选）", "type": "select", "default": ""},
        ],
    },
    # ML models
    "ml_lr": {
        "id": "ml_lr", "name": "逻辑回归", "icon": "LR",
        "category": "ml_models",
        "description": "二分类临床预测模型，ROC/校准曲线/列线图",
        "example_dataset": "ml_lr_example",
        "params": [
            {"key": "target", "label": "目标变量", "type": "select", "default": "outcome"},
        ],
    },
    "ml_lasso": {
        "id": "ml_lasso", "name": "Lasso/岭回归", "icon": "LSR",
        "category": "ml_models",
        "description": "L1/L2正则化回归，高维特征选择",
        "example_dataset": "ml_lasso_example",
        "params": [
            {"key": "target", "label": "目标变量", "type": "select", "default": "outcome"},
            {"key": "regularization", "label": "正则化", "type": "select", "default": "lasso",
             "options": ["lasso", "ridge"]},
        ],
    },
    "ml_knn": {
        "id": "ml_knn", "name": "K近邻 (KNN)", "icon": "KNN",
        "category": "ml_models",
        "description": "多分类疾病诊断，决策边界与K值优化",
        "example_dataset": "ml_knn_example",
        "params": [
            {"key": "target", "label": "目标变量", "type": "select", "default": "disease"},
        ],
    },
    "ml_xgboost": {
        "id": "ml_xgboost", "name": "XGBoost", "icon": "XGB",
        "category": "ml_models",
        "description": "梯度提升树，SHAP特征重要性与学习曲线",
        "example_dataset": "ml_xgboost_example",
        "params": [
            {"key": "target", "label": "目标变量", "type": "select", "default": "outcome"},
        ],
    },
    "ml_rf": {
        "id": "ml_rf", "name": "随机森林", "icon": "RF",
        "category": "ml_models",
        "description": "集成学习，OOB误差与部分依赖图",
        "example_dataset": "ml_rf_example",
        "params": [
            {"key": "target", "label": "目标变量", "type": "select", "default": "complication"},
        ],
    },
    "ml_svm": {
        "id": "ml_svm", "name": "支持向量机 (SVM)", "icon": "SVM",
        "category": "ml_models",
        "description": "核函数比较、决策边界、支持向量",
        "example_dataset": "ml_svm_example",
        "params": [
            {"key": "target", "label": "目标变量", "type": "select", "default": "disease_subtype"},
        ],
    },
    "ml_dt": {
        "id": "ml_dt", "name": "决策树", "icon": "DT",
        "category": "ml_models",
        "description": "临床决策规则、树结构可视化",
        "example_dataset": "ml_dt_example",
        "params": [
            {"key": "target", "label": "目标变量", "type": "select", "default": "high_risk"},
        ],
    },
    "ml_cnn": {
        "id": "ml_cnn", "name": "1D-CNN 时序分析", "icon": "CNN",
        "category": "ml_models",
        "description": "多指标时序数据的深度学习分类",
        "example_dataset": "ml_cnn_example",
        "params": [
            {"key": "target", "label": "目标变量", "type": "select", "default": "label"},
        ],
    },
    # Integrated tools
    "feature_engineering": {
        "id": "feature_engineering", "name": "特征工程", "icon": "FEG",
        "category": "tools",
        "description": "缺失值处理、标准化、编码与衍生",
        "example_dataset": "feature_engineering_example",
        "params": [],
    },
    "model_comparison": {
        "id": "model_comparison", "name": "多模型比较", "icon": "CMP",
        "category": "tools",
        "description": "多模型ROC/PR比较与性能评估",
        "example_dataset": "model_comparison_example",
        "params": [
            {"key": "target", "label": "目标变量", "type": "select", "default": "outcome"},
        ],
    },
    "dim_reduction": {
        "id": "dim_reduction", "name": "降维分析", "icon": "DIM",
        "category": "tools",
        "description": "PCA、t-SNE 高维数据可视化",
        "example_dataset": "dim_reduction_example",
        "params": [
            {"key": "group_var", "label": "分组变量", "type": "select", "default": "group"},
        ],
    },
    "cluster": {
        "id": "cluster", "name": "聚类分析", "icon": "CLS",
        "category": "tools",
        "description": "K-means、层次聚类、GMM、共识聚类",
        "example_dataset": "cluster_example",
        "params": [
            {"key": "n_clusters", "label": "簇数", "type": "number", "default": 4},
        ],
    },
}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


# ── Method Catalog API ─────────────────────────────────────

@app.get("/api/methods")
def list_methods() -> dict:
    return {"methods": list(METHOD_CATALOG.values())}


@app.get("/api/methods/{method_id}")
def get_method(method_id: str) -> dict:
    if method_id not in METHOD_CATALOG:
        raise HTTPException(status_code=404, detail=f"Method '{method_id}' not found")
    return METHOD_CATALOG[method_id]


# ── Upload ─────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> dict:
    try:
        meta = await save_upload(file)
        df = read_file(meta["path"], meta["filename"])
        var_types = classify_variables(df)
        summary = summarize_dataset(df, var_types)
        sheets = get_sheet_names(meta["path"], meta["filename"])
        preview = df.head(10).fillna("").to_dict(orient="records")
        return {
            "upload_id": meta["upload_id"],
            "filename": meta["filename"],
            "file_type": meta["ext"],
            "sheet_names": sheets if sheets else None,
            "row_count": len(df),
            "col_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "variable_types": var_types,
            "preview": preview,
            "missing_percent": summary["missing_percent"],
            "summary": summary,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


# ── Examples ────────────────────────────────────────────────

@app.get("/api/examples")
def list_examples() -> list[dict]:
    return get_example_datasets()


@app.get("/api/examples/{name}")
def get_example(name: str) -> dict:
    filepath = EXAMPLES_DIR / f"{name}.csv"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Example '{name}' not found")
    df = pd.read_csv(filepath)
    var_types = classify_variables(df)
    summary = summarize_dataset(df, var_types)
    return {
        "name": name,
        "filename": f"{name}.csv",
        "row_count": len(df),
        "col_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "variable_types": var_types,
        "preview": df.head(10).fillna("").to_dict(orient="records"),
        "missing_percent": summary["missing_percent"],
        "summary": summary,
    }


@app.get("/api/examples/{name}/download")
def download_example(name: str) -> FileResponse:
    filepath = EXAMPLES_DIR / f"{name}.csv"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Example not found")
    return FileResponse(filepath, media_type="text/csv", filename=f"{name}.csv")


# ── Main Analysis API ──────────────────────────────────────

@app.post("/api/analyze")
def run_analysis(req: MethodRequest) -> dict:
    """Run a complete method analysis pipeline."""
    method_id = req.method_id
    if method_id not in METHOD_CATALOG:
        raise HTTPException(status_code=404, detail=f"Method '{method_id}' not found")

    # Use method's example dataset as default
    dataset_name = req.dataset_name
    if not dataset_name and not req.upload_id:
        dataset_name = METHOD_CATALOG[method_id].get("example_dataset", "gee_example")

    df = _get_df_simple({
        "upload_id": req.upload_id,
        "dataset_name": dataset_name,
        "use_demo": req.use_demo,
        "sheet_name": req.sheet_name,
    })

    params = req.params or {}
    # Set default params from catalog
    for p in METHOD_CATALOG[method_id].get("params", []):
        if p["key"] not in params:
            params[p["key"]] = p.get("default", "")

    # Route to appropriate service
    if method_id in STATS_ROUTER:
        results = STATS_ROUTER[method_id](df, params)
    elif method_id in ML_ROUTER:
        results = ML_ROUTER[method_id](df, params)
    else:
        raise HTTPException(status_code=400, detail=f"Method '{method_id}' not implemented")

    if not results.get("diagnostics"):
        results["diagnostics"] = make_fallback_diagnostics(
            df,
            params,
            METHOD_CATALOG[method_id]["name"],
        )

    # Add discussion if missing
    if not results.get("discussion"):
        results["discussion"] = generate_discussion(method_id, results, df)

    # Add descriptive stats
    var_types = classify_variables(df)
    summary = summarize_dataset(df, var_types)
    results["data_summary"] = summary
    results["method_id"] = method_id
    results["method_name"] = METHOD_CATALOG[method_id]["name"]

    return _sanitize_numpy(results)


@app.post("/api/analyze/report")
def generate_report(req: MethodRequest) -> dict:
    """Generate a full markdown report for a method."""
    method_id = req.method_id
    dataset_name = req.dataset_name
    if not dataset_name and not req.upload_id:
        dataset_name = METHOD_CATALOG.get(method_id, {}).get("example_dataset", "gee_example")
    df = _get_df_simple({
        "upload_id": req.upload_id,
        "dataset_name": dataset_name,
        "use_demo": req.use_demo,
        "sheet_name": req.sheet_name,
    })
    params = req.params or {}
    for p in METHOD_CATALOG[method_id].get("params", []):
        if p["key"] not in params:
            params[p["key"]] = p.get("default", "")

    if method_id in STATS_ROUTER:
        results = STATS_ROUTER[method_id](df, params)
    elif method_id in ML_ROUTER:
        results = ML_ROUTER[method_id](df, params)
    else:
        raise HTTPException(status_code=400, detail=f"Method '{method_id}' not implemented")

    if not results.get("diagnostics"):
        results["diagnostics"] = make_fallback_diagnostics(
            df,
            params,
            METHOD_CATALOG[method_id]["name"],
        )

    report = generate_method_report(method_id, results, df)
    return {"report": report, "method_id": method_id}


# ── Data Info ──────────────────────────────────────────────

@app.post("/api/dataset/data")
def dataset_data(req: dict) -> dict:
    df = _get_df_simple(req)
    var_types = classify_variables(df)
    summary = summarize_dataset(df, var_types)
    return {
        "row_count": len(df),
        "col_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "variable_types": var_types,
        "summary": summary,
        "data": _df_to_column_data(df),
    }


@app.post("/api/data/descriptive-charts")
def descriptive_charts(req: dict) -> dict:
    df = _get_df_simple(req)
    charts = make_descriptive_charts(df)
    return {"charts": charts}


# ── Tables ─────────────────────────────────────────────────

@app.post("/api/table/baseline")
def baseline_table(req: TableRequest) -> dict:
    df = _get_df(req)
    if req.table_type == "descriptive":
        return build_descriptive_table(df, req.variables, req.decimal_places)
    elif req.table_type == "missing":
        return build_missing_table(df)
    # default: baseline
    group_var = req.group_var
    if not group_var or group_var not in df.columns:
        cat_cols = [c for c in df.columns if 2 <= df[c].nunique() <= 10]
        if cat_cols:
            group_var = cat_cols[0]
        else:
            raise HTTPException(status_code=400, detail="No suitable group variable found")
    return build_baseline_table(df, group_var, req.variables, req.decimal_places, req.p_digits)


# ── Export ─────────────────────────────────────────────────

@app.post("/api/export/table-csv")
def export_table_csv(req: ExportRequest) -> FileResponse:
    data = req.table_data or []
    dest = export_to_csv(data, "table_export")
    return FileResponse(dest, media_type="text/csv", filename="table_export.csv")


@app.post("/api/export/table-excel")
def export_table_excel(req: ExportRequest) -> FileResponse:
    data = req.table_data or []
    dest = export_to_excel(data, "table_export")
    return FileResponse(
        dest,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="table_export.xlsx",
    )


@app.post("/api/export/table-html")
def export_table_html(req: ExportRequest) -> Response:
    data = req.table_data or []
    html = export_to_html_table(data)
    return Response(content=html, media_type="text/html")


@app.post("/api/export/chart-config")
def export_chart_config_endpoint(req: ExportRequest) -> FileResponse:
    data = req.chart_data or {}
    dest = export_chart_config(data, "chart_config")
    return FileResponse(dest, media_type="application/json", filename="chart_config.json")


# ── Helpers ────────────────────────────────────────────────

def make_fallback_diagnostics(df: pd.DataFrame, params: dict, method_name: str) -> list[dict]:
    """Create generic, method-aware diagnostics when a runner has no custom plots."""
    try:
        import plotly.graph_objects as go
    except Exception:
        return []

    diagnostics: list[dict] = []
    key_cols = _diagnostic_key_columns(df, params)
    if not key_cols:
        key_cols = list(df.columns[: min(12, len(df.columns))])

    # Completeness is a useful diagnostic for every clinical workflow.
    completeness = (df[key_cols].notna().mean() * 100).sort_values()
    fig_complete = go.Figure()
    fig_complete.add_trace(go.Bar(
        x=np.round(completeness.values, 1),
        y=list(completeness.index),
        orientation="h",
        marker=dict(
            color=np.round(completeness.values, 1),
            colorscale=[[0, "#c0616e"], [0.7, "#f6c453"], [1, "#0d7377"]],
            cmin=0,
            cmax=100,
            line=dict(color="white", width=0.7),
        ),
        text=[f"{v:.1f}%" for v in completeness.values],
        textposition="auto",
        hovertemplate="%{y}<br>非缺失率=%{x:.1f}%<extra></extra>",
    ))
    fig_complete.update_layout(
        title=f"{method_name} · 数据完整性诊断",
        xaxis_title="非缺失率 (%)",
        yaxis_title="变量",
        height=520,
        margin=dict(l=120, r=40, t=72, b=56),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    diagnostics.append({"title": "数据完整性诊断", "plotly": fig_complete.to_json()})

    distribution_fig = _make_distribution_diagnostic(df, params, key_cols, method_name, go)
    if distribution_fig is not None:
        diagnostics.append({"title": "关键变量分布诊断", "plotly": distribution_fig.to_json()})

    return diagnostics


def _diagnostic_key_columns(df: pd.DataFrame, params: dict) -> list[str]:
    keys = [
        "outcome_var", "predictor_var", "treatment_var", "group_var", "time_var",
        "event_var", "subject_var", "cluster_var", "baseline_var", "size_var",
        "weight_var", "strata_var", "target", "id_var",
    ]
    cols: list[str] = []
    for key in keys:
        value = params.get(key)
        if isinstance(value, list):
            cols.extend([v for v in value if isinstance(v, str)])
        elif isinstance(value, str):
            cols.append(value)
    for value in params.get("value_vars", []) or []:
        if isinstance(value, str):
            cols.append(value)
    seen = set()
    return [c for c in cols if c in df.columns and not (c in seen or seen.add(c))][:16]


def _make_distribution_diagnostic(df: pd.DataFrame, params: dict, key_cols: list[str], method_name: str, go):
    numeric_cols = [
        c for c in key_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if df[c].notna().sum() > 0
        ][:4]

    group_col = _first_valid_column(df, params, ["group_var", "treatment_var", "target"])
    if group_col and numeric_cols:
        value_col = next((c for c in numeric_cols if c != group_col), None)
        if value_col:
            plot_df = df[[group_col, value_col]].dropna()
        else:
            plot_df = pd.DataFrame()
        if not plot_df.empty and group_col in plot_df.columns and value_col in plot_df.columns:
            group_series = plot_df[group_col].astype(str)
            top_groups = group_series.value_counts().head(12).index
            plot_df = plot_df[group_series.isin(top_groups)]
            fig = go.Figure()
            for group_name, group_df in plot_df.groupby(plot_df[group_col].astype(str), sort=False):
                fig.add_trace(go.Box(
                    y=group_df[value_col],
                    name=str(group_name),
                    boxmean=True,
                    marker=dict(size=4),
                    line=dict(width=1.4),
                    hovertemplate=f"{group_col}={group_name}<br>{value_col}=%{{y:.3g}}<extra></extra>",
                ))
            fig.update_layout(
                title=f"{method_name} · {value_col} 分组分布",
                xaxis_title=group_col,
                yaxis_title=value_col,
                height=520,
                margin=dict(l=72, r=40, t=72, b=86),
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            return fig

    if numeric_cols:
        fig = go.Figure()
        for col in numeric_cols[:4]:
            values = df[col].dropna()
            if values.empty:
                continue
            fig.add_trace(go.Histogram(
                x=values,
                name=col,
                opacity=0.72,
                nbinsx=32,
                hovertemplate=f"{col}<br>值=%{{x:.3g}}<br>计数=%{{y}}<extra></extra>",
            ))
        if fig.data:
            fig.update_layout(
                title=f"{method_name} · 关键数值变量分布",
                xaxis_title="取值",
                yaxis_title="计数",
                barmode="overlay",
                height=520,
                margin=dict(l=72, r=40, t=72, b=64),
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            return fig

    cat_col = group_col or next((c for c in key_cols if c in df.columns), None)
    if cat_col:
        counts = df[cat_col].astype(str).replace({"nan": "缺失"}).value_counts().head(15)
        fig = go.Figure(go.Bar(
            x=counts.index,
            y=counts.values,
            marker=dict(color="#0d7377", line=dict(color="white", width=0.7)),
            hovertemplate="%{x}<br>样本数=%{y}<extra></extra>",
        ))
        fig.update_layout(
            title=f"{method_name} · {cat_col} 样本结构",
            xaxis_title=cat_col,
            yaxis_title="样本数",
            height=520,
            margin=dict(l=72, r=40, t=72, b=96),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        return fig

    return None


def _first_valid_column(df: pd.DataFrame, params: dict, keys: list[str]) -> str | None:
    for key in keys:
        value = params.get(key)
        if isinstance(value, str) and value in df.columns:
            return value
    return None

def _sanitize_numpy(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

def _get_df(req: AnalyzeRequest | TableRequest) -> pd.DataFrame:
    if req.use_demo or not req.upload_id:
        ds = req.dataset_name or "gee_example"
        filepath = EXAMPLES_DIR / f"{ds}.csv"
        if not filepath.exists():
            filepath = next(EXAMPLES_DIR.glob("*.csv"))
        return pd.read_csv(filepath)
    upload_id = req.upload_id
    filepath = None
    filename = ""
    for f in UPLOADS_DIR.iterdir():
        if f.stem.startswith(upload_id):
            filepath = str(f)
            filename = f.name.split("_", 1)[1] if "_" in f.name else f.name
            break
    if not filepath:
        raise HTTPException(status_code=404, detail="Upload not found")
    return read_file(filepath, filename, req.sheet_name if hasattr(req, "sheet_name") else None)


def _get_df_simple(req: dict) -> pd.DataFrame:
    if req.get("use_demo") or not req.get("upload_id"):
        ds = req.get("dataset_name") or "gee_example"
        filepath = EXAMPLES_DIR / f"{ds}.csv"
        if not filepath.exists():
            filepath = next(EXAMPLES_DIR.glob("*.csv"))
        return pd.read_csv(filepath)
    upload_id = req.get("upload_id", "")
    filepath = None
    filename = ""
    for f in UPLOADS_DIR.iterdir():
        if f.stem.startswith(upload_id):
            filepath = str(f)
            filename = f.name.split("_", 1)[1] if "_" in f.name else f.name
            break
    if not filepath:
        raise HTTPException(status_code=404, detail="Upload not found")
    return read_file(filepath, filename, req.get("sheet_name"))


def _df_to_column_data(df: pd.DataFrame) -> dict[str, list]:
    def _json_value(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        return value
    return {str(col): [_json_value(v) for v in df[col].tolist()] for col in df.columns}
