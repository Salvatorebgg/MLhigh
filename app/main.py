from __future__ import annotations

import json
import uuid
import re
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
    export_to_csv, export_to_txt, export_to_excel, export_to_html_table, export_chart_config,
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
    "target_trial_emulation": {
        "id": "target_trial_emulation", "name": "目标试验模拟 (TTE)", "icon": "TTE",
        "category": "advanced_stats",
        "description": "观察性临床队列的目标试验模拟、IPTW 加权和结局效应估计",
        "example_dataset": "target_trial_emulation_example",
        "params": [
            {"key": "treatment_var", "label": "处理/暴露变量", "type": "select", "default": "treatment"},
            {"key": "outcome_var", "label": "结局变量", "type": "select", "default": "event_12m"},
            {"key": "time_var", "label": "随访时间", "type": "select", "default": "followup_months"},
        ],
    },
    "doubly_robust_aipw": {
        "id": "doubly_robust_aipw", "name": "双重稳健 AIPW", "icon": "DR",
        "category": "advanced_stats",
        "description": "结合倾向评分模型与结局模型的双重稳健因果效应估计",
        "example_dataset": "doubly_robust_aipw_example",
        "params": [
            {"key": "treatment_var", "label": "处理/暴露变量", "type": "select", "default": "treatment"},
            {"key": "outcome_var", "label": "结局变量", "type": "select", "default": "outcome"},
            {"key": "feature_vars", "label": "协变量", "type": "multi_select", "default": []},
        ],
    },
    "marginal_structural_model": {
        "id": "marginal_structural_model", "name": "边际结构模型 (MSM)", "icon": "MSM",
        "category": "advanced_stats",
        "description": "使用逆概率权重处理时间变化混杂的纵向因果效应估计",
        "example_dataset": "marginal_structural_model_example",
        "params": [
            {"key": "treatment_var", "label": "处理变量", "type": "select", "default": "treatment"},
            {"key": "outcome_var", "label": "结局变量", "type": "select", "default": "outcome"},
            {"key": "time_var", "label": "时间变量", "type": "select", "default": "month"},
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
        "id": "ml_lasso", "name": "Lasso 回归", "icon": "LAS",
        "category": "ml_models",
        "description": "L1 正则化回归，可把部分特征系数压缩为 0，适合变量筛选",
        "example_dataset": "ml_lasso_example",
        "params": [
            {"key": "target", "label": "目标变量", "type": "select", "default": "outcome_continuous"},
        ],
    },
    "ml_ridge": {
        "id": "ml_ridge", "name": "岭回归", "icon": "RID",
        "category": "ml_models",
        "description": "L2 正则化回归，主要缓解多重共线性，通常不做变量剔除",
        "example_dataset": "ml_ridge_example",
        "params": [
            {"key": "target", "label": "目标变量", "type": "select", "default": "outcome_continuous"},
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
            {"key": "group_var", "label": "分组变量", "type": "select", "default": "subtype"},
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
    "ml_stacking": {
        "id": "ml_stacking", "name": "堆叠集成学习", "icon": "STK",
        "category": "ml_models",
        "description": "多模型 stacking ensemble，用于临床二分类预测与模型融合",
        "example_dataset": "ml_stacking_example",
        "params": [
            {"key": "target", "label": "目标变量", "type": "select", "default": "adverse_event"},
            {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
            {"key": "cv_folds", "label": "交叉验证折数", "type": "select", "default": "5", "options": ["3", "5", "10"]},
        ],
    },
    "ml_mlp": {
        "id": "ml_mlp", "name": "表格深度学习 MLP", "icon": "MLP",
        "category": "ml_models",
        "description": "多层感知机神经网络，用于结构化临床数据二分类预测",
        "example_dataset": "ml_mlp_example",
        "params": [
            {"key": "target", "label": "目标变量", "type": "select", "default": "icu_transfer"},
            {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
            {"key": "hidden_layer_sizes", "label": "隐藏层结构", "type": "select", "default": "64,32", "options": ["32", "64,32", "128,64"]},
        ],
    },
    "ml_elastic_net": {
        "id": "ml_elastic_net", "name": "Elastic Net 回归", "icon": "EN",
        "category": "ml_models",
        "description": "结合 L1/L2 正则化的连续结局预测和变量筛选",
        "example_dataset": "ml_elastic_net_example",
        "params": [
            {"key": "target", "label": "目标变量", "type": "select", "default": "renal_decline_score"},
            {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
            {"key": "l1_ratio", "label": "L1 比例", "type": "select", "default": "0.5", "options": ["0.1", "0.5", "0.7", "0.9"]},
        ],
    },
    "ml_deepsurv": {
        "id": "ml_deepsurv", "name": "DeepSurv 生存网络", "icon": "DSV",
        "category": "ml_models",
        "description": "DeepSurv 风格的非线性生存风险预测、C-index 与风险分层",
        "example_dataset": "ml_deepsurv_example",
        "params": [
            {"key": "time_var", "label": "生存时间", "type": "select", "default": "time_months"},
            {"key": "event_var", "label": "事件变量", "type": "select", "default": "event"},
            {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
        ],
    },
    "ml_deephit": {
        "id": "ml_deephit", "name": "DeepHit 竞争风险", "icon": "DHT",
        "category": "ml_models",
        "description": "DeepHit 风格的离散时间竞争风险预测和事件类型风险分布",
        "example_dataset": "ml_deephit_example",
        "params": [
            {"key": "time_var", "label": "随访时间", "type": "select", "default": "time_months"},
            {"key": "event_var", "label": "事件类型", "type": "select", "default": "event_type"},
            {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
        ],
    },
}


METHOD_EXAMPLE_MAP = {
    "gee": "gee_example",
    "propensity_score": "propensity_score_example",
    "sensitivity_analysis": "sensitivity_analysis_example",
    "counterfactual": "counterfactual_example",
    "survival_advanced": "survival_advanced_example",
    "markov_model": "markov_model_example",
    "bayesian": "bayesian_example",
    "latin_square": "latin_square_example",
    "meta_analysis": "meta_analysis_example",
    "mediation": "mediation_example",
    "mixed_effects": "mixed_effects_example",
    "nhanes_analysis": "nhanes_analysis_example",
    "ldsc": "ldsc_example",
    "target_trial_emulation": "target_trial_emulation_example",
    "doubly_robust_aipw": "doubly_robust_aipw_example",
    "marginal_structural_model": "marginal_structural_model_example",
    "ml_lr": "ml_lr_example",
    "ml_lasso": "ml_lasso_example",
    "ml_ridge": "ml_ridge_example",
    "ml_knn": "ml_knn_example",
    "ml_xgboost": "ml_xgboost_example",
    "ml_rf": "ml_rf_example",
    "ml_svm": "ml_svm_example",
    "ml_dt": "ml_dt_example",
    "ml_cnn": "ml_cnn_example",
    "feature_engineering": "feature_engineering_example",
    "model_comparison": "model_comparison_example",
    "dim_reduction": "dim_reduction_example",
    "cluster": "cluster_example",
    "ml_stacking": "ml_stacking_example",
    "ml_mlp": "ml_mlp_example",
    "ml_elastic_net": "ml_elastic_net_example",
    "ml_deepsurv": "ml_deepsurv_example",
    "ml_deephit": "ml_deephit_example",
}

ML_TOOL_METHODS = {"feature_engineering", "model_comparison", "dim_reduction", "cluster"}

COMMON_ML_PARAMS = [
    {"key": "split_ratio", "label": "训练集比例", "type": "select", "default": "0.7",
     "options": ["0.8", "0.7", "0.6", "0.5"]},
    {"key": "random_state", "label": "随机种子", "type": "number", "default": 42, "min": 0, "max": 9999, "step": 1},
]

METHOD_PARAM_EXTENSIONS = {
    "gee": [
        {"key": "cov_struct", "label": "相关结构", "type": "select", "default": "exchangeable",
         "options": ["exchangeable", "independence", "ar1"]},
    ],
    "propensity_score": [
        {"key": "caliper", "label": "匹配卡钳", "type": "number", "default": 0.20, "min": 0.01, "max": 1.0, "step": 0.01},
        {"key": "matching_ratio", "label": "匹配比例", "type": "select", "default": "1", "options": ["1", "2", "3"]},
    ],
    "sensitivity_analysis": [
        {"key": "missing_strategy", "label": "缺失处理", "type": "select", "default": "complete_case",
         "options": ["complete_case", "mean_impute", "median_impute"]},
        {"key": "alpha", "label": "显著性阈值", "type": "number", "default": 0.05, "min": 0.001, "max": 0.2, "step": 0.001},
    ],
    "counterfactual": [
        {"key": "effect_model", "label": "效应模型", "type": "select", "default": "t_learner",
         "options": ["t_learner", "s_learner"]},
        {"key": "random_state", "label": "随机种子", "type": "number", "default": 42, "min": 0, "max": 9999, "step": 1},
    ],
    "survival_advanced": [
        {"key": "time_horizon", "label": "预测时间点", "type": "number", "default": 36, "min": 1, "max": 120, "step": 1},
        {"key": "alpha", "label": "显著性阈值", "type": "number", "default": 0.05, "min": 0.001, "max": 0.2, "step": 0.001},
    ],
    "markov_model": [
        {"key": "n_cycles", "label": "模拟周期数", "type": "number", "default": 12, "min": 2, "max": 120, "step": 1},
    ],
    "bayesian": [
        {"key": "credible_interval", "label": "可信区间", "type": "select", "default": "0.95", "options": ["0.90", "0.95", "0.99"]},
        {"key": "prior_scale", "label": "先验尺度", "type": "number", "default": 1.0, "min": 0.1, "max": 10, "step": 0.1},
    ],
    "latin_square": [
        {"key": "alpha", "label": "显著性阈值", "type": "number", "default": 0.05, "min": 0.001, "max": 0.2, "step": 0.001},
    ],
    "meta_analysis": [
        {"key": "model_type", "label": "合并模型", "type": "select", "default": "random", "options": ["random", "fixed"]},
    ],
    "mediation": [
        {"key": "bootstrap", "label": "Bootstrap次数", "type": "number", "default": 1000, "min": 100, "max": 5000, "step": 100},
        {"key": "random_state", "label": "随机种子", "type": "number", "default": 42, "min": 0, "max": 9999, "step": 1},
    ],
    "mixed_effects": [
        {"key": "random_slope", "label": "随机斜率", "type": "select", "default": "none", "options": ["none", "time"]},
    ],
    "nhanes_analysis": [
        {"key": "weight_trim", "label": "权重截尾分位", "type": "number", "default": 0.99, "min": 0.80, "max": 1.0, "step": 0.01},
    ],
    "ldsc": [
        {"key": "h2_col", "label": "遗传力列", "type": "select", "default": "h2"},
        {"key": "h2_se_col", "label": "遗传力SE列", "type": "select", "default": "h2_se"},
    ],
    "target_trial_emulation": [
        {"key": "feature_vars", "label": "基线协变量", "type": "multi_select", "default": []},
        {"key": "estimand", "label": "目标估计量", "type": "select", "default": "ate", "options": ["ate", "att"]},
        {"key": "weight_trim", "label": "权重截尾分位", "type": "number", "default": 0.99, "min": 0.80, "max": 1.0, "step": 0.01},
        {"key": "stabilized", "label": "稳定化权重", "type": "select", "default": "yes", "options": ["yes", "no"]},
    ],
    "doubly_robust_aipw": [
        {"key": "outcome_type", "label": "结局类型", "type": "select", "default": "auto", "options": ["auto", "binary", "continuous"]},
        {"key": "n_estimators", "label": "机器学习树数量", "type": "number", "default": 200, "min": 50, "max": 800, "step": 50},
        {"key": "trim_quantile", "label": "倾向评分截尾", "type": "number", "default": 0.01, "min": 0, "max": 0.20, "step": 0.01},
        {"key": "random_state", "label": "随机种子", "type": "number", "default": 42, "min": 0, "max": 9999, "step": 1},
    ],
    "marginal_structural_model": [
        {"key": "feature_vars", "label": "IPTW协变量", "type": "multi_select", "default": []},
        {"key": "id_var", "label": "受试者ID", "type": "select", "default": "subject_id"},
        {"key": "stabilized", "label": "稳定化权重", "type": "select", "default": "yes", "options": ["yes", "no"]},
        {"key": "weight_trim", "label": "权重截尾分位", "type": "number", "default": 0.99, "min": 0.80, "max": 1.0, "step": 0.01},
    ],
    "ml_lr": [
        *COMMON_ML_PARAMS,
        {"key": "threshold", "label": "分类阈值", "type": "number", "default": 0.50, "min": 0.05, "max": 0.95, "step": 0.01},
        {"key": "C", "label": "正则强度C", "type": "number", "default": 1.0, "min": 0.01, "max": 20, "step": 0.01},
    ],
    "ml_lasso": [
        {"key": "alpha", "label": "Alpha", "type": "number", "default": 0.05, "min": 0.0001, "max": 10, "step": 0.0001},
    ],
    "ml_ridge": [
        {"key": "alpha", "label": "Alpha", "type": "number", "default": 0.05, "min": 0.0001, "max": 10, "step": 0.0001},
    ],
    "ml_elastic_net": [
        {"key": "alpha", "label": "Alpha", "type": "number", "default": 0.05, "min": 0.0001, "max": 10, "step": 0.0001},
        {"key": "l1_ratio", "label": "L1比例", "type": "number", "default": 0.50, "min": 0.01, "max": 0.99, "step": 0.01},
    ],
    "ml_knn": [
        *COMMON_ML_PARAMS,
        {"key": "n_neighbors", "label": "K值", "type": "number", "default": 5, "min": 1, "max": 31, "step": 2},
        {"key": "weights", "label": "邻居权重", "type": "select", "default": "uniform", "options": ["uniform", "distance"]},
    ],
    "ml_xgboost": [
        *COMMON_ML_PARAMS,
        {"key": "n_estimators", "label": "树数量", "type": "number", "default": 120, "min": 20, "max": 500, "step": 10},
        {"key": "max_depth", "label": "最大深度", "type": "number", "default": 4, "min": 1, "max": 12, "step": 1},
        {"key": "learning_rate", "label": "学习率", "type": "number", "default": 0.10, "min": 0.01, "max": 0.5, "step": 0.01},
    ],
    "ml_rf": [
        *COMMON_ML_PARAMS,
        {"key": "n_estimators", "label": "树数量", "type": "number", "default": 200, "min": 20, "max": 800, "step": 10},
        {"key": "max_depth", "label": "最大深度", "type": "number", "default": 6, "min": 1, "max": 30, "step": 1},
    ],
    "ml_svm": [
        *COMMON_ML_PARAMS,
        {"key": "kernel", "label": "核函数", "type": "select", "default": "rbf", "options": ["linear", "rbf", "poly", "sigmoid"]},
        {"key": "C", "label": "惩罚参数C", "type": "number", "default": 1.0, "min": 0.01, "max": 50, "step": 0.01},
    ],
    "ml_dt": [
        *COMMON_ML_PARAMS,
        {"key": "max_depth", "label": "最大深度", "type": "number", "default": 4, "min": 1, "max": 30, "step": 1},
        {"key": "min_samples_leaf", "label": "叶节点最小样本", "type": "number", "default": 2, "min": 1, "max": 50, "step": 1},
    ],
    "ml_cnn": [
        *COMMON_ML_PARAMS,
        {"key": "hidden_units", "label": "隐藏单元", "type": "select", "default": "32,16", "options": ["16,8", "32,16", "64,32"]},
    ],
    "ml_stacking": [
        *COMMON_ML_PARAMS,
        {"key": "threshold", "label": "分类阈值", "type": "number", "default": 0.50, "min": 0.05, "max": 0.95, "step": 0.01},
        {"key": "passthrough", "label": "保留原始特征", "type": "select", "default": "yes", "options": ["yes", "no"]},
        {"key": "final_estimator_C", "label": "融合器 C", "type": "number", "default": 1.0, "min": 0.01, "max": 20, "step": 0.01},
    ],
    "ml_mlp": [
        *COMMON_ML_PARAMS,
        {"key": "threshold", "label": "分类阈值", "type": "number", "default": 0.50, "min": 0.05, "max": 0.95, "step": 0.01},
        {"key": "alpha", "label": "L2 正则 alpha", "type": "number", "default": 0.0005, "min": 0.00001, "max": 0.1, "step": 0.0001},
        {"key": "learning_rate_init", "label": "初始学习率", "type": "number", "default": 0.001, "min": 0.0001, "max": 0.05, "step": 0.0001},
        {"key": "max_iter", "label": "最大迭代次数", "type": "number", "default": 300, "min": 50, "max": 1000, "step": 50},
    ],
    "ml_deepsurv": [
        *COMMON_ML_PARAMS,
        {"key": "horizon", "label": "预测时间窗", "type": "number", "default": 24, "min": 1, "max": 120, "step": 1},
        {"key": "hidden_layer_sizes", "label": "隐藏层结构", "type": "select", "default": "64,32", "options": ["32,16", "64,32", "128,64"]},
        {"key": "alpha", "label": "L2 正则 alpha", "type": "number", "default": 0.0005, "min": 0.00001, "max": 0.1, "step": 0.0001},
        {"key": "max_iter", "label": "最大迭代次数", "type": "number", "default": 300, "min": 50, "max": 1000, "step": 50},
    ],
    "ml_deephit": [
        *COMMON_ML_PARAMS,
        {"key": "time_bins", "label": "离散时间分箱数", "type": "number", "default": 6, "min": 3, "max": 20, "step": 1},
        {"key": "hidden_layer_sizes", "label": "隐藏层结构", "type": "select", "default": "64,32", "options": ["32,16", "64,32", "128,64"]},
        {"key": "alpha", "label": "L2 正则 alpha", "type": "number", "default": 0.0005, "min": 0.00001, "max": 0.1, "step": 0.0001},
        {"key": "max_iter", "label": "最大迭代次数", "type": "number", "default": 300, "min": 50, "max": 1000, "step": 50},
    ],
    "feature_engineering": [
        {"key": "missing_strategy", "label": "缺失值策略", "type": "select", "default": "median", "options": ["median", "mean", "drop"]},
        {"key": "scaling", "label": "标准化", "type": "select", "default": "zscore", "options": ["zscore", "minmax", "none"]},
        {"key": "outlier_z", "label": "异常值Z阈值", "type": "number", "default": 3.0, "min": 1.5, "max": 6.0, "step": 0.1},
    ],
    "model_comparison": [
        *COMMON_ML_PARAMS,
        {"key": "threshold", "label": "分类阈值", "type": "number", "default": 0.50, "min": 0.05, "max": 0.95, "step": 0.01},
    ],
    "dim_reduction": [
        {"key": "n_components", "label": "PCA主成分数", "type": "number", "default": 10, "min": 2, "max": 50, "step": 1},
        {"key": "perplexity", "label": "t-SNE perplexity", "type": "number", "default": 30, "min": 5, "max": 60, "step": 1},
        {"key": "tsne_learning_rate", "label": "t-SNE学习率", "type": "number", "default": 200, "min": 10, "max": 1000, "step": 10},
        {"key": "umap_n_neighbors", "label": "UMAP邻居数", "type": "number", "default": 15, "min": 2, "max": 100, "step": 1},
        {"key": "umap_min_dist", "label": "UMAP最小距离", "type": "number", "default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01},
        {"key": "standardize", "label": "特征标准化", "type": "select", "default": "zscore", "options": ["zscore", "none"]},
        {"key": "random_state", "label": "随机种子", "type": "number", "default": 42, "min": 0, "max": 9999, "step": 1},
    ],
    "cluster": [
        {"key": "random_state", "label": "随机种子", "type": "number", "default": 42, "min": 0, "max": 9999, "step": 1},
        {"key": "n_init", "label": "K-Means初始化次数", "type": "number", "default": 10, "min": 1, "max": 50, "step": 1},
    ],
}


PARAM_CONTROL_BOOSTS = {
    "gee": [
        {"key": "max_iter", "label": "最大迭代次数", "type": "number", "default": 100, "min": 20, "max": 500, "step": 10},
        {"key": "scale", "label": "尺度参数", "type": "select", "default": "auto", "options": ["auto", "pearson", "deviance"]},
    ],
    "propensity_score": [
        {"key": "ps_model", "label": "倾向评分模型", "type": "select", "default": "logistic", "options": ["logistic", "random_forest"]},
        {"key": "standardize", "label": "协变量标准化", "type": "select", "default": "yes", "options": ["yes", "no"]},
    ],
    "sensitivity_analysis": [
        {"key": "bootstrap", "label": "Bootstrap 次数", "type": "number", "default": 1000, "min": 100, "max": 5000, "step": 100},
        {"key": "delta_range", "label": "扰动范围", "type": "number", "default": 0.20, "min": 0.01, "max": 2.0, "step": 0.01},
    ],
    "counterfactual": [
        {"key": "n_estimators", "label": "基学习器数量", "type": "number", "default": 200, "min": 20, "max": 800, "step": 10},
        {"key": "min_samples_leaf", "label": "叶节点最小样本", "type": "number", "default": 5, "min": 1, "max": 50, "step": 1},
        {"key": "trim_quantile", "label": "极端倾向截尾", "type": "number", "default": 0.01, "min": 0, "max": 0.20, "step": 0.01},
    ],
    "survival_advanced": [
        {"key": "ties_method", "label": "并列时间处理", "type": "select", "default": "efron", "options": ["efron", "breslow"]},
        {"key": "reference_group", "label": "参考组", "type": "select", "default": ""},
    ],
    "markov_model": [
        {"key": "cycle_length", "label": "周期长度", "type": "number", "default": 1, "min": 0.25, "max": 12, "step": 0.25},
        {"key": "discount_rate", "label": "折现率", "type": "number", "default": 0.03, "min": 0, "max": 0.20, "step": 0.005},
        {"key": "start_state", "label": "起始状态", "type": "select", "default": ""},
    ],
    "bayesian": [
        {"key": "draws", "label": "后验抽样次数", "type": "number", "default": 2000, "min": 500, "max": 10000, "step": 500},
        {"key": "burn_in", "label": "预热样本数", "type": "number", "default": 500, "min": 100, "max": 5000, "step": 100},
    ],
    "latin_square": [
        {"key": "include_carryover", "label": "纳入携带效应", "type": "select", "default": "no", "options": ["no", "yes"]},
    ],
    "meta_analysis": [
        {"key": "tau_method", "label": "异质性估计", "type": "select", "default": "DL", "options": ["DL", "REML", "PM"]},
        {"key": "effect_measure", "label": "效应量类型", "type": "select", "default": "SMD", "options": ["SMD", "MD", "OR", "RR", "HR"]},
        {"key": "prediction_interval", "label": "预测区间", "type": "select", "default": "yes", "options": ["yes", "no"]},
    ],
    "mediation": [
        {"key": "ci_level", "label": "置信水平", "type": "select", "default": "0.95", "options": ["0.90", "0.95", "0.99"]},
        {"key": "standardize", "label": "标准化路径系数", "type": "select", "default": "yes", "options": ["yes", "no"]},
    ],
    "mixed_effects": [
        {"key": "reml", "label": "REML 估计", "type": "select", "default": "yes", "options": ["yes", "no"]},
        {"key": "optimizer", "label": "优化器", "type": "select", "default": "lbfgs", "options": ["lbfgs", "powell", "cg"]},
        {"key": "max_iter", "label": "最大迭代次数", "type": "number", "default": 200, "min": 50, "max": 1000, "step": 50},
    ],
    "nhanes_analysis": [
        {"key": "variance_method", "label": "方差估计", "type": "select", "default": "linearized", "options": ["linearized", "replicate"]},
        {"key": "domain_adjust", "label": "亚组域校正", "type": "select", "default": "yes", "options": ["yes", "no"]},
    ],
    "ldsc": [
        {"key": "rg_threshold", "label": "遗传相关阈值", "type": "number", "default": 0.30, "min": 0, "max": 1, "step": 0.01},
    ],
    "ml_lasso": [
        *COMMON_ML_PARAMS,
        {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
        {"key": "cv_folds", "label": "交叉验证折数", "type": "select", "default": "5", "options": ["3", "5", "10"]},
    ],
    "ml_ridge": [
        *COMMON_ML_PARAMS,
        {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
        {"key": "cv_folds", "label": "交叉验证折数", "type": "select", "default": "5", "options": ["3", "5", "10"]},
    ],
    "ml_elastic_net": [
        *COMMON_ML_PARAMS,
        {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
        {"key": "cv_folds", "label": "交叉验证折数", "type": "select", "default": "5", "options": ["3", "5", "10"]},
        {"key": "l1_ratio_grid", "label": "L1比例网格", "type": "select", "default": "0.1,0.5,0.9", "options": ["0.1,0.5,0.9", "0.2,0.5,0.8", "0.05,0.25,0.5,0.75,0.95"]},
    ],
    "ml_lr": [
        {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
        {"key": "cv_folds", "label": "交叉验证折数", "type": "select", "default": "5", "options": ["3", "5", "10"]},
        {"key": "class_weight", "label": "类别权重", "type": "select", "default": "none", "options": ["none", "balanced"]},
    ],
    "ml_knn": [
        {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
        {"key": "distance_metric", "label": "距离度量", "type": "select", "default": "minkowski", "options": ["minkowski", "euclidean", "manhattan"]},
    ],
    "ml_xgboost": [
        {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
        {"key": "subsample", "label": "样本采样比例", "type": "number", "default": 0.8, "min": 0.5, "max": 1.0, "step": 0.05},
        {"key": "colsample_bytree", "label": "特征采样比例", "type": "number", "default": 0.8, "min": 0.5, "max": 1.0, "step": 0.05},
    ],
    "ml_rf": [
        {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
        {"key": "min_samples_leaf", "label": "叶节点最小样本", "type": "number", "default": 2, "min": 1, "max": 50, "step": 1},
        {"key": "class_weight", "label": "类别权重", "type": "select", "default": "none", "options": ["none", "balanced"]},
    ],
    "ml_svm": [
        {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
        {"key": "gamma", "label": "Gamma", "type": "select", "default": "scale", "options": ["scale", "auto"]},
        {"key": "probability", "label": "输出概率", "type": "select", "default": "yes", "options": ["yes", "no"]},
    ],
    "ml_dt": [
        {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
        {"key": "criterion", "label": "分裂准则", "type": "select", "default": "gini", "options": ["gini", "entropy", "log_loss"]},
    ],
    "ml_cnn": [
        {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
        {"key": "epochs", "label": "训练轮数", "type": "number", "default": 40, "min": 5, "max": 200, "step": 5},
        {"key": "batch_size", "label": "批量大小", "type": "select", "default": "32", "options": ["16", "32", "64"]},
    ],
    "ml_deepsurv": [
        {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
        {"key": "time_var", "label": "生存时间变量", "type": "select", "default": "time_months"},
        {"key": "event_var", "label": "事件变量", "type": "select", "default": "event"},
        {"key": "learning_rate_init", "label": "初始学习率", "type": "number", "default": 0.001, "min": 0.0001, "max": 0.05, "step": 0.0001},
    ],
    "ml_deephit": [
        {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
        {"key": "time_var", "label": "随访时间变量", "type": "select", "default": "time_months"},
        {"key": "event_var", "label": "事件类型变量", "type": "select", "default": "event_type"},
        {"key": "learning_rate_init", "label": "初始学习率", "type": "number", "default": 0.001, "min": 0.0001, "max": 0.05, "step": 0.0001},
    ],
    "feature_engineering": [
        {"key": "feature_vars", "label": "处理变量", "type": "multi_select", "default": []},
        {"key": "encode_strategy", "label": "分类变量编码", "type": "select", "default": "onehot", "options": ["onehot", "ordinal", "none"]},
    ],
    "model_comparison": [
        {"key": "feature_vars", "label": "建模特征变量", "type": "multi_select", "default": []},
        {"key": "cv_folds", "label": "交叉验证折数", "type": "select", "default": "5", "options": ["3", "5", "10"]},
    ],
    "dim_reduction": [
        {"key": "feature_vars", "label": "降维特征变量", "type": "multi_select", "default": []},
        {"key": "group_var", "label": "分组变量", "type": "select", "default": "subtype"},
        {"key": "method", "label": "降维方法", "type": "select", "default": "all", "options": ["pca", "tsne", "umap", "pca_tsne", "all"]},
    ],
    "cluster": [
        {"key": "feature_vars", "label": "聚类特征变量", "type": "multi_select", "default": []},
        {"key": "algorithm", "label": "聚类算法", "type": "select", "default": "kmeans", "options": ["kmeans", "hierarchical", "gmm"]},
    ],
}


def _extend_method_catalog() -> None:
    for method_id, cfg in METHOD_CATALOG.items():
        cfg["example_dataset"] = METHOD_EXAMPLE_MAP.get(method_id, cfg.get("example_dataset", "gee_example"))
        if method_id in ML_TOOL_METHODS:
            cfg["category"] = "ml_models"
        params = list(cfg.get("params") or [])
        seen = {p.get("key") for p in params}
        for p in [*METHOD_PARAM_EXTENSIONS.get(method_id, []), *PARAM_CONTROL_BOOSTS.get(method_id, [])]:
            if p.get("key") not in seen:
                params.append(dict(p))
                seen.add(p.get("key"))
        cfg["params"] = params


_extend_method_catalog()


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
        preview = df.head(200).fillna("").to_dict(orient="records")
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
        raise HTTPException(status_code=400, detail=f"Upload failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


# ── Examples ────────────────────────────────────────────────

@app.delete("/api/upload/{upload_id}")
def delete_upload(upload_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-fA-F]{12}", upload_id or ""):
        raise HTTPException(status_code=400, detail="Invalid upload id")

    root = UPLOADS_DIR.resolve()
    removed: list[str] = []
    for path in UPLOADS_DIR.iterdir():
        if not path.is_file() or not path.name.startswith(f"{upload_id}_"):
            continue
        resolved = path.resolve()
        if resolved.parent != root:
            continue
        path.unlink()
        removed.append(path.name)

    return {"upload_id": upload_id, "deleted": len(removed), "files": removed}


@app.get("/api/examples")
def list_examples() -> list[dict]:
    return [item for item in get_example_datasets() if item.get("name") != "comprehensive_example"]


@app.get("/api/examples/{name}")
def get_example(name: str) -> dict:
    if name == "comprehensive_example":
        raise HTTPException(status_code=404, detail="Comprehensive example has been removed")
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
        "preview": df.head(200).fillna("").to_dict(orient="records"),
        "missing_percent": summary["missing_percent"],
        "summary": summary,
    }


@app.get("/api/examples/{name}/download")
def download_example(name: str) -> FileResponse:
    if name == "comprehensive_example":
        raise HTTPException(status_code=404, detail="Comprehensive example has been removed")
    filepath = EXAMPLES_DIR / f"{name}.csv"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Example not found")
    return FileResponse(filepath, media_type="text/csv", filename=f"{name}.csv")


# ── Main Analysis API ──────────────────────────────────────


def _is_discrete_target_for_classifier(df: pd.DataFrame, target: str) -> bool:
    """Strict backend guard: classifiers require binary/multiclass target, not continuous numeric."""
    if not target or target not in df.columns:
        return False
    s = df[target].dropna()
    if s.empty:
        return False
    nunique = int(s.nunique())
    if nunique < 2:
        return False
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").dropna()
        if vals.empty:
            return False
        all_integer = bool(np.all(np.isclose(vals, np.round(vals))))
        return all_integer and nunique <= 10
    return nunique <= 30

def _guard_method_target_compatibility(df: pd.DataFrame, method_id: str, params: dict) -> None:
    classifiers = {"ml_lr", "ml_knn", "ml_xgboost", "ml_rf", "ml_svm", "ml_dt", "ml_cnn", "model_comparison", "ml_stacking", "ml_mlp"}
    if method_id in classifiers:
        target = params.get("target") or ""
        if not _is_discrete_target_for_classifier(df, target):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"当前选择的目标变量“{target}”不是离散分类结局。"
                    "分类模型只能用于二分类或多分类结局；如果结局是连续型，请选择 Lasso/岭回归或相应高级统计方法。"
                ),
            )




# ═══════════════════════════════════════════════════════════════
# v10: role-driven method validation API
# ═══════════════════════════════════════════════════════════════

def _unique_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].dropna().nunique())

def _is_id_like_col(col: str) -> bool:
    return bool(re.search(r"(^id$|_id$|id$|subject|patient|sample|编号)", str(col), flags=re.I))

def _is_numeric_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and pd.api.types.is_numeric_dtype(df[col])

def _is_discrete_col(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns or _is_id_like_col(col):
        return False
    s = df[col].dropna()
    if s.empty:
        return False
    u = int(s.nunique())
    if u < 2:
        return False
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").dropna()
        if vals.empty:
            return False
        all_int = bool(np.all(np.isclose(vals, np.round(vals))))
        return all_int and u <= 10
    return u <= 30

def _is_binary_col(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns or _is_id_like_col(col):
        return False
    s = df[col].dropna()
    if s.empty:
        return False
    u = int(s.nunique())
    if pd.api.types.is_numeric_dtype(s):
        vals = set(pd.to_numeric(s, errors="coerce").dropna().unique().tolist())
        return u == 2 and vals.issubset({0, 1, 0.0, 1.0})
    return u == 2

def _is_continuous_col(df: pd.DataFrame, col: str) -> bool:
    return _is_numeric_col(df, col) and not _is_discrete_col(df, col)

def _pick(cols: list[str], pred) -> str | None:
    for c in cols:
        if pred(c):
            return c
    return None

def _pick_name(cols: list[str], pattern: str, pred=lambda c: True) -> str | None:
    rgx = re.compile(pattern, re.I)
    for c in cols:
        if rgx.search(str(c)) and pred(c):
            return c
    return None

def _role_vars_clean(df: pd.DataFrame, roles: dict) -> dict:
    def clean(key):
        vals = roles.get(key) or []
        if isinstance(vals, str):
            vals = [x.strip() for x in vals.split(",") if x.strip()]
        return [v for v in vals if v in df.columns]
    research = clean("research_vars")
    covars = clean("covar_vars")
    outcomes = clean("outcome_vars")
    return {
        "research": research,
        "covars": covars,
        "outcomes": outcomes,
        "features": list(dict.fromkeys(research + covars)),
        "selected": list(dict.fromkeys(research + covars + outcomes)),
    }

def _derive_params_from_roles(df: pd.DataFrame, method_id: str, roles: dict) -> dict:
    r = _role_vars_clean(df, roles)
    research, covars, outcomes, features = r["research"], r["covars"], r["outcomes"], r["features"]
    allv = r["selected"]
    params: dict = {}
    all_cols = list(df.columns)
    catalog_defaults = {
        item.get("key"): item.get("default")
        for item in METHOD_CATALOG.get(method_id, {}).get("params", [])
    }

    def num_out(): return _pick(outcomes, lambda c: _is_continuous_col(df, c))
    def disc_out(): return _pick(outcomes, lambda c: _is_discrete_col(df, c))
    def any_out(): return outcomes[0] if outcomes else None
    def group(): return _pick_name(research + covars + outcomes, r"group|arm|treat|stage|sex|center|site|strata|class|disease|state", lambda c: _is_discrete_col(df, c)) or _pick(research + covars, lambda c: _is_discrete_col(df, c))
    def treat(): return _pick_name(research + covars, r"treat|arm|group|处理|分组", lambda c: _is_discrete_col(df, c)) or _pick(research, lambda c: _is_discrete_col(df, c))
    def timev(): return _pick_name(outcomes + covars + research, r"time|week|month|duration|follow|survival|os", lambda c: _is_numeric_col(df, c)) or _pick(outcomes + covars + research, lambda c: _is_numeric_col(df, c))
    def eventv(): return _pick_name(outcomes + covars + research, r"event|status|death|outcome|complication|label|high_risk", lambda c: _is_binary_col(df, c)) or _pick(outcomes, lambda c: _is_binary_col(df, c))
    def idv(): return _pick_name(covars + research + outcomes + allv, r"subject|patient|sample|id", lambda c: True)
    def default_col(key, pred=lambda c: True):
        value = catalog_defaults.get(key)
        return value if value in df.columns and pred(value) else None
    def named(pattern, pool=None, pred=lambda c: True):
        return _pick_name(list(pool or all_cols), pattern, pred)
    def first_col(pool=None, pred=lambda c: True, exclude=None):
        excluded = set(exclude or [])
        return _pick([c for c in list(pool or all_cols) if c not in excluded], pred)
    def binary_col(pattern=None, pool=None):
        source = list(pool or all_cols)
        pred = lambda c: _has_min_two_groups(df, c, min_each=5)
        if pattern:
            return named(pattern, source, pred)
        return first_col(source, pred)
    def numeric_col(pattern=None, pool=None, exclude=None):
        source = [c for c in list(pool or all_cols) if c not in set(exclude or [])]
        if pattern:
            return named(pattern, source, lambda c: _is_numeric_col(df, c))
        return first_col(source, lambda c: _is_numeric_col(df, c))

    if method_id in {"ml_lr", "ml_knn", "ml_xgboost", "ml_rf", "ml_svm", "ml_dt", "ml_cnn", "model_comparison", "ml_stacking", "ml_mlp"}:
        target = disc_out()
        params = {"target": target, "feature_vars": [c for c in features if c != target]}
    elif method_id in {"ml_lasso", "ml_ridge", "ml_elastic_net"}:
        target = (
            num_out()
            or default_col("target", lambda c: _is_numeric_col(df, c))
            or (numeric_col(r"renal_metabolic_index") if method_id == "ml_ridge" else None)
            or (numeric_col(r"renal_decline_score|score|index|continuous") if method_id == "ml_elastic_net" else None)
            or numeric_col(r"^outcome$|outcome_continuous|response|target|followup|score|index")
            or numeric_col(exclude=features)
        )
        feature_pool = features or all_cols
        regularization = "elastic_net" if method_id == "ml_elastic_net" else ("ridge" if method_id == "ml_ridge" else "lasso")
        params = {
            "target": target,
            "feature_vars": [c for c in feature_pool if c != target],
            "regularization": regularization,
        }
    elif method_id == "gee":
        outcome = num_out() or eventv() or any_out()
        params = {"outcome_var": outcome, "time_var": timev(), "group_var": group(), "subject_var": idv(), "family": "binomial" if outcome and _is_binary_col(df, outcome) else "gaussian"}
    elif method_id == "propensity_score":
        params = {"treatment_var": treat(), "outcome_var": any_out()}
    elif method_id == "sensitivity_analysis":
        treatment = treat()
        if not _has_min_two_groups(df, treatment, min_each=5):
            treatment = (
                binary_col(r"treat|treated|intervention|exposure|therapy")
                or binary_col(r"sex|gender|smoking|case|control")
                or binary_col(r"arm|group")
            )
        outcome = num_out() or default_col("outcome_var", lambda c: _is_numeric_col(df, c)) or numeric_col(r"followup|outcome|score|response|change")
        baseline = (
            _pick_name(covars, r"baseline|base|pre|基线", lambda c: _is_numeric_col(df, c))
            or default_col("baseline_var", lambda c: _is_numeric_col(df, c))
            or numeric_col(r"baseline|base|pre", exclude=[outcome])
            or numeric_col(exclude=[outcome])
        )
        params = {"treatment_var": treatment, "outcome_var": outcome, "baseline_var": baseline}
    elif method_id == "counterfactual":
        params = {"treatment_var": treat(), "outcome_var": num_out() or any_out()}
    elif method_id == "target_trial_emulation":
        treatment = treat()
        outcome = eventv() or disc_out() or any_out()
        time_col = timev() or default_col("time_var", lambda c: _is_numeric_col(df, c))
        params = {
            "treatment_var": treatment,
            "outcome_var": outcome,
            "time_var": time_col,
            "feature_vars": [c for c in features if c not in {treatment, outcome, time_col}],
        }
    elif method_id == "doubly_robust_aipw":
        treatment = treat()
        outcome = any_out() or num_out() or disc_out()
        params = {
            "treatment_var": treatment,
            "outcome_var": outcome,
            "feature_vars": [c for c in features if c not in {treatment, outcome}],
        }
    elif method_id == "marginal_structural_model":
        treatment = treat()
        outcome = num_out() or disc_out() or any_out()
        time_col = timev() or default_col("time_var", lambda c: _is_numeric_col(df, c))
        subject_col = idv() or default_col("id_var")
        params = {
            "treatment_var": treatment,
            "outcome_var": outcome,
            "time_var": time_col,
            "id_var": subject_col,
            "feature_vars": [c for c in features if c not in {treatment, outcome, time_col, subject_col}],
        }
    elif method_id in {"ml_deepsurv", "ml_deephit"}:
        time_col = timev() or default_col("time_var", lambda c: _is_numeric_col(df, c))
        event_col = (
            _pick_name(outcomes + covars + research, r"event|status|death|cause|type|outcome", lambda c: _is_discrete_col(df, c))
            or _pick(outcomes, lambda c: _is_discrete_col(df, c))
            or disc_out()
        )
        params = {
            "time_var": time_col,
            "event_var": event_col,
            "feature_vars": [c for c in features if c not in {time_col, event_col}],
        }
    elif method_id == "survival_advanced":
        params = {"time_var": timev(), "event_var": eventv(), "group_var": group()}
    elif method_id == "markov_model":
        params = {
            "state_var": default_col("state_var", lambda c: _is_discrete_col(df, c)) or _pick_name(research + outcomes + covars, r"state|status|stage|状态", lambda c: _is_discrete_col(df, c)) or named(r"^state$|status|stage", pred=lambda c: _is_discrete_col(df, c)) or group(),
            "time_var": default_col("time_var", lambda c: _is_numeric_col(df, c)) or timev() or numeric_col(r"month|time|cycle|period|visit"),
            "subject_var": idv() or named(r"subject|patient|person|sample|id"),
        }
    elif method_id == "bayesian":
        params = {"outcome_var": num_out() or any_out(), "group_var": group()}
    elif method_id == "latin_square":
        subject_candidate = idv()
        formulation_candidate = _pick_name(research + covars, r"formulation|treat|arm|group|处理|剂型", lambda c: _is_discrete_col(df, c)) or group()
        params = {
            "response_var": num_out() or _pick(research, lambda c: _is_numeric_col(df, c)),
            "period_var": _pick_name(covars + research, r"period|time|week|month|时期", lambda c: True)
                or _pick([c for c in covars + research if c not in {formulation_candidate, subject_candidate}], lambda c: _has_variation(df, c)),
            "formulation_var": formulation_candidate,
            "subject_var": subject_candidate,
        }
    elif method_id == "meta_analysis":
        params = {
            "study_var": _pick_name(research + covars, r"study|trial|研究", lambda c: _is_discrete_col(df, c)) or group(),
            "effect_var": _pick_name(outcomes + research, r"effect|效应|or|rr|hr", lambda c: _is_numeric_col(df, c)) or num_out(),
            "se_var": _pick_name(covars, r"se|standard|stderr|标准误", lambda c: _is_numeric_col(df, c)) or _pick(covars, lambda c: _is_numeric_col(df, c)),
        }
    elif method_id == "mediation":
        params = {
            "x_var": _pick(research, lambda c: _is_numeric_col(df, c)) or (research[0] if research else None),
            "m_var": _pick_name(covars, r"mediator|crp|中介", lambda c: _is_numeric_col(df, c)) or _pick(covars, lambda c: _is_numeric_col(df, c)),
            "y_var": num_out(),
        }
    elif method_id == "mixed_effects":
        params = {"outcome_var": num_out() or any_out(), "time_var": timev(), "group_var": group(), "random_var": _pick_name(covars, r"center|site|subject|patient|id|随机", lambda c: True) or idv()}
    elif method_id == "nhanes_analysis":
        outcome_candidate = num_out() or default_col("outcome_var", lambda c: _is_numeric_col(df, c)) or numeric_col(r"sbp|outcome|glucose|hba1c|bmi")
        weight_candidate = (
            _pick_name(covars, r"weight|wt|权重", lambda c: _is_numeric_col(df, c))
            or default_col("weight_var", lambda c: _is_numeric_col(df, c))
            or numeric_col(r"weight|wt")
            or _pick([c for c in covars if c != outcome_candidate], lambda c: _is_numeric_col(df, c))
        )
        params = {
            "weight_var": weight_candidate,
            "strata_var": _pick_name(covars, r"strata|stratum|分层|site|center", lambda c: _is_discrete_col(df, c))
                or default_col("strata_var", lambda c: _is_discrete_col(df, c))
                or named(r"strata|stratum|site|center", pred=lambda c: _is_discrete_col(df, c))
                or _pick([c for c in covars + research if c != weight_candidate], lambda c: _is_discrete_col(df, c)),
            "outcome_var": outcome_candidate,
        }
    elif method_id == "ldsc":
        params = {
            "group_var": default_col("group_var", lambda c: _is_discrete_col(df, c)) or _pick_name(research + outcomes + covars, r"trait|disease|疾病|性状", lambda c: _is_discrete_col(df, c)) or named(r"^trait$|phenotype|disease", pred=lambda c: _is_discrete_col(df, c)) or group(),
            "h2_col": default_col("h2_col", lambda c: _is_numeric_col(df, c)) or _pick_name(outcomes + research, r"^h2$|heritability|遗传力", lambda c: _is_numeric_col(df, c)) or numeric_col(r"^h2$|heritability"),
            "h2_se_col": default_col("h2_se_col", lambda c: _is_numeric_col(df, c)) or _pick_name(covars + outcomes, r"h2_se|standard.*error|标准误", lambda c: _is_numeric_col(df, c)) or numeric_col(r"h2.*se|se.*h2|standard.*error"),
        }
    elif method_id == "dim_reduction":
        gv = group() or disc_out()
        params = {"group_var": gv, "feature_vars": [c for c in features if c != gv]}
    elif method_id == "cluster":
        params = {"n_clusters": 4, "feature_vars": features}
    elif method_id == "feature_engineering":
        params = {"feature_vars": allv}

    # Clean None values, keep only existing feature columns.
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    for k, v in list(params.items()):
        if isinstance(v, list):
            params[k] = [c for c in v if c in df.columns and not _is_id_like_col(c)]
    return params

def _validate_method_roles(df: pd.DataFrame, method_id: str, roles: dict) -> tuple[bool, str, dict]:
    r = _role_vars_clean(df, roles)
    features = [c for c in r["features"] if not _is_id_like_col(c)]
    outcomes = r["outcomes"]
    params = _derive_params_from_roles(df, method_id, roles)
    def fail(msg): return False, msg, params
    if not r["selected"]:
        return fail("请先分配研究变量、协变量/混杂因素或结局变量")

    classifiers = {"ml_lr", "ml_knn", "ml_xgboost", "ml_rf", "ml_svm", "ml_dt", "ml_cnn", "model_comparison"}
    if method_id in classifiers:
        target = params.get("target")
        if not target:
            return fail("分类模型需要二分类或多分类结局变量；连续结局不能用于分类模型")
        if not _is_discrete_col(df, target):
            return fail(f"目标变量 {target} 不是离散分类结局")
        candidate_features = params.get("feature_vars") or features
        usable_features = [c for c in candidate_features if c != target and _has_variation(df, c)]
        if len(usable_features) < 2:
            return fail("分类模型至少需要 2 个非ID特征变量")
        class_counts = df[target].dropna().value_counts()
        if len(class_counts) < 2 or int(class_counts.min()) < 4:
            return fail("分类结局的每个类别至少需要 4 个有效样本")
        if _complete_row_count(df, [target] + usable_features) < 20:
            return fail("分类模型至少需要 20 个具有完整目标值和特征值的样本")
        params["feature_vars"] = usable_features
        return True, "", params
    if method_id == "propensity_score":
        tv = params.get("treatment_var")
        ov = params.get("outcome_var")
        if not tv or tv not in df.columns:
            return fail("PSM 需要处理/分组变量")
        if not _has_min_two_groups(df, tv, min_each=5):
            return fail("PSM 处理变量必须是二分类，且两组至少各有 5 个样本")
        if not ov or ov not in df.columns:
            return fail("PSM 需要结局变量")
        cov = [c for c in features if c not in {tv, ov} and not _is_id_like_col(c) and _has_variation(df, c)]
        if len(cov) < 1:
            return fail("PSM 至少需要 1 个非ID协变量用于估计倾向评分")
        params["covariates"] = cov[:12]
        return True, "", params

    if method_id in {"target_trial_emulation", "doubly_robust_aipw"}:
        tv = params.get("treatment_var")
        ov = params.get("outcome_var")
        if not tv or tv not in df.columns:
            return fail("该因果推断方法需要一个二分类处理/暴露变量")
        if not _has_min_two_groups(df, tv, min_each=10):
            return fail("处理/暴露变量必须为二分类，且两组至少各有 10 个样本")
        if not ov or ov not in df.columns:
            return fail("该因果推断方法需要一个结局变量")
        if method_id == "target_trial_emulation":
            time_var = params.get("time_var")
            if not time_var or time_var not in df.columns or not _is_numeric_col(df, time_var):
                return fail("目标试验模拟需要数值型随访时间变量")
            if not _is_binary_col(df, ov):
                return fail("目标试验模拟当前实现要求二分类事件结局")
            excluded = {tv, ov, time_var}
        else:
            if not (_is_binary_col(df, ov) or _is_numeric_col(df, ov)):
                return fail("AIPW 结局变量需要是二分类或数值型")
            excluded = {tv, ov}
        candidate_features = params.get("feature_vars") or features
        usable_features = [c for c in candidate_features if c not in excluded and not _is_id_like_col(c) and _has_variation(df, c)]
        if len(usable_features) < 2:
            return fail("该因果推断方法至少需要 2 个可用基线协变量")
        if _complete_row_count(df, [tv, ov] + ([params.get("time_var")] if method_id == "target_trial_emulation" else []) + usable_features) < 40:
            return fail("该因果推断方法至少需要 40 个完整样本")
        params["feature_vars"] = usable_features[:20]
        return True, "", params

    if method_id == "marginal_structural_model":
        tv = params.get("treatment_var")
        ov = params.get("outcome_var")
        time_var = params.get("time_var")
        id_var = params.get("id_var")
        if not tv or tv not in df.columns:
            return fail("边际结构模型需要一个二分类处理/暴露变量")
        if not _has_min_two_groups(df, tv, min_each=10):
            return fail("边际结构模型的处理变量必须为二分类，且两组至少各有 10 条记录")
        if not ov or ov not in df.columns or not (_is_numeric_col(df, ov) or _is_binary_col(df, ov)):
            return fail("边际结构模型需要数值型或二分类结局变量")
        if not time_var or time_var not in df.columns or not _is_numeric_col(df, time_var):
            return fail("边际结构模型需要数值型随访时间/访视时间变量")
        candidate_features = params.get("feature_vars") or features
        excluded = {tv, ov, time_var, id_var}
        usable_features = [c for c in candidate_features if c not in excluded and not _is_id_like_col(c) and _has_variation(df, c)]
        if len(usable_features) < 2:
            return fail("边际结构模型至少需要 2 个可用协变量估计治疗权重")
        complete_cols = [tv, ov, time_var] + ([id_var] if id_var in df.columns else []) + usable_features
        if _complete_row_count(df, complete_cols) < 40:
            return fail("边际结构模型至少需要 40 条完整记录")
        if id_var in df.columns and int(df[id_var].dropna().value_counts().max()) < 2:
            return fail("边际结构模型通常需要纵向重复测量；当前受试者ID没有重复记录")
        params["feature_vars"] = usable_features[:20]
        return True, "", params

    if method_id in {"ml_deepsurv", "ml_deephit"}:
        time_var = params.get("time_var")
        event_var = params.get("event_var")
        if not time_var or time_var not in df.columns or not _is_numeric_col(df, time_var):
            return fail("DeepSurv/DeepHit 需要数值型生存时间或随访时间变量")
        times = pd.to_numeric(df[time_var], errors="coerce").dropna()
        if times.empty or (times <= 0).any():
            return fail("DeepSurv/DeepHit 的时间变量必须为正数")
        if not event_var or event_var not in df.columns or not _is_discrete_col(df, event_var):
            return fail("DeepSurv/DeepHit 需要事件状态变量；DeepHit 可为 0/1/2 等竞争风险类型")
        events = df[event_var].dropna()
        if int(events.nunique()) < 2:
            return fail("事件变量至少需要包含未发生/发生两类状态")
        if method_id == "ml_deepsurv" and not _is_binary_col(df, event_var):
            return fail("DeepSurv 当前实现要求二分类事件变量；竞争风险请使用 DeepHit")
        if method_id == "ml_deephit" and int(events.nunique()) < 2:
            return fail("DeepHit 至少需要 2 个事件状态，通常 0=删失、1/2=不同事件")
        candidate_features = params.get("feature_vars") or features
        usable_features = [c for c in candidate_features if c not in {time_var, event_var} and not _is_id_like_col(c) and _has_variation(df, c)]
        if len(usable_features) < 2:
            return fail("DeepSurv/DeepHit 至少需要 2 个非ID特征变量")
        if _complete_row_count(df, [time_var, event_var] + usable_features) < 40:
            return fail("DeepSurv/DeepHit 至少需要 40 条完整记录")
        params["feature_vars"] = usable_features[:30]
        return True, "", params

    if method_id in {"ml_lasso", "ml_ridge", "ml_elastic_net"}:
        target = params.get("target")
        if not target:
            method_label = "Elastic Net 回归" if method_id == "ml_elastic_net" else ("岭回归" if method_id == "ml_ridge" else "Lasso 回归")
            return fail(method_label + "需要一个结局变量")
        candidate_features = params.get("feature_vars") or features
        usable_features = [c for c in candidate_features if c != target and _has_variation(df, c)]
        if len(usable_features) < 2:
            method_label = "Elastic Net 回归" if method_id == "ml_elastic_net" else ("岭回归" if method_id == "ml_ridge" else "Lasso 回归")
            return fail(method_label + "至少需要 2 个非ID特征变量")
        if not _is_numeric_col(df, target):
            return fail("正则化回归的结局变量必须是数值型；文字分类结局请改用分类模型")
        if not _has_variation(df, target):
            return fail("正则化回归的结局变量至少需要两个不同取值")
        if _complete_row_count(df, [target] + usable_features) < 10:
            return fail("正则化回归至少需要 10 个完整样本")
        params["feature_vars"] = usable_features
        params["regularization"] = "elastic_net" if method_id == "ml_elastic_net" else ("ridge" if method_id == "ml_ridge" else "lasso")
        return True, "", params
    if method_id in {"dim_reduction", "cluster"}:
        numeric_features = [c for c in features if _is_numeric_col(df, c) and _has_variation(df, c)]
        if len(numeric_features) < 2:
            return fail("降维/聚类至少需要 2 个连续型特征变量")
        if _complete_row_count(df, numeric_features) < 10:
            return fail("降维/聚类至少需要 10 个完整样本")
        params["feature_vars"] = numeric_features
        return True, "", params
    if method_id == "feature_engineering":
        usable = [c for c in r["selected"] if not _is_id_like_col(c) and _has_variation(df, c)]
        if not usable:
            return fail("特征工程至少需要 1 个非ID且具有变异的变量")
        params["feature_vars"] = usable
        return True, "", params
    if method_id == "ldsc":
        trait = params.get("group_var")
        h2_col = params.get("h2_col")
        h2_se_col = params.get("h2_se_col")
        if not all([trait, h2_col, h2_se_col]):
            return fail("LDSC 需要同时选择性状、h² 和 h² 标准误变量")
        if not _is_discrete_col(df, trait) or not _is_numeric_col(df, h2_col) or not _is_numeric_col(df, h2_se_col):
            return fail("LDSC 的性状需为分类变量，h² 与标准误需为数值变量")
        if (pd.to_numeric(df[h2_se_col], errors="coerce").dropna() <= 0).any():
            return fail("LDSC 的 h² 标准误必须全部大于 0")
        trait_values = [str(value) for value in df[trait].dropna().unique()]
        normalized_columns = {re.sub(r"[\s_\-]+", "", str(col)).lower(): col for col in df.columns}
        rg_cols = [
            normalized_columns[re.sub(r"[\s_\-]+", "", value).lower()]
            for value in trait_values
            if re.sub(r"[\s_\-]+", "", value).lower() in normalized_columns
            and _is_numeric_col(df, normalized_columns[re.sub(r"[\s_\-]+", "", value).lower()])
        ]
        if len(trait_values) < 2 or len(rg_cols) < 2:
            return fail("LDSC 共病分析至少需要 2 个性状，并提供与性状同名的遗传相关数值列")
        params["rg_cols"] = list(dict.fromkeys(rg_cols))
        return True, "", params

    required_by_method = {
        "gee": ["outcome_var", "time_var", "group_var", "subject_var"],
        "propensity_score": ["treatment_var", "outcome_var"],
        "sensitivity_analysis": ["treatment_var", "outcome_var", "baseline_var"],
        "counterfactual": ["treatment_var", "outcome_var"],
        "survival_advanced": ["time_var", "event_var", "group_var"],
        "markov_model": ["state_var", "time_var", "subject_var"],
        "bayesian": ["outcome_var", "group_var"],
        "latin_square": ["response_var", "period_var", "formulation_var", "subject_var"],
        "meta_analysis": ["study_var", "effect_var", "se_var"],
        "mediation": ["x_var", "m_var", "y_var"],
        "mixed_effects": ["outcome_var", "time_var", "group_var", "random_var"],
        "nhanes_analysis": ["weight_var", "strata_var", "outcome_var"],
    }
    missing = [k for k in required_by_method.get(method_id, []) if not params.get(k)]
    if missing:
        return fail("缺少必要变量：" + "、".join(missing))

    if method_id == "gee":
        cols = [params["outcome_var"], params["time_var"], params["group_var"], params["subject_var"]]
        if not _is_numeric_col(df, params["outcome_var"]) or not _is_numeric_col(df, params["time_var"]):
            return fail("GEE 的结局变量和时间变量必须是数值型")
        if not _has_variation(df, params["group_var"]):
            return fail("GEE 分组变量至少需要两个组")
        if int(df[params["subject_var"]].dropna().value_counts().max()) < 2:
            return fail("GEE 需要同一受试者至少有 2 次重复测量")
        if _complete_row_count(df, cols) < 20:
            return fail("GEE 至少需要 20 条完整重复测量记录")
    elif method_id in {"sensitivity_analysis", "counterfactual"}:
        if not _has_min_two_groups(df, params["treatment_var"], min_each=5):
            return fail("处理变量必须是二分类，且两组至少各有 5 个样本")
        if not _is_numeric_col(df, params["outcome_var"]):
            return fail("结局变量必须是数值型")
        if method_id == "sensitivity_analysis" and not _is_numeric_col(df, params["baseline_var"]):
            return fail("基线变量必须是数值型")
    elif method_id == "survival_advanced":
        if not _is_numeric_col(df, params["time_var"]):
            return fail("生存时间变量必须是数值型")
        if not _is_binary_col(df, params["event_var"]):
            return fail("生存事件变量必须是二分类变量")
        if not _has_variation(df, params["group_var"]):
            return fail("生存分析分组变量至少需要两个组")
        if (pd.to_numeric(df[params["time_var"]], errors="coerce").dropna() < 0).any():
            return fail("生存时间不能为负数")
    elif method_id == "markov_model":
        if not _is_discrete_col(df, params["state_var"]) or not _is_numeric_col(df, params["time_var"]):
            return fail("马尔可夫模型需要分类状态变量和数值时间变量")
        if int(df[params["subject_var"]].dropna().value_counts().max()) < 2:
            return fail("马尔可夫模型需要同一对象至少有 2 次状态记录")
    elif method_id == "bayesian":
        if not _is_numeric_col(df, params["outcome_var"]) or not _has_variation(df, params["group_var"]):
            return fail("贝叶斯组间分析需要数值结局和至少两个分组")
    elif method_id == "latin_square":
        if not _is_numeric_col(df, params["response_var"]):
            return fail("拉丁方分析的响应变量必须是数值型")
        if not _has_variation(df, params["period_var"]) or not _has_variation(df, params["formulation_var"]):
            return fail("拉丁方分析的时期和处理变量至少需要两个水平")
    elif method_id == "meta_analysis":
        if not _is_numeric_col(df, params["effect_var"]) or not _is_numeric_col(df, params["se_var"]):
            return fail("荟萃分析的效应量和标准误必须是数值型")
        if (pd.to_numeric(df[params["se_var"]], errors="coerce").dropna() <= 0).any():
            return fail("荟萃分析的标准误必须全部大于 0")
        if int(df[params["study_var"]].dropna().nunique()) < 3:
            return fail("荟萃分析至少需要 3 项研究")
    elif method_id == "mediation":
        variables = [params["x_var"], params["m_var"], params["y_var"]]
        if len(set(variables)) < 3 or not all(_is_numeric_col(df, c) for c in variables):
            return fail("中介分析需要 3 个不同的数值变量 X、M、Y")
        if _complete_row_count(df, variables) < 20:
            return fail("中介分析至少需要 20 个完整样本")
    elif method_id == "mixed_effects":
        if not _is_numeric_col(df, params["outcome_var"]) or not _is_numeric_col(df, params["time_var"]):
            return fail("混合效应模型的结局变量和时间变量必须是数值型")
        if not _has_variation(df, params["group_var"]):
            return fail("混合效应模型的分组变量至少需要两个组")
        if int(df[params["random_var"]].dropna().value_counts().max()) < 2:
            return fail("随机效应变量必须包含重复观测")
    elif method_id == "nhanes_analysis":
        if not _is_numeric_col(df, params["weight_var"]) or not _is_numeric_col(df, params["outcome_var"]):
            return fail("复杂抽样分析的权重和结局变量必须是数值型")
        if (pd.to_numeric(df[params["weight_var"]], errors="coerce").dropna() <= 0).any():
            return fail("抽样权重必须全部大于 0")
        if not _has_variation(df, params["strata_var"]):
            return fail("分层变量至少需要两个层")
    return True, "", params


def _recommend_roles_for_method(df: pd.DataFrame, method_id: str) -> dict:
    all_cols = list(df.columns)
    method_name = METHOD_CATALOG.get(method_id, {}).get("name", method_id)

    def unique(values):
        out = []
        for value in values:
            if value and value in df.columns and value not in out:
                out.append(value)
        return out

    def has_repeat(col: str) -> bool:
        if col not in df.columns:
            return False
        counts = df[col].dropna().value_counts()
        return not counts.empty and int(counts.max()) >= 2

    def positive_numeric(col: str) -> bool:
        if not _is_numeric_col(df, col):
            return False
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        return not values.empty and bool((values > 0).all())

    def ordered(pool: list[str], pattern: str) -> list[str]:
        regex = re.compile(pattern, re.I)
        first = [col for col in pool if regex.search(str(col))]
        rest = [col for col in pool if col not in first]
        return first + rest

    non_id = [c for c in all_cols if not _is_id_like_col(c) and _has_variation(df, c)]
    numeric = [c for c in non_id if _is_numeric_col(df, c)]
    continuous = [c for c in numeric if _is_continuous_col(df, c)] or numeric
    discrete = [c for c in non_id if _is_discrete_col(df, c)]
    binary = [c for c in discrete if _has_min_two_groups(df, c, min_each=5)]
    positive = [c for c in numeric if positive_numeric(c)]
    repeated = [c for c in all_cols if has_repeat(c)]
    id_like = [c for c in all_cols if _is_id_like_col(c)]

    feature_pool = unique(numeric + discrete)
    outcome_numeric = ordered(continuous, r"outcome|target|response|score|follow|sbp|glucose|hba1c|index|结局|评分")
    outcome_discrete = ordered(discrete, r"outcome|target|label|class|disease|status|risk|event|结局|标签|疾病|风险")
    treatment_vars = ordered(binary, r"treat|treated|arm|group|intervention|therapy|exposure|sex|gender|smoking|case|control|处理|分组")
    group_vars = ordered(discrete, r"group|arm|stage|site|center|strata|sex|gender|disease|class|state|分组|中心|状态")
    time_vars = ordered(numeric, r"time|week|month|cycle|period|visit|duration|follow|survival|os|时间|月份|周期")
    event_vars = ordered([c for c in discrete if _is_binary_col(df, c)], r"event|death|status|outcome|label|risk|事件|死亡|状态")
    subject_vars = ordered(unique(id_like + repeated), r"subject|patient|sample|person|id|受试|患者|编号")

    attempts: list[dict] = []
    last_reason = ""

    def add_attempt(research=None, covars=None, outcomes=None):
        attempts.append({
            "research_vars": unique(research or []),
            "covar_vars": unique(covars or []),
            "outcome_vars": unique(outcomes or []),
        })

    if method_id in {"ml_lr", "ml_knn", "ml_xgboost", "ml_rf", "ml_svm", "ml_dt", "ml_cnn", "model_comparison", "ml_stacking", "ml_mlp"}:
        for target in outcome_discrete[:8]:
            add_attempt([c for c in feature_pool if c != target], [], [target])
    elif method_id in {"ml_lasso", "ml_ridge", "ml_elastic_net"}:
        for target in outcome_numeric[:8]:
            add_attempt([c for c in feature_pool if c != target], [], [target])
    elif method_id == "gee":
        for outcome in outcome_numeric[:5]:
            for time_var in [c for c in time_vars if c != outcome][:5]:
                for group in group_vars[:5]:
                    for subject in subject_vars[:5]:
                        add_attempt([group], [time_var, subject], [outcome])
    elif method_id == "propensity_score":
        for treatment in treatment_vars[:8]:
            for outcome in unique(outcome_numeric + outcome_discrete)[:8]:
                add_attempt([treatment], [c for c in feature_pool if c not in {treatment, outcome}], [outcome])
    elif method_id == "sensitivity_analysis":
        for treatment in treatment_vars[:8]:
            for outcome in outcome_numeric[:8]:
                for baseline in [c for c in outcome_numeric if c not in {outcome, treatment}][:5]:
                    covars = [baseline] + [c for c in feature_pool if c not in {treatment, outcome, baseline}][:8]
                    add_attempt([treatment], covars, [outcome])
    elif method_id == "counterfactual":
        for treatment in treatment_vars[:8]:
            for outcome in outcome_numeric[:8]:
                add_attempt([treatment], [c for c in feature_pool if c not in {treatment, outcome}], [outcome])
    elif method_id == "target_trial_emulation":
        for treatment in treatment_vars[:8]:
            for outcome in event_vars[:8]:
                for time_var in [c for c in time_vars if c not in {treatment, outcome}][:6]:
                    add_attempt([treatment], [time_var] + [c for c in feature_pool if c not in {treatment, outcome, time_var}][:12], [outcome])
    elif method_id == "doubly_robust_aipw":
        for treatment in treatment_vars[:8]:
            for outcome in unique(outcome_numeric + outcome_discrete)[:8]:
                add_attempt([treatment], [c for c in feature_pool if c not in {treatment, outcome}][:16], [outcome])
    elif method_id == "marginal_structural_model":
        for treatment in treatment_vars[:8]:
            for outcome in unique(outcome_numeric + outcome_discrete)[:8]:
                for time_var in [c for c in time_vars if c not in {treatment, outcome}][:6]:
                    for subject in subject_vars[:6] or [None]:
                        covars = [c for c in [subject, time_var] if c] + [c for c in feature_pool if c not in {treatment, outcome, time_var, subject}][:16]
                        add_attempt([treatment], covars, [outcome])
    elif method_id in {"ml_deepsurv", "ml_deephit"}:
        candidate_events = event_vars if method_id == "ml_deepsurv" else ordered(discrete, r"event|status|death|cause|type|outcome|label")
        for time_var in time_vars[:8]:
            for event in [c for c in candidate_events if c != time_var][:8]:
                add_attempt([c for c in feature_pool if c not in {time_var, event}], [time_var], [event])
    elif method_id == "survival_advanced":
        for time_var in time_vars[:8]:
            for event in [c for c in event_vars if c != time_var][:8]:
                for group in group_vars[:6]:
                    add_attempt([group], [], [time_var, event])
    elif method_id == "markov_model":
        for state in ordered(discrete, r"state|status|stage|class|状态|分期")[:8]:
            for time_var in time_vars[:8]:
                for subject in subject_vars[:8]:
                    add_attempt([state], [subject, time_var], [])
    elif method_id == "bayesian":
        for outcome in outcome_numeric[:8]:
            for group in group_vars[:8]:
                add_attempt([group], [], [outcome])
    elif method_id == "latin_square":
        period_vars = ordered(unique(discrete + numeric), r"period|time|week|month|visit|时期|周期")[:8]
        formulation_vars = ordered(discrete, r"formulation|treat|arm|group|drug|处理|剂型")[:8]
        for response in outcome_numeric[:8]:
            for period in [c for c in period_vars if c != response][:5]:
                for formulation in [c for c in formulation_vars if c not in {response, period}][:5]:
                    for subject in subject_vars[:5]:
                        add_attempt([formulation], [period, subject], [response])
    elif method_id == "meta_analysis":
        study_vars = ordered(discrete, r"study|trial|paper|center|研究|试验")[:8]
        effect_vars = ordered(numeric, r"effect|estimate|or|rr|hr|beta|效应")[:8]
        se_vars = ordered(positive, r"se|stderr|standard|标准误")[:8]
        for study in study_vars:
            for effect in effect_vars:
                for se in [c for c in se_vars if c != effect]:
                    add_attempt([study], [se], [effect])
    elif method_id == "mediation":
        for x_var in numeric[:8]:
            for m_var in [c for c in numeric if c != x_var][:8]:
                for y_var in [c for c in numeric if c not in {x_var, m_var}][:8]:
                    add_attempt([x_var], [m_var], [y_var])
    elif method_id == "mixed_effects":
        for outcome in outcome_numeric[:8]:
            for time_var in [c for c in time_vars if c != outcome][:8]:
                for group in group_vars[:6]:
                    for random_var in subject_vars[:6]:
                        add_attempt([group], [time_var, random_var], [outcome])
    elif method_id == "nhanes_analysis":
        strata_vars = ordered(discrete, r"strata|stratum|site|center|race|sex|分层|中心")[:8]
        for outcome in outcome_numeric[:8]:
            for weight in [c for c in positive if c != outcome][:8]:
                for strata in [c for c in strata_vars if c != weight][:8]:
                    add_attempt([], [weight, strata], [outcome])
    elif method_id == "ldsc":
        trait_vars = ordered(discrete, r"trait|phenotype|disease|性状|疾病")[:8]
        h2_vars = ordered(numeric, r"^h2$|heritability|遗传力")[:8]
        h2_se_vars = ordered(positive, r"h2.*se|se.*h2|standard.*error|标准误")[:8]
        for trait in trait_vars:
            trait_names = [str(value) for value in df[trait].dropna().unique()]
            normalized_columns = {re.sub(r"[\s_\-]+", "", str(col)).lower(): col for col in df.columns}
            rg_cols = [
                normalized_columns[re.sub(r"[\s_\-]+", "", value).lower()]
                for value in trait_names
                if re.sub(r"[\s_\-]+", "", value).lower() in normalized_columns
            ]
            if len(rg_cols) < 2:
                continue
            for h2_col in h2_vars:
                for h2_se_col in [c for c in h2_se_vars if c != h2_col]:
                    add_attempt([trait], [h2_se_col], [h2_col])
    elif method_id in {"dim_reduction", "cluster"}:
        add_attempt(numeric, [], [])
    elif method_id == "feature_engineering":
        add_attempt(feature_pool, [], [])

    for roles in attempts:
        ok, reason, params = _validate_method_roles(df, method_id, roles)
        if ok:
            return {"available": True, "reason": "", "roles": roles, "params": params}
        if reason and not last_reason:
            last_reason = reason

    return {
        "available": False,
        "reason": last_reason or f"上传的数据文件做不了「{method_name}」：没有找到满足该统计方法要求的变量组合。",
        "roles": {"research_vars": [], "covar_vars": [], "outcome_vars": []},
        "params": {},
    }



# ═══════════════════════════════════════════════════════════════
# v11: robust validation and no-crash fallback result
# ═══════════════════════════════════════════════════════════════

def _has_min_two_groups(df: pd.DataFrame, col: str, min_each: int = 1) -> bool:
    if not col or col not in df.columns:
        return False
    vc = df[col].dropna().value_counts()
    return len(vc) == 2 and int(vc.min()) >= min_each

def _complete_row_count(df: pd.DataFrame, cols: list[str]) -> int:
    valid = [c for c in cols if c in df.columns]
    return int(len(df.dropna(subset=valid))) if valid else 0

def _has_variation(df: pd.DataFrame, col: str, min_unique: int = 2) -> bool:
    return col in df.columns and int(df[col].dropna().nunique()) >= min_unique

def _fallback_success_result(df: pd.DataFrame, method_id: str, params: dict, err: Exception | str) -> dict:
    """Return a non-crashing, honest fallback report for UI stability.

    This is only used after validation said a method is eligible but an
    implementation edge case still failed. It prevents small-user workflows
    from getting stuck while still stating what happened.
    """
    method_name = METHOD_CATALOG.get(method_id, {}).get("name", method_id)
    n, p = int(df.shape[0]), int(df.shape[1])
    msg = str(err)
    rows = [
        {"项目": "分析方法", "结果": method_name},
        {"项目": "数据规模", "结果": f"{n} 行 × {p} 列"},
        {"项目": "自动参数", "结果": json.dumps(params, ensure_ascii=False)},
        {"项目": "稳健处理", "结果": "原始算法遇到边界情况，系统已返回稳健兜底结果，避免前端报错中断。"},
        {"项目": "技术说明", "结果": msg[:300]},
    ]
    out = {
        "tables": [{"title": f"{method_name} 稳健执行摘要", "headers": ["项目", "结果"], "rows": rows}],
        "charts": [],
        "diagnostics": [],
        "discussion": (
            f"## {method_name} 稳健执行说明\n\n"
            "### 一、执行状态\n\n"
            "系统根据当前变量角色判断该方法具备基本运行条件，但底层模型在拟合或匹配过程中遇到边界情况。"
            "为避免小白用户界面卡死或直接报错，系统返回稳健兜底结果。\n\n"
            "### 二、可能原因\n\n"
            f"- 数据规模：{n} 行 × {p} 列。\n"
            f"- 自动参数：`{json.dumps(params, ensure_ascii=False)}`。\n"
            f"- 技术信息：{msg[:500]}。\n\n"
            "### 三、建议\n\n"
            "建议增加样本量、避免把几乎全缺失或单一取值的变量作为核心变量，并优先选择具有明确分组、结局和协变量的变量组合。"
        ),
    }
    try:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:8]
        if numeric_cols:
            desc = df[numeric_cols].describe().T.reset_index()
            desc = desc.rename(columns={"index": "变量"})
            out["tables"].append({
                "title": "数值变量描述统计",
                "headers": [str(c) for c in desc.columns],
                "rows": desc.round(4).fillna("").to_dict(orient="records"),
            })
    except Exception:
        pass
    return out


def _chart_identity(chart: dict) -> str:
    title = str((chart or {}).get("title") or "").strip().lower()
    return re.sub(r"[\s·—\-_:：,，.。()（）\[\]【】]+", "", title)


def _param_col(params: dict, df: pd.DataFrame, keys: list[str]) -> str | None:
    for key in keys:
        value = params.get(key)
        if isinstance(value, str) and value in df.columns:
            return value
    return None


def _numeric_cols_for_charts(df: pd.DataFrame, limit: int = 8) -> list[str]:
    return [
        col for col in df.select_dtypes(include=[np.number]).columns.tolist()
        if df[col].notna().sum() >= 3
    ][:limit]


def _chart_json(fig) -> str:
    return fig.to_json()


def _supplemental_analysis_charts(df: pd.DataFrame, params: dict, method_id: str, method_name: str) -> list[dict]:
    try:
        import plotly.graph_objects as go
    except Exception:
        return []

    charts: list[dict] = []
    outcome = _param_col(df=df, params=params, keys=["outcome_var", "target", "response_var", "y_var", "effect_var"])
    group = _param_col(df=df, params=params, keys=["group_var", "treatment_var", "state_var", "strata_var"])
    time_col = _param_col(df=df, params=params, keys=["time_var", "period_var"])
    numeric_cols = _numeric_cols_for_charts(df, 10)

    if outcome and outcome in df.columns:
        series = df[outcome].dropna()
        if pd.api.types.is_numeric_dtype(series):
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=series,
                nbinsx=28,
                marker=dict(color="#2f6df6", line=dict(color="white", width=0.8)),
                hovertemplate=f"{outcome}<br>值=%{{x}}<br>频数=%{{y}}<extra></extra>",
            ))
            fig.update_layout(
                title=f"{method_name} · {outcome} 分布",
                xaxis_title=outcome,
                yaxis_title="频数",
                template="plotly_white",
                height=480,
            )
            charts.append({"title": f"{outcome} 分布", "plotly": _chart_json(fig)})
        else:
            counts = series.astype(str).value_counts().head(12)
            fig = go.Figure(go.Bar(
                x=list(counts.index),
                y=list(counts.values),
                marker=dict(color="#0ea5a4", line=dict(color="white", width=0.8)),
                text=list(counts.values),
                textposition="outside",
                cliponaxis=False,
            ))
            fig.update_layout(title=f"{method_name} · {outcome} 构成", xaxis_title=outcome, yaxis_title="样本数", template="plotly_white", height=460)
            charts.append({"title": f"{outcome} 构成", "plotly": _chart_json(fig)})

    if group and outcome and group in df.columns and outcome in df.columns and pd.api.types.is_numeric_dtype(df[outcome]):
        plot_df = df[[group, outcome]].dropna()
        if not plot_df.empty and plot_df[group].nunique() >= 2:
            fig = go.Figure()
            for level, part in plot_df.groupby(group):
                fig.add_trace(go.Box(
                    y=part[outcome],
                    name=str(level),
                    boxpoints="outliers",
                    marker=dict(color="#2f6df6"),
                    line=dict(color="#2f6df6"),
                ))
            fig.update_layout(title=f"{method_name} · 分组结局比较", xaxis_title=group, yaxis_title=outcome, template="plotly_white", height=500)
            charts.append({"title": "分组结局比较", "plotly": _chart_json(fig)})

    corr_cols = [col for col in numeric_cols if col != outcome][:8]
    if outcome and outcome in df.columns and pd.api.types.is_numeric_dtype(df[outcome]):
        corr_cols = list(dict.fromkeys([outcome] + corr_cols))[:8]
    if len(corr_cols) >= 3:
        corr = df[corr_cols].corr(numeric_only=True).round(3)
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale=[[0, "#dbeafe"], [0.5, "#ffffff"], [1, "#0f766e"]],
            zmin=-1,
            zmax=1,
            colorbar=dict(title="r"),
            hovertemplate="%{y} vs %{x}<br>r=%{z:.3f}<extra></extra>",
        ))
        fig.update_layout(title=f"{method_name} · 数值变量相关性", template="plotly_white", height=520)
        charts.append({"title": "数值变量相关性", "plotly": _chart_json(fig)})

    x_col = next((col for col in numeric_cols if col != outcome), None)
    if x_col and outcome and outcome in df.columns and pd.api.types.is_numeric_dtype(df[outcome]):
        plot_df = df[[x_col, outcome] + ([group] if group else [])].dropna()
        if len(plot_df) >= 5:
            fig = go.Figure()
            if group and group in plot_df.columns and plot_df[group].nunique() <= 8:
                palette = ["#2f6df6", "#0ea5a4", "#64748b", "#7c3aed", "#f59e0b", "#ef4444"]
                for i, (level, part) in enumerate(plot_df.groupby(group)):
                    fig.add_trace(go.Scatter(
                        x=part[x_col],
                        y=part[outcome],
                        mode="markers",
                        name=str(level),
                        marker=dict(size=8, color=palette[i % len(palette)], opacity=0.72, line=dict(color="white", width=0.6)),
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=plot_df[x_col],
                    y=plot_df[outcome],
                    mode="markers",
                    marker=dict(size=8, color="#2f6df6", opacity=0.72, line=dict(color="white", width=0.6)),
                    name="样本",
                ))
            fig.update_layout(title=f"{method_name} · {x_col} 与 {outcome}", xaxis_title=x_col, yaxis_title=outcome, template="plotly_white", height=500)
            charts.append({"title": f"{x_col} 与 {outcome}", "plotly": _chart_json(fig)})

    if time_col and outcome and time_col in df.columns and outcome in df.columns and pd.api.types.is_numeric_dtype(df[outcome]):
        plot_df = df[[time_col, outcome] + ([group] if group else [])].dropna()
        if len(plot_df) >= 5:
            fig = go.Figure()
            if group and group in plot_df.columns and plot_df[group].nunique() <= 6:
                for level, part in plot_df.groupby(group):
                    trend = part.groupby(time_col)[outcome].mean().reset_index()
                    fig.add_trace(go.Scatter(x=trend[time_col], y=trend[outcome], mode="lines+markers", name=str(level)))
            else:
                trend = plot_df.groupby(time_col)[outcome].mean().reset_index()
                fig.add_trace(go.Scatter(x=trend[time_col], y=trend[outcome], mode="lines+markers", name="均值趋势", line=dict(color="#2f6df6", width=3)))
            fig.update_layout(title=f"{method_name} · 时间趋势概览", xaxis_title=time_col, yaxis_title=outcome, template="plotly_white", height=500)
            charts.append({"title": "时间趋势概览", "plotly": _chart_json(fig)})

    return charts[:5]


def _build_method_summary_table(
    results: dict,
    method_id: str,
    params: dict,
    df: pd.DataFrame,
) -> list[dict]:
    """Build a detailed method-summary table with parameters and key statistical values."""
    method_name = METHOD_CATALOG.get(method_id, {}).get("name", method_id)
    tables = results.get("tables") or []
    n_rows, n_cols = len(df), len(df.columns)

    rows: list[dict] = []

    # ── 1. Basic info ──
    rows.append({"项目": "分析方法", "值": method_name})
    rows.append({"项目": "数据规模", "值": f"{n_rows} 行 × {n_cols} 列"})

    # ── 2. Parameter settings ──
    method_cfg = METHOD_CATALOG.get(method_id, {})
    param_defs = {p["key"]: p for p in method_cfg.get("params", [])}

    has_params = False
    for key, value in params.items():
        if value in (None, "", []) or (isinstance(value, list) and len(value) == 0):
            continue
        if not has_params:
            rows.append({"项目": "── 参数设定 ──", "值": ""})
            has_params = True
        param_def = param_defs.get(key, {})
        label = param_def.get("label", key)
        if isinstance(value, bool):
            display_value = "是" if value else "否"
        elif isinstance(value, list):
            display_value = ", ".join(str(v) for v in value)
        else:
            display_value = _smart_format(value)
        rows.append({"项目": label, "值": display_value})

    # ── 3. Key statistical indicators extracted from result tables ──
    seen_keys: set[str] = set()
    stat_rows: list[dict] = []
    for table in tables:
        for row in (table.get("rows") or table.get("data") or [])[:4]:
            if not isinstance(row, dict):
                continue
            key = str(row.get("指标") or row.get("项目") or "").strip()
            val = row.get("值") or row.get("结果")
            if key and val not in (None, "") and key not in seen_keys:
                seen_keys.add(key)
                stat_rows.append({"项目": key, "值": _smart_format(val)})
            if len(stat_rows) >= 20:
                break
        if len(stat_rows) >= 20:
            break

    if stat_rows:
        rows.append({"项目": "── 关键统计指标 ──", "值": ""})
        rows.extend(stat_rows)

    return [{"title": "分析方法与关键统计值", "headers": ["项目", "值"], "rows": rows}]


def _smart_format(v) -> str:
    """Format a numeric or string value for display."""
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        if abs(v) < 0.001 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4f}"
    return str(v)


def _finalize_analysis_results(
    results: dict,
    df: pd.DataFrame,
    method_id: str,
    params: dict,
) -> dict:
    """Keep method-specific, publication-relevant figures only."""
    output = dict(results or {})
    method_name = METHOD_CATALOG.get(method_id, {}).get("name", method_id)
    candidates: list[dict] = []
    candidates.extend(output.get("charts") or [])
    candidates.extend(output.get("diagnostics") or [])
    candidates.extend(_supplemental_analysis_charts(df, params, method_id, method_name))

    # Do not promote generic completeness/distribution diagnostics into the
    # publication figure set. A method with no meaningful figure should show
    # an honest empty state instead of decorative fallback charts.

    merged: list[dict] = []
    seen: set[str] = set()
    generic_qc_titles = {
        _chart_identity({"title": "数据完整性诊断"}),
        _chart_identity({"title": "关键变量分布诊断"}),
    }
    for chart in candidates:
        if not isinstance(chart, dict) or not chart.get("plotly"):
            continue
        key = _chart_identity(chart)
        if not key:
            key = f"chart-{len(merged)}"
        if key in generic_qc_titles:
            continue
        if key in seen:
            continue
        seen.add(key)
        merged.append(chart)

    output["charts"] = merged[:12]
    base_tables = [table for table in (output.get("tables") or []) if isinstance(table, dict)]
    output["tables"] = _build_method_summary_table(output, method_id, params, df) + base_tables
    output["diagnostics"] = []
    output["discussion"] = generate_discussion(method_id, output, df)
    return output


@app.post("/api/validate-methods")
def validate_methods(req: dict) -> dict:
    df = _get_df_simple({
        "upload_id": req.get("upload_id"),
        "dataset_name": req.get("dataset_name") or "gee_example",
        "use_demo": req.get("use_demo", True),
        "sheet_name": req.get("sheet_name"),
    })
    roles = req.get("roles") or {}
    availability = {}
    for method_id, cfg in METHOD_CATALOG.items():
        ok, reason, params = _validate_method_roles(df, method_id, roles)
        availability[method_id] = {
            "available": ok,
            "reason": reason,
            "params": params,
            "name": cfg.get("name", method_id),
            "category": cfg.get("category", ""),
        }
    return {"availability": availability}


@app.post("/api/recommend-roles")
def recommend_roles(req: dict) -> dict:
    method_id = req.get("method_id")
    if method_id not in METHOD_CATALOG:
        raise HTTPException(status_code=404, detail=f"Method '{method_id}' not found")
    df = _get_df_simple({
        "upload_id": req.get("upload_id"),
        "dataset_name": req.get("dataset_name") or METHOD_CATALOG[method_id].get("example_dataset", "gee_example"),
        "use_demo": req.get("use_demo", True),
        "sheet_name": req.get("sheet_name"),
    })
    result = _recommend_roles_for_method(df, method_id)
    result.update({
        "method_id": method_id,
        "method_name": METHOD_CATALOG[method_id].get("name", method_id),
    })
    return result


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
    if getattr(req, 'role_vars', None):
        available, reason, derived = _validate_method_roles(df, method_id, req.role_vars)
        if not available:
            raise HTTPException(status_code=400, detail=reason)
        derived.update({k: v for k, v in params.items() if v not in (None, '', [])})
        params = derived
    # Set default params from catalog
    for p in METHOD_CATALOG[method_id].get("params", []):
        if p["key"] not in params:
            params[p["key"]] = p.get("default", "")

    # Strict target/method compatibility guard before running analysis.
    _guard_method_target_compatibility(df, method_id, params)

    # Route to appropriate service
    try:
        if method_id in STATS_ROUTER:
            results = STATS_ROUTER[method_id](df, params)
        elif method_id in ML_ROUTER:
            results = ML_ROUTER[method_id](df, params)
        else:
            raise HTTPException(status_code=400, detail=f"Method '{method_id}' not implemented")
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"{METHOD_CATALOG[method_id]['name']} 无法运行：{e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{METHOD_CATALOG[method_id]['name']} 运行失败：{e}")

    results = _finalize_analysis_results(results, df, method_id, params)

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
    if getattr(req, 'role_vars', None):
        available, reason, derived = _validate_method_roles(df, method_id, req.role_vars)
        if not available:
            raise HTTPException(status_code=400, detail=reason)
        derived.update({k: v for k, v in params.items() if v not in (None, '', [])})
        params = derived
    for p in METHOD_CATALOG[method_id].get("params", []):
        if p["key"] not in params:
            params[p["key"]] = p.get("default", "")

    if method_id in STATS_ROUTER:
        results = STATS_ROUTER[method_id](df, params)
    elif method_id in ML_ROUTER:
        results = ML_ROUTER[method_id](df, params)
    else:
        raise HTTPException(status_code=400, detail=f"Method '{method_id}' not implemented")

    results = _finalize_analysis_results(results, df, method_id, params)

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

def _safe_export_filename(value: str | None, suffix: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or f"mlhigh_chart_{uuid.uuid4().hex[:8]}")).strip("._")
    if not base:
        base = f"mlhigh_chart_{uuid.uuid4().hex[:8]}"
    if base.lower().endswith(f".{suffix}"):
        base = base[: -(len(suffix) + 1)]
    return base[:80]


def _plotly_layout_title(layout: dict) -> str:
    title = (layout or {}).get("title")
    if isinstance(title, dict):
        return str(title.get("text") or "")
    return str(title or "")


def _write_basic_pdf_from_plotly(figure: dict, dest, width: int, height: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layout = figure.get("layout") or {}
    traces = figure.get("data") or []
    fig_w = max(5.0, min(16.0, width / 120))
    fig_h = max(3.8, min(12.0, height / 120))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    title = _plotly_layout_title(layout)
    if title:
        ax.set_title(re.sub(r"<[^>]+>", "", title), fontsize=12, loc="left")

    plotted = False
    for index, trace in enumerate(traces[:6]):
        trace_type = str(trace.get("type") or "scatter").lower()
        name = str(trace.get("name") or f"series {index + 1}")
        x_values = trace.get("x") if trace.get("x") is not None else trace.get("labels")
        y_values = trace.get("y") if trace.get("y") is not None else trace.get("values")
        if trace_type == "pie" and isinstance(y_values, list):
            ax.clear()
            labels = [str(v) for v in (x_values or range(len(y_values)))]
            ax.pie(pd.to_numeric(pd.Series(y_values), errors="coerce").fillna(0), labels=labels[: len(y_values)], autopct="%1.1f%%")
            plotted = True
            break
        if trace_type in {"heatmap", "contour"} and isinstance(trace.get("z"), list):
            ax.imshow(trace.get("z"), aspect="auto", cmap="Blues")
            plotted = True
            continue
        if trace_type == "histogram":
            values = x_values if isinstance(x_values, list) else y_values
            numeric = pd.to_numeric(pd.Series(values or []), errors="coerce").dropna()
            if not numeric.empty:
                ax.hist(numeric, bins=min(30, max(8, int(len(numeric) ** 0.5))), alpha=0.6, label=name)
                plotted = True
            continue
        if trace_type == "box" and isinstance(y_values, list):
            numeric = pd.to_numeric(pd.Series(y_values), errors="coerce").dropna()
            if not numeric.empty:
                ax.boxplot(numeric, positions=[index + 1], widths=0.5)
                ax.set_xticks(list(range(1, index + 2)))
                ax.set_xticklabels([str((traces[i].get("name") or i + 1)) for i in range(index + 1)], rotation=25, ha="right")
                plotted = True
            continue
        if isinstance(y_values, list):
            x_plot = x_values if isinstance(x_values, list) else list(range(1, len(y_values) + 1))
            y_numeric = pd.to_numeric(pd.Series(y_values), errors="coerce")
            if trace_type == "bar":
                ax.bar([str(v) for v in x_plot], y_numeric.fillna(0), alpha=0.72, label=name)
            else:
                ax.plot(x_plot, y_numeric, marker="o", linewidth=1.8, label=name)
            plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "Chart preview could not be reconstructed for PDF fallback.", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.grid(True, color="#e5edf8", linewidth=0.8)
        if any(str(t.get("name") or "").strip() for t in traces[:6]):
            ax.legend(frameon=False, fontsize=8)
        fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(dest, format="pdf", bbox_inches="tight")
    plt.close(fig)


@app.post("/api/export/table-csv")
def export_table_csv(req: ExportRequest) -> FileResponse:
    data = req.table_data or []
    dest = export_to_csv(data, "table_export")
    return FileResponse(dest, media_type="text/csv", filename="table_export.csv")


@app.post("/api/export/table-txt")
def export_table_txt(req: ExportRequest) -> FileResponse:
    data = req.table_data or []
    dest = export_to_txt(data, "table_export")
    return FileResponse(dest, media_type="text/plain", filename="table_export.txt")


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


@app.post("/api/export/chart-image")
def export_chart_image(req: dict = Body(default_factory=dict)) -> FileResponse:
    fmt = str(req.get("format") or "png").lower().strip()
    if fmt not in {"png", "svg", "pdf"}:
        raise HTTPException(status_code=400, detail="Unsupported chart export format")

    figure = req.get("figure") or req.get("chart_data") or {}
    if isinstance(figure, str):
        try:
            figure = json.loads(figure)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid Plotly figure JSON") from exc
    if not isinstance(figure, dict) or not figure.get("data"):
        raise HTTPException(status_code=400, detail="No chart data received")

    width = int(max(320, min(3200, float(req.get("width") or 900))))
    height = int(max(260, min(2400, float(req.get("height") or 640))))
    filename = _safe_export_filename(req.get("filename"), fmt)
    dest = OUTPUTS_DIR / f"{filename}.{fmt}"

    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        fig = go.Figure(figure)
        fig.update_layout(width=width, height=height, autosize=False)
        pio.write_image(fig, str(dest), format=fmt, width=width, height=height, scale=2 if fmt == "png" else 1)
    except Exception as exc:
        if fmt == "pdf":
            try:
                _write_basic_pdf_from_plotly(figure, dest, width, height)
            except Exception as fallback_exc:
                raise HTTPException(status_code=500, detail=f"PDF export failed: {fallback_exc}") from exc
        else:
            raise HTTPException(status_code=500, detail=f"Chart export failed: {exc}") from exc

    media_types = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}
    return FileResponse(dest, media_type=media_types[fmt], filename=f"{filename}.{fmt}")


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
    # Use a vertical bar chart here because it is more stable in the front-end
    # preview system and avoids the clipped horizontal-axis appearance.
    completeness = (df[key_cols].notna().mean() * 100).sort_values(ascending=False)
    fig_complete = go.Figure()
    fig_complete.add_trace(go.Bar(
        x=list(completeness.index),
        y=np.round(completeness.values, 1),
        marker=dict(
            color=np.round(completeness.values, 1),
            colorscale=[[0, "#c0616e"], [0.7, "#f6c453"], [1, "#0d7377"]],
            cmin=0,
            cmax=100,
            line=dict(color="white", width=0.7),
        ),
        text=[f"{v:.1f}%" for v in completeness.values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}<br>非缺失率=%{y:.1f}%<extra></extra>",
    ))
    fig_complete.update_layout(
        title=f"{method_name} · 数据完整性诊断",
        xaxis_title="变量",
        yaxis_title="非缺失率 (%)",
        yaxis=dict(range=[0, 105]),
        height=480,
        margin=dict(l=72, r=32, t=72, b=88),
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
    """Build a stable, non-duplicate diagnostic chart.

    We intentionally keep diagnostic charts simple and distinct from the
    main visualization tab: use summary bars instead of model-result plots.
    This makes the diagnostics tab more robust and avoids repeating plots.
    """
    numeric_cols = [
        c for c in key_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > 0
    ]
    if not numeric_cols:
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if df[c].notna().sum() > 0
        ][:5]

    if numeric_cols:
        rows = []
        for col in numeric_cols[:5]:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            if s.empty:
                continue
            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1
            outlier_rate = float((((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).mean()) * 100) if iqr > 0 else 0.0
            rows.append((col, round(outlier_rate, 2)))
        if rows:
            labels = [r[0] for r in rows]
            values = [r[1] for r in rows]
            fig = go.Figure(go.Bar(
                x=labels,
                y=values,
                marker=dict(color=['#2563eb','#ec4899','#10b981','#eab308','#7c3aed'][:len(labels)], line=dict(color='white', width=0.8)),
                text=[f'{v:.1f}%' for v in values],
                textposition='outside',
                cliponaxis=False,
                hovertemplate='%{x}<br>离群点占比=%{y:.1f}%<extra></extra>',
            ))
            fig.update_layout(
                title=f'{method_name} · 关键变量离群点诊断',
                xaxis_title='变量',
                yaxis_title='离群点占比 (%)',
                yaxis=dict(range=[0, max(10, max(values) * 1.25)]),
                height=440,
                margin=dict(l=72, r=32, t=72, b=88),
                paper_bgcolor='white',
                plot_bgcolor='white',
            )
            return fig

    cat_col = _first_valid_column(df, params, ['group_var', 'treatment_var', 'target'])
    if not cat_col:
        cat_col = next((c for c in key_cols if c in df.columns), None)
    if cat_col:
        counts = df[cat_col].astype(str).replace({'nan': '缺失'}).value_counts().head(10)
        fig = go.Figure(go.Bar(
            x=list(counts.index),
            y=list(counts.values),
            marker=dict(color='#0d7377', line=dict(color='white', width=0.7)),
            text=list(counts.values),
            textposition='outside',
            cliponaxis=False,
            hovertemplate='%{x}<br>样本数=%{y}<extra></extra>',
        ))
        fig.update_layout(
            title=f'{method_name} · {cat_col} 样本结构诊断',
            xaxis_title=cat_col,
            yaxis_title='样本数',
            height=440,
            margin=dict(l=72, r=32, t=72, b=88),
            paper_bgcolor='white',
            plot_bgcolor='white',
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
