from __future__ import annotations

import numpy as np
import pandas as pd


def generate_discussion(method_id: str, results: dict, df: pd.DataFrame) -> str:
    """Generate deep discussion based on analysis results."""
    if results.get("discussion"):
        return results["discussion"]

    n_rows = len(df)
    n_cols = len(df.columns)
    n_tables = len(results.get("tables", []))
    n_charts = len(results.get("charts", []))
    n_diag = len(results.get("diagnostics", []))

    method_names = {
        "gee": "广义估计方程 (GEE)",
        "propensity_score": "倾向性评分匹配 (PSM)",
        "sensitivity_analysis": "稳健性与敏感性分析",
        "counterfactual": "反事实推断与因果推断",
        "survival_advanced": "复杂生存与疾病进程建模",
        "markov_model": "马尔可夫模型",
        "bayesian": "贝叶斯统计分析",
        "latin_square": "拉丁方裂项方差分析",
        "meta_analysis": "荟萃分析",
        "mediation": "中介效应分析",
        "mixed_effects": "混合效应模型",
        "nhanes_analysis": "复杂抽样分析",
        "ml_lr": "逻辑回归",
        "ml_lasso": "Lasso/岭回归",
        "ml_knn": "K近邻 (KNN)",
        "ml_xgboost": "XGBoost",
        "ml_rf": "随机森林",
        "ml_svm": "支持向量机 (SVM)",
        "ml_dt": "决策树",
        "ml_cnn": "1D-CNN 时序分析",
        "feature_engineering": "特征工程",
        "model_comparison": "多模型比较",
        "dim_reduction": "降维分析 (PCA/t-SNE)",
        "cluster": "聚类分析",
    }

    name = method_names.get(method_id, method_id)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    missing_pct = round(df.isna().sum().sum() / (n_rows * n_cols) * 100, 1) if n_rows * n_cols > 0 else 0

    return (
        f"## {name} 分析结果与讨论\n\n"
        f"### 一、方法学概述\n"
        f"本分析采用 {name} 方法，基于 {n_rows} 条记录和 {n_cols} 个变量的数据集完成全流程建模分析。"
        f"数据集包含 {len(num_cols)} 个连续型变量和 {len(cat_cols)} 个分类型变量，"
        f"整体缺失率为 {missing_pct}%。\n\n"
        f"### 二、主要发现\n"
        f"系统自动完成了 {name} 的全流程分析，生成 {n_tables} 个结果表格、"
        f"{n_charts} 张可视化图表和 {n_diag} 项诊断评估。"
        f"详细的统计结果和模型参数请参阅各标签页中的表格和图表。\n\n"
        f"### 三、临床意义\n"
        f"数据驱动的分析结果需要结合临床背景进行审慎解读。"
        f"统计学显著性不等同于临床显著性，建议结合效应量大小、置信区间宽度和临床最小重要差异（MCID）"
        f"综合评估结果的实际意义。\n\n"
        f"### 四、局限性\n"
        f"- 结果的外部推广性取决于样本的代表性\n"
        f"- 观察性数据无法直接推断因果关系\n"
        f"- 建议在独立队列中进行外部验证\n\n"
        f"### 五、结论\n"
        f"基于 {n_rows} 例样本的 {name} 分析已完成。"
        f"详细结果请参阅各标签页，建议结合领域知识对统计发现进行审慎解释。"
    )


def generate_method_report(method_id: str, results: dict, df: pd.DataFrame) -> str:
    """Generate a full method report in markdown format."""
    title_map = {
        "gee": "广义估计方程 (GEE) 分析报告",
        "propensity_score": "倾向性评分匹配 (PSM) 分析报告",
        "sensitivity_analysis": "稳健性与敏感性分析报告",
        "counterfactual": "反事实推断与因果推断分析报告",
        "survival_advanced": "复杂生存与疾病进程建模分析报告",
        "markov_model": "马尔可夫模型分析报告",
        "bayesian": "贝叶斯统计分析报告",
        "latin_square": "拉丁方裂项方差分析报告",
        "meta_analysis": "荟萃分析报告",
        "mediation": "中介效应分析报告",
        "mixed_effects": "混合效应模型分析报告",
        "nhanes_analysis": "复杂抽样分析报告",
        "ml_lr": "逻辑回归模型分析报告",
        "ml_lasso": "Lasso/岭回归模型分析报告",
        "ml_knn": "K近邻 (KNN) 分类分析报告",
        "ml_xgboost": "XGBoost 模型分析报告",
        "ml_rf": "随机森林模型分析报告",
        "ml_svm": "支持向量机 (SVM) 分析报告",
        "ml_dt": "决策树模型分析报告",
        "ml_cnn": "1D-CNN 时序分析报告",
        "feature_engineering": "特征工程分析报告",
        "model_comparison": "多模型比较分析报告",
        "dim_reduction": "降维分析报告",
        "cluster": "聚类分析报告",
    }

    discussion = generate_discussion(method_id, results, df)
    n_tables = len(results.get("tables", []))
    n_charts = len(results.get("charts", []))

    report = f"# {title_map.get(method_id, method_id + ' 分析报告')}\n\n"
    report += f"**数据概况：** {len(df)} 条记录，{len(df.columns)} 个变量\n\n"
    report += f"**分析输出：** {n_tables} 个结果表格，{n_charts} 张可视化图表\n\n"
    report += "---\n\n"
    report += discussion
    report += "\n\n---\n\n*此报告由 MLhigh 临床高级统计与机器学习平台自动生成*"
    return report
