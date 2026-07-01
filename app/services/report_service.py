from __future__ import annotations

import numpy as np
import pandas as pd


METHOD_NAMES = {
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
    "ldsc": "LDSC 共病分析",
    "ml_lr": "逻辑回归",
    "ml_lasso": "Lasso 回归",
    "ml_ridge": "岭回归",
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


def _stringify(v) -> str:
    if isinstance(v, float):
        if abs(v) < 0.001 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4f}"
    return str(v)


def _find_value_in_tables(tables: list, *keys: str):
    """Search result tables for a key (指标=xxx, 值=yyy rows) and return value."""
    for table in (tables or []):
        rows = table.get("rows") or table.get("data") or []
        for row in rows:
            if isinstance(row, dict):
                for key in keys:
                    if row.get("指标") == key or row.get("项目") == key:
                        return row.get("值") or row.get("结果")
    return None


def _find_best_single_value(tables: list, *candidates):
    """Try multiple candidate keys, return first found."""
    for key in candidates:
        v = _find_value_in_tables(tables, key)
        if v is not None:
            return v
    return None


def _all_table_keys(tables: list) -> set:
    keys = set()
    for table in (tables or []):
        for row in (table.get("rows") or table.get("data") or []):
            if isinstance(row, dict):
                keys.add(row.get("指标") or row.get("项目") or "")
    return keys


def _describe_rocauc(auc_val) -> str:
    """Interpret AUC value with clinical grading."""
    try:
        auc = float(auc_val)
    except (TypeError, ValueError):
        return ""
    if auc >= 0.97:
        grade = "极强区分能力，在临床上可作为高可靠诊断工具的核心依据"
    elif auc >= 0.90:
        grade = "优秀区分能力，区分患者与非患者的效果很好，适合临床推广"
    elif auc >= 0.80:
        grade = "良好区分能力，对临床辅助决策有明确价值，可结合其他指标综合判断"
    elif auc >= 0.70:
        grade = "可接受区分能力，有一定临床应用价值，但假阳性/假阴性仍需关注"
    elif auc >= 0.60:
        grade = "区分能力一般，单独使用时临床决策价值有限，建议结合更多特征"
    else:
        grade = "区分能力较弱，该模型在当前数据下难以有效区分两类人群，需考虑增加更有区分度的变量或更换建模策略"
    return f"AUC = {auc:.3f}，{grade}。"


def _describe_metric(name_cn: str, val) -> str:
    """Describe a metric value in plain Chinese."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    return f"{name_cn} = {v:.3f}"


def _extract_chart_titles(results: dict) -> list:
    return [str(c.get("title") or "") for c in (results.get("charts") or []) if c.get("title")]


def _find_charts_containing(charts: list, *keywords: str) -> list:
    found = []
    for c in charts:
        title = str(c.get("title") or "").lower()
        if any(kw.lower() in title for kw in keywords):
            found.append(c.get("title"))
    return found


def _find_auc_from_results(results: dict) -> float | None:
    """Try to extract AUC value from tables or charts."""
    tables = results.get("tables") or []
    auc = _find_value_in_tables(tables, "AUC-ROC", "AUC", "ROC-AUC")
    if auc is not None:
        try:
            return float(auc)
        except (TypeError, ValueError):
            pass
    # Try parsing from chart titles
    for chart in (results.get("charts") or []):
        title = str(chart.get("title") or "")
        if "auc" in title.lower():
            import re
            m = re.search(r'AUC\s*[=:]?\s*([0-9.]+)', title, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
    return None


def _result_snapshot_detailed(results: dict, limit: int = 8) -> list[str]:
    """Extract key metrics from tables grouped by category."""
    lines: list = []
    tables = results.get("tables") or []
    for table in tables:
        title = str(table.get("title") or "结果表")
        row_items = []
        for row in (table.get("rows") or [])[:4]:
            if not isinstance(row, dict):
                continue
            key = row.get("指标") or row.get("项目") or ""
            val = row.get("值") or row.get("结果") or ""
            if key and val not in (None, ""):
                row_items.append(f"{key}: {_stringify(val)}")
        if row_items:
            lines.append(f"**{title}**：{'；'.join(row_items)}")
        if len(lines) >= limit:
            break
    return lines


def generate_discussion(method_id: str, results: dict, df: pd.DataFrame) -> str:
    """Generate detailed, clinically meaningful interpretation for each method."""
    name = METHOD_NAMES.get(method_id, method_id)
    n_rows, n_cols = len(df), len(df.columns)
    tables = results.get("tables") or []
    charts = results.get("charts") or []
    n_tables, n_charts = len(tables), len(charts)
    chart_titles = [str(c.get("title") or "") for c in charts if c.get("title")]
    existing = str(results.get("discussion") or "").strip()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    missing_pct = round(df.isna().sum().sum() / max(1, n_rows * n_cols) * 100, 1)

    # ── Method-specific detailed interpretation ──
    method_body = _method_specific_interpretation(method_id, results, df, name)
    auc_val = _find_auc_from_results(results)

    # ── Build the full discussion ──
    parts = [f"## {name} · 结果解读\n"]

    # 1) Overview
    parts.append("### 一、分析概览\n")
    parts.append(f"本次分析共纳入 **{n_rows} 条记录、{n_cols} 个变量**，其中数值变量 {len(num_cols)} 个、分类/文本变量 {len(cat_cols)} 个，整体缺失率为 **{missing_pct}%**。\n")

    # 2) Key metrics snapshot
    snapshots = _result_snapshot_detailed(results, limit=6)
    if snapshots:
        parts.append("### 二、关键指标速览\n")
        for line in snapshots:
            parts.append(f"- {line}\n")
        parts.append("")

    # 3) AUC specific interpretation
    if auc_val is not None:
        parts.append("### 三、模型区分度 (ROC 曲线)\n")
        parts.append(f"{_describe_rocauc(auc_val)}\n")
        parts.append("ROC 曲线下的面积 (AUC) 反映模型将随机抽取的一个阳性样本排在随机抽取的阴性样本之前的概率。"
                     "理想 AUC 接近 1，AUC = 0.5 表示模型无区分能力等同于随机猜测。"
                     "AUC 越高，模型在灵敏度和特异度之间取得更好平衡的空间越大。\n")
        if auc_val >= 0.90:
            parts.append("该模型的区分能力达到优秀水平，如果校准也良好，可作为临床辅助决策的核心工具。\n")
        elif auc_val >= 0.80:
            parts.append("该模型区分能力良好，适合用于筛查或辅助诊断场景，建议结合校准曲线评估概率的准确性。\n")
        elif auc_val >= 0.70:
            parts.append("该模型有一定区分能力，可用于风险分层初筛，但不建议直接用于确诊决策。建议尝试增加有区分力的特征或调整模型结构。\n")
        else:
            parts.append("当前模型的区分能力有限。建议从以下几个方面改进：(1) 增加与结局强相关的生物学指标或临床特征；(2) 检查变量编码方式是否正确；(3) 尝试非线性模型或集成方法。\n")

        calib_title = _find_charts_containing(charts, "校准")
        if calib_title:
            parts.append(f"本次分析已生成校准曲线（「{calib_title[0]}」），校准曲线反映预测概率与真实发生率的一致程度。"
                        "若校准曲线靠近对角线，说明概率预测可靠；若系统性偏离，需进行概率校准（如 Platt scaling 或 isotonic regression）。\n")

    # 4) Method-specific detailed interpretation (the meat)
    parts.append("### 四、专项分析解读\n")
    parts.append(method_body)

    # 5) DCA interpretation if relevant
    dca_charts = _find_charts_containing(charts, "DCA", "决策曲线", "临床净获益")
    if dca_charts:
        parts.append("\n### 五、临床决策曲线 (DCA) 解读\n")
        parts.append("DCA 曲线显示在不同阈值概率下使用模型进行临床决策所带来的净获益。"
                     "净获益 = (真阳性获益 − 假阳性代价) / 总人数。横坐标是阈值概率，纵坐标是净获益。\n")
        parts.append("- **如果 DCA 曲线在较大阈值范围内高于\"全部治疗\"和\"全部不治疗\"两条参考线**，说明模型在临床上确实有价值。\n")
        parts.append("- **如果 DCA 曲线仅在很窄的阈值范围内高于参考线**，说明模型的临床实用性有限，适合特定高风险人群而非普适筛查。\n")
        parts.append("- **如果 DCA 曲线始终低于某条参考线**，说明在该阈值范围内使用模型反而不如统一处理策略。\n")
        parts.append("临床决策时，DCA 提供的净获益信息比 AUC 和校准更直接地回答\"用这个模型到底能让多少患者受益、多少患者受损\"。\n")

    auc_target = 5 if auc_val is not None else 7 if dca_charts else 6

    # 6) Clinical significance
    parts.append(f"### {['六','六','七'][min(auc_target - 5, 2)] if auc_target > 5 else '六'}. 临床解读要点\n")
    parts.append("- **效应方向**：首先确认主效应是正向还是负向，是否与临床预期一致。\n")
    parts.append("- **效应量大小**：不应仅仅关注 P 值。大样本中小差异也可显著但临床意义不大，建议结合效应量和置信区间综合判断。\n")
    parts.append("- **置信区间**：窄区间意味着估计精确，宽区间则不确定性较高。若区间跨过无效应线应谨慎解读。\n")
    parts.append("- **统计学指标说明**：P 值反映观察到的效应由随机误差产生的概率，统计学显著（P < 0.05）不等于临床重要。"
                    "效应量（如 OR、HR、Cohen d、η² 等）反映关联或差异的实际大小。置信区间提供效应估计的不确定性范围。\n")
    parts.append("- **可比性与混杂**：观察性分析中需评估组间基线可比性。若通过 PSM/IPW 等方法平衡后结果一致，结论更稳健。\n")
    parts.append("- **外推性**：结论适用于与当前样本相似的人群，不同中心、年代、纳排标准的群体可能得到不同效应估计。\n")

    # 7) Recommendations
    parts.append(f"### {chr(ord('六') + (auc_target - 4) if chr(ord('六') + (auc_target - 4)).isalpha() else '八')}. 建议与注意事项\n")
    parts.append("- 请核对变量编码方向、参考组设定和异常值处理是否正确。\n")
    parts.append("- 建议在正式论文或报告中补充亚组分析、交互效应检验或敏感性分析，以增强结论的稳健性。\n")
    parts.append("- 如果是预测模型，建议在独立外部数据中验证其区分度和校准度。\n")
    parts.append("- 在观察性研究中避免使用\"导致\"\"引起\"等因果措辞，优先使用\"相关\"\"关联\"\"提示\"。\n")

    discussion = "".join(parts)
    return discussion


def _method_specific_interpretation(method_id: str, results: dict, df: pd.DataFrame, name: str) -> str:
    """Generate method-specific clinical interpretation based on actual results."""
    tables = results.get("tables") or []
    charts = results.get("charts") or []
    n_rows, n_cols = len(df), len(df.columns)

    if method_id.startswith("ml_"):
        return _ml_interpretation(method_id, results, df, name)

    if method_id == "model_comparison":
        return _model_comparison_interpretation(results, df)

    if method_id == "cluster":
        return _cluster_interpretation(results, df)

    if method_id == "dim_reduction":
        return _dim_reduction_interpretation(results, df)

    if method_id in ("gee", "mixed_effects"):
        return _longitudinal_interpretation(method_id, results, df, name)

    if method_id in ("propensity_score", "counterfactual", "sensitivity_analysis"):
        return _causal_interpretation(method_id, results, df, name)

    if method_id in ("survival_advanced", "markov_model"):
        return _survival_interpretation(method_id, results, df, name)

    if method_id in ("meta_analysis",):
        return _meta_interpretation(results, df, name)

    if method_id in ("bayesian",):
        return _bayesian_interpretation(results, df, name)

    if method_id == "latin_square":
        return _latin_square_interpretation(results, df, name)

    if method_id == "mediation":
        return _mediation_interpretation(results, df, name)

    if method_id == "nhanes_analysis":
        return _nhanes_interpretation(results, df, name)

    if method_id == "ldsc":
        return _ldsc_interpretation(results, df, name)

    # Generic fallback
    return _generic_interpretation(results, df, name)


def _ml_interpretation(method_id: str, results: dict, df: pd.DataFrame, name: str) -> str:
    """Detailed ML classifier interpretation."""
    tables = results.get("tables") or []
    charts = results.get("charts") or []

    accuracy = _find_value_in_tables(tables, "Accuracy")
    auc_val = _find_value_in_tables(tables, "AUC-ROC", "AUC")
    f1 = _find_value_in_tables(tables, "F1(macro)")
    precision = _find_value_in_tables(tables, "Precision(macro)")
    recall = _find_value_in_tables(tables, "Recall(macro)")

    parts = []
    parts.append(f"**{name}** 是一种{'经典线性' if method_id == 'ml_lr' else '非参数' if method_id == 'ml_knn' else '集成学习' if method_id in ('ml_rf', 'ml_xgboost') else '机器学习'}分类方法。\n")

    if method_id == "ml_lr":
        parts.append("逻辑回归通过 logistic 函数将线性预测值映射为 0–1 之间的概率，广泛用于疾病风险预测和二分类结局建模。"
                    "其优势在于系数可解释：每个特征的系数（log-odds）可以直接转换为优势比 (OR = exp(β))，便于临床解读。\n")
    elif method_id == "ml_rf":
        parts.append("随机森林通过 Bootstrap 采样构建多棵决策树，最后投票或平均得到预测结果。"
                    "它善于捕获非线性关系和特征交互，且对缺失值和异常值有一定容忍度。"
                    "特征重要性排名可以帮助识别对预测贡献最大的变量，但重要性不等于独立因果效应。\n")
    elif method_id == "ml_xgboost":
        parts.append("XGBoost 是一种梯度提升树方法，通过逐轮训练弱学习器来逐步降低预测误差。"
                    "它在临床预测建模中表现通常很出色，尤其在特征较多、关系复杂的数据上。"
                    "需要注意控制学习率和树深度以防止过拟合——模型在训练集上几乎完美但测试集表现明显下降时即为过拟合信号。\n")
    elif method_id == "ml_svm":
        parts.append("SVM 寻找最大化类别间隔的超平面，通过核函数映射到高维空间处理非线性可分问题。"
                    "在小样本、高维度场景下表现较好，但对特征尺度敏感，建议在建模前进行标准化。\n")
    elif method_id == "ml_knn":
        parts.append("KNN 基于距离最近邻的样本投票进行分类。方法直观、无需训练过程，但对 K 值的选择和特征标准化非常敏感。"
                    "K 值过小容易受噪声影响，K 值过大则决策边界过于平滑失去区分力。"
                    "建议在 K 值敏感性曲线上观察不同 K 值下的表现变化。\n")
    elif method_id == "ml_dt":
        parts.append("决策树通过递归分裂生成可解释的分类规则，树结构可以直接转化为临床决策路径。"
                    "但单棵树容易过拟合，通常作为随机森林和 XGBoost 的基础组件而非独立建模工具。\n")
    elif method_id == "ml_cnn":
        parts.append("1D-CNN 通过卷积核在时间维度上滑动提取局部时序模式，适用于纵向重复测量或时间序列分类任务。"
                    "卷积层可以自动学习不同时间窗口内的特征组合，减少手工特征工程。\n")

    # Performance assessment
    parts.append("\n**模型性能评估：**\n")
    metrics_text = []
    if auc_val is not None:
        metrics_text.append(_describe_rocauc(auc_val))
    if accuracy is not None:
        metrics_text.append(_describe_metric("准确率 (Accuracy)", accuracy))
    if f1 is not None:
        metrics_text.append(f"F1 分数 = {_stringify(f1)}（精确率和召回率的调和均值，在类别不均衡时比准确率更有参考价值）")
    if precision is not None:
        metrics_text.append(_describe_metric("精确率 (Precision)", precision) + "（预测为阳性中真正阳性的比例）")
    if recall is not None:
        metrics_text.append(_describe_metric("召回率 (Recall)", recall) + "（真实阳性中被正确检出的比例）")
    if metrics_text:
        parts.append("；".join(metrics_text) + "。\n")
    else:
        parts.append("具体指标详见结果表格。\n")

    # ROC specific
    if auc_val is not None:
        parts.append("\n**ROC 曲线临床意义：**\n")
        parts.append(f"AUC 值为 {_stringify(auc_val)}。ROC 曲线的横坐标为 1−特异度（假阳性率），纵坐标为灵敏度（真阳性率）。"
                    "曲线越靠近左上角，说明模型在保持高灵敏度的同时假阳性率越低。"
                    "临床应用中，可以通过改变分类阈值在灵敏度与特异度之间权衡——"
                    "例如筛查场景需要高灵敏度（宁可多误报也要少漏诊），而确诊场景需要高特异度（避免假阳性导致不必要治疗）。\n")

    # Calibration
    if _find_charts_containing(charts, "校准"):
        parts.append("\n**校准曲线解读：**\n")
        parts.append("校准曲线将预测概率按大小分桶，比较每桶的平均预测概率与实际阳性比例。"
                    "如果校准曲线在对角线上方说明预测概率偏低（低估风险），在对角线下方说明预测概率偏高（高估风险）。"
                    "校准不良的模型即使 AUC 高也可能给出误导性的概率估计，在临床决策中需谨慎使用。\n")

    # PR curve
    if _find_charts_containing(charts, "PR"):
        parts.append("\n**PR 曲线（精确率-召回率曲线）：**当正负样本严重不均衡时（如罕见病预测），PR 曲线比 ROC 曲线更有参考价值。"
                    "PR 曲线下的面积越大，模型在平衡精确率和召回率方面的表现越好。\n")

    # Feature importance
    imp_charts = _find_charts_containing(charts, "特征重要性", "系数图", "特征系数")
    if imp_charts:
        parts.append("\n**特征贡献解读：**\n")
        parts.append("特征重要性图展示了各变量对模型预测的贡献程度。排名靠前的特征是区分结局最核心的变量。"
                    "但需注意：(1) 重要性反映的是预测贡献而非独立因果效应；(2) 高度相关的特征之间重要性可能被分散；"
                    "(3) 如果一个特征在临床上非常重要但重要性排名靠后，可能说明它在当前数据中变异不够或与其他特征共线。\n")

    # DCA
    if _find_charts_containing(charts, "DCA", "决策曲线"):
        parts.append("\n**DCA 决策曲线临床指导：**如果 DCA 显示模型在某阈值范围内净获益高于\"全治\"和\"全不治\"策略，"
                    "则说明借助该模型进行临床决策确实可以减少不必要的治疗或及时干预高危患者。"
                    "阈值范围越宽、净获益越大，模型的临床实用价值越高。\n")

    return "".join(parts)


def _model_comparison_interpretation(results: dict, df: pd.DataFrame) -> str:
    tables = results.get("tables") or []
    parts = ["多模型比较通过统一的数据划分和评估指标，在同一标准下排名各模型的预测表现。\n\n"]
    parts.append("**如何阅读比较结果：**\n")
    parts.append("- 优先看主要指标（通常为 AUC 或 F1）的排名，排名靠前的模型在当前数据上综合表现更好。\n")
    parts.append("- 同时关注稳定性：如果各模型之间指标差异很小（如 AUC 差 0.01–0.02），则模型选择不那么关键，可优先选更简单、更可解释的模型。\n")
    parts.append("- 逻辑回归虽然预测性能未必最优，但系数可解释性好，在论文中更易通过同行评议。XGBoost 和随机森林通常预测更好但解释性较弱。\n")
    parts.append("- 如果数据量小、特征数多，集成方法可能过拟合；此时简单模型反而泛化更好。\n")

    # Extract ranking
    ranking_rows = []
    for table in tables:
        for row in (table.get("rows") or []):
            if isinstance(row, dict) and ("AUC" in str(row.get("指标") or "") or "排名" in str(row.get("指标") or "")):
                ranking_rows.append(row)
    if ranking_rows:
        parts.append("\n**比较结果：**具体排名和分项指标详见结果表格。建议结合各模型的混淆矩阵和校准曲线，"
                    "综合判断哪个模型更适合您的临床场景——筛查、诊断或预后评估。\n")
    return "".join(parts)


def _cluster_interpretation(results: dict, df: pd.DataFrame) -> str:
    parts = ["聚类分析是一种无监督学习方法，根据样本在特征空间中的相似性将人群分为若干亚组。\n\n"]
    parts.append("**聚类结果的临床解读注意事项：**\n")
    parts.append("- 聚类分出的亚组不等同于已证实的临床亚型，需要从以下几个维度验证：各簇的结局是否存在实质性差异、簇的生物学合理性和可重复性。\n")
    parts.append("- 查看各簇在关键变量上的分布差异（通过箱线图、雷达图或平行坐标图），判断哪些特征主导了分组结果。\n")
    parts.append("- 如果某个簇仅包含少量样本（如 < 总人群 5%），应警惕异常值、数据错误或过度分割的可能。\n")
    parts.append("- 建议在独立数据集中验证聚类结构的稳定性——改变随机种子或子样本后，簇的个数和特征是否保持一致？\n")
    parts.append("- 在论文中应把聚类结果定位为\"探索性发现\"或\"潜在人群分层\"而非最终诊断分类。\n")
    return "".join(parts)


def _dim_reduction_interpretation(results: dict, df: pd.DataFrame) -> str:
    parts = ["降维分析通过保留主要方差方向或局部邻域结构，将高维数据映射到 2–3 维空间以便可视化和探索。\n\n"]
    parts.append("**降维图解读：**\n")
    parts.append("- PCA 降维图：如果不同颜色（标签）的点在二维图上自然分开，说明主要方差方向与分组标签一致；如果混杂在一起，说明分组信息不在主方差方向上。\n")
    parts.append("- t-SNE 降维图：t-SNE 擅长保留局部邻域结构，适合发现细粒度聚类。但 t-SNE 的距离和簇大小不可直接解释，只能作为定性观察工具。\n")
    parts.append("- 如果降维图表现出明显的簇结构，可以后续用聚类算法正式分组。如果点分布均匀无明显分离，说明数据中可能不存在明显的亚组结构。\n")
    parts.append("- 注意：降维失去的方差比例反映信息损失程度。若前两个主成分仅解释 < 30% 总方差，二维图可能不足以捕捉数据结构全貌。\n")
    return "".join(parts)


def _longitudinal_interpretation(method_id: str, results: dict, df: pd.DataFrame, name: str) -> str:
    parts = [f"**{name}** 专门处理重复测量数据中的组内相关性，给出总体平均效应的稳健估计。\n\n"]
    parts.append("**解读重点：**\n")
    parts.append("- **时间效应**：观察结局指标在随访期间的整体变化趋势。如果时间效应显著且方向为正，说明干预后结局随时间改善。\n")
    parts.append("- **组别效应**：比较不同处理组之间的总体差异。组间差异的估计值和置信区间是临床解读的核心——差异的大小决定临床意义。\n")
    parts.append("- **交互效应**：时间 × 组别交互反映处理效果是否随时间变化。交互显著意味着组间差异在不同随访时间点不一致，"
                    "需进一步分析各时间点的简单效应，而不应将组间平均效应作为唯一结论。\n")
    parts.append("- **工作相关结构**：GEE 引入工作相关矩阵来处理同一受试者重复测量之间的相关性。"
                    "即使相关结构指定不完全正确，GEE 的均值参数估计仍然是一致的（稳健标准误），这是 GEE 在临床纵向数据中广泛使用的重要优势。\n")
    parts.append("- 当交互效应不显著时，可以主要报告组间总体平均差。当交互效应显著时，应重点描述不同时间点的效应变化模式。\n")
    return "".join(parts)


def _causal_interpretation(method_id: str, results: dict, df: pd.DataFrame, name: str) -> str:
    parts = [f"**{name}** 旨在减少观察性研究中的选择偏倚和混杂，使处理组和对照组在基线协变量上更具可比性。\n\n"]
    if "propensity" in method_id or method_id == "propensity_score":
        parts.append("**PSM 的核心机制与解读：**\n")
        parts.append("- 倾向评分 (Propensity Score) 是给定协变量后接受处理的概率估计。匹配后两组的协变量分布应趋于一致。\n")
        parts.append("- **平衡性检查**：匹配前后的标准化均值差异 (SMD) 是判断匹配质量的关键指标。SMD < 0.1 通常认为平衡良好；"
                        "如果匹配后仍有变量 SMD > 0.2，说明该变量未充分平衡，结论应加以限定。\n")
        parts.append("- **ATT（处理组平均处理效应）**：衡量实际接受治疗者相对于若未接受治疗的反事实结局的差异。ATT 是最常报告的量。\n")
        parts.append("- **敏感性分析**：即使匹配后平衡良好，仍可能存在未测量混杂。如果结果对轻微的未测量混杂敏感（如 Rosenbaum 界限 Γ 接近 1），"
                        "结论的因果解释应更加保守。如果 Γ > 2 时结论才改变，则结果较为稳健。\n")
    elif "sensitivity" in method_id:
        parts.append("**敏感性分析的核心逻辑：**\n")
        parts.append("- 在基准模型估计完成后，系统性地改变关键假设或参数（如扰动幅度 δ、未测量混杂强度），观察主效应是否保持方向和统计显著性。\n")
        parts.append("- 如果主效应在较大扰动范围内方向一致且未跨过无效应线，则结论稳健；如果轻微扰动即导致方向反转或不显著，则结论脆弱。\n")
        parts.append("- 敏感性分析不可证明因果，但可以量化结论对假设的依赖程度——这是观察性研究质量控制的重要工具。\n")
    elif "counterfactual" in method_id:
        parts.append("**反事实框架的临床解读：**\n")
        parts.append("- 反事实推理试图回答\"假如同一患者接受了另一治疗方案，结局会如何\"。这是因果推断的黄金标准思维。\n")
        parts.append("- 分析给出 ATE、ATT 或 ATU 等不同因果参数，区别在于估计的目标人群不同。ATT 适用于\"当前接受治疗的人群\"，ATE 适用于\"全人群\"。\n")
        parts.append("- 因果识别的核心假设是无未测量混杂 (unconfoundedness) 和正数性 (positivity)。如果存在强烈的未测量混杂因素（如医生的治疗偏好），"
                        "因果结论仍然需要谨慎对待。\n")
    parts.append("\n**临床表述建议：**\n")
    parts.append("- 匹配或调整后应写成\"在控制已知协变量后，处理组与对照组的差异估计为……\"而非直接\"治疗导致/引起……\"。\n")
    parts.append("- 报告中应同时呈现匹配前后的平衡性统计和效应估计，让读者自行判断匹配质量。\n")
    return "".join(parts)


def _survival_interpretation(method_id: str, results: dict, df: pd.DataFrame, name: str) -> str:
    parts = [f"**{name}** 用于分析从起点到事件发生的时间数据，同时处理删失（观察期内未发生事件或失访）。\n\n"]
    parts.append("**生存分析的临床解读要素：**\n")
    parts.append("- **中位生存时间**：50% 患者仍存活（或无事件）的时间点。中位生存期越长，预后越好。\n")
    parts.append("- **1 年/3 年/5 年生存率**：在特定时间点的累积生存概率。这些绝对值比风险比更直观——"
                    "风险比 0.7 听起来不错，但如果 5 年生存率仅从 60% 提升到 68%，绝对获益为 8 个百分点。\n")
    parts.append("- **风险比 (HR)**：HR < 1 表示治疗组风险低于对照组（获益），HR > 1 表示风险增高。"
                    "HR = 0.75 意味着治疗组在任何时间点发生事件的瞬时风险比对照组低 25%。\n")
    parts.append("- **比例风险假设**：Cox 模型假设 HR 不随时间变化。可通过 Schoenfeld 残差图或时间交互项检验。"
                    "如果比例风险不成立（例如治疗效果在早期强、后期减弱），应改用时间分层 Cox 或参数生存模型。\n")
    parts.append("- **删失机制**：如果删失与结局相关（如有信息删失），模型估计可能偏倚。应评估随访完成率和删失原因的分布。\n")
    if method_id == "markov_model":
        parts.append("\n**马尔可夫模型补充解读：**\n")
        parts.append("- 马尔可夫模型用转移矩阵描述患者在不同疾病状态之间移动的概率。\n")
        parts.append("- 转移概率反映在当前周期内从状态 i 转为状态 j 的可能性。对角线概率越大，状态越稳定。\n")
        parts.append("- 长期稳态分布可揭示患者群体在无限时间后最可能处于的各状态比例，对卫生经济学评估和资源配置有重要意义。\n")
    return "".join(parts)


def _meta_interpretation(results: dict, df: pd.DataFrame, name: str) -> str:
    parts = ["荟萃分析通过加权合并多项独立研究的结果，提高效应估计的精度和统计效力。\n\n"]
    parts.append("**森林图解读：**\n")
    parts.append("- 每个方块代表一项研究的效应估计，方块大小反映该研究在合并中的权重（通常与样本量和精确度成正比）。\n")
    parts.append("- 横线代表置信区间，穿过无效应线的研究在该次分析中不显著。\n")
    parts.append("- 菱形（汇总估计）和垂直线的关系：菱形完全不跨过垂直线 = 汇总效应显著。菱形越窄，合并估计越精确。\n")
    parts.append("- **异质性 I²**：I² < 25% 为低异质性、25–50% 为中等、> 75% 为高异质性。高异质性时建议用随机效应模型，"
                    "并探索异质性来源（如研究设计、人群特征、测量方式等）。\n")
    parts.append("- **发表偏倚**：漏斗图不对称可能提示小样本阴性结果未被发表。但漏斗图不对称也可能因真实异质性或效应量与研究精度的关联造成。\n")
    return "".join(parts)


def _bayesian_interpretation(results: dict, df: pd.DataFrame, name: str) -> str:
    parts = ["贝叶斯分析将先验知识与当前数据结合，生成后验分布。相较于频率学派，贝叶斯直接给出\"参数大于某值的概率\"，更贴近临床直觉。\n\n"]
    parts.append("**贝叶斯结果解读：**\n")
    parts.append("- **后验均值/中位数**：参数的最可能取值。由于加入了先验，贝叶斯估计通常比纯频率估计更稳定，尤其是在小样本中。\n")
    parts.append("- **95% 可信区间（CrI）**：有 95% 概率参数真值落在此区间内。这与频率学派的置信区间（如果重复实验无数次，95% 的区间会包含真值）含义不同。\n")
    parts.append("- **后验概率**：可直接表述为\"治疗有效的后验概率为 97%\"，而非\"我们拒绝治疗无效的假设\"。这对临床沟通和理解更友好。\n")
    parts.append("- **先验选择**：无信息先验让数据说话，有信息先验需要从既往文献或专家意见中合理构建。"
                    "如果先验和后验差异很大，说明数据与先验不一致，应仔细审查数据质量和先验合理性。\n")
    parts.append("- **收敛诊断**：MCMC 链是否收敛决定了后验估计是否可信。R-hat 接近 1、有效样本量足够大是收敛的基本信号。\n")
    return "".join(parts)


def _latin_square_interpretation(results: dict, df: pd.DataFrame, name: str) -> str:
    parts = ["拉丁方设计通过交叉平衡行（受试者/区组）、列（时期/序列）和处理因素，同时控制两种系统性变异来源。\n\n"]
    parts.append("**方差分解表解读：**\n")
    parts.append("- **处理效应**：在排除了行和列的变异后，不同处理水平之间的差异是否显著。这是研究关心的核心效应。\n")
    parts.append("- **行效应（受试者/区组）**：反映个体或区组间的系统性差异。显著的行效应提示受试者基线存在差异，需要纳入模型控制。\n")
    parts.append("- **列效应（时期/序列）**：反映时间或测量顺序的系统性变化。显著的列效应提示存在学习效应、疲劳效应或随时间漂移。\n")
    parts.append("- 如果处理效应显著而交互效应不显著，可集中报告各处理水平的校正均值及其差值。\n")
    return "".join(parts)


def _mediation_interpretation(results: dict, df: pd.DataFrame, name: str) -> str:
    parts = ["中介分析将总效应分解为直接效应（X → Y，不经过中介）和间接效应（X → M → Y，通过中介变量传递）。\n\n"]
    parts.append("**中介分析的关键指标：**\n")
    parts.append("- **间接效应 a × b**：这是中介效应的核心。如果 Bootstrap 置信区间不包含 0，说明中介效应在统计上显著。\n")
    parts.append("- **中介比例**：间接效应 / 总效应 × 100%，反映中介路径解释了总效应的多大比例。比例越高，该中介变量的机制解释力越强。\n")
    parts.append("- **注意**：中介分析建立在因果假设上（X→M、M→Y 以及无未测量混杂），统计显著不等于已证实因果通路。"
                    "建议将中介结果表述为\"与假设机制一致\"或\"可能通过……路径发挥作用\"，而非\"中介变量已被证明是因果通路\"。\n")
    parts.append("- 如果直接效应仍显著（不完全中介），说明还有其他未测量的中介路径或直接通路。\n")
    return "".join(parts)


def _nhanes_interpretation(results: dict, df: pd.DataFrame, name: str) -> str:
    parts = ["复杂抽样分析用于处理来自分层、整群或多阶段抽样的调查数据（如 NHANES、CHARLS 等全国代表性调查数据）。\n\n"]
    parts.append("**与普通回归的关键区别：**\n")
    parts.append("- **抽样权重**：每个样本代表总体中一定数量的人群。不纳入权重会导致估计偏倚—某些亚组被过度代表而另一些被低估。\n")
    parts.append("- **分层 (Strata)**：原始抽样设计的分层变量。正确的分层指定可以减少标准误，提高估计精度。\n")
    parts.append("- **PSU（初级抽样单位）**：同一 PSU 内的样本可能存在相关性。如果不纳入 PSU 聚类，标准误将被低估（过度显著）。\n")
    parts.append("- 报告时应强调\"全国代表性估计\"而非\"样本内比较\"，并说明加权方法、分层和聚类结构。\n")
    return "".join(parts)


def _ldsc_interpretation(results: dict, df: pd.DataFrame, name: str) -> str:
    parts = ["LDSC（LD Score Regression）利用 GWAS 汇总统计量的 LD Score 回归估计遗传力和遗传相关。\n\n"]
    parts.append("**LDSC 核心指标解读：**\n")
    parts.append("- **遗传力 h²**：表型方差中由常见 SNP 解释的比例。h² ≈ 0 说明常见变异对该性状几乎没有贡献；h² 高说明遗传贡献大。\n")
    parts.append("- **遗传相关 rg**：两个性状共享的遗传基础。rg ≈ 1 意味着两个性状几乎完全受相同基因变异影响；rg ≈ 0 说明几乎无共享。\n")
    parts.append("- **rg 为负值**：一个性状的遗传风险倾向与另一个性状的保护倾向相关（如某基因变异同时导致疾病 A 风险升高但疾病 B 风险降低）。\n")
    parts.append("- **标准误（SE）**：h² 和 rg 的 SE 反映估计精度。若 SE 较大（接近或大于估计值），结论应非常保守。\n")
    parts.append("- 注意：LDSC 估计的是 tagged SNP 贡献的遗传力，不是全基因组遗传力。罕见变异、结构变异等不在常见 SNP array 范围内的遗传贡献不会被捕获。\n")
    return "".join(parts)


def _generic_interpretation(results: dict, df: pd.DataFrame, name: str) -> str:
    tables = results.get("tables") or []
    parts = [f"**{name}** 已完成分析。\n\n"]
    parts.append("**通用解读框架：**\n")
    parts.append("- **效应方向与大小**：首先确认主效应是正向还是负向，效应量是否达到临床最小重要差异。\n")
    parts.append("- **统计显著性与临床重要性**：不要混淆 P < 0.05 与有临床价值。大样本下微小差异也可显著，应结合效应量综合判断。\n")
    parts.append("- **置信区间**：区间的宽度反映估计精密度。若区间跨过无效应线，即使点估计偏离也应谨慎解读。\n")
    parts.append("- **模型假设**：每个统计方法都有前提假设（如正态性、方差齐性、独立性等）。若假设不满足，应考虑替代方法或对结论加以限定。\n")
    return "".join(parts)


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
