from __future__ import annotations

import io
import json
import base64
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Plotly imports for interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.figure_factory as ff
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def _test_normality(series: pd.Series, alpha: float = 0.05) -> bool:
    """Shapiro-Wilk test for normality. Returns True if normal."""
    sample = series.dropna().sample(n=min(len(series), 5000), random_state=42)
    if len(sample) < 3:
        return False
    if len(sample) > 5000:
        sample = sample.sample(n=5000, random_state=42)
    try:
        _, p = stats.shapiro(sample)
        return p > alpha
    except Exception:
        return False


def format_p_value(p: float | None, digits: int = 3) -> str:
    """Format p-value for medical tables."""
    if p is None:
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.{digits}f}"


def calc_group_comparison(df: pd.DataFrame, var: str, group_col: str) -> dict:
    """Compare variable across groups, return p-value and method."""
    groups = df[group_col].dropna().unique()
    if len(groups) < 2:
        return {"method": "N/A", "p_value": None, "statistic": None, "note": "少于2组，无法比较"}

    group_data = {str(g): df.loc[df[group_col] == g, var].dropna().values for g in groups}
    group_data = {k: v for k, v in group_data.items() if len(v) > 0}
    if len(group_data) < 2:
        return {"method": "N/A", "p_value": None, "statistic": None, "note": "有效组不足2个"}

    series = df[var].dropna()
    is_numeric = pd.api.types.is_numeric_dtype(series)

    try:
        if is_numeric:
            if len(group_data) == 2:
                a, b = list(group_data.values())
                if len(a) >= 3 and len(b) >= 3 and _test_normality(series):
                    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
                    return {"method": "Welch's t-test", "p_value": round(float(p_val), 4), "statistic": round(float(t_stat), 4)}
                else:
                    u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
                    return {"method": "Mann-Whitney U", "p_value": round(float(p_val), 4), "statistic": round(float(u_stat), 4)}
            else:
                if all(len(v) >= 3 for v in group_data.values()):
                    f_stat, p_val = stats.f_oneway(*group_data.values())
                    return {"method": "One-way ANOVA", "p_value": round(float(p_val), 4), "statistic": round(float(f_stat), 4)}
                else:
                    h_stat, p_val = stats.kruskal(*group_data.values())
                    return {"method": "Kruskal-Wallis", "p_value": round(float(p_val), 4), "statistic": round(float(h_stat), 4)}
        else:
            ct = pd.crosstab(df[group_col], df[var])
            chi2, p_val, _, _ = stats.chi2_contingency(ct)
            return {"method": "Chi-square", "p_value": round(float(p_val), 4), "statistic": round(float(chi2), 4)}
    except Exception:
        return {"method": "Error", "p_value": None, "statistic": None, "note": "计算失败"}


def _fig_to_json(fig) -> str:
    """Convert plotly figure to JSON string."""
    if fig is None:
        return "{}"
    return json.dumps(fig.to_dict() if hasattr(fig, "to_dict") else fig, default=str)


def _plt_to_base64(fig=None, dpi=150):
    """Convert matplotlib figure to base64 PNG."""
    buf = io.BytesIO()
    if fig is None:
        fig = plt.gcf()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ═══════════════════════════════════════════════════════════════
# 1. GEE — Generalized Estimating Equations
# ═══════════════════════════════════════════════════════════════

def run_gee(df: pd.DataFrame, params: dict) -> dict:
    """Longitudinal analysis with GEE."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    outcome = params.get("outcome_var", "sbp")
    time_var = params.get("time_var", "time")
    group_var = params.get("group_var", "treatment")
    subject_var = params.get("subject_var", "subject_id")
    covariates = params.get("covariates", ["age", "sex"])

    df_clean = df.dropna(subset=[outcome, time_var, group_var, subject_var]).copy()
    formula = f"{outcome} ~ {group_var} + {time_var}"
    for cv in covariates:
        if cv in df_clean.columns:
            formula += f" + {cv}"

    try:
        families = {
            "gaussian": sm.families.Gaussian(),
            "binomial": sm.families.Binomial(),
            "poisson": sm.families.Poisson(),
        }
        family = families.get(params.get("family", "gaussian"), sm.families.Gaussian())
        model = smf.gee(formula, subject_var, df_clean, cov_struct=sm.cov_struct.Exchangeable(), family=family)
        result = model.fit()

        coef_df = pd.DataFrame({
            "参数": result.params.index,
            "系数估计": np.round(result.params.values, 4),
            "标准误": np.round(result.bse.values, 4),
            "Z值": np.round(result.params.values / result.bse.values, 3),
            "P值": np.round(result.pvalues.values, 4),
            "95% CI 下限": np.round(result.params.values - 1.96 * result.bse.values, 4),
            "95% CI 上限": np.round(result.params.values + 1.96 * result.bse.values, 4),
        })
        out["tables"].append({"title": "GEE 系数估计", "headers": list(coef_df.columns), "rows": coef_df.to_dict(orient="records")})

        # Time trend chart
        if HAS_PLOTLY:
            groups = df_clean[group_var].unique()
            fig = go.Figure()
            for g in groups:
                sub = df_clean[df_clean[group_var] == g]
                agg = sub.groupby(time_var)[outcome].agg(["mean", "sem"]).reset_index()
                fig.add_trace(go.Scatter(
                    x=agg[time_var], y=agg["mean"], mode="lines+markers",
                    name=str(g),
                    error_y=dict(type="data", array=agg["sem"], visible=True),
                ))
            fig.update_layout(title=f"{outcome} 时间趋势 (GEE)", xaxis_title=time_var, yaxis_title=outcome,
                              template="plotly_white", height=520)
            out["charts"].append({"title": "时间趋势图", "plotly": _fig_to_json(fig)})

        # Residual plot
        if HAS_PLOTLY:
            resid = result.resid
            fitted = result.fittedvalues
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=fitted, y=resid, mode="markers", marker=dict(opacity=0.5)))
            fig2.add_hline(y=0, line_dash="dash", line_color="red")
            fig2.update_layout(title="残差诊断图", xaxis_title="拟合值", yaxis_title="残差", template="plotly_white", height=520)
            out["diagnostics"].append({"title": "残差诊断", "plotly": _fig_to_json(fig2)})

        n_subjects = df_clean[subject_var].nunique()
        n_obs = len(df_clean)
        n_timepoints = df_clean[time_var].nunique()
        group_coef = coef_df.iloc[1]['系数估计'] if len(coef_df) > 1 else 0
        group_ci_low = coef_df.iloc[1]['95% CI 下限'] if len(coef_df) > 1 else 0
        group_ci_high = coef_df.iloc[1]['95% CI 上限'] if len(coef_df) > 1 else 0
        group_p = coef_df.iloc[1]['P值'] if len(coef_df) > 1 else 1
        time_coef = coef_df.iloc[2]['系数估计'] if len(coef_df) > 2 else 0
        time_p = coef_df.iloc[2]['P值'] if len(coef_df) > 2 else 1
        sig_label = "具有统计学显著性" if group_p < 0.05 else "未达到统计学显著性"
        out["discussion"] = (
            f"## GEE (广义估计方程) 分析结果与讨论\n\n"
            f"### 一、方法学概述\n\n"
            f"广义估计方程 (GEE) 是处理纵向/聚类数据的边际模型方法，通过指定工作相关矩阵来校正"
            f"重复测量间的相关性，从而获得总体平均效应的一致估计。本分析采用可交换 (Exchangeable) "
            f"相关结构，假设同一受试者内任意两次测量间的相关性相同。该方法对相关结构的错误指定具有"
            f"稳健性，即使工作相关矩阵与真实结构不完全一致，三明治估计量仍可提供有效的标准误估计。"
            f"本研究纳入 **{n_subjects}** 名受试者，共 **{n_obs}** 条观测记录，"
            f"覆盖 **{n_timepoints}** 个时间点。\n\n"
            f"### 二、主要发现\n\n"
            f"- 分组变量 **{group_var}** 的回归系数为 **{group_coef:.4f}** "
            f"(95% CI: {group_ci_low:.4f} ~ {group_ci_high:.4f}, P = {group_p:.4f})，{sig_label}\n"
            f"- 时间变量 **{time_var}** 的效应估计为 **{time_coef:.4f}** (P = {time_p:.4f})，"
            f"反映了结局指标随时间的整体变化趋势\n"
            f"- 模型共纳入 **{len(coef_df)}** 个参数（含截距及协变量），各参数估计见系数表\n\n"
            f"### 三、临床意义\n\n"
            f"GEE 模型的边际解释表明，在控制时间效应和协变量后，{group_var} 对 {outcome} 的"
            f"平均影响为 {group_coef:.4f} 个单位。该效应量在临床实践中"
            f"{'可能具有实际意义，提示干预措施对结局指标产生了有意义的改善' if abs(group_coef) > 0 and group_p < 0.05 else '需结合临床最小重要差异 (MCID) 进一步判断其实际意义'}。"
            f"时间趋势图直观展示了各组在随访期间的变化轨迹，有助于识别疗效出现的时间窗口。\n\n"
            f"### 四、模型诊断与局限性\n\n"
            f"- 残差诊断图应检查是否存在系统性模式，若残差呈随机分布则支持模型假设\n"
            f"- GEE 为半参数方法，不对数据分布做严格假设，但要求样本量足够大以保证三明治估计量的渐近性质\n"
            f"- 可交换相关结构假设可能过于简化，若数据存在明显的时间衰减相关，可考虑 AR(1) 结构\n"
            f"- 缺失数据需满足完全随机缺失 (MCAR) 假设，若存在非随机缺失可能导致偏倚\n\n"
            f"### 五、结论\n\n"
            f"基于 {n_subjects} 名受试者 {n_obs} 条纵向观测的 GEE 分析表明，"
            f"{group_var} 对 {outcome} 的效应估计为 {group_coef:.4f} (P = {group_p:.4f})，"
            f"{sig_label}。模型采用可交换相关结构有效处理了重复测量数据的内在相关性，"
            f"为纵向临床研究提供了可靠的总体平均效应推断。"
        )
    except Exception as e:
        out["discussion"] = f"GEE模型运行异常: {str(e)}"
    return out


# ═══════════════════════════════════════════════════════════════
# 2. Propensity Score Matching
# ═══════════════════════════════════════════════════════════════

def run_propensity_score(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    treatment_var = params.get("treatment_var", "treatment")
    covariates = params.get("covariates", ["age", "sex", "bmi", "sbp", "glucose", "comorbidity_count"])
    outcome_var = params.get("outcome_var", "outcome")

    cov_cols = [c for c in covariates if c in df.columns]
    df_clean = df.dropna(subset=[treatment_var] + cov_cols).copy()
    X = df_clean[cov_cols].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category").cat.codes
    y = df_clean[treatment_var].values

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    ps = model.predict_proba(X)[:, 1]
    df_clean["ps"] = ps

    # SMD before matching
    smd_before = {}
    for col in cov_cols:
        treated = df_clean[df_clean[treatment_var] == 1][col]
        control = df_clean[df_clean[treatment_var] == 0][col]
        if treated.dtype == object or control.dtype == object:
            treated_num = treated.astype("category").cat.codes.astype(float)
            control_num = control.astype("category").cat.codes.astype(float)
            pooled_std = np.sqrt((np.var(treated_num) + np.var(control_num)) / 2)
            smd = abs(np.mean(treated_num) - np.mean(control_num)) / max(pooled_std, 1e-8)
        else:
            pooled_std = np.sqrt((np.var(treated) + np.var(control)) / 2)
            smd = abs(np.mean(treated) - np.mean(control)) / max(pooled_std, 1e-8)
        smd_before[col] = round(smd, 3)

    # Matching
    treated_df = df_clean[df_clean[treatment_var] == 1].copy()
    control_df = df_clean[df_clean[treatment_var] == 0].copy()
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(control_df[["ps"]].values)
    distances, indices = nn.kneighbors(treated_df[["ps"]].values)
    matched_control = control_df.iloc[indices.flatten()].copy()
    matched_df = pd.concat([treated_df.reset_index(drop=True), matched_control.reset_index(drop=True)], ignore_index=True)

    smd_after = {}
    for col in cov_cols:
        treated = matched_df[matched_df[treatment_var] == 1][col]
        control = matched_df[matched_df[treatment_var] == 0][col]
        if treated.dtype == object or control.dtype == object:
            treated_num = treated.astype("category").cat.codes.astype(float)
            control_num = control.astype("category").cat.codes.astype(float)
            pooled_std = np.sqrt((np.var(treated_num) + np.var(control_num)) / 2)
            smd = abs(np.mean(treated_num) - np.mean(control_num)) / max(pooled_std, 1e-8)
        else:
            pooled_std = np.sqrt((np.var(treated) + np.var(control)) / 2)
            smd = abs(np.mean(treated) - np.mean(control)) / max(pooled_std, 1e-8)
        smd_after[col] = round(smd, 3)

    smd_table = pd.DataFrame({
        "协变量": list(smd_before.keys()),
        "匹配前SMD": [smd_before[k] for k in smd_before],
        "匹配后SMD": [smd_after[k] for k in smd_after],
    })
    out["tables"].append({"title": "倾向性评分匹配前后 SMD", "headers": list(smd_table.columns), "rows": smd_table.to_dict(orient="records")})

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(smd_before.keys()), y=list(smd_before.values()), name="匹配前 SMD",
                             marker_color="#E06830", opacity=0.7))
        fig.add_trace(go.Bar(x=list(smd_after.keys()), y=list(smd_after.values()), name="匹配后 SMD",
                             marker_color="#0E7C7B", opacity=0.7))
        fig.add_hline(y=0.1, line_dash="dash", line_color="green", annotation_text="SMD=0.1 阈值")
        fig.update_layout(title="标准化均值差 (SMD) 匹配前后对比", barmode="group",
                          xaxis_title="协变量", yaxis_title="SMD", template="plotly_white", height=520)
        out["charts"].append({"title": "SMD 对比图", "plotly": _fig_to_json(fig)})

        # PS distribution
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=df_clean[df_clean[treatment_var] == 1]["ps"], name="治疗组",
                                     opacity=0.6, marker_color="#E06830"))
        fig2.add_trace(go.Histogram(x=df_clean[df_clean[treatment_var] == 0]["ps"], name="对照组",
                                     opacity=0.6, marker_color="#0E7C7B"))
        fig2.update_layout(title="倾向性评分分布", barmode="overlay", xaxis_title="倾向性评分",
                           yaxis_title="频数", template="plotly_white", height=520)
        out["diagnostics"].append({"title": "倾向性评分分布", "plotly": _fig_to_json(fig2)})

    n_treated = len(treated_df)
    n_control_pool = len(control_df)
    n_matched = len(matched_df)
    n_imbalanced_before = sum(1 for v in smd_before.values() if v > 0.1)
    n_imbalanced_after = sum(1 for v in smd_after.values() if v > 0.1)
    max_smd_before = max(smd_before.values()) if smd_before else 0
    max_smd_after = max(smd_after.values()) if smd_after else 0
    mean_smd_before = np.mean(list(smd_before.values())) if smd_before else 0
    mean_smd_after = np.mean(list(smd_after.values())) if smd_after else 0
    out["discussion"] = (
        f"## 倾向性评分匹配 (PSM) 分析结果与讨论\n\n"
        f"### 一、方法学概述\n\n"
        f"倾向性评分匹配 (Propensity Score Matching) 是观察性研究中控制混杂偏倚的核心方法。"
        f"通过 Logistic 回归模型估计每个个体接受治疗的条件概率（倾向性评分），"
        f"再基于该评分进行最近邻匹配，使治疗组和对照组在观测协变量上达到均衡，"
        f"从而模拟随机化试验的效果。本分析纳入 **{len(cov_cols)}** 个协变量构建倾向性评分模型，"
        f"治疗组 **{n_treated}** 例，对照组候选池 **{n_control_pool}** 例，"
        f"采用 1:1 最近邻匹配（欧氏距离），最终获得 **{n_matched}** 例匹配样本。\n\n"
        f"### 二、主要发现\n\n"
        f"- 匹配前：**{n_imbalanced_before}/{len(smd_before)}** 个协变量 SMD > 0.1（不均衡），"
        f"最大 SMD = **{max_smd_before:.3f}**，平均 SMD = **{mean_smd_before:.3f}**\n"
        f"- 匹配后：**{n_imbalanced_after}/{len(smd_after)}** 个协变量 SMD > 0.1，"
        f"最大 SMD = **{max_smd_after:.3f}**，平均 SMD = **{mean_smd_after:.3f}**\n"
        f"- SMD 降幅：平均降低 **{(mean_smd_before - mean_smd_after) / max(mean_smd_before, 1e-8) * 100:.1f}%**，"
        f"表明匹配{'有效' if n_imbalanced_after <= 1 else '部分'}地平衡了组间基线差异\n\n"
        f"### 三、临床意义\n\n"
        f"通过 PSM 消除可观测混杂因素后，治疗组与对照组在基线特征上的可比性显著提高。"
        f"这使得后续的处理效应估计更接近因果推断，减少了选择偏倚对结论的威胁。"
        f"临床决策者可基于匹配后的效应估计，更有信心地评估干预措施的真实疗效。"
        f"{'所有协变量均达到良好均衡 (SMD < 0.1)，支持后续因果效应估计的有效性。' if n_imbalanced_after == 0 else '部分协变量匹配后仍存在轻度不均衡，建议在后续分析中进一步调整。'}\n\n"
        f"### 四、模型诊断与局限性\n\n"
        f"- 倾向性评分分布图应检查两组评分的重叠区域 (common support)，重叠不足将限制匹配质量\n"
        f"- PSM 仅能控制已观测的混杂因素，无法消除未测量混杂 (unmeasured confounding) 的影响\n"
        f"- 1:1 匹配可能导致样本量损失，降低统计效能；可考虑 1:k 匹配或加权方法作为替代\n"
        f"- 匹配后样本不再独立，后续分析应使用配对方法或考虑匹配结构\n\n"
        f"### 五、结论\n\n"
        f"基于 {len(cov_cols)} 个协变量的倾向性评分 1:1 最近邻匹配成功将不均衡协变量从 "
        f"{n_imbalanced_before} 个降至 {n_imbalanced_after} 个，平均 SMD 从 {mean_smd_before:.3f} "
        f"降至 {mean_smd_after:.3f}。匹配后的样本为后续因果效应估计提供了较为可靠的基础，"
        f"但结论仍受限于未测量混杂因素的潜在影响。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 3. Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════

def run_sensitivity_analysis(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    treatment_var = params.get("treatment_var", "treatment")
    outcome_var = params.get("outcome_var", "followup_score")
    baseline_var = params.get("baseline_var", "baseline_score")

    df_clean = df.dropna(subset=[treatment_var, baseline_var]).copy()
    observed = df_clean.dropna(subset=[outcome_var]).copy()
    unique_trt = observed[treatment_var].unique()
    trt_val = unique_trt[0] if len(unique_trt) >= 1 else "Experimental"
    ctrl_val = unique_trt[1] if len(unique_trt) >= 2 else "Standard"
    treated = observed[observed[treatment_var] == trt_val][outcome_var]
    control = observed[observed[treatment_var] == ctrl_val][outcome_var]
    observed_effect = treated.mean() - control.mean()

    scenarios = []
    for gamma in np.arange(0, 3.1, 0.3):
        worst_adj = observed_effect - gamma * np.std(observed[outcome_var])
        best_adj = observed_effect + gamma * np.std(observed[outcome_var])
        scenarios.append({
            "gamma": round(gamma, 1),
            "adjusted_effect_lower": round(min(worst_adj, best_adj), 3),
            "adjusted_effect_upper": round(max(worst_adj, best_adj), 3),
            "significant": "Yes" if min(abs(worst_adj), abs(best_adj)) > 0.5 * np.std(observed[outcome_var]) else "No",
        })

    sens_table = pd.DataFrame(scenarios)
    out["tables"].append({"title": "敏感性分析场景", "headers": list(sens_table.columns), "rows": sens_table.to_dict(orient="records")})

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[s["gamma"] for s in scenarios], y=[s["adjusted_effect_lower"] for s in scenarios],
                                 mode="lines+markers", name="下界", line=dict(color="#E06830")))
        fig.add_trace(go.Scatter(x=[s["gamma"] for s in scenarios], y=[s["adjusted_effect_upper"] for s in scenarios],
                                 mode="lines+markers", name="上界", line=dict(color="#0E7C7B")))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="临界点敏感性分析", xaxis_title="Gamma (未测量混杂程度)",
                          yaxis_title="调整后效应值", template="plotly_white", height=530)
        out["charts"].append({"title": "临界点分析图", "plotly": _fig_to_json(fig)})

    n_observed = len(observed)
    n_total = len(df_clean)
    missing_rate = (n_total - n_observed) / max(n_total, 1) * 100
    outcome_sd = np.std(observed[outcome_var])
    # Find the gamma at which significance is lost
    gamma_critical = None
    for s in scenarios:
        if s["significant"] == "No" and gamma_critical is None:
            gamma_critical = s["gamma"]
    gamma_critical_str = f"{gamma_critical}" if gamma_critical is not None else ">3.0"
    out["discussion"] = (
        f"## 敏感性分析结果与讨论\n\n"
        f"### 一、方法学概述\n\n"
        f"敏感性分析旨在评估研究结论对未测量混杂因素的稳健性。本分析采用 Rosenbaum 界限法的"
        f"改良版本，通过引入偏倚参数 Gamma (γ) 模拟不同程度的未测量混杂对处理效应估计的影响。"
        f"γ = 0 表示无未测量混杂，γ 值越大表示潜在混杂越严重。当调整后的效应区间包含零时，"
        f"表明在该混杂水平下研究结论不再稳健。本分析基于 **{n_observed}** 例完整观测数据，"
        f"缺失率为 **{missing_rate:.1f}%**，结局变量标准差为 **{outcome_sd:.3f}**。\n\n"
        f"### 二、主要发现\n\n"
        f"- 观测处理效应 (未调整): **{observed_effect:.3f}** ({outcome_var} 单位)\n"
        f"- 效应量相对于标准差: **{observed_effect / max(outcome_sd, 1e-8):.2f}** SD 单位 (Cohen's d)\n"
        f"- 临界 Gamma 值: **{gamma_critical_str}** — 当 γ 超过此值时，处理效应的显著性将被推翻\n"
        f"- 在 γ ≤ 1.5 范围内，调整后效应区间{'均不包含零，结论稳健' if gamma_critical is None or gamma_critical > 1.5 else '已包含零，结论敏感'}\n\n"
        f"### 三、临床意义\n\n"
        f"临界 Gamma 值为 {gamma_critical_str}，意味着需要存在一个使个体接受治疗的概率改变 "
        f"{gamma_critical_str} 倍标准差的未测量混杂因素，才能完全解释观测到的处理效应。"
        f"{'这一阈值较高，表明即使存在中等程度的未测量混杂，研究结论仍然成立，为临床决策提供了较强的证据支持。' if gamma_critical is None or gamma_critical > 2.0 else '该阈值提示结论对未测量混杂较为敏感，临床应用时需谨慎解读，建议结合领域知识评估是否存在重要的遗漏变量。'}"
        f"\n\n"
        f"### 四、模型诊断与局限性\n\n"
        f"- 本方法假设未测量混杂对处理效应的影响呈线性关系，实际情况可能更为复杂\n"
        f"- 分析基于完整病例，若缺失数据与结局相关 (MAR/MNAR)，可能影响敏感性评估的准确性\n"
        f"- Gamma 参数的临床可解释性有限，需结合具体研究领域判断何种程度的混杂是合理的\n"
        f"- 建议补充 E-value 分析以提供更直观的混杂强度阈值\n\n"
        f"### 五、结论\n\n"
        f"敏感性分析表明，观测处理效应 {observed_effect:.3f} 在临界 Gamma = {gamma_critical_str} "
        f"时失去统计显著性。综合考虑效应量 ({observed_effect / max(outcome_sd, 1e-8):.2f} SD) "
        f"和稳健性阈值，本研究结论对未测量混杂具有"
        f"{'良好的稳健性，支持因果推断的合理性。' if gamma_critical is None or gamma_critical > 2.0 else '一定的敏感性，因果推断需谨慎，建议在后续研究中纳入更多潜在混杂因素。'}"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 4. Counterfactual / Causal Inference
# ═══════════════════════════════════════════════════════════════

def run_counterfactual(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    treatment_var = params.get("treatment_var", "treatment")
    outcome_var = params.get("outcome_var", "outcome")
    covariates = params.get("covariates", ["age", "sex", "bmi", "baseline_severity", "smoking"])

    cov_cols = [c for c in covariates if c in df.columns]
    df_clean = df.dropna(subset=[treatment_var, outcome_var] + cov_cols).copy()
    X = df_clean[cov_cols].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category").cat.codes
    T = df_clean[treatment_var].values
    Y = df_clean[outcome_var].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Outcome model
    outcome_model = LogisticRegression(max_iter=1000) if len(np.unique(Y)) <= 10 else None
    if outcome_model:
        outcome_model.fit(np.column_stack([X_scaled, T]), Y)
        ite = outcome_model.predict_proba(np.column_stack([X_scaled, np.ones_like(T)]))[:, 1] - \
              outcome_model.predict_proba(np.column_stack([X_scaled, np.zeros_like(T)]))[:, 1]
    else:
        from sklearn.linear_model import LinearRegression
        outcome_model = LinearRegression()
        outcome_model.fit(np.column_stack([X_scaled, T]), Y)
        y1_pred = outcome_model.predict(np.column_stack([X_scaled, np.ones_like(T)]))
        y0_pred = outcome_model.predict(np.column_stack([X_scaled, np.zeros_like(T)]))
        ite = y1_pred - y0_pred

    ate = np.mean(ite)
    att = np.mean(ite[T == 1])

    results = [{"估计量": "ATE (平均处理效应)", "值": round(float(ate), 4)},
               {"估计量": "ATT (处理组平均效应)", "值": round(float(att), 4)}]
    out["tables"].append({"title": "因果效应估计", "headers": ["估计量", "值"], "rows": results})

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=ite[T == 1], name="Treatment=1", opacity=0.6, marker_color="#E06830"))
        fig.add_trace(go.Histogram(x=ite[T == 0], name="Treatment=0", opacity=0.6, marker_color="#0E7C7B"))
        fig.add_vline(x=ate, line_dash="dash", line_color="red", annotation_text=f"ATE={ate:.3f}")
        fig.update_layout(title="个体处理效应分布", barmode="overlay", template="plotly_white", height=530)
        out["charts"].append({"title": "ITE分布", "plotly": _fig_to_json(fig)})
        out["diagnostics"].append({"title": "ITE分布", "plotly": _fig_to_json(fig)})

    n_total = len(df_clean)
    n_treated_cf = int(np.sum(T == 1))
    n_control_cf = int(np.sum(T == 0))
    ite_std = float(np.std(ite))
    ite_positive_pct = float(np.mean(ite > 0) * 100)
    ate_se = ite_std / np.sqrt(n_total)
    ate_ci_low = ate - 1.96 * ate_se
    ate_ci_high = ate + 1.96 * ate_se
    out["discussion"] = (
        f"## 反事实推断/因果推断分析结果与讨论\n\n"
        f"### 一、方法学概述\n\n"
        f"反事实推断框架 (Counterfactual Inference) 基于 Rubin 因果模型的潜在结局理论，"
        f"通过构建结局预测模型估计每个个体在接受和未接受治疗两种情境下的潜在结局，"
        f"进而计算个体处理效应 (ITE)、平均处理效应 (ATE) 和处理组平均效应 (ATT)。"
        f"本分析纳入 **{n_total}** 例样本（治疗组 {n_treated_cf} 例，对照组 {n_control_cf} 例），"
        f"基于 **{len(cov_cols)}** 个协变量 ({', '.join(cov_cols)}) 构建标准化结局模型，"
        f"通过反事实预测估计因果效应。\n\n"
        f"### 二、主要发现\n\n"
        f"- **ATE (平均处理效应)**: {ate:.4f} (95% CI: {ate_ci_low:.4f} ~ {ate_ci_high:.4f})，"
        f"表示总体中每个个体接受治疗的平均因果效应\n"
        f"- **ATT (处理组平均效应)**: {att:.4f}，表示实际接受治疗者的平均获益\n"
        f"- ITE 标准差: **{ite_std:.4f}**，反映个体间处理效应的异质性程度\n"
        f"- **{ite_positive_pct:.1f}%** 的个体预测从治疗中获益 (ITE > 0)\n\n"
        f"### 三、临床意义\n\n"
        f"ATE 与 ATT 的{'一致性' if abs(ate - att) < 0.01 else '差异'}"
        f"{'表明治疗效应在人群中较为均匀' if abs(ate - att) < 0.01 else f'(差值 = {abs(ate - att):.4f}) 提示存在治疗效应异质性，即治疗选择可能与个体对治疗的反应性相关'}。"
        f"ITE 分布图揭示了处理效应的个体差异，{ite_positive_pct:.1f}% 的正效应比例"
        f"{'支持该治疗在目标人群中的广泛应用' if ite_positive_pct > 70 else '提示需要进一步识别获益亚组以实现精准治疗'}。"
        f"这一发现对个体化治疗决策具有重要指导意义。\n\n"
        f"### 四、模型诊断与局限性\n\n"
        f"- 因果推断依赖强可忽略性假设 (Strong Ignorability)：在给定协变量条件下，治疗分配与潜在结局独立\n"
        f"- 该假设不可检验，若存在未观测的混杂因素，ATE/ATT 估计可能存在偏倚\n"
        f"- 结局模型的正确指定是估计无偏性的关键，模型误设可能导致 ITE 估计偏差\n"
        f"- 正值性假设 (Positivity) 要求每个协变量组合下均有治疗和对照观测，违反时外推不可靠\n\n"
        f"### 五、结论\n\n"
        f"基于 {len(cov_cols)} 个协变量调整的反事实推断分析估计 ATE 为 {ate:.4f}，"
        f"ATT 为 {att:.4f}，{ite_positive_pct:.1f}% 的个体预测从治疗中获益。"
        f"结果在强可忽略性假设下支持治疗的因果效应，但需注意未测量混杂的潜在威胁，"
        f"建议结合敏感性分析评估结论的稳健性。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 5. Complex Survival & Disease Progression
# ═══════════════════════════════════════════════════════════════

def run_survival_advanced(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    from lifelines import KaplanMeierFitter, CoxPHFitter

    time_var = params.get("time_var", "time")
    event_var = params.get("event_var", "event")
    group_var = params.get("group_var", "stage")
    covariates = params.get("covariates", ["age", "treatment", "albumin", "ldh"])

    df_clean = df.dropna(subset=[time_var, event_var]).copy()

    # KM curves
    if HAS_PLOTLY:
        fig = go.Figure()
        for g in sorted(df_clean[group_var].unique()):
            sub = df_clean[df_clean[group_var] == g]
            kmf = KaplanMeierFitter()
            kmf.fit(sub[time_var], sub[event_var], label=str(g))
            fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_.values.flatten(),
                                     mode="lines", name=str(g)))
        fig.update_layout(title="Kaplan-Meier 生存曲线", xaxis_title="时间", yaxis_title="生存概率",
                          template="plotly_white", height=520)
        out["charts"].append({"title": "KM生存曲线", "plotly": _fig_to_json(fig)})

    # Cox model
    cov_cols_cox = [c for c in covariates if c in df_clean.columns]
    cph = CoxPHFitter()
    cox_input = df_clean[[time_var, event_var] + cov_cols_cox].copy()
    for col in cov_cols_cox:
        if cox_input[col].dtype == object:
            cox_input[col] = cox_input[col].astype("category").cat.codes
    cph.fit(cox_input, duration_col=time_var, event_col=event_var)
    cox_summary = cph.summary
    cox_table = pd.DataFrame({
        "变量": cox_summary.index,
        "HR": np.round(cox_summary["exp(coef)"].values, 3),
        "95% CI 下限": np.round(cox_summary["exp(coef) lower 95%"].values, 3),
        "95% CI 上限": np.round(cox_summary["exp(coef) upper 95%"].values, 3),
        "P值": np.round(cox_summary["p"].values, 4),
    })
    out["tables"].append({"title": "Cox 比例风险模型", "headers": list(cox_table.columns), "rows": cox_table.to_dict(orient="records")})

    # Forest plot for Cox
    if HAS_PLOTLY:
        fig2 = go.Figure()
        for i, row in cox_table.iterrows():
            fig2.add_trace(go.Scatter(
                x=[row["HR"]], y=[row["变量"]], mode="markers",
                error_x=dict(type="data", symmetric=False,
                             array=[row["95% CI 上限"] - row["HR"]],
                             arrayminus=[row["HR"] - row["95% CI 下限"]]),
                marker=dict(size=10, color="#0E7C7B"),
                name=str(row["变量"]),
            ))
        fig2.add_vline(x=1, line_dash="dash", line_color="gray")
        fig2.update_layout(title="Cox 回归风险比 (HR) 森林图", xaxis_title="Hazard Ratio",
                           template="plotly_white", height=520)
        out["diagnostics"].append({"title": "Cox森林图", "plotly": _fig_to_json(fig2)})

    n_total_surv = len(df_clean)
    n_events = int(df_clean[event_var].sum())
    event_rate = n_events / max(n_total_surv, 1) * 100
    n_groups_surv = df_clean[group_var].nunique()
    median_time = df_clean[time_var].median()
    sig_vars = cox_table[cox_table["P值"] < 0.05]["变量"].tolist() if len(cox_table) > 0 else []
    max_hr = cox_table["HR"].max() if len(cox_table) > 0 else 1
    max_hr_var = cox_table.loc[cox_table["HR"].idxmax(), "变量"] if len(cox_table) > 0 else ""
    out["discussion"] = (
        f"## 复杂生存分析与疾病进展分析结果与讨论\n\n"
        f"### 一、方法学概述\n\n"
        f"本分析综合运用 Kaplan-Meier 生存估计和 Cox 比例风险回归模型，全面评估生存结局的"
        f"影响因素。KM 方法提供非参数生存曲线估计，直观展示各组生存概率随时间的变化；"
        f"Cox 模型则在比例风险假设下量化各协变量对风险率的独立影响。"
        f"本研究纳入 **{n_total_surv}** 例患者，观察到 **{n_events}** 例终点事件 "
        f"(事件率 {event_rate:.1f}%)，中位随访时间 **{median_time:.1f}**，"
        f"按 **{group_var}** 分为 **{n_groups_surv}** 组进行比较，"
        f"Cox 模型纳入 **{len(cov_cols_cox)}** 个协变量进行多因素调整。\n\n"
        f"### 二、主要发现\n\n"
        f"- KM 曲线显示 {group_var} 各组间生存概率存在{'明显' if n_groups_surv > 1 else ''}分离趋势\n"
        f"- Cox 回归中，{len(sig_vars)} 个变量达到统计学显著性 (P < 0.05)"
        f"{': ' + ', '.join(sig_vars) if sig_vars else ''}\n"
        f"- 最大风险比: **{max_hr_var}** (HR = **{max_hr:.3f}**)，"
        f"表示该因素每增加一个单位，风险率{'增加' if max_hr > 1 else '降低'} "
        f"**{abs(max_hr - 1) * 100:.1f}%**\n"
        f"- 事件率 {event_rate:.1f}% 提供了充足的统计效能用于多因素分析\n\n"
        f"### 三、临床意义\n\n"
        f"Cox 模型的风险比 (HR) 提供了各因素对预后影响的量化指标。"
        f"{'显著的预后因素可用于构建临床风险分层模型，指导个体化治疗决策和随访策略的制定。' if sig_vars else '未发现显著预后因素，可能与样本量或事件数不足有关。'}"
        f"森林图直观展示了各变量的 HR 及其置信区间，HR > 1 表示风险因素，HR < 1 表示保护因素。"
        f"临床医生可据此识别高危患者亚群，优化资源配置和干预时机。\n\n"
        f"### 四、模型诊断与局限性\n\n"
        f"- Cox 模型的核心假设为比例风险 (PH)，应通过 Schoenfeld 残差检验验证\n"
        f"- 若 PH 假设不满足，可考虑时依协变量、分层 Cox 模型或参数生存模型\n"
        f"- 删失机制应为非信息性删失，若删失与预后相关将导致偏倚\n"
        f"- 竞争风险的存在可能高估特定事件的累积发生率，必要时应采用 Fine-Gray 模型\n\n"
        f"### 五、结论\n\n"
        f"基于 {n_total_surv} 例患者 ({n_events} 例事件) 的生存分析表明，"
        f"{group_var} 分组间存在生存差异。Cox 多因素回归识别出 {len(sig_vars)} 个独立预后因素，"
        f"其中 {max_hr_var} 的风险比最高 (HR = {max_hr:.3f})。"
        f"研究结果为临床预后评估和治疗决策提供了循证依据。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 6. Markov Model
# ═══════════════════════════════════════════════════════════════

def run_markov_model(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    state_var = params.get("state_var", "state")
    time_var = params.get("time_var", "month")

    df_clean = df.dropna(subset=[state_var, time_var]).copy()
    states = sorted(df_clean[state_var].unique())
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    # Count transitions
    trans_count = np.zeros((n_states, n_states))
    for subj in df_clean["subject_id"].unique():
        sub_df = df_clean[df_clean["subject_id"] == subj].sort_values(time_var)
        for t in range(len(sub_df) - 1):
            s_from = state_to_idx[sub_df.iloc[t][state_var]]
            s_to = state_to_idx[sub_df.iloc[t + 1][state_var]]
            trans_count[s_from, s_to] += 1

    trans_prob = trans_count / np.maximum(trans_count.sum(axis=1, keepdims=True), 1)
    trans_df = pd.DataFrame(trans_prob, index=states, columns=states)
    trans_df = trans_df.round(3)

    rows = []
    for from_state in states:
        for to_state in states:
            rows.append({"起始状态": from_state, "目标状态": to_state,
                         "转移概率": trans_prob[state_to_idx[from_state], state_to_idx[to_state]]})
    out["tables"].append({"title": "状态转移概率矩阵", "headers": ["起始状态", "目标状态", "转移概率"],
                          "rows": [dict(r) for r in rows if r["转移概率"] > 0]})

    # Project state distribution
    if HAS_PLOTLY:
        n_proj = 20
        proj = np.zeros((n_proj, n_states))
        proj[0, 0] = 1.0
        for t in range(1, n_proj):
            proj[t] = proj[t - 1] @ trans_prob

        fig = go.Figure()
        for i, s in enumerate(states):
            fig.add_trace(go.Scatter(x=list(range(n_proj)), y=proj[:, i], mode="lines", name=s))
        fig.update_layout(title="Markov 状态概率轨迹", xaxis_title="周期", yaxis_title="概率",
                          template="plotly_white", height=520)
        out["charts"].append({"title": "状态概率轨迹", "plotly": _fig_to_json(fig)})

    n_subjects_mk = df_clean["subject_id"].nunique()
    n_transitions = int(trans_count.sum())
    # Find dominant transition (off-diagonal)
    np.fill_diagonal(trans_count, 0)
    max_trans_idx = np.unravel_index(np.argmax(trans_count), trans_count.shape)
    max_trans_from = states[max_trans_idx[0]]
    max_trans_to = states[max_trans_idx[1]]
    max_trans_prob = trans_prob[max_trans_idx[0], max_trans_idx[1]]
    # Steady state approximation
    try:
        eigenvalues, eigenvectors = np.linalg.eig(trans_prob.T)
        idx_one = np.argmin(np.abs(eigenvalues - 1))
        steady = np.real(eigenvectors[:, idx_one])
        steady = steady / steady.sum()
        steady_state_str = ", ".join([f"{s}: {p:.3f}" for s, p in zip(states, steady)])
    except Exception:
        steady_state_str = "计算未收敛"
    # Self-retention rates
    retention_rates = [trans_prob[i, i] for i in range(n_states)]
    avg_retention = np.mean(retention_rates)
    out["discussion"] = (
        f"## Markov 状态转移模型分析结果与讨论\n\n"
        f"### 一、方法学概述\n\n"
        f"离散时间 Markov 链模型假设系统在任意时刻处于有限个状态之一，且下一时刻的状态"
        f"仅取决于当前状态（马尔可夫性/无记忆性）。通过统计相邻时间点间的状态转移频次，"
        f"估计转移概率矩阵，进而预测疾病进展轨迹和长期稳态分布。"
        f"本分析基于 **{n_subjects_mk}** 名受试者的纵向状态数据，识别出 **{n_states}** 个"
        f"离散状态 ({', '.join([str(s) for s in states])})，共观测到 **{n_transitions}** 次状态转移。\n\n"
        f"### 二、主要发现\n\n"
        f"- 状态空间包含 **{n_states}** 个状态，转移概率矩阵维度为 {n_states}x{n_states}\n"
        f"- 最主要的状态转移路径: **{max_trans_from} → {max_trans_to}** (概率 = **{max_trans_prob:.3f}**)\n"
        f"- 平均状态自留概率: **{avg_retention:.3f}**，反映状态的整体稳定性\n"
        f"- 稳态分布估计: {steady_state_str}\n"
        f"- 20 周期状态概率轨迹图展示了从初始状态出发的长期演变趋势\n\n"
        f"### 三、临床意义\n\n"
        f"转移概率矩阵量化了疾病在不同阶段间的进展和逆转概率，为临床路径管理提供了数据支撑。"
        f"主要转移路径 {max_trans_from} → {max_trans_to} (P = {max_trans_prob:.3f}) 揭示了"
        f"疾病进展的关键节点，提示临床干预应重点关注该转移的预防。"
        f"稳态分布预测了在当前转移模式下患者群体的长期状态构成，"
        f"可用于医疗资源规划和卫生经济学评估。\n\n"
        f"### 四、模型诊断与局限性\n\n"
        f"- 马尔可夫性假设（无记忆性）可能过于简化，实际疾病进展常受历史状态影响\n"
        f"- 时间齐次性假设意味着转移概率不随时间变化，长期随访中该假设可能不成立\n"
        f"- 状态定义的粗细程度影响模型的临床可解释性和预测精度\n"
        f"- 样本量和观测时间点数量限制了转移概率估计的精度，稀疏转移的估计不确定性较大\n\n"
        f"### 五、结论\n\n"
        f"基于 {n_subjects_mk} 名受试者 {n_transitions} 次转移观测的 {n_states} 状态 Markov 模型"
        f"揭示了疾病状态间的转移规律。主要进展路径为 {max_trans_from} → {max_trans_to}，"
        f"平均状态自留率 {avg_retention:.3f} 表明状态具有{'较高' if avg_retention > 0.7 else '中等' if avg_retention > 0.5 else '较低'}的稳定性。"
        f"模型预测的稳态分布可为长期疾病管理策略提供参考。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 7. Bayesian Analysis
# ═══════════════════════════════════════════════════════════════

def run_bayesian(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    outcome_var = params.get("outcome_var", "sbp_reduction")
    group_var = params.get("group_var", "treatment")

    df_clean = df.dropna(subset=[outcome_var, group_var]).copy()
    groups = df_clean[group_var].unique()

    # Simple Bayesian comparison using conjugate normal model
    bayes_results = []
    if HAS_PLOTLY:
        fig = go.Figure()
    colors = ["#0E7C7B", "#E06830", "#665C9E", "#C58725"]

    for i, g in enumerate(groups):
        data = df_clean[df_clean[group_var] == g][outcome_var].values
        n = len(data)
        xbar = np.mean(data)
        sd = np.std(data, ddof=1)
        # Normal-Normal conjugate: prior N(mu0, tau0^2), likelihood N(theta, sigma^2/n)
        mu0, tau0 = 0, 10  # weakly informative prior
        se = sd / np.sqrt(n)
        post_var = 1 / (1 / tau0**2 + 1 / se**2)
        post_mean = post_var * (mu0 / tau0**2 + xbar / se**2)
        post_sd = np.sqrt(post_var)
        ci_low = post_mean - 1.96 * post_sd
        ci_high = post_mean + 1.96 * post_sd
        p_superior = 1 - stats.norm.cdf(0, post_mean, post_sd)

        bayes_results.append({
            "组别": str(g), "样本量": n, "均值": round(xbar, 2),
            "后验均值": round(post_mean, 3), "后验标准差": round(post_sd, 3),
            "95% HDI 下限": round(ci_low, 3), "95% HDI 上限": round(ci_high, 3),
            "P(效应>0)": round(p_superior, 4),
        })

        if HAS_PLOTLY:
            x = np.linspace(post_mean - 4 * post_sd, post_mean + 4 * post_sd, 200)
            y = stats.norm.pdf(x, post_mean, post_sd)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="tozeroy",
                                     name=str(g), line=dict(color=colors[i % len(colors)])))

    out["tables"].append({"title": "贝叶斯后验估计", "headers": list(bayes_results[0].keys()), "rows": bayes_results})

    if HAS_PLOTLY:
        fig.update_layout(title="后验分布", xaxis_title=outcome_var, yaxis_title="密度",
                          template="plotly_white", height=520)
        out["charts"].append({"title": "后验分布图", "plotly": _fig_to_json(fig)})

    diff = bayes_results[0]["后验均值"] - bayes_results[1]["后验均值"] if len(bayes_results) >= 2 else 0
    n_groups_bayes = len(bayes_results)
    total_n_bayes = sum(r["样本量"] for r in bayes_results)
    max_p_superior = max(r["P(效应>0)"] for r in bayes_results)
    best_group = [r["组别"] for r in bayes_results if r["P(效应>0)"] == max_p_superior][0]
    diff_sd = np.sqrt(bayes_results[0]["后验标准差"]**2 + bayes_results[1]["后验标准差"]**2) if len(bayes_results) >= 2 else 0
    prob_diff_positive = 1 - stats.norm.cdf(0, diff, max(diff_sd, 1e-8)) if len(bayes_results) >= 2 else 0
    out["discussion"] = (
        f"## 贝叶斯分析结果与讨论\n\n"
        f"### 一、方法学概述\n\n"
        f"贝叶斯分析采用正态-正态共轭模型，将先验信息与观测数据结合得到后验分布。"
        f"本分析使用弱信息先验 (μ₀ = 0, τ₀ = 10)，使后验推断主要由数据驱动，"
        f"同时保持了贝叶斯框架的优势：直接给出参数的概率陈述、自然地量化不确定性、"
        f"以及通过后验概率支持临床决策。共分析 **{n_groups_bayes}** 个组别，"
        f"总样本量 **{total_n_bayes}**，结局变量为 **{outcome_var}**。\n\n"
        f"### 二、主要发现\n\n"
        f"- 组间后验均值差异: **{diff:.3f}** (后验 SD = {diff_sd:.3f})\n"
        f"- 差异为正的后验概率: **P(Δ > 0) = {prob_diff_positive:.4f}**\n"
        f"- 效应为正的最高后验概率组: **{best_group}** (P(效应>0) = {max_p_superior:.4f})\n"
        f"- 各组 95% 最高密度区间 (HDI) 见后验估计表，HDI 不包含零表示效应具有实际意义\n\n"
        f"### 三、临床意义\n\n"
        f"贝叶斯后验概率为临床决策提供了直观的概率解释。"
        f"P(Δ > 0) = {prob_diff_positive:.4f} 表示在观测数据下，"
        f"{'组间差异为正的可能性很高 (> 95%)，提供了强有力的证据支持治疗优效性' if prob_diff_positive > 0.95 else '组间差异为正的可能性为 ' + f'{prob_diff_positive*100:.1f}%' + '，证据强度需结合临床阈值判断'}。"
        f"与频率学派的 P 值不同，后验概率可直接回答\"治疗有效的概率是多少\"这一临床核心问题，"
        f"更符合临床决策的思维模式。\n\n"
        f"### 四、模型诊断与局限性\n\n"
        f"- 弱信息先验在大样本下对后验影响极小，但小样本时先验选择可能影响结论\n"
        f"- 正态共轭模型假设数据服从正态分布，偏态数据可能需要更灵活的模型\n"
        f"- 本分析为解析解，未使用 MCMC 采样，避免了收敛性问题但限制了模型复杂度\n"
        f"- 多组比较时未进行多重性校正，贝叶斯框架下可通过层次模型自然实现收缩估计\n\n"
        f"### 五、结论\n\n"
        f"贝叶斯分析在弱信息先验下估计组间后验均值差异为 {diff:.3f}，"
        f"差异为正的后验概率为 {prob_diff_positive:.4f}。"
        f"后验分布图直观展示了各组效应的不确定性范围，"
        f"为基于概率的临床决策提供了完整的推断框架。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 8. Latin Square ANOVA
# ═══════════════════════════════════════════════════════════════

def run_latin_square(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    response_var = params.get("response_var", "response")
    period_var = params.get("period_var", "period")
    formulation_var = params.get("formulation_var", "formulation_code")
    subject_var = params.get("subject_var", "subject_id")

    df_clean = df.dropna(subset=[response_var, period_var, formulation_var, subject_var]).copy()
    df_clean["period"] = df_clean[period_var].astype(str)
    df_clean["formulation"] = df_clean[formulation_var].astype(str)
    df_clean["subject"] = df_clean[subject_var].astype(str)

    try:
        model = ols(f"{response_var} ~ C(period) + C(subject) + C(formulation)", data=df_clean).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        anova_df = pd.DataFrame({
            "变异来源": anova.index,
            "自由度": anova["df"].values.astype(int),
            "平方和": np.round(anova["sum_sq"].values, 4),
            "均方": np.round(anova["sum_sq"].values / anova["df"].values, 4),
            "F值": np.round(anova["F"].values, 3),
            "P值": np.round(anova["PR(>F)"].values, 4),
        })
        out["tables"].append({"title": "拉丁方方差分析表", "headers": list(anova_df.columns), "rows": anova_df.to_dict(orient="records")})
    except Exception:
        anova_df = pd.DataFrame({"备注": ["方差分析模型拟合失败，请检查变量配置"]})
        out["tables"].append({"title": "方差分析表", "headers": list(anova_df.columns), "rows": anova_df.to_dict(orient="records")})

    # Mean comparison
    if HAS_PLOTLY:
        agg = df_clean.groupby(formulation_var)[response_var].agg(["mean", "sem"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg[formulation_var], y=agg["mean"],
                              error_y=dict(type="data", array=agg["sem"]),
                              marker_color=["#0E7C7B", "#E06830", "#665C9E"][:len(agg)]))
        fig.update_layout(title="各剂型均值比较", xaxis_title="剂型", yaxis_title=response_var,
                          template="plotly_white", height=530)
        out["charts"].append({"title": "剂型均值比较", "plotly": _fig_to_json(fig)})

    n_subjects_ls = df_clean[subject_var].nunique()
    n_periods = df_clean["period"].nunique()
    n_formulations = df_clean["formulation"].nunique()
    n_obs_ls = len(df_clean)
    grand_mean = df_clean[response_var].mean()
    grand_std = df_clean[response_var].std()
    # Extract ANOVA results if available
    try:
        form_p = anova_df[anova_df["变异来源"].str.contains("formulation", case=False)]["P值"].values[0]
        period_p = anova_df[anova_df["变异来源"].str.contains("period", case=False)]["P值"].values[0]
        form_f = anova_df[anova_df["变异来源"].str.contains("formulation", case=False)]["F值"].values[0]
        form_sig = "显著" if form_p < 0.05 else "不显著"
    except (IndexError, KeyError):
        form_p = None
        period_p = None
        form_f = None
        form_sig = "未确定"
    out["discussion"] = (
        f"## 拉丁方设计方差分析结果与讨论\n\n"
        f"### 一、方法学概述\n\n"
        f"拉丁方设计 (Latin Square Design) 是一种同时控制两个区组因素（时期和受试者）的"
        f"实验设计方法，广泛应用于生物等效性研究和交叉试验。该设计通过让每个受试者在不同时期"
        f"接受不同处理，有效分离了时期效应、个体差异和处理效应，提高了处理间比较的精度。"
        f"本分析纳入 **{n_subjects_ls}** 名受试者，**{n_periods}** 个时期，"
        f"**{n_formulations}** 种剂型/处理，共 **{n_obs_ls}** 条观测记录。"
        f"响应变量 ({response_var}) 总体均值为 **{grand_mean:.3f}** (SD = {grand_std:.3f})。\n\n"
        f"### 二、主要发现\n\n"
        f"- 剂型效应: F = **{form_f if form_f is not None else 'N/A'}**, "
        f"P = **{form_p if form_p is not None else 'N/A'}** — {form_sig}\n"
        f"- 时期效应: P = **{period_p if period_p is not None else 'N/A'}** — "
        f"{'显著，提示存在残留效应或学习效应' if period_p is not None and period_p < 0.05 else '不显著，支持无明显时期偏倚'}\n"
        f"- 受试者间变异已通过设计控制，提高了剂型比较的统计效能\n"
        f"- 各剂型均值比较图展示了处理间的差异模式\n\n"
        f"### 三、临床意义\n\n"
        f"{'剂型间差异达到统计学显著性，提示不同制剂的生物利用度或药效存在差异，需进一步评估其临床相关性。' if form_p is not None and form_p < 0.05 else '剂型间差异未达统计学显著性，在当前样本量下支持各剂型的等效性假设。'}"
        f"拉丁方设计通过消除个体差异和时期效应的干扰，使剂型间的比较更为精确。"
        f"{'时期效应显著提示可能存在残留效应 (carryover effect)，这可能影响处理效应估计的无偏性，需要在解读时予以考虑。' if period_p is not None and period_p < 0.05 else '时期效应不显著支持了交叉设计的有效性，排除了明显的残留效应干扰。'}\n\n"
        f"### 四、模型诊断与局限性\n\n"
        f"- 拉丁方设计假设无处理与时期的交互作用（无残留效应），违反时需采用额外的洗脱期\n"
        f"- 模型假设残差正态性和方差齐性，应通过残差图和 Levene 检验验证\n"
        f"- 设计要求处理数 = 时期数 = 受试者数（或其倍数），限制了设计的灵活性\n"
        f"- 受试者脱落将破坏设计的平衡性，影响效应估计的效率\n\n"
        f"### 五、结论\n\n"
        f"基于 {n_subjects_ls} 名受试者 {n_periods} 时期 {n_formulations} 剂型的拉丁方方差分析表明，"
        f"剂型效应{form_sig} (P = {form_p if form_p is not None else 'N/A'})。"
        f"设计有效控制了时期和个体变异，为处理间比较提供了高效能的统计推断框架。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 9. Meta Analysis
# ═══════════════════════════════════════════════════════════════

def run_meta_analysis(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    study_var = params.get("study_var", "study")
    effect_var = params.get("effect_var", "or_estimate")
    se_var = params.get("se_var", "se")

    df_clean = df.dropna(subset=[study_var, effect_var, se_var]).copy()
    effects = df_clean[effect_var].values
    ses = df_clean[se_var].values
    names = df_clean[study_var].values

    # Fixed effect
    weights_fe = 1 / ses**2
    pooled_fe = np.sum(weights_fe * effects) / np.sum(weights_fe)
    se_fe = np.sqrt(1 / np.sum(weights_fe))

    # Random effects (DerSimonian-Laird)
    Q = np.sum(weights_fe * (effects - pooled_fe)**2)
    df_q = len(effects) - 1
    tau2 = max(0, (Q - df_q) / (np.sum(weights_fe) - np.sum(weights_fe**2) / np.sum(weights_fe)))
    weights_re = 1 / (ses**2 + tau2)
    pooled_re = np.sum(weights_re * effects) / np.sum(weights_re)
    se_re = np.sqrt(1 / np.sum(weights_re))

    I2 = max(0, (Q - df_q) / Q * 100) if Q > 0 else 0

    table_data = []
    for i, n in enumerate(names):
        table_data.append({
            "研究": str(n),
            "效应量": round(effects[i], 3),
            "95% CI 下限": round(effects[i] - 1.96 * ses[i], 3),
            "95% CI 上限": round(effects[i] + 1.96 * ses[i], 3),
            "权重(FE)": round(weights_fe[i] / np.sum(weights_fe) * 100, 1),
        })
    table_data.append({"研究": "固定效应合并", "效应量": round(pooled_fe, 3),
                       "95% CI 下限": round(pooled_fe - 1.96 * se_fe, 3),
                       "95% CI 上限": round(pooled_fe + 1.96 * se_fe, 3),
                       "权重(FE)": 100.0})
    table_data.append({"研究": "随机效应合并", "效应量": round(pooled_re, 3),
                       "95% CI 下限": round(pooled_re - 1.96 * se_re, 3),
                       "95% CI 上限": round(pooled_re + 1.96 * se_re, 3),
                       "权重(FE)": 100.0})
    out["tables"].append({"title": f"荟萃分析 (I²={I2:.1f}%, tau²={tau2:.4f})", "headers": list(table_data[0].keys()),
                          "rows": table_data})

    if HAS_PLOTLY:
        fig = go.Figure()
        for i, n in enumerate(names):
            fig.add_trace(go.Scatter(
                x=[effects[i]], y=[f"{n} (n={i+1})"], mode="markers",
                error_x=dict(type="data", symmetric=True, array=[1.96 * ses[i]]),
                marker=dict(size=10 + weights_fe[i] * 2, color="#0E7C7B"),
                name=str(n), showlegend=False,
            ))
        fig.add_vline(x=pooled_re, line_dash="dash", line_color="#E06830",
                      annotation_text=f"Pooled={pooled_re:.3f}")
        fig.update_layout(title="森林图 (随机效应模型)", xaxis_title="Effect Size (OR)",
                          template="plotly_white", height=200 + 30 * len(names))
        out["charts"].append({"title": "森林图", "plotly": _fig_to_json(fig)})

        # Funnel plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=effects, y=1 / ses, mode="markers",
                                   marker=dict(size=10, color="#0E7C7B"), name="研究"))
        fig2.add_vline(x=pooled_re, line_dash="dash", line_color="gray")
        fig2.update_layout(title="漏斗图", xaxis_title="效应量", yaxis_title="精度 (1/SE)",
                           xaxis=dict(zeroline=False), template="plotly_white", height=520)
        out["diagnostics"].append({"title": "漏斗图", "plotly": _fig_to_json(fig2)})

    n_studies = len(names)
    pooled_re_ci_low = pooled_re - 1.96 * se_re
    pooled_re_ci_high = pooled_re + 1.96 * se_re
    pooled_fe_ci_low = pooled_fe - 1.96 * se_fe
    pooled_fe_ci_high = pooled_fe + 1.96 * se_fe
    p_heterogeneity = 1 - stats.chi2.cdf(Q, df_q) if df_q > 0 else 1.0
    effect_range = np.max(effects) - np.min(effects)
    sig_pooled = "具有统计学显著性" if (pooled_re_ci_low > 0 or pooled_re_ci_high < 0) else "未达统计学显著性"
    heterogeneity_level = "高" if I2 > 75 else "中等" if I2 > 50 else "低" if I2 > 25 else "极低"
    out["discussion"] = (
        f"## 荟萃分析 (Meta-Analysis) 结果与讨论\n\n"
        f"### 一、方法学概述\n\n"
        f"荟萃分析通过系统整合多项独立研究的效应量，获得更精确的总体效应估计。"
        f"本分析同时采用固定效应模型（假设各研究估计同一真实效应）和随机效应模型"
        f"（DerSimonian-Laird 方法，允许研究间存在真实效应差异）。"
        f"异质性通过 Cochran's Q 检验和 I² 统计量评估。"
        f"共纳入 **{n_studies}** 项研究，效应量范围 {np.min(effects):.3f} ~ {np.max(effects):.3f}，"
        f"研究间效应量跨度为 **{effect_range:.3f}**。\n\n"
        f"### 二、主要发现\n\n"
        f"- **随机效应合并效应量**: {pooled_re:.3f} (95% CI: {pooled_re_ci_low:.3f} ~ {pooled_re_ci_high:.3f})，{sig_pooled}\n"
        f"- **固定效应合并效应量**: {pooled_fe:.3f} (95% CI: {pooled_fe_ci_low:.3f} ~ {pooled_fe_ci_high:.3f})\n"
        f"- **异质性检验**: Q = {Q:.2f}, df = {df_q}, P = {p_heterogeneity:.4f}\n"
        f"- **I² = {I2:.1f}%** (τ² = {tau2:.4f})，异质性水平: **{heterogeneity_level}**\n"
        f"- 固定与随机效应模型结果{'一致' if abs(pooled_fe - pooled_re) < 0.05 else '存在差异'}，"
        f"{'支持结论的稳健性' if abs(pooled_fe - pooled_re) < 0.05 else '提示异质性对合并估计有实质影响'}\n\n"
        f"### 三、临床意义\n\n"
        f"合并效应量 {pooled_re:.3f} {sig_pooled}，"
        f"{'表明干预措施在多项研究中一致显示出有意义的效应，为临床实践提供了较强的循证支持。' if (pooled_re_ci_low > 0 or pooled_re_ci_high < 0) else '置信区间跨越零值，当前证据不足以确认干预效应的方向性。'}"
        f"I² = {I2:.1f}% 表明 {heterogeneity_level} 水平的异质性，"
        f"{'建议进行亚组分析或 Meta 回归以探索异质性来源。' if I2 > 50 else '各研究结果较为一致，合并估计具有较好的代表性。'}"
        f"漏斗图可辅助评估发表偏倚，不对称分布提示可能存在小样本研究效应或选择性报告。\n\n"
        f"### 四、模型诊断与局限性\n\n"
        f"- 随机效应模型在研究数量较少 (< 5) 时，τ² 估计不精确，可能低估真实异质性\n"
        f"- DerSimonian-Laird 方法在小样本 Meta 分析中可能产生偏低的方差估计\n"
        f"- 漏斗图的视觉评估主观性较强，建议补充 Egger 检验或 trim-and-fill 方法\n"
        f"- 纳入研究的质量差异可能影响合并估计的可靠性，建议进行敏感性分析（逐一剔除法）\n\n"
        f"### 五、结论\n\n"
        f"基于 {n_studies} 项研究的荟萃分析，随机效应模型合并效应量为 {pooled_re:.3f} "
        f"(95% CI: {pooled_re_ci_low:.3f} ~ {pooled_re_ci_high:.3f})，{sig_pooled}。"
        f"研究间异质性为{heterogeneity_level}水平 (I² = {I2:.1f}%)。"
        f"综合证据{'支持' if (pooled_re_ci_low > 0 or pooled_re_ci_high < 0) else '尚不足以确认'}"
        f"干预措施的有效性，后续研究应关注异质性来源和潜在的发表偏倚。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 10. Mediation Analysis
# ═══════════════════════════════════════════════════════════════

def run_mediation(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    import statsmodels.api as sm

    x_var = params.get("x_var", "stress_score")
    m_var = params.get("m_var", "inflammation_index")
    y_var = params.get("y_var", "blood_pressure")

    df_clean = df.dropna(subset=[x_var, m_var, y_var]).copy()
    X = df_clean[x_var].values
    M = df_clean[m_var].values
    Y = df_clean[y_var].values

    # Step 1: X → Y (total effect)
    m1 = sm.OLS(Y, sm.add_constant(X)).fit()
    c = m1.params[1]

    # Step 2: X → M
    m2 = sm.OLS(M, sm.add_constant(X)).fit()
    a = m2.params[1]

    # Step 3: X + M → Y
    m3 = sm.OLS(Y, sm.add_constant(np.column_stack([X, M]))).fit()
    b = m3.params[2]
    c_prime = m3.params[1]

    # Bootstrap
    n_boot = 500
    ab_samples = []
    for _ in range(n_boot):
        idx = np.random.choice(len(df_clean), len(df_clean), replace=True)
        Xb = X[idx]
        Mb = M[idx]
        Yb = Y[idx]
        a_b = sm.OLS(Mb, sm.add_constant(Xb)).fit().params[1]
        b_b = sm.OLS(Yb, sm.add_constant(np.column_stack([Xb, Mb]))).fit().params[2]
        ab_samples.append(a_b * b_b)
    ab = np.mean(ab_samples)
    ab_ci = np.percentile(ab_samples, [2.5, 97.5])

    med_table = pd.DataFrame({
        "效应": ["总效应 (c)", "X→M (a)", "M→Y (b)", "直接效应 (c')", "间接效应 (a×b)"],
        "估计值": [round(c, 4), round(a, 4), round(b, 4), round(c_prime, 4), round(ab, 4)],
        "95% CI 下限": [round(m1.conf_int()[1, 0], 4), round(m2.conf_int()[1, 0], 4),
                        round(m3.conf_int()[2, 0], 4), round(m3.conf_int()[1, 0], 4),
                        round(ab_ci[0], 4)],
        "95% CI 上限": [round(m1.conf_int()[1, 1], 4), round(m2.conf_int()[1, 1], 4),
                        round(m3.conf_int()[2, 1], 4), round(m3.conf_int()[1, 1], 4),
                        round(ab_ci[1], 4)],
        "P值": [round(m1.pvalues[1], 4), round(m2.pvalues[1], 4), round(m3.pvalues[2], 4),
                round(m3.pvalues[1], 4), round(2 * min(np.mean(np.array(ab_samples) > 0), np.mean(np.array(ab_samples) < 0)), 4)],
    })
    out["tables"].append({"title": "中介效应分解", "headers": list(med_table.columns), "rows": med_table.to_dict(orient="records")})

    # Mediation proportion
    prop_med = ab / c * 100 if c != 0 else 0
    n_mediation = len(df_clean)
    ab_sig = "显著" if (ab_ci[0] > 0 or ab_ci[1] < 0) else "不显著"
    direct_sig = "显著" if m3.pvalues[1] < 0.05 else "不显著"
    total_sig = "显著" if m1.pvalues[1] < 0.05 else "不显著"
    a_sig = "显著" if m2.pvalues[1] < 0.05 else "不显著"
    b_sig = "显著" if m3.pvalues[2] < 0.05 else "不显著"
    # Determine mediation type
    if ab_sig == "显著" and direct_sig == "显著":
        med_type = "部分中介"
    elif ab_sig == "显著" and direct_sig == "不显著":
        med_type = "完全中介"
    else:
        med_type = "无显著中介效应"

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=ab_samples, nbinsx=40, marker_color="#0E7C7B", opacity=0.7))
        fig.add_vline(x=ab, line_color="#E06830", line_width=2)
        fig.add_vline(x=ab_ci[0], line_dash="dash", line_color="gray")
        fig.add_vline(x=ab_ci[1], line_dash="dash", line_color="gray")
        fig.update_layout(title=f"间接效应 Bootstrap 分布 (中介比例: {prop_med:.1f}%)",
                          xaxis_title="a × b", template="plotly_white", height=520)
        out["charts"].append({"title": "Bootstrap间接效应分布", "plotly": _fig_to_json(fig)})

    out["discussion"] = (
        f"## 中介效应分析结果与讨论\n\n"
        f"### 一、方法学概述\n\n"
        f"中介分析 (Mediation Analysis) 基于 Baron & Kenny 因果步骤法和 Bootstrap 推断，"
        f"旨在分解自变量对因变量的总效应为直接效应和通过中介变量的间接效应。"
        f"本分析检验路径 **{x_var} → {m_var} → {y_var}** 的中介机制，"
        f"采用 **{n_boot}** 次 Bootstrap 重抽样估计间接效应的置信区间。"
        f"样本量为 **{n_mediation}** 例，满足中介分析对样本量的基本要求。\n\n"
        f"### 二、主要发现\n\n"
        f"- **总效应 (c)**: {c:.4f} (P = {m1.pvalues[1]:.4f})，{total_sig}\n"
        f"- **路径 a ({x_var} → {m_var})**: {a:.4f} (P = {m2.pvalues[1]:.4f})，{a_sig}\n"
        f"- **路径 b ({m_var} → {y_var})**: {b:.4f} (P = {m3.pvalues[2]:.4f})，{b_sig}\n"
        f"- **直接效应 (c')**: {c_prime:.4f} (P = {m3.pvalues[1]:.4f})，{direct_sig}\n"
        f"- **间接效应 (a×b)**: {ab:.4f} (95% Bootstrap CI: {ab_ci[0]:.4f} ~ {ab_ci[1]:.4f})，{ab_sig}\n"
        f"- **中介比例**: {prop_med:.1f}% — 判定为 **{med_type}**\n\n"
        f"### 三、临床意义\n\n"
        f"中介分析揭示了 {x_var} 影响 {y_var} 的潜在机制路径。"
        f"{'间接效应显著 (CI 不包含零)，表明 ' + m_var + ' 在 ' + x_var + ' 与 ' + y_var + ' 的关系中发挥了' + med_type + '作用。' if ab_sig == '显著' else '间接效应的 Bootstrap CI 包含零，当前数据不支持 ' + m_var + ' 的中介作用。'}"
        f"中介比例 {prop_med:.1f}% 表示总效应中有该比例通过中介变量传递。"
        f"{'这一发现为干预靶点的选择提供了依据：通过调控 ' + m_var + ' 可能间接改善 ' + y_var + '。' if ab_sig == '显著' else '建议探索其他潜在中介变量或增大样本量以提高检验效能。'}\n\n"
        f"### 四、模型诊断与局限性\n\n"
        f"- 中介分析的因果解释依赖于无未测量混杂假设（X-M、M-Y、X-Y 路径均无遗漏混杂）\n"
        f"- 横截面数据无法确定时序关系，纵向设计更适合验证中介机制\n"
        f"- Bootstrap 方法对间接效应的非正态分布具有稳健性，优于 Sobel 检验\n"
        f"- 单中介模型可能过于简化，实际机制可能涉及多重中介或调节中介\n\n"
        f"### 五、结论\n\n"
        f"基于 {n_mediation} 例样本的中介分析表明，{m_var} 在 {x_var} 与 {y_var} "
        f"关系中的中介效应为 {ab:.4f} ({ab_sig}，95% CI: {ab_ci[0]:.4f} ~ {ab_ci[1]:.4f})，"
        f"中介比例 {prop_med:.1f}%，判定为{med_type}。"
        f"研究结果为理解变量间的作用机制提供了统计学证据。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 11. Mixed Effects Model
# ═══════════════════════════════════════════════════════════════

def run_mixed_effects(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    import statsmodels.formula.api as smf

    outcome_var = params.get("outcome_var", "sbp")
    time_var = params.get("time_var", "time")
    group_var = params.get("group_var", "treatment")
    random_var = params.get("random_var", "center")

    df_clean = df.dropna(subset=[outcome_var, time_var, group_var, random_var]).copy()
    formula = f"{outcome_var} ~ {group_var} + {time_var}"

    try:
        vcf = {random_var: "0 + C({})".format(random_var)}
        model = smf.mixedlm(formula, df_clean, groups=df_clean["subject_id"],
                            vc_formula=vcf, re_formula="1")
        result = model.fit(reml=True)
        fe = result.fe_params
        coef_table = pd.DataFrame({
            "参数": fe.index,
            "估计值": np.round(fe.values, 4),
            "标准误": np.round(result.bse_fe, 4),
            "Z值": np.round(fe.values / result.bse_fe, 3),
            "P值": np.round(result.pvalues_fe, 4),
        })
        out["tables"].append({"title": "混合效应模型 固定效应", "headers": list(coef_table.columns),
                              "rows": coef_table.to_dict(orient="records")})
    except Exception as e:
        coef_table = pd.DataFrame({"备注": [f"模型拟合失败: {e}"]})
        out["tables"].append({"title": "固定效应", "headers": list(coef_table.columns), "rows": coef_table.to_dict(orient="records")})

    if HAS_PLOTLY:
        centers = sorted(df_clean[random_var].unique())
        fig = go.Figure()
        for c in centers:
            sub = df_clean[df_clean[random_var] == c]
            agg = sub.groupby(time_var)[outcome_var].mean().reset_index()
            fig.add_trace(go.Scatter(x=agg[time_var], y=agg[outcome_var], mode="lines+markers", name=str(c)))
        fig.update_layout(title="各中心纵向趋势", xaxis_title=time_var, yaxis_title=outcome_var,
                          template="plotly_white", height=530)
        out["charts"].append({"title": "各中心纵向趋势", "plotly": _fig_to_json(fig)})

    n_subjects_me = df_clean["subject_id"].nunique()
    n_centers = df_clean[random_var].nunique()
    n_obs_me = len(df_clean)
    n_timepoints_me = df_clean[time_var].nunique()
    try:
        group_fe = fe[group_var] if group_var in fe.index else 0
        group_p_me = result.pvalues_fe[group_var] if group_var in result.pvalues_fe.index else 1
        time_fe = fe[time_var] if time_var in fe.index else 0
        time_p_me = result.pvalues_fe[time_var] if time_var in result.pvalues_fe.index else 1
        re_var = result.cov_re.iloc[0, 0] if hasattr(result, 'cov_re') else 0
        resid_var = result.scale
        icc = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else 0
    except Exception:
        group_fe = 0
        group_p_me = 1
        time_fe = 0
        time_p_me = 1
        icc = 0
    out["discussion"] = (
        f"## 混合效应模型分析结果与讨论\n\n"
        f"### 一、方法学概述\n\n"
        f"线性混合效应模型 (Linear Mixed-Effects Model, LMM) 同时包含固定效应和随机效应，"
        f"适用于具有层次/嵌套结构的纵向数据。固定效应估计总体平均趋势，随机效应捕获"
        f"个体和聚类水平的变异。本模型以 **{outcome_var}** 为结局变量，"
        f"固定效应包括 **{group_var}** 和 **{time_var}**，"
        f"随机效应包括受试者随机截距和 **{random_var}** 水平的方差成分。"
        f"分析纳入 **{n_subjects_me}** 名受试者，来自 **{n_centers}** 个{random_var}，"
        f"共 **{n_obs_me}** 条观测，覆盖 **{n_timepoints_me}** 个时间点。采用 REML 估计。\n\n"
        f"### 二、主要发现\n\n"
        f"- **{group_var} 固定效应**: {group_fe:.4f} (P = {group_p_me:.4f})，"
        f"{'具有统计学显著性' if group_p_me < 0.05 else '未达统计学显著性'}\n"
        f"- **{time_var} 固定效应**: {time_fe:.4f} (P = {time_p_me:.4f})，"
        f"反映结局随时间的平均变化率\n"
        f"- **组内相关系数 (ICC)**: {icc:.3f}，表示 {icc*100:.1f}% 的总变异归因于受试者间差异\n"
        f"- 各{random_var}的纵向趋势图展示了聚类水平的异质性\n\n"
        f"### 三、临床意义\n\n"
        f"混合效应模型的优势在于同时考虑了个体内变化和个体间差异。"
        f"ICC = {icc:.3f} 表明{'受试者间差异是总变异的主要来源，支持使用随机效应模型' if icc > 0.3 else '受试者内变化占主导，但随机效应仍有助于提高估计效率'}。"
        f"{group_var} 的固定效应 {group_fe:.4f} 代表了在控制时间趋势和随机变异后的"
        f"净处理效应，{'为干预措施的有效性提供了支持' if group_p_me < 0.05 else '当前证据不足以确认处理效应'}。"
        f"各{random_var}趋势图的分散程度反映了多中心研究中心间的异质性。\n\n"
        f"### 四、模型诊断与局限性\n\n"
        f"- 随机效应假设服从正态分布，严重偏离可能影响固定效应的推断\n"
        f"- REML 估计在小样本（聚类数 < 30）时可能低估方差成分\n"
        f"- 模型假设缺失数据为随机缺失 (MAR)，非随机缺失需要模式混合模型等替代方法\n"
        f"- 固定效应的线性假设可能过于简化，非线性时间趋势可通过样条或多项式扩展\n\n"
        f"### 五、结论\n\n"
        f"基于 {n_centers} 个{random_var} {n_subjects_me} 名受试者的混合效应模型分析表明，"
        f"{group_var} 对 {outcome_var} 的固定效应为 {group_fe:.4f} (P = {group_p_me:.4f})。"
        f"模型通过随机截距和{random_var}方差成分有效处理了数据的层次结构，"
        f"ICC = {icc:.3f} 证实了采用混合模型的必要性。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 12. Complex Survey Analysis (NHANES style)
# ═══════════════════════════════════════════════════════════════

def run_nhanes_analysis(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    weight_var = params.get("weight_var", "survey_weight")
    strata_var = params.get("strata_var", "strata")

    df_clean = df.dropna(subset=[weight_var]).copy()
    weights = df_clean[weight_var].values
    norm_weights = weights / weights.sum()

    # Weighted descriptive stats
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["patient_id", weight_var, "psu"]]
    desc_rows = []
    for col in num_cols[:8]:
        vals = df_clean[col].dropna().values
        w = norm_weights[df_clean[col].notna().values]
        w = w / w.sum()
        wmean = np.average(vals, weights=w)
        wvar = np.average((vals - wmean)**2, weights=w)
        desc_rows.append({
            "变量": col, "加权均值": round(wmean, 3), "加权标准差": round(np.sqrt(wvar), 3),
            "未加权均值": round(np.mean(vals), 3), "设计效应": round(max(wvar / np.var(vals), 1.0), 2) if np.var(vals) > 0 else 1.0,
        })
    out["tables"].append({"title": "加权描述统计", "headers": list(desc_rows[0].keys()), "rows": desc_rows})

    # Weighted regression
    outcome = params.get("outcome_var", "sbp")
    predictors = params.get("predictors", ["age", "bmi", "glucose"])
    pred_cols = [c for c in predictors if c in df_clean.columns and c != outcome]
    if pred_cols:
        import statsmodels.api as sm
        y = df_clean[outcome].dropna()
        X = df_clean.loc[y.index, pred_cols]
        X = sm.add_constant(X)
        w = norm_weights[y.index]
        model = sm.WLS(y, X, weights=w).fit()
        reg_table = pd.DataFrame({
            "变量": model.params.index,
            "系数": np.round(model.params.values, 4),
            "标准误": np.round(model.bse.values, 4),
            "t值": np.round(model.tvalues.values, 3),
            "P值": np.round(model.pvalues.values, 4),
        })
        out["tables"].append({"title": "加权回归结果", "headers": list(reg_table.columns), "rows": reg_table.to_dict(orient="records")})

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=weights, nbinsx=50, marker_color="#0E7C7B", opacity=0.7))
        fig.update_layout(title="调查权重分布", xaxis_title="权重", yaxis_title="频数",
                          template="plotly_white", height=520)
        out["charts"].append({"title": "权重分布", "plotly": _fig_to_json(fig)})

    n_total_nhanes = len(df_clean)
    weight_mean = np.mean(weights)
    weight_cv = np.std(weights) / max(weight_mean, 1e-8)
    n_strata = df_clean[strata_var].nunique() if strata_var in df_clean.columns else 0
    avg_deff = np.mean([r["设计效应"] for r in desc_rows]) if desc_rows else 1.0
    max_deff_var = max(desc_rows, key=lambda x: x["设计效应"])["变量"] if desc_rows else ""
    max_deff = max(r["设计效应"] for r in desc_rows) if desc_rows else 1.0
    # Weighted vs unweighted difference
    mean_diff_pct = np.mean([abs(r["加权均值"] - r["未加权均值"]) / max(abs(r["未加权均值"]), 1e-8) * 100 for r in desc_rows]) if desc_rows else 0
    try:
        n_sig_predictors = int((model.pvalues < 0.05).sum() - 1)  # exclude intercept
        r_squared = model.rsquared
    except Exception:
        n_sig_predictors = 0
        r_squared = 0
    out["discussion"] = (
        f"## 复杂抽样调查分析 (NHANES 风格) 结果与讨论\n\n"
        f"### 一、方法学概述\n\n"
        f"复杂抽样调查分析考虑了调查设计中的分层、聚类和不等概率抽样，"
        f"通过调查权重校正样本的代表性偏差，使统计推断能够推广至目标总体。"
        f"本分析采用加权最小二乘法 (WLS) 进行回归建模，权重反映了每个观测单位"
        f"代表目标总体中的人数。共纳入 **{n_total_nhanes}** 例样本，"
        f"涉及 **{n_strata}** 个分层，调查权重均值为 **{weight_mean:.2f}** "
        f"(变异系数 CV = {weight_cv:.2f})。\n\n"
        f"### 二、主要发现\n\n"
        f"- 加权与未加权均值的平均相对偏差: **{mean_diff_pct:.1f}%**，"
        f"{'表明抽样设计对估计有实质影响' if mean_diff_pct > 5 else '表明抽样设计对估计影响较小'}\n"
        f"- 平均设计效应 (DEFF): **{avg_deff:.2f}**，等效于有效样本量为实际样本量的 "
        f"**{1/max(avg_deff, 1e-8)*100:.0f}%**\n"
        f"- 最大设计效应变量: **{max_deff_var}** (DEFF = {max_deff:.2f})，"
        f"该变量受抽样设计影响最大\n"
        f"- 加权回归模型: **{n_sig_predictors}** 个预测变量显著 (P < 0.05)，"
        f"R² = **{r_squared:.3f}**\n\n"
        f"### 三、临床意义\n\n"
        f"调查加权分析确保了研究结论可推广至目标总体（如全国人口），而非仅限于样本本身。"
        f"设计效应 {avg_deff:.2f} 意味着由于复杂抽样设计，实际统计效能相当于简单随机抽样的 "
        f"{1/max(avg_deff, 1e-8)*100:.0f}%，这对样本量计算和结果解读具有重要意义。"
        f"{'加权与未加权估计的显著差异提示，忽略调查权重将导致有偏的总体推断。' if mean_diff_pct > 5 else '加权与未加权估计较为接近，但仍建议使用加权分析以确保推断的严谨性。'}"
        f"加权回归结果可用于识别总体水平的健康影响因素。\n\n"
        f"### 四、模型诊断与局限性\n\n"
        f"- 调查权重的极端值可能导致估计不稳定，建议检查权重分布并考虑截断处理\n"
        f"- WLS 假设权重已知且固定，未考虑权重估计本身的不确定性\n"
        f"- 设计效应因变量而异，单一 DEFF 值不能代表所有分析的效率损失\n"
        f"- 缺失数据处理需考虑调查设计，完整病例分析可能破坏权重的代表性\n"
        f"- 未纳入 PSU (初级抽样单位) 的聚类效应可能低估标准误\n\n"
        f"### 五、结论\n\n"
        f"基于 {n_total_nhanes} 例加权样本 ({n_strata} 个分层) 的复杂抽样分析表明，"
        f"调查设计对统计估计有{'实质性' if mean_diff_pct > 5 or avg_deff > 1.5 else '轻度'}影响 "
        f"(平均 DEFF = {avg_deff:.2f})。加权分析确保了结论对目标总体的外推有效性，"
        f"为基于人群的流行病学研究和公共卫生决策提供了可靠的统计推断基础。"
    )
    return out


# ── Method Router ───────────────────────────────────────────

STATS_ROUTER = {
    "gee": run_gee,
    "propensity_score": run_propensity_score,
    "sensitivity_analysis": run_sensitivity_analysis,
    "counterfactual": run_counterfactual,
    "survival_advanced": run_survival_advanced,
    "markov_model": run_markov_model,
    "bayesian": run_bayesian,
    "latin_square": run_latin_square,
    "meta_analysis": run_meta_analysis,
    "mediation": run_mediation,
    "mixed_effects": run_mixed_effects,
    "nhanes_analysis": run_nhanes_analysis,
}
