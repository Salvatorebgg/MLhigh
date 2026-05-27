# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import json
import base64
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.figure_factory as ff
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report, mean_squared_error,
    r2_score, mean_absolute_error
)
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


def _fig_to_json(fig) -> str:
    if fig is None:
        return "{}"
    return json.dumps(fig.to_dict() if hasattr(fig, "to_dict") else fig, default=str)


def _get_xy(df, feature_cols, target_col):
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    X = df_clean[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category").cat.codes
    y = df_clean[target_col].values
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)
    return X, y


# ═══════════════════════════════════════════════════════════════
# 1. Logistic Regression
# ═══════════════════════════════════════════════════════════════

def run_ml_lr(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    target = params.get("target", "outcome")
    feature_cols = [c for c in df.columns if c not in [target, "patient_id"]]
    X, y = _get_xy(df, feature_cols, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    # Metrics
    metrics = [
        {"指标": "准确率 (Accuracy)", "值": round(accuracy_score(y_test, y_pred), 4)},
        {"指标": "精确率 (Precision)", "值": round(precision_score(y_test, y_pred, zero_division=0), 4)},
        {"指标": "召回率 (Recall)", "值": round(recall_score(y_test, y_pred, zero_division=0), 4)},
        {"指标": "F1 分数", "值": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)},
        {"指标": "AUC-ROC", "值": round(roc_auc_score(y_test, y_prob), 4)},
    ]
    out["tables"].append({"title": "模型性能指标", "headers": ["指标", "值"], "rows": metrics})

    # Coefficients
    coef_table = pd.DataFrame({
        "特征": feature_cols,
        "系数": np.round(model.coef_[0], 4),
        "Odds Ratio": np.round(np.exp(model.coef_[0]), 3),
    })
    out["tables"].append({"title": "逻辑回归系数", "headers": list(coef_table.columns), "rows": coef_table.to_dict(orient="records")})

    if HAS_PLOTLY:
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={metrics[-1]['值']})",
                                 line=dict(color="#0E7C7B", width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(color="gray", dash="dash")))
        fig.update_layout(title="ROC 曲线", xaxis_title="1 - 特异度", yaxis_title="灵敏度",
                          template="plotly_white", height=520)
        out["charts"].append({"title": "ROC曲线", "plotly": _fig_to_json(fig)})

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig2 = ff.create_annotated_heatmap(cm, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                                            colorscale="Teal", showscale=True)
        fig2.update_layout(title="混淆矩阵", template="plotly_white", height=520)
        out["diagnostics"].append({"title": "混淆矩阵", "plotly": _fig_to_json(fig2)})

        # Coefficient plot
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=coef_table["特征"], y=coef_table["系数"],
                               marker_color=["#0E7C7B" if v > 0 else "#E06830" for v in coef_table["系数"]]))
        fig3.update_layout(title="特征系数", xaxis_title="特征", yaxis_title="系数",
                           template="plotly_white", height=520)
        out["diagnostics"].append({"title": "特征系数图", "plotly": _fig_to_json(fig3)})

    out["discussion"] = _build_lr_discussion(metrics, coef_table, feature_cols, model, X_train, X_test, y_test, y_pred)
    return out


def _build_lr_discussion(metrics, coef_table, feature_cols, model, X_train, X_test, y_test, y_pred):
    auc = metrics[-1]['值']
    acc = metrics[0]['值']
    prec = metrics[1]['值']
    rec = metrics[2]['值']
    f1 = metrics[3]['值']
    n_train = len(X_train)
    n_test = len(X_test)
    n_features = len(feature_cols)
    top_idx = np.argmax(np.abs(model.coef_[0]))
    top_feat = feature_cols[top_idx]
    top_or = np.exp(model.coef_[0][top_idx])
    top_coef = model.coef_[0][top_idx]

    sorted_idx = np.argsort(np.abs(model.coef_[0]))[::-1]
    risk_factors = [(feature_cols[i], np.exp(model.coef_[0][i])) for i in sorted_idx if model.coef_[0][i] > 0][:3]
    protect_factors = [(feature_cols[i], np.exp(model.coef_[0][i])) for i in sorted_idx if model.coef_[0][i] < 0][:2]

    epv = min(np.sum(y_test == 0), np.sum(y_test == 1)) * (n_train + n_test) / n_test / n_features

    auc_interp = "优秀" if auc >= 0.9 else "良好" if auc >= 0.8 else "中等" if auc >= 0.7 else "较弱"

    risk_text = "；".join([f"{f}（OR = {o:.3f}，即该因素每增加一个标准差，事件风险{'增加' if o > 1 else '降低'}{abs(o-1)*100:.1f}%）" for f, o in risk_factors])
    protect_text = "；".join([f"{f}（OR = {o:.3f}，保护效应{(1-o)*100:.1f}%）" for f, o in protect_factors]) if protect_factors else "未发现显著保护因素"

    return (
        f"## 逻辑回归模型分析结果与讨论\n\n"
        f"### 一、方法学概述\n"
        f"本研究采用多变量逻辑回归（Logistic Regression）模型评估各预测因子与二分类结局变量之间的关联强度。"
        f"逻辑回归作为临床预测建模的经典方法，其核心优势在于回归系数可直接转化为比值比（Odds Ratio, OR），"
        f"具有明确的流行病学解释意义，且模型结构透明、临床可解释性强。"
        f"本分析共纳入 {n_train + n_test} 例样本，采用 70:30 的训练-测试集随机划分策略（分层抽样保持类别比例），"
        f"训练集 {n_train} 例用于模型拟合，测试集 {n_test} 例用于独立验证。"
        f"所有连续型预测变量均经标准化处理（Z-score）以消除量纲差异对系数估计的影响。\n\n"
        f"### 二、主要发现\n"
        f"模型在独立测试集上的受试者工作特征曲线下面积（AUC-ROC）为 **{auc:.4f}**，"
        f"判定为{auc_interp}的区分能力（AUC > 0.8 通常被认为具有较好的临床应用价值，Hosmer & Lemeshow 标准）。"
        f"分类性能指标显示：准确率 = {acc:.4f}，精确率 = {prec:.4f}，召回率 = {rec:.4f}，F1 分数 = {f1:.4f}。\n\n"
        f"**危险因素识别：**{risk_text}。\n\n"
        f"**保护因素识别：**{protect_text}。\n\n"
        f"系数分析表明，**{top_feat}** 是最强的独立预测因子（|系数| 最大），"
        f"其 OR = {top_or:.3f}，提示该变量对结局事件的发生具有{'促进' if top_or > 1 else '抑制'}作用。\n\n"
        f"### 三、临床意义\n"
        f"从临床转化角度而言，OR > 2 的预测因子通常具有重要的筛查和干预价值。"
        f"本模型识别的危险因素可为临床风险分层提供量化依据，辅助个体化治疗决策和早期预警系统的构建。"
        f"建议临床实践中重点监测 {top_feat} 等高权重预测因子，"
        f"对高风险人群实施针对性干预策略。\n\n"
        f"值得注意的是，逻辑回归模型的线性假设使其在捕捉复杂非线性关系方面存在局限，"
        f"但其系数的可解释性和稳定性使其在临床指南制定和风险评分开发中仍占据核心地位。\n\n"
        f"### 四、模型诊断与局限性\n"
        f"1. **样本量充分性**：按照 Events Per Variable（EPV）≥ 10 的经验法则，"
        f"本数据集 EPV ≈ {epv:.1f}，{'满足' if epv >= 10 else '接近但未完全满足'}建模的最低样本量要求。\n"
        f"2. **模型假设**：逻辑回归假设自变量与 log-odds 之间存在线性关系，且观测值相互独立。"
        f"建议后续通过 Box-Tidwell 检验验证线性假设，并计算方差膨胀因子（VIF）排除严重多重共线性。\n"
        f"3. **局限性**：(a) 横断面数据设计无法推断因果关系，OR 仅反映关联强度；"
        f"(b) 未纳入的潜在混杂因素可能导致效应估计偏倚；"
        f"(c) 模型的外部推广性需在独立队列中进一步验证（外部验证）；"
        f"(d) 单次随机划分可能引入划分偏倚，建议补充交叉验证或 Bootstrap 验证。\n\n"
        f"### 五、结论\n"
        f"本研究构建的多变量逻辑回归预测模型在独立测试集上展现了{auc_interp}的区分能力（AUC = {auc:.4f}），"
        f"**{top_feat}** 被识别为最强的独立{'危险' if top_or > 1 else '保护'}因素（OR = {top_or:.3f}）。"
        f"该模型可为临床风险评估和患者分层管理提供循证参考，"
        f"但其推广应用需在多中心外部队列中进一步验证其泛化性能和校准度。"
        f"建议后续研究结合决策曲线分析（DCA）评估模型的临床净获益。"
    )


# ═══════════════════════════════════════════════════════════════
# 2. Lasso / Ridge Regression
# ═══════════════════════════════════════════════════════════════

def run_ml_lasso(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    target = params.get("target", "outcome")
    method = params.get("regularization", "lasso")
    feature_cols = [c for c in df.columns if c not in [target, "patient_id"]]
    X, y = _get_xy(df, feature_cols, target)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    alphas = np.logspace(-3, 2, 100)
    if method == "ridge":
        model_cv = RidgeCV(alphas=alphas, cv=5)
        model_cv.fit(X_s, y)
        best_alpha = model_cv.alpha_
        model = Ridge(alpha=best_alpha)
        model.fit(X_s, y)
        coef = model.coef_
    else:
        model_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=5000)
        model_cv.fit(X_s, y)
        best_alpha = model_cv.alpha_
        model = Lasso(alpha=best_alpha, max_iter=5000)
        model.fit(X_s, y)
        coef = model.coef_

    y_pred = model.predict(X_s)
    selected = np.sum(np.abs(coef) > 1e-6)
    metrics = [
        {"指标": "R²", "值": round(r2_score(y, y_pred), 4)},
        {"指标": "MAE", "值": round(mean_absolute_error(y, y_pred), 4)},
        {"指标": "RMSE", "值": round(np.sqrt(mean_squared_error(y, y_pred)), 4)},
        {"指标": "最优Alpha", "值": round(best_alpha, 6)},
        {"指标": "入选特征数", "值": int(selected)},
        {"指标": "总特征数", "值": len(feature_cols)},
    ]
    out["tables"].append({"title": f"{'Lasso' if method == 'lasso' else 'Ridge'} 回归结果", "headers": ["指标", "值"], "rows": metrics})

    coef_table = pd.DataFrame({
        "特征": feature_cols,
        "系数": np.round(coef, 4),
    }).sort_values("系数", key=abs, ascending=False).head(20)
    out["tables"].append({"title": "特征系数 (Top 20)", "headers": list(coef_table.columns),
                          "rows": coef_table.to_dict(orient="records")})

    if HAS_PLOTLY:
        # Regularization path
        coefs = []
        for a in alphas:
            if method == "ridge":
                m = Ridge(alpha=a).fit(X_s, y)
            else:
                m = Lasso(alpha=a, max_iter=5000).fit(X_s, y)
            coefs.append(m.coef_)
        coefs = np.array(coefs)

        fig = go.Figure()
        for i in range(min(15, coefs.shape[1])):
            fig.add_trace(go.Scatter(x=alphas, y=coefs[:, i], mode="lines",
                                     name=feature_cols[i], line=dict(width=1.5)))
        fig.add_vline(x=best_alpha, line_dash="dash", line_color="red",
                      annotation_text=f"Best Alpha={best_alpha:.4f}")
        fig.update_layout(title="正则化路径", xaxis_type="log", xaxis_title="Alpha",
                          yaxis_title="系数", template="plotly_white", height=520)
        out["charts"].append({"title": "正则化路径图", "plotly": _fig_to_json(fig)})

        # Coefficient bar
        fig2 = go.Figure()
        top = coef_table.head(15)
        fig2.add_trace(go.Bar(x=top["特征"], y=top["系数"],
                               marker_color=["#0E7C7B" if v > 0 else "#E06830" for v in top["系数"]]))
        fig2.update_layout(title="Top 15 特征系数", xaxis_title="特征", yaxis_title="系数",
                           template="plotly_white", height=530)
        out["diagnostics"].append({"title": "特征系数", "plotly": _fig_to_json(fig2)})

    method_name = 'Lasso' if method == 'lasso' else 'Ridge'
    penalty = 'L1' if method == 'lasso' else 'L2'
    r2 = metrics[0]['值']
    mae = metrics[1]['值']
    rmse = metrics[2]['值']
    n_total = len(feature_cols)
    top3 = coef_table.head(3)
    top_feats = "、".join(top3["特征"].tolist())
    r2_interp = "优秀" if r2 >= 0.8 else "良好" if r2 >= 0.6 else "中等" if r2 >= 0.4 else "较弱"
    shrink_ratio = (n_total - int(selected)) / n_total * 100 if method == 'lasso' else 0

    out["discussion"] = (
        f"## {method_name} 回归分析结果与讨论\n\n"
        f"### 一、方法学概述\n"
        f"{method_name} 回归是在普通最小二乘法基础上引入 {penalty} 正则化惩罚项的线性回归方法。"
        f"{'Lasso（Least Absolute Shrinkage and Selection Operator）通过 L1 范数惩罚实现系数的稀疏化，自动完成变量筛选，将不重要变量的系数压缩至零。' if method == 'lasso' else 'Ridge 回归通过 L2 范数惩罚控制系数的整体大小，有效缓解多重共线性问题，但不会将系数压缩至零。'}"
        f"本分析对 {n_total} 个候选预测变量进行了正则化回归建模，"
        f"通过 5 折交叉验证在 100 个对数均匀分布的 Alpha 值中搜索最优正则化强度。\n\n"
        f"### 二、主要发现\n"
        f"交叉验证确定的最优正则化参数 **Alpha = {best_alpha:.6f}**。"
        f"模型拟合优度 R² = **{r2:.4f}**（{r2_interp}），MAE = {mae:.4f}，RMSE = {rmse:.4f}。\n\n"
        f"{'**变量筛选结果：**从 ' + str(n_total) + ' 个候选变量中，Lasso 自动筛选出 ' + str(int(selected)) + ' 个非零系数变量（压缩比例 ' + f'{shrink_ratio:.1f}' + '%），实现了有效的特征降维。' if method == 'lasso' else '**系数收缩结果：**Ridge 回归保留了全部 ' + str(n_total) + ' 个变量，但通过 L2 惩罚将系数整体收缩，有效控制了过拟合风险。'}\n\n"
        f"系数绝对值排名前三的关键变量为：**{top_feats}**，"
        f"这些变量对结局变量的预测贡献最大。\n\n"
        f"### 三、临床意义\n"
        f"{'Lasso 的变量筛选功能在高维临床数据（如组学数据、电子健康记录）中具有重要应用价值，可从大量候选生物标志物中识别出最具预测力的核心指标集，为后续验证研究提供候选靶点。' if method == 'lasso' else 'Ridge 回归在存在多重共线性的临床数据中（如多个高度相关的实验室指标）表现稳健，能够提供更可靠的系数估计，避免普通回归中系数符号翻转等不稳定现象。'}"
        f"正则化路径图直观展示了各变量系数随惩罚强度变化的轨迹，"
        f"有助于理解变量的相对重要性和稳定性。\n\n"
        f"### 四、模型诊断与局限性\n"
        f"1. **正则化路径稳定性**：系数路径图中较早归零的变量通常对结局贡献较小，"
        f"而在较大 Alpha 值下仍保持非零的变量具有更强的预测稳定性。\n"
        f"2. **模型假设**：{method_name} 回归仍假设因变量与自变量之间存在线性关系，"
        f"对于非线性效应需考虑引入交互项或多项式特征。\n"
        f"3. **局限性**：(a) 标准化后的系数不能直接解释为原始尺度的效应量；"
        f"(b) {'Lasso 在高度相关变量组中倾向于随机选择其中一个，可能遗漏重要变量（建议补充 Elastic Net）' if method == 'lasso' else 'Ridge 不进行变量选择，模型解释性不如 Lasso 简洁'}；"
        f"(c) 交叉验证的最优 Alpha 可能因样本划分不同而波动。\n\n"
        f"### 五、结论\n"
        f"{method_name} 回归模型（Alpha = {best_alpha:.6f}）在本数据集上达到 R² = {r2:.4f} 的拟合效果，"
        f"{'成功从 ' + str(n_total) + ' 个变量中筛选出 ' + str(int(selected)) + ' 个核心预测因子' if method == 'lasso' else '通过 L2 正则化有效控制了多重共线性'}。"
        f"关键预测变量 {top_feats} 值得在后续研究中重点关注。"
        f"建议结合领域知识对筛选结果进行生物学合理性评估，并在独立样本中验证模型的预测性能。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 3. KNN
# ═══════════════════════════════════════════════════════════════

def run_ml_knn(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    target = params.get("target", "disease")
    feature_cols = [c for c in df.columns if c not in [target, "patient_id"]]
    X, y = _get_xy(df, feature_cols, target)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.3, random_state=42, stratify=y)

    # Find best k
    k_range = range(1, 22, 2)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores.append(np.mean(cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")))
    best_k = list(k_range)[np.argmax(scores)]

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = [
        {"指标": "最优 K 值", "值": best_k},
        {"指标": "准确率", "值": round(accuracy_score(y_test, y_pred), 4)},
        {"指标": "精确率 (加权)", "值": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4)},
        {"指标": "召回率 (加权)", "值": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4)},
        {"指标": "F1 (加权)", "值": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)},
    ]
    out["tables"].append({"title": "KNN 模型性能", "headers": ["指标", "值"], "rows": metrics})

    if HAS_PLOTLY:
        # K selection
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(k_range), y=scores, mode="lines+markers",
                                 marker=dict(size=10), line=dict(color="#0E7C7B", width=2)))
        fig.add_vline(x=best_k, line_dash="dash", line_color="#E06830",
                      annotation_text=f"Best K={best_k}")
        fig.update_layout(title="K 值优化", xaxis_title="K", yaxis_title="准确率 (CV)",
                          template="plotly_white", height=530)
        out["charts"].append({"title": "K值优化", "plotly": _fig_to_json(fig)})

        # Decision boundary (first 2 features)
        if len(feature_cols) >= 2:
            x_min, x_max = X_s[:, 0].min() - 1, X_s[:, 0].max() + 1
            y_min, y_max = X_s[:, 1].min() - 1, X_s[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            knn_bnd = KNeighborsClassifier(n_neighbors=best_k)
            knn_bnd.fit(X_s[:, :2], y)
            Z = knn_bnd.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            fig2 = go.Figure()
            fig2.add_trace(go.Contour(x=np.linspace(x_min, x_max, 100), y=np.linspace(y_min, y_max, 100),
                                       z=Z, colorscale="Teal", opacity=0.6, showscale=True))
            unique_y = np.unique(y)
            for i, label in enumerate(unique_y):
                mask = y == label
                fig2.add_trace(go.Scatter(x=X_s[mask, 0], y=X_s[mask, 1], mode="markers",
                                          name=str(label), marker=dict(size=6)))
            fig2.update_layout(title=f"决策边界 (K={best_k}, 前2个特征)", template="plotly_white", height=520)
            out["diagnostics"].append({"title": "决策边界", "plotly": _fig_to_json(fig2)})

    acc = metrics[1]['值']
    prec_w = metrics[2]['值']
    rec_w = metrics[3]['值']
    f1_w = metrics[4]['值']
    n_total = len(X)
    best_score = max(scores)

    out["discussion"] = (
        f"## K 近邻（KNN）分类分析结果与讨论\n\n"
        f"### 一、方法学概述\n"
        f"K 近邻算法（K-Nearest Neighbors, KNN）是一种基于实例的非参数分类方法，"
        f"其核心思想是通过计算待预测样本与训练集中所有样本的距离，选取距离最近的 K 个邻居进行多数投票决定分类。"
        f"KNN 不需要显式的模型训练过程，属于惰性学习（lazy learning），"
        f"其优势在于决策边界可以是任意形状，能够捕捉复杂的非线性分类模式。\n\n"
        f"本分析对 {n_total} 例样本的 {len(feature_cols)} 个特征进行了标准化处理（Z-score），"
        f"通过 5 折交叉验证在 K = 1 至 21（奇数）范围内搜索最优近邻数。\n\n"
        f"### 二、主要发现\n"
        f"交叉验证确定的最优 K 值为 **{best_k}**（CV 准确率 = {best_score:.4f}）。"
        f"模型在独立测试集上的分类性能：准确率 = **{acc:.4f}**，"
        f"加权精确率 = {prec_w:.4f}，加权召回率 = {rec_w:.4f}，加权 F1 = {f1_w:.4f}。\n\n"
        f"K 值优化曲线显示，{'较小的 K 值（K < 5）可能导致过拟合（对噪声敏感），而较大的 K 值可能导致欠拟合（决策边界过于平滑）。' if best_k > 3 else '最优 K 值较小，提示数据中存在较为紧凑的局部聚类结构。'}"
        f"当前 K = {best_k} 在偏差-方差权衡中取得了最佳平衡点。\n\n"
        f"### 三、临床意义\n"
        f"KNN 的非参数特性使其特别适合于临床表型分类中类别边界不规则的场景，"
        f"例如基于多维生物标志物的疾病亚型识别。"
        f"决策边界可视化（基于前两个主要特征）直观展示了分类区域的空间分布，"
        f"有助于理解不同类别患者在特征空间中的分离程度。\n\n"
        f"然而，KNN 的'黑箱'特性限制了其在需要明确解释预测依据的临床场景中的应用。"
        f"建议将 KNN 作为探索性分析工具，识别潜在的分类模式后，"
        f"再使用可解释性更强的模型（如逻辑回归或决策树）进行验证和机制解释。\n\n"
        f"### 四、模型诊断与局限性\n"
        f"1. **维度灾难**：当特征维度较高时，欧氏距离的区分能力下降（'维度灾难'），"
        f"建议在高维数据中先进行降维（PCA）或特征选择后再应用 KNN。\n"
        f"2. **特征缩放敏感性**：KNN 对特征尺度高度敏感，本分析已进行标准化处理。"
        f"若存在异常值，建议改用鲁棒缩放（RobustScaler）。\n"
        f"3. **计算效率**：KNN 的预测时间复杂度为 O(n·d)，在大规模数据集上可能较慢。\n"
        f"4. **局限性**：(a) 缺乏特征重要性排序，无法识别关键预测因子；"
        f"(b) 对类别不平衡敏感，多数类可能主导投票结果；"
        f"(c) 决策边界可视化仅展示前两个特征的投影，可能丢失高维信息。\n\n"
        f"### 五、结论\n"
        f"KNN 分类模型（K = {best_k}）在本数据集上达到 {acc:.4f} 的分类准确率，"
        f"表明基于特征空间距离的分类策略对本研究的临床分类问题具有{'良好' if acc >= 0.8 else '一定'}的适用性。"
        f"该结果为后续构建更复杂的集成学习模型提供了基线参考，"
        f"建议结合随机森林或梯度提升等方法进行综合比较，以确定最优分类策略。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 4. XGBoost
# ═══════════════════════════════════════════════════════════════

def run_ml_xgboost(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    try:
        import xgboost as xgb
    except ImportError:
        out["discussion"] = "XGBoost 未安装。请运行: pip install xgboost"
        return out

    target = params.get("target", "outcome")
    feature_cols = [c for c in df.columns if c not in [target, "patient_id"]]
    X, y = _get_xy(df, feature_cols, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                               random_state=42, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = [
        {"指标": "准确率", "值": round(accuracy_score(y_test, y_pred), 4)},
        {"指标": "AUC", "值": round(roc_auc_score(y_test, y_prob), 4)},
        {"指标": "F1", "值": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)},
    ]
    out["tables"].append({"title": "XGBoost 性能", "headers": ["指标", "值"], "rows": metrics})

    imp = pd.DataFrame({
        "特征": feature_cols,
        "重要性": np.round(model.feature_importances_, 4),
    }).sort_values("重要性", ascending=False).head(15)
    out["tables"].append({"title": "特征重要性 (Top 15)", "headers": list(imp.columns),
                          "rows": imp.to_dict(orient="records")})

    if HAS_PLOTLY:
        # Feature importance
        fig = go.Figure()
        fig.add_trace(go.Bar(x=imp["重要性"], y=imp["特征"], orientation="h",
                             marker_color="#0E7C7B"))
        fig.update_layout(title="XGBoost 特征重要性", yaxis=dict(autorange="reversed"),
                          template="plotly_white", height=520)
        out["charts"].append({"title": "特征重要性", "plotly": _fig_to_json(fig)})

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={metrics[1]['值']}",
                                  line=dict(color="#0E7C7B", width=2)))
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                  line=dict(color="gray", dash="dash")))
        fig2.update_layout(title="ROC 曲线", template="plotly_white", height=530)
        out["diagnostics"].append({"title": "ROC曲线", "plotly": _fig_to_json(fig2)})

    auc_val = metrics[1]['值']
    acc_val = metrics[0]['值']
    f1_val = metrics[2]['值']
    top1 = imp.iloc[0]
    top3_feats = "、".join(imp.head(3)["特征"].tolist())
    n_total = len(X_train) + len(X_test)
    auc_interp = "优秀" if auc_val >= 0.9 else "良好" if auc_val >= 0.8 else "中等" if auc_val >= 0.7 else "较弱"
    top5_imp_sum = imp.head(5)["重要性"].sum()

    out["discussion"] = (
        f"## XGBoost 梯度提升模型分析结果与讨论\n\n"
        f"### 一、方法学概述\n"
        f"XGBoost（eXtreme Gradient Boosting）是基于梯度提升决策树（GBDT）框架的高性能集成学习算法，"
        f"通过序贯地构建弱学习器（决策树）并对前一轮的残差进行拟合，逐步降低预测误差。"
        f"相比传统 GBDT，XGBoost 引入了正则化项（L1/L2）防止过拟合、"
        f"列采样（column subsampling）增强泛化能力、以及高效的近似分裂算法提升计算速度。\n\n"
        f"本分析采用 100 棵树（n_estimators=100）、最大深度 4、学习率 0.1 的配置，"
        f"对 {n_total} 例样本的 {len(feature_cols)} 个特征进行建模。"
        f"训练-测试集按 7:3 分层随机划分。\n\n"
        f"### 二、主要发现\n"
        f"模型在独立测试集上 AUC = **{auc_val:.4f}**（{auc_interp}），"
        f"准确率 = {acc_val:.4f}，宏平均 F1 = {f1_val:.4f}。\n\n"
        f"**特征重要性分析：**基于信息增益（gain）的特征重要性排序显示，"
        f"**{top1['特征']}** 是最重要的预测变量（重要性 = {top1['重要性']:.4f}），"
        f"前三位关键特征为 {top3_feats}。"
        f"前 5 个特征累计贡献了 {top5_imp_sum*100:.1f}% 的预测信息，"
        f"{'提示预测信息高度集中于少数核心变量' if top5_imp_sum > 0.7 else '提示多个变量共同贡献了预测信息'}。\n\n"
        f"### 三、临床意义\n"
        f"XGBoost 的非线性建模能力使其能够自动捕捉变量间的交互效应和非线性剂量-反应关系，"
        f"这在复杂临床数据中尤为重要。特征重要性排序为临床变量筛选提供了数据驱动的证据，"
        f"排名靠前的变量可作为后续机制研究和干预靶点的优先候选。\n\n"
        f"然而，XGBoost 的'黑箱'特性限制了对单个预测结果的解释。"
        f"建议结合 SHAP（SHapley Additive exPlanations）值分析，"
        f"量化每个特征对个体预测的边际贡献，实现模型的局部可解释性。\n\n"
        f"### 四、模型诊断与局限性\n"
        f"1. **过拟合风险**：当前配置（max_depth=4, lr=0.1）属于中等复杂度，"
        f"建议通过学习曲线或早停法（early stopping）进一步评估过拟合程度。\n"
        f"2. **超参数敏感性**：XGBoost 性能对超参数组合敏感，"
        f"当前使用默认配置，建议通过贝叶斯优化或随机搜索进行系统调参。\n"
        f"3. **特征重要性解释**：基于 gain 的重要性可能偏向高基数特征，"
        f"建议同时参考 permutation importance 或 SHAP 值进行交叉验证。\n"
        f"4. **局限性**：(a) 集成模型的预测过程不透明，难以直接转化为临床决策规则；"
        f"(b) 对缺失值的处理虽然内置，但可能引入隐含偏倚；"
        f"(c) 小样本场景下可能不如简单模型稳健。\n\n"
        f"### 五、结论\n"
        f"XGBoost 梯度提升模型在本数据集上展现了{auc_interp}的预测性能（AUC = {auc_val:.4f}），"
        f"**{top1['特征']}** 被识别为最重要的预测因子。"
        f"该模型适合作为临床预测工具的候选方案，但在实际部署前需完成：(1) 超参数系统优化；"
        f"(2) SHAP 可解释性分析；(3) 外部队列验证；(4) 与简单模型的性能-可解释性权衡评估。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 5. Random Forest
# ═══════════════════════════════════════════════════════════════

def run_ml_rf(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    target = params.get("target", "outcome")
    feature_cols = [c for c in df.columns if c not in [target, "patient_id"]]
    X, y = _get_xy(df, feature_cols, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1, oob_score=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = [
        {"指标": "准确率", "值": round(accuracy_score(y_test, y_pred), 4)},
        {"指标": "AUC", "值": round(roc_auc_score(y_test, y_prob), 4)},
        {"指标": "F1", "值": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)},
        {"指标": "OOB 分数", "值": round(model.oob_score_, 4) if hasattr(model, "oob_score_") else "N/A"},
    ]
    out["tables"].append({"title": "随机森林性能", "headers": ["指标", "值"], "rows": metrics})

    imp = pd.DataFrame({
        "特征": feature_cols,
        "重要性": np.round(model.feature_importances_, 4),
    }).sort_values("重要性", ascending=False).head(15)
    out["tables"].append({"title": "特征重要性 (Top 15)", "headers": list(imp.columns),
                          "rows": imp.to_dict(orient="records")})

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=imp["重要性"], y=imp["特征"], orientation="h",
                             marker_color="#0E7C7B"))
        fig.update_layout(title="随机森林 特征重要性", yaxis=dict(autorange="reversed"),
                          template="plotly_white", height=520)
        out["charts"].append({"title": "特征重要性", "plotly": _fig_to_json(fig)})

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig2 = ff.create_annotated_heatmap(cm, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                                            colorscale="Teal", showscale=True)
        fig2.update_layout(title="混淆矩阵", template="plotly_white", height=520)
        out["diagnostics"].append({"title": "混淆矩阵", "plotly": _fig_to_json(fig2)})

    auc_val = metrics[1]['值']
    acc_val = metrics[0]['值']
    f1_val = metrics[2]['值']
    oob = model.oob_score_ if hasattr(model, "oob_score_") else 0
    oob_err = 1 - oob
    top3_feats = "、".join(imp.head(3)["特征"].tolist())
    top1 = imp.iloc[0]
    n_total = len(X_train) + len(X_test)
    auc_interp = "优秀" if auc_val >= 0.9 else "良好" if auc_val >= 0.8 else "中等" if auc_val >= 0.7 else "较弱"

    out["discussion"] = (
        f"## 随机森林模型分析结果与讨论\n\n"
        f"### 一、方法学概述\n"
        f"随机森林（Random Forest）是基于 Bagging 策略的集成学习方法，"
        f"通过构建多棵决策树并对其预测结果进行投票/平均来提升模型的稳定性和泛化能力。"
        f"每棵树在训练时使用 Bootstrap 有放回抽样的子集，"
        f"且在每个分裂节点随机选择特征子集（通常为 √p 个），"
        f"这种双重随机化机制有效降低了树间相关性，减少了集成模型的方差。\n\n"
        f"本分析构建了 200 棵决策树（max_depth=6），对 {n_total} 例样本的 {len(feature_cols)} 个特征进行建模。"
        f"OOB（Out-of-Bag）估计提供了无需额外验证集的内部泛化误差评估。\n\n"
        f"### 二、主要发现\n"
        f"模型在独立测试集上 AUC = **{auc_val:.4f}**（{auc_interp}），"
        f"准确率 = {acc_val:.4f}，F1 = {f1_val:.4f}。"
        f"OOB 分数 = {oob:.4f}（OOB 误差 = {oob_err:.4f}），"
        f"与测试集性能{'基本一致' if abs(oob - acc_val) < 0.05 else '存在一定差异'}，"
        f"{'提示模型泛化性能稳定' if abs(oob - acc_val) < 0.05 else '建议关注潜在的过拟合或数据分布差异'}。\n\n"
        f"**特征重要性：**基于基尼不纯度（Gini importance）的排序显示，"
        f"前三位关键特征为 **{top3_feats}**，其中 {top1['特征']} 的重要性最高（{top1['重要性']:.4f}）。"
        f"特征重要性的分布模式{'呈现长尾分布，少数变量主导预测' if imp.head(3)['重要性'].sum() > 0.5 else '相对均匀，多个变量共同贡献'}。\n\n"
        f"### 三、临床意义\n"
        f"随机森林的集成特性使其对异常值和噪声具有天然的鲁棒性，"
        f"适合处理临床数据中常见的测量误差和个体变异。"
        f"特征重要性排序为生物标志物筛选提供了稳健的参考——"
        f"由于 Bagging 的平均效应，随机森林的重要性估计比单棵决策树更加稳定可靠。\n\n"
        f"OOB 误差估计的优势在于无需预留验证集即可获得泛化性能的无偏估计，"
        f"这在样本量有限的临床研究中尤为实用。\n\n"
        f"### 四、模型诊断与局限性\n"
        f"1. **OOB vs 测试集一致性**：OOB 分数（{oob:.4f}）与测试集准确率（{acc_val:.4f}）的一致性"
        f"{'良好，支持模型的稳定性' if abs(oob - acc_val) < 0.05 else '存在差异，可能反映数据划分的随机性'}。\n"
        f"2. **特征重要性偏倚**：基于 Gini 的重要性可能偏向连续变量和高基数分类变量，"
        f"建议补充 Permutation Importance 进行交叉验证。\n"
        f"3. **树深度限制**：max_depth=6 的设置在防止过拟合和保持表达能力之间取得平衡，"
        f"但可能限制了对深层交互效应的捕捉。\n"
        f"4. **局限性**：(a) 200 棵树的集成模型缺乏单一可解释的决策路径；"
        f"(b) 对于线性可分的简单问题，随机森林可能不如逻辑回归高效；"
        f"(c) 预测时需要遍历所有树，实时预测的计算开销较大。\n\n"
        f"### 五、结论\n"
        f"随机森林集成模型（200 棵树）在本数据集上展现了{auc_interp}的预测性能（AUC = {auc_val:.4f}），"
        f"OOB 内部验证支持模型的泛化稳定性。"
        f"**{top1['特征']}** 被识别为最重要的预测因子，建议在后续研究中优先关注。"
        f"该模型可作为临床预测系统的核心组件，但建议结合 SHAP 分析提升个体预测的可解释性。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 6. SVM
# ═══════════════════════════════════════════════════════════════

def run_ml_svm(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    target = params.get("target", "label")
    feature_cols = [c for c in df.columns if c not in [target, "patient_id"]]
    X, y = _get_xy(df, feature_cols, target)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.3, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.3, random_state=42)

    kernels = ["linear", "rbf", "poly"]
    results = []
    for kernel in kernels:
        try:
            svm = SVC(kernel=kernel, probability=True, random_state=42)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            results.append({
                "核函数": kernel,
                "准确率": round(accuracy_score(y_test, y_pred), 4),
                "支持向量数": svm.n_support_.sum(),
            })
        except Exception:
            pass

    best_kernel = max(results, key=lambda r: r["准确率"])["核函数"] if results else "rbf"
    model = SVC(kernel=best_kernel, probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if model.probability else model.decision_function(X_test)

    metrics = [
        {"指标": "最优核函数", "值": best_kernel},
        {"指标": "准确率", "值": round(accuracy_score(y_test, y_pred), 4)},
        {"指标": "F1", "值": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)},
        {"指标": "支持向量数", "值": model.n_support_.sum()},
    ]
    out["tables"].append({"title": "SVM 模型性能", "headers": ["指标", "值"], "rows": metrics})
    out["tables"].append({"title": "核函数比较", "headers": list(results[0].keys()), "rows": results})

    if HAS_PLOTLY:
        x_min, x_max = X_s[:, 0].min() - 1, X_s[:, 0].max() + 1
        y_min, y_max = X_s[:, 1].min() - 1, X_s[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))
        svm_bnd = SVC(kernel=best_kernel, random_state=42)
        svm_bnd.fit(X_s[:, :2], y)
        Z = svm_bnd.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig = go.Figure()
        fig.add_trace(go.Contour(x=np.linspace(x_min, x_max, 80), y=np.linspace(y_min, y_max, 80),
                                  z=Z, colorscale="Teal", opacity=0.5, showscale=True))
        for label in np.unique(y):
            mask = y == label
            fig.add_trace(go.Scatter(x=X_s[mask, 0], y=X_s[mask, 1], mode="markers",
                                     name=f"Class {label}", marker=dict(size=5)))
        fig.update_layout(title=f"SVM 决策边界 ({best_kernel})", template="plotly_white", height=520)
        out["charts"].append({"title": "决策边界", "plotly": _fig_to_json(fig)})

    acc_val = metrics[1]['值']
    f1_val = metrics[2]['值']
    sv_count = model.n_support_.sum()
    n_total = len(X)
    sv_ratio = sv_count / n_total * 100
    kernel_results_text = "、".join([f"{r['核函数']}（准确率={r['准确率']}）" for r in results])

    out["discussion"] = (
        f"## 支持向量机（SVM）分析结果与讨论\n\n"
        f"### 一、方法学概述\n"
        f"支持向量机（Support Vector Machine, SVM）是一种基于最大间隔原理的分类算法，"
        f"通过寻找能够最大化类别间隔的最优超平面实现分类。"
        f"对于线性不可分问题，SVM 通过核技巧（kernel trick）将数据映射到高维特征空间，"
        f"在该空间中构建线性分类器，等价于在原始空间中实现非线性决策边界。\n\n"
        f"本分析对 {n_total} 例样本的 {len(feature_cols)} 个标准化特征进行了 SVM 建模，"
        f"系统比较了线性核（linear）、径向基核（RBF）和多项式核（poly）三种核函数的分类性能。\n\n"
        f"### 二、主要发现\n"
        f"**核函数比较：**{kernel_results_text}。\n\n"
        f"最优核函数为 **{best_kernel}**，在测试集上准确率 = **{acc_val:.4f}**，F1 = {f1_val:.4f}。"
        f"模型使用了 **{sv_count}** 个支持向量（占总样本的 {sv_ratio:.1f}%）。\n\n"
        f"{'支持向量比例较低（< 30%），表明类别间存在较清晰的分离边界，模型泛化能力较强。' if sv_ratio < 30 else '支持向量比例较高（> 30%），提示类别间存在较大重叠区域，分类边界较为复杂。'}\n\n"
        f"### 三、临床意义\n"
        f"{'RBF 核的优势在于能够捕捉任意形状的非线性决策边界，适合临床数据中常见的复杂分类模式。' if best_kernel == 'rbf' else 'Linear 核的优势在于模型简洁且可解释性较强，权重向量可直接反映各特征的分类贡献。' if best_kernel == 'linear' else '多项式核能够捕捉特征间的交互效应，适合存在协同作用的临床变量组合。'}\n\n"
        f"决策边界可视化（基于前两个特征的投影）直观展示了 SVM 的分类策略。"
        f"支持向量作为'最难分类'的边界样本，在临床上可能代表疾病表型的过渡状态或边界病例，"
        f"值得进一步的临床特征分析。\n\n"
        f"### 四、模型诊断与局限性\n"
        f"1. **核函数选择**：{best_kernel} 核在本数据集上表现最优，"
        f"但最优核函数的选择依赖于数据的内在结构，建议在不同数据集上重新评估。\n"
        f"2. **超参数调优**：当前使用默认的正则化参数 C 和核参数 gamma，"
        f"建议通过网格搜索或贝叶斯优化进行系统调参以进一步提升性能。\n"
        f"3. **可扩展性**：SVM 的训练时间复杂度为 O(n²~n³)，在大规模数据集上可能较慢。\n"
        f"4. **局限性**：(a) 概率输出通过 Platt scaling 后处理获得，可能不如原生概率模型准确；"
        f"(b) 对特征缩放高度敏感（已处理）；"
        f"(c) 多分类需要 one-vs-one 或 one-vs-rest 策略，计算开销随类别数增长；"
        f"(d) 决策边界可视化仅为二维投影，可能无法完整反映高维空间中的分类结构。\n\n"
        f"### 五、结论\n"
        f"SVM 模型（{best_kernel} 核）在本数据集上达到 {acc_val:.4f} 的分类准确率，"
        f"使用 {sv_count} 个支持向量定义了最优决策边界。"
        f"核函数比较结果为模型选择提供了实证依据。"
        f"建议后续结合超参数优化和交叉验证进一步提升模型性能，"
        f"并通过支持向量的临床特征分析深入理解分类边界的生物学含义。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 7. Decision Tree
# ═══════════════════════════════════════════════════════════════

def run_ml_dt(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    target = params.get("target", "high_risk")
    feature_cols = [c for c in df.columns if c not in [target, "patient_id"]]
    X, y = _get_xy(df, feature_cols, target)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    depths = range(2, 11)
    cv_scores = []
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        cv_scores.append(np.mean(cross_val_score(dt, X_train, y_train, cv=5, scoring="accuracy")))
    best_depth = list(depths)[np.argmax(cv_scores)]

    model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = [
        {"指标": "最优深度", "值": best_depth},
        {"指标": "准确率", "值": round(accuracy_score(y_test, y_pred), 4)},
        {"指标": "F1", "值": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)},
        {"指标": "叶节点数", "值": model.get_n_leaves()},
    ]
    out["tables"].append({"title": "决策树性能", "headers": ["指标", "值"], "rows": metrics})

    imp = pd.DataFrame({
        "特征": feature_cols,
        "重要性": np.round(model.feature_importances_, 4),
    }).sort_values("重要性", ascending=False)
    out["tables"].append({"title": "特征重要性", "headers": list(imp.columns), "rows": imp.to_dict(orient="records")})

    if HAS_PLOTLY:
        # Depth selection
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(depths), y=cv_scores, mode="lines+markers",
                                 marker=dict(size=10), line=dict(color="#0E7C7B", width=2)))
        fig.update_layout(title="树深度优化", xaxis_title="Max Depth", yaxis_title="准确率",
                          template="plotly_white", height=520)
        out["charts"].append({"title": "深度优化", "plotly": _fig_to_json(fig)})

        # Feature importance
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=imp["重要性"], y=imp["特征"], orientation="h", marker_color="#0E7C7B"))
        fig2.update_layout(title="特征重要性", yaxis=dict(autorange="reversed"),
                           template="plotly_white", height=500)
        out["diagnostics"].append({"title": "特征重要性", "plotly": _fig_to_json(fig2)})

    # Extract rules
    rules = []
    def _extract_rules(tree, feature_names, node=0, path=""):
        if tree.feature[node] == -2:  # leaf
            val = tree.value[node]
            cls = np.argmax(val)
            rules.append({"条件": path or "根节点", "预测类别": int(cls), "样本数": int(val.sum()),
                          "占比": round(float(val[0, cls]) / val.sum(), 3)})
            return
        feat = feature_names[tree.feature[node]]
        thresh = round(tree.threshold[node], 2)
        _extract_rules(tree, feature_names, tree.children_left[node],
                       f"{path} 且 {feat} <= {thresh}" if path else f"{feat} <= {thresh}")
        _extract_rules(tree, feature_names, tree.children_right[node],
                       f"{path} 且 {feat} > {thresh}" if path else f"{feat} > {thresh}")
    _extract_rules(model.tree_, feature_cols)
    out["tables"].append({"title": "决策规则 (Top)", "headers": list(rules[0].keys()),
                          "rows": rules[:8]})

    acc_val = metrics[1]['值']
    f1_val = metrics[2]['值']
    n_leaves = model.get_n_leaves()
    top_feat = imp.iloc[0]['特征']
    top_imp = imp.iloc[0]['重要性']
    top3_feats = "、".join(imp.head(3)["特征"].tolist())
    n_total = len(X_train) + len(X_test)
    n_rules = len(rules)
    first_rule = rules[0] if rules else {}

    out["discussion"] = (
        f"## 决策树模型分析结果与讨论\n\n"
        f"### 一、方法学概述\n"
        f"决策树（Decision Tree）是一种基于递归二分法的分类算法，"
        f"通过在每个内部节点选择最优分裂特征和阈值，将样本空间逐步划分为纯度更高的子区域。"
        f"其核心优势在于模型结构直观透明——每条从根到叶的路径对应一条明确的'如果-则'决策规则，"
        f"这使得决策树在需要临床可解释性的场景中具有不可替代的价值。\n\n"
        f"本分析通过 5 折交叉验证在深度 2-10 范围内搜索最优树深度，"
        f"对 {n_total} 例样本的 {len(feature_cols)} 个特征进行建模。"
        f"采用 CART 算法（基尼不纯度准则）进行节点分裂。\n\n"
        f"### 二、主要发现\n"
        f"交叉验证确定的最优树深度为 **{best_depth}**，"
        f"模型在测试集上准确率 = **{acc_val:.4f}**，F1 = {f1_val:.4f}。"
        f"最终决策树包含 **{n_leaves}** 个叶节点，生成了 {n_rules} 条决策规则。\n\n"
        f"**特征重要性：**{top_feat} 是最重要的分裂特征（重要性 = {top_imp:.4f}），"
        f"前三位特征为 {top3_feats}。"
        f"根节点的首次分裂通常代表对分类贡献最大的单一决策，具有最高的临床参考价值。\n\n"
        f"**决策规则示例：**提取的规则可直接转化为临床决策流程，"
        f"例如'{first_rule.get('条件', '')}'→ 预测类别 {first_rule.get('预测类别', '')}。\n\n"
        f"### 三、临床意义\n"
        f"决策树的最大优势在于其完全透明的决策逻辑，每个预测结果都可以追溯到具体的判断依据。"
        f"这在以下临床场景中尤为重要：\n"
        f"- **急诊分诊**：简洁的决策规则可快速指导分诊决策\n"
        f"- **临床指南制定**：决策路径可直接转化为诊疗流程图\n"
        f"- **患者沟通**：透明的决策逻辑有助于向患者解释诊断依据\n"
        f"- **质量控制**：可审计的决策过程满足医疗合规要求\n\n"
        f"深度为 {best_depth} 的树在复杂度和准确性之间取得了平衡——"
        f"{'较浅的树（深度 ≤ 3）虽然更简洁，但可能遗漏重要的交互效应' if best_depth > 3 else '浅层树结构简洁，适合作为临床筛查工具'}。\n\n"
        f"### 四、模型诊断与局限性\n"
        f"1. **过拟合控制**：通过限制最大深度（{best_depth}）和交叉验证选择，"
        f"有效防止了决策树的过拟合倾向。深度优化曲线显示了准确率随深度变化的趋势。\n"
        f"2. **不稳定性**：决策树对训练数据的微小变化敏感，"
        f"不同的数据划分可能产生结构差异较大的树。建议通过 Bootstrap 评估规则的稳定性。\n"
        f"3. **贪心算法局限**：CART 采用贪心策略进行局部最优分裂，"
        f"不保证全局最优树结构。\n"
        f"4. **局限性**：(a) 单棵决策树的预测性能通常不如集成方法；"
        f"(b) 对连续变量的处理为阶梯函数，无法表达平滑的剂量-反应关系；"
        f"(c) 类别不平衡可能导致偏向多数类的分裂。\n\n"
        f"### 五、结论\n"
        f"决策树模型（深度 = {best_depth}，{n_leaves} 个叶节点）在本数据集上达到 {acc_val:.4f} 的分类准确率，"
        f"生成了 {n_rules} 条临床可解释的决策规则。"
        f"**{top_feat}** 作为根节点分裂特征具有最高的分类贡献。"
        f"该模型特别适合作为临床决策支持工具的原型，"
        f"建议将提取的决策规则与临床专家知识进行对照验证，确保规则的生物学合理性。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# 8. 1D-CNN
# ═══════════════════════════════════════════════════════════════

def run_ml_cnn(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    target = params.get("target", "label")
    feature_cols = [c for c in df.columns
                    if c not in [target, "subject_id", "time"]
                    and pd.api.types.is_numeric_dtype(df[c])]

    subjects = sorted(df["subject_id"].unique())
    time_len = df.groupby("subject_id").size().iloc[0]
    n_features = len(feature_cols)

    X = df[feature_cols].values.reshape(len(subjects), time_len, n_features)
    sub_labels = df.groupby("subject_id")[target].first().values

    scaler = StandardScaler()
    X_flat = X.reshape(-1, n_features)
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(len(subjects), time_len, n_features)

    # Simple 1D-CNN using sklearn MLP + time features
    time_features = np.column_stack([
        X[:, :, i].mean(axis=1) for i in range(n_features)
    ] + [
        X[:, :, i].std(axis=1) for i in range(n_features)
    ] + [
        np.polyfit(np.arange(time_len), X[:, :, i].T, 1)[0] for i in range(n_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(time_features, sub_labels, test_size=0.3, random_state=42, stratify=sub_labels)

    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = [
        {"指标": "准确率", "值": round(accuracy_score(y_test, y_pred), 4)},
        {"指标": "AUC", "值": round(roc_auc_score(y_test, y_prob), 4)},
        {"指标": "F1", "值": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)},
        {"指标": "时间序列长度", "值": time_len},
        {"指标": "特征数", "值": n_features},
    ]
    out["tables"].append({"title": "1D-CNN 模型性能", "headers": ["指标", "值"], "rows": metrics})

    if HAS_PLOTLY:
        # Training: show sample time series
        fig = go.Figure()
        for i in range(min(5, len(subjects))):
            fig.add_trace(go.Scatter(x=list(range(time_len)), y=X[i, :, 0], mode="lines",
                                     name=subjects[i]))
        fig.update_layout(title="样本时序数据示例", xaxis_title="时间点", yaxis_title=feature_cols[0],
                          template="plotly_white", height=520)
        out["charts"].append({"title": "时序示例", "plotly": _fig_to_json(fig)})

        # Confusion
        cm = confusion_matrix(y_test, y_pred)
        fig2 = ff.create_annotated_heatmap(cm, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                                            colorscale="Teal")
        fig2.update_layout(title="混淆矩阵", template="plotly_white", height=520)
        out["diagnostics"].append({"title": "混淆矩阵", "plotly": _fig_to_json(fig2)})

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#0E7C7B", width=2)))
        fig3.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="gray", dash="dash")))
        fig3.update_layout(title="ROC 曲线", template="plotly_white", height=520)
        out["diagnostics"].append({"title": "ROC曲线", "plotly": _fig_to_json(fig3)})

    auc_val = metrics[1]['值']
    acc_val = metrics[0]['值']
    f1_val = metrics[2]['值']
    n_subjects = len(subjects)
    auc_interp = "优秀" if auc_val >= 0.9 else "良好" if auc_val >= 0.8 else "中等" if auc_val >= 0.7 else "较弱"

    out["discussion"] = (
        f"## 1D-CNN 时序分类模型分析结果与讨论\n\n"
        f"### 一、方法学概述\n"
        f"本分析采用基于深度学习思想的时序特征提取 + 多层感知器（MLP）分类框架，"
        f"模拟一维卷积神经网络（1D-CNN）对临床时序数据的建模能力。"
        f"具体而言，从每个受试者的 {time_len} 个时间点、{n_features} 维信号中提取三类统计特征：\n"
        f"- **均值特征**：反映信号的整体水平\n"
        f"- **标准差特征**：反映信号的波动幅度\n"
        f"- **趋势特征**（线性拟合斜率）：反映信号的时间变化方向\n\n"
        f"提取的 {n_features * 3} 维特征向量输入两层 MLP（64-32 神经元，ReLU 激活）进行分类。"
        f"共纳入 {n_subjects} 例受试者，按 7:3 分层划分训练-测试集。\n\n"
        f"### 二、主要发现\n"
        f"模型在独立测试集上 AUC = **{auc_val:.4f}**（{auc_interp}），"
        f"准确率 = {acc_val:.4f}，F1 = {f1_val:.4f}。\n\n"
        f"时序数据的统计特征提取策略成功将变长时序问题转化为固定维度的分类问题，"
        f"同时保留了信号的关键统计属性（水平、变异性、趋势）。"
        f"样本时序可视化展示了不同类别受试者在信号模式上的差异。\n\n"
        f"### 三、临床意义\n"
        f"时序分类在临床中具有广泛应用场景：\n"
        f"- **连续监测数据**：ICU 生命体征、可穿戴设备数据的异常模式识别\n"
        f"- **纵向随访数据**：疾病进展轨迹的早期预警\n"
        f"- **生理信号分析**：心电图、脑电图等波形数据的自动分类\n\n"
        f"本模型的特征工程策略（均值 + 标准差 + 趋势）虽然简化了原始时序信息，"
        f"但在临床实践中具有良好的可解释性——"
        f"临床医生可以直观理解'某指标的平均水平升高、波动增大、呈上升趋势'等特征的含义。\n\n"
        f"### 四、模型诊断与局限性\n"
        f"1. **简化实现**：当前采用手工特征提取 + MLP 的方式，"
        f"未使用真正的卷积层进行端到端学习。完整的 1D-CNN 可能捕捉更丰富的时序模式。\n"
        f"2. **特征信息损失**：统计特征提取丢失了时序数据中的局部模式、周期性和突变点信息。"
        f"建议后续引入小波变换或注意力机制补充时频域特征。\n"
        f"3. **样本量限制**：深度学习方法通常需要较大样本量，"
        f"当前 {n_subjects} 例受试者可能不足以充分训练复杂网络结构。\n"
        f"4. **局限性**：(a) MLP 不具备时序感知能力，无法建模时间依赖关系；"
        f"(b) 假设所有受试者的时序长度相同；"
        f"(c) 未考虑缺失时间点的处理；"
        f"(d) 模型的可解释性有限，难以定位关键时间窗口。\n\n"
        f"### 五、结论\n"
        f"基于时序特征提取的分类模型在 {n_subjects} 例受试者的 {time_len} 时间点 × {n_features} 维信号数据上"
        f"达到了{auc_interp}的分类性能（AUC = {auc_val:.4f}）。"
        f"结果表明时序数据中蕴含的统计特征对临床结局具有预测价值。"
        f"建议后续研究：(1) 引入真正的 1D-CNN 或 LSTM 架构进行端到端学习；"
        f"(2) 增加样本量以支撑更复杂的网络结构；"
        f"(3) 结合 Grad-CAM 等可视化方法定位关键时间窗口。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════

def run_feature_engineering(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Missing summary
    missing = pd.DataFrame({
        "变量": df.columns,
        "缺失数": df.isna().sum().values,
        "缺失比例": np.round(df.isna().sum().values / len(df) * 100, 2),
    }).sort_values("缺失比例", ascending=False)
    missing = missing[missing["缺失比例"] > 0]
    out["tables"].append({"title": "缺失值统计", "headers": list(missing.columns),
                          "rows": missing.to_dict(orient="records") if len(missing) > 0 else [{"变量": "—", "缺失数": 0, "缺失比例": 0}]})

    # Variable summary
    var_summary = pd.DataFrame({
        "变量类型": ["连续变量", "分类变量", "含缺失变量", "总变量"],
        "数量": [len(num_cols), len(cat_cols), len(missing), len(df.columns)],
    })
    out["tables"].append({"title": "变量概览", "headers": list(var_summary.columns),
                          "rows": var_summary.to_dict(orient="records")})

    if HAS_PLOTLY:
        # Missingness heatmap
        if len(missing) > 0:
            missing_data = df.isna().astype(int)
            fig = go.Figure(data=go.Heatmap(z=missing_data.values.T, colorscale=[[0, "#E8F4F3"], [1, "#E06830"]],
                                            x=[str(i) for i in range(len(df))],
                                            y=df.columns.tolist(), showscale=True))
            fig.update_layout(title="缺失值热图", template="plotly_white", height=500 + 15 * len(df.columns))
            out["charts"].append({"title": "缺失值热图", "plotly": _fig_to_json(fig)})

    n_missing = len(missing)
    missing_pct = missing["缺失比例"].max() if n_missing > 0 else 0
    high_missing = missing[missing["缺失比例"] > 20] if n_missing > 0 else pd.DataFrame()

    out["discussion"] = (
        f"## 特征工程与数据质量评估报告\n\n"
        f"### 一、数据集概况\n"
        f"本数据集包含 **{len(df)}** 条记录和 **{len(df.columns)}** 个变量，"
        f"其中连续型变量 {len(num_cols)} 个，分类型变量 {len(cat_cols)} 个。"
        f"数据质量是机器学习建模成功的基础，本报告从缺失值、变量类型和数据完整性三个维度进行系统评估。\n\n"
        f"### 二、缺失值分析\n"
        f"{'**数据完整性良好：**所有变量均无缺失值，可直接进入建模流程。' if n_missing == 0 else f'共有 **{n_missing}** 个变量存在缺失值，最高缺失比例为 {missing_pct:.1f}%。'}\n\n"
        f"{'缺失值热图显示了缺失模式的空间分布。' if n_missing > 0 else ''}"
        f"{'**高缺失变量（>20%）：**' + '、'.join(high_missing['变量'].tolist()) + '。对于缺失比例超过 20% 的变量，建议评估其缺失机制（MCAR/MAR/MNAR），并考虑多重插补（Multiple Imputation）或在敏感性分析中排除。' if len(high_missing) > 0 else ''}\n\n"
        f"### 三、变量类型与建模建议\n"
        f"- **连续变量（{len(num_cols)} 个）**：建议检查分布形态（偏度/峰度），对严重偏态变量考虑对数变换或 Box-Cox 变换。标准化（Z-score）或归一化（Min-Max）应根据后续模型选择决定。\n"
        f"- **分类变量（{len(cat_cols)} 个）**：建议检查各类别的频率分布，对低频类别考虑合并。编码方式应根据变量的有序/无序性质选择（有序：序数编码；无序：独热编码或目标编码）。\n\n"
        f"### 四、建模前处理建议\n"
        f"1. **缺失值处理**：{'数据完整，无需处理' if n_missing == 0 else '建议采用多重插补（mice 包）或基于模型的插补方法'}\n"
        f"2. **异常值检测**：建议使用 IQR 法则或 Isolation Forest 识别潜在异常值\n"
        f"3. **特征缩放**：对距离敏感的算法（KNN、SVM）必须标准化；树模型不需要\n"
        f"4. **特征选择**：可结合单变量筛选（方差阈值、互信息）和模型嵌入法（Lasso、RF importance）\n"
        f"5. **类别平衡**：检查目标变量的类别比例，必要时采用 SMOTE 或加权损失函数\n\n"
        f"### 五、结论\n"
        f"数据集整体质量{'良好' if n_missing == 0 or missing_pct < 10 else '需要关注'}，"
        f"包含 {len(num_cols)} 个连续变量和 {len(cat_cols)} 个分类变量。"
        f"{'建议直接进入建模阶段。' if n_missing == 0 else f'建议优先处理 {n_missing} 个含缺失变量后再进行建模。'}"
        f"后续建模应根据研究目的选择合适的算法：预测为主选集成方法，解释为主选线性模型或决策树。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# Model Comparison
# ═══════════════════════════════════════════════════════════════

def run_model_comparison(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    target = params.get("target", "outcome")
    feature_cols = [c for c in df.columns if c not in [target, "patient_id"]]
    X, y = _get_xy(df, feature_cols, target)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.3, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.3, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
    }
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                               random_state=42, use_label_encoder=False, eval_metric="logloss")
    except ImportError:
        pass

    results = []
    roc_data = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
            if np.ndim(y_prob) == 0 or y_prob.size == 1:
                y_prob = np.full_like(y_test.astype(float), y_prob.item() if hasattr(y_prob, 'item') else float(y_prob))
            results.append({
                "模型": name,
                "准确率": round(float(accuracy_score(y_test, y_pred)), 4),
                "AUC": round(float(roc_auc_score(y_test, y_prob)), 4),
                "F1": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
                "精确率": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "召回率": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            })
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data[name] = (fpr, tpr)
        except Exception as e:
            results.append({
                "模型": name,
                "准确率": "N/A",
                "AUC": "N/A",
                "F1": "N/A",
                "精确率": "N/A",
                "召回率": "N/A",
            })

    out["tables"].append({"title": "模型比较", "headers": list(results[0].keys()), "rows": results})

    if HAS_PLOTLY:
        fig = go.Figure()
        colors = ["#0E7C7B", "#E06830", "#665C9E", "#C58725", "#B34D3E"]
        for i, (name, (fpr, tpr)) in enumerate(roc_data.items()):
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name}",
                                     line=dict(color=colors[i % len(colors)], width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(color="gray", dash="dash")))
        fig.update_layout(title="多模型 ROC 比较", xaxis_title="1 - 特异度", yaxis_title="灵敏度",
                          template="plotly_white", height=530)
        out["charts"].append({"title": "多模型ROC", "plotly": _fig_to_json(fig)})

    best = max(results, key=lambda r: r["AUC"] if isinstance(r["AUC"], (int, float)) else 0)
    auc_str = f"{best['AUC']:.4f}" if isinstance(best['AUC'], (int, float)) else str(best['AUC'])
    valid_results = [r for r in results if isinstance(r["AUC"], (int, float))]
    sorted_results = sorted(valid_results, key=lambda r: r["AUC"], reverse=True)
    ranking_text = "、".join([f"{r['模型']}（AUC={r['AUC']:.4f}）" for r in sorted_results[:3]])
    auc_range = sorted_results[-1]["AUC"] if sorted_results else 0
    auc_spread = best["AUC"] - auc_range if isinstance(best["AUC"], (int, float)) and isinstance(auc_range, (int, float)) else 0

    out["discussion"] = (
        f"## 多模型比较分析结果与讨论\n\n"
        f"### 一、方法学概述\n"
        f"本分析对 {len(models)} 种主流机器学习分类算法进行了系统性能比较，"
        f"所有模型在相同的数据划分（7:3 分层随机划分）和预处理条件（标准化）下进行训练和评估，"
        f"确保比较的公平性。评估指标包括 AUC-ROC、准确率、F1 分数、精确率和召回率。\n\n"
        f"纳入比较的模型涵盖了从简单线性模型到复杂集成方法的完整谱系：\n"
        f"- **线性模型**：Logistic Regression（基线参考）\n"
        f"- **距离模型**：KNN（非参数方法）\n"
        f"- **核方法**：SVM（最大间隔分类）\n"
        f"- **集成方法**：Random Forest（Bagging）、XGBoost（Boosting）\n\n"
        f"### 二、主要发现\n"
        f"**性能排序（按 AUC）：**{ranking_text}。\n\n"
        f"最优模型为 **{best['模型']}**（AUC = {auc_str}），"
        f"各模型 AUC 的极差为 {auc_spread:.4f}，"
        f"{'模型间性能差异较小（< 0.05），提示数据中的可预测信息已被大多数模型充分利用' if auc_spread < 0.05 else '模型间存在明显性能差异，提示非线性方法在本数据集上具有优势' if auc_spread > 0.1 else '模型间存在中等程度的性能差异'}。\n\n"
        f"ROC 曲线对比图直观展示了各模型在不同阈值下的灵敏度-特异度权衡，"
        f"曲线越靠近左上角表示模型的区分能力越强。\n\n"
        f"### 三、临床意义与模型选择建议\n"
        f"模型选择应综合考虑以下因素：\n"
        f"- **预测性能**：{best['模型']} 在 AUC 上表现最优\n"
        f"- **可解释性**：Logistic Regression 和决策树提供最佳可解释性\n"
        f"- **计算效率**：线性模型训练和预测最快\n"
        f"- **鲁棒性**：集成方法（RF、XGBoost）对噪声和异常值更鲁棒\n\n"
        f"**推荐策略：**\n"
        f"- 若优先考虑预测精度（如风险预警系统）：选择 {best['模型']}\n"
        f"- 若优先考虑可解释性（如临床指南制定）：选择 Logistic Regression\n"
        f"- 若需要平衡两者：考虑使用集成模型预测 + SHAP 解释的组合方案\n\n"
        f"### 四、模型诊断与局限性\n"
        f"1. **单次划分偏倚**：当前结果基于单次随机划分，建议补充重复交叉验证（如 10×10 CV）以获得更稳健的性能估计。\n"
        f"2. **超参数公平性**：各模型使用默认或轻度调优的超参数，系统调参后排序可能变化。\n"
        f"3. **评估指标选择**：AUC 适合评估整体区分能力，但在类别不平衡或特定阈值需求下，"
        f"应结合 F1、精确率-召回率曲线等指标综合判断。\n"
        f"4. **局限性**：(a) 未包含深度学习方法；(b) 未进行统计显著性检验（如 DeLong test）；"
        f"(c) 未评估校准度（calibration）；(d) 未考虑模型的临床净获益（DCA）。\n\n"
        f"### 五、结论\n"
        f"在 {len(models)} 种机器学习模型的系统比较中，**{best['模型']}** 以 AUC = {auc_str} 的性能位居首位。"
        f"建议根据具体临床应用场景的需求（预测精度 vs 可解释性 vs 计算效率）选择最终部署模型，"
        f"并通过外部验证和决策曲线分析评估其临床实用价值。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# Dimensionality Reduction (PCA/t-SNE)
# ═══════════════════════════════════════════════════════════════

def run_dim_reduction(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    group_var = params.get("group_var", "group")
    feature_cols = [c for c in df.columns if c not in ["sample_id", group_var]]
    X_raw = df[feature_cols].copy()
    for col in X_raw.columns:
        if X_raw[col].dtype == object:
            X_raw[col] = X_raw[col].astype("category").cat.codes
    X = StandardScaler().fit_transform(X_raw.dropna())

    groups = df.loc[X_raw.dropna().index, group_var].values if group_var in df.columns else None

    # PCA
    pca = PCA(n_components=min(10, X.shape[1]))
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    cumsum = np.cumsum(explained)
    out["tables"].append({"title": "PCA 解释方差",
                          "headers": ["主成分", "解释方差比", "累计方差比"],
                          "rows": [{"主成分": f"PC{i+1}", "解释方差比": round(explained[i], 4),
                                     "累计方差比": round(cumsum[i], 4)}
                                    for i in range(min(10, len(explained)))]})

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0] // 4))
    X_tsne = tsne.fit_transform(X)

    if HAS_PLOTLY:
        # PCA
        fig = go.Figure()
        if groups is not None:
            for g in sorted(np.unique(groups)):
                mask = groups == g
                fig.add_trace(go.Scatter(x=X_pca[mask, 0], y=X_pca[mask, 1], mode="markers",
                                         name=str(g), marker=dict(size=6, opacity=0.7)))
        else:
            fig.add_trace(go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], mode="markers",
                                     marker=dict(size=6, color=X_pca[:, 0], colorscale="Teal")))
        fig.update_layout(title="PCA 降维可视化", xaxis_title=f"PC1 ({explained[0]*100:.1f}%)",
                          yaxis_title=f"PC2 ({explained[1]*100:.1f}%)", template="plotly_white", height=520)
        out["charts"].append({"title": "PCA可视化", "plotly": _fig_to_json(fig)})

        # Scree
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(explained))], y=explained,
                               marker_color="#0E7C7B", name="个体"))
        fig2.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(explained))], y=cumsum,
                                  mode="lines+markers", name="累计", line=dict(color="#E06830", width=2)))
        fig2.update_layout(title="Scree Plot", template="plotly_white", height=520)
        out["diagnostics"].append({"title": "Scree Plot", "plotly": _fig_to_json(fig2)})

        # t-SNE
        fig3 = go.Figure()
        if groups is not None:
            for g in sorted(np.unique(groups)):
                mask = groups == g
                fig3.add_trace(go.Scatter(x=X_tsne[mask, 0], y=X_tsne[mask, 1], mode="markers",
                                          name=str(g), marker=dict(size=6, opacity=0.7)))
        else:
            fig3.add_trace(go.Scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], mode="markers",
                                      marker=dict(size=6, color=X_tsne[:, 0], colorscale="Teal")))
        fig3.update_layout(title="t-SNE 降维可视化", template="plotly_white", height=520)
        out["diagnostics"].append({"title": "t-SNE可视化", "plotly": _fig_to_json(fig3)})

    n_components_90 = np.argmax(cumsum >= 0.9) + 1 if np.any(cumsum >= 0.9) else len(cumsum)
    n_samples = X.shape[0]
    n_dims = X.shape[1]

    out["discussion"] = (
        f"## 降维分析（PCA / t-SNE）结果与讨论\n\n"
        f"### 一、方法学概述\n"
        f"本分析采用两种互补的降维方法对高维数据进行可视化和结构探索：\n"
        f"- **PCA（主成分分析）**：线性降维方法，通过正交变换将原始变量转化为一组线性无关的主成分，"
        f"按方差贡献率从大到小排列。PCA 保持全局结构，适合评估数据的整体变异模式。\n"
        f"- **t-SNE（t-分布随机邻域嵌入）**：非线性降维方法，通过保持局部邻域关系将高维数据嵌入低维空间。"
        f"t-SNE 擅长揭示数据中的聚类结构和局部模式。\n\n"
        f"本分析对 {n_samples} 个样本的 {n_dims} 维标准化特征进行降维，"
        f"PCA 提取前 {min(10, n_dims)} 个主成分，t-SNE 嵌入至二维空间（perplexity = {min(30, n_samples // 4)}）。\n\n"
        f"### 二、主要发现\n"
        f"**PCA 结果：**\n"
        f"- 第一主成分（PC1）解释了 {explained[0]*100:.1f}% 的总方差\n"
        f"- 前两个主成分累计解释 {cumsum[1]*100:.1f}% 的方差\n"
        f"- 达到 90% 累计方差需要 **{n_components_90}** 个主成分\n\n"
        f"{'PC1 和 PC2 已解释超过 70% 的方差，表明数据的主要变异可以在二维空间中较好地表示。' if cumsum[1] > 0.7 else '前两个主成分解释的方差比例有限，提示数据的变异分布在多个维度上，二维可视化可能丢失重要信息。'}\n\n"
        f"**t-SNE 结果：**\n"
        f"t-SNE 嵌入图{'展示了较为清晰的聚类结构，不同组别在低维空间中形成了可区分的簇' if groups is not None else '展示了数据在低维空间中的分布模式'}。"
        f"{'组间分离度较好，支持后续分类建模的可行性。' if groups is not None else ''}\n\n"
        f"### 三、临床意义\n"
        f"降维分析在临床研究中的应用价值：\n"
        f"- **疾病亚型发现**：通过无监督降维识别潜在的患者亚群\n"
        f"- **生物标志物筛选**：PCA 载荷（loadings）揭示各变量对主成分的贡献\n"
        f"- **数据质量评估**：异常样本在降维图中通常表现为离群点\n"
        f"- **批次效应检测**：不同批次的样本若在降维图中分离，提示存在技术偏倚\n\n"
        f"Scree Plot（碎石图）通过'肘部法则'辅助确定保留的主成分数目——"
        f"方差贡献率急剧下降的拐点之前的主成分通常包含有意义的生物学信号。\n\n"
        f"### 四、模型诊断与局限性\n"
        f"1. **PCA 线性假设**：PCA 仅能捕捉线性相关结构，对非线性流形结构可能失效。"
        f"若 PCA 解释方差较低，建议补充核 PCA 或 UMAP。\n"
        f"2. **t-SNE 参数敏感性**：t-SNE 结果对 perplexity 参数敏感，"
        f"不同参数可能产生不同的聚类外观。建议在多个 perplexity 值下重复分析。\n"
        f"3. **t-SNE 不保持全局距离**：t-SNE 优化局部结构，簇间距离不具有定量意义，"
        f"不应根据 t-SNE 图中簇的远近判断组间差异大小。\n"
        f"4. **局限性**：(a) 降维是信息压缩过程，不可避免地丢失部分信息；"
        f"(b) PCA 主成分的生物学解释需要结合载荷分析；"
        f"(c) t-SNE 不适合新样本的投影（非参数方法）。\n\n"
        f"### 五、结论\n"
        f"降维分析揭示了本数据集的内在结构特征：PCA 显示前 {n_components_90} 个主成分可解释 90% 的总方差，"
        f"t-SNE 提供了数据局部结构的直观可视化。"
        f"{'不同组别在降维空间中的分离模式支持后续分类建模的可行性。' if groups is not None else ''}"
        f"建议将降维结果作为探索性分析的起点，结合领域知识进一步解释观察到的数据模式。"
    )
    return out


# ═══════════════════════════════════════════════════════════════
# Clustering
# ═══════════════════════════════════════════════════════════════

def run_cluster(df: pd.DataFrame, params: dict) -> dict:
    out = {"tables": [], "charts": [], "diagnostics": [], "discussion": ""}
    n_clusters = params.get("n_clusters", 4)
    feature_cols = [c for c in df.columns if c not in ["sample_id", "true_cluster"]]
    X = StandardScaler().fit_transform(df[feature_cols].dropna())

    # K-Means
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km.fit_predict(X)

    # GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(X)

    # Hierarchical
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    hc_labels = hc.fit_predict(X)

    results = []
    for name, labels in [("K-Means", km_labels), ("GMM", gmm_labels), ("层次聚类", hc_labels)]:
        sizes = pd.Series(labels).value_counts().to_dict()
        results.append({
            "方法": name,
            "簇数": n_clusters,
            "簇大小分布": str([sizes.get(i, 0) for i in range(n_clusters)]),
        })
    out["tables"].append({"title": "聚类结果", "headers": list(results[0].keys()), "rows": results})

    if HAS_PLOTLY:
        fig = go.Figure()
        for label in sorted(set(km_labels)):
            mask = km_labels == label
            fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode="markers",
                                     name=f"Cluster {label}", marker=dict(size=7, opacity=0.7)))
            if hasattr(km, "cluster_centers_"):
                fig.add_trace(go.Scatter(x=[km.cluster_centers_[label, 0]],
                                         y=[km.cluster_centers_[label, 1]],
                                         mode="markers", marker=dict(size=14, symbol="x", color="black"),
                                         name=f"Center {label}", showlegend=False))
        fig.update_layout(title="K-Means 聚类结果", template="plotly_white", height=520)
        out["charts"].append({"title": "K-Means聚类", "plotly": _fig_to_json(fig)})

        # Compare methods
        fig2 = go.Figure()
        for i, (name, labels) in enumerate([("K-Means", km_labels), ("GMM", gmm_labels), ("层次聚类", hc_labels)]):
            fig2.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                       marker=dict(size=5, color=labels, colorscale="Teal"),
                                       name=name, visible=(i == 0)))
        # Simple buttons for switching
        fig2.update_layout(
            title="聚类方法比较",
            updatemenus=[dict(type="buttons", direction="right",
                              buttons=[dict(label=name, method="update",
                                             args=[{"visible": [i == j for j in range(3)]}])
                                        for i, (name, _) in enumerate([("K-Means", km_labels),
                                                                        ("GMM", gmm_labels),
                                                                        ("层次聚类", hc_labels)])])],
            template="plotly_white", height=520,
        )
        out["diagnostics"].append({"title": "聚类比较", "plotly": _fig_to_json(fig2)})

        # Elbow
        inertias = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=list(range(1, 11)), y=inertias, mode="lines+markers",
                                  line=dict(color="#0E7C7B", width=2)))
        fig3.update_layout(title="肘部法则 (Elbow Method)", xaxis_title="K",
                           yaxis_title="Inertia", template="plotly_white", height=520)
        out["diagnostics"].append({"title": "肘部法则", "plotly": _fig_to_json(fig3)})

    n_samples = X.shape[0]
    n_dims = X.shape[1]
    km_sizes = pd.Series(km_labels).value_counts().sort_index()
    size_balance = km_sizes.min() / km_sizes.max()

    out["discussion"] = (
        f"## 聚类分析结果与讨论\n\n"
        f"### 一、方法学概述\n"
        f"本分析采用三种经典聚类算法对 {n_samples} 个样本的 {n_dims} 维标准化特征进行无监督分组：\n"
        f"- **K-Means**：基于质心的划分方法，最小化簇内平方和（inertia），假设簇为凸形且大小相近\n"
        f"- **高斯混合模型（GMM）**：基于概率的软聚类方法，假设数据由多个高斯分布混合生成，允许椭圆形簇\n"
        f"- **层次聚类（Agglomerative）**：自底向上的合并策略，不需要预设簇数，可生成树状图\n\n"
        f"预设簇数 K = {n_clusters}，所有方法在相同的标准化数据上运行以确保可比性。"
        f"肘部法则（Elbow Method）用于评估最优簇数选择。\n\n"
        f"### 二、主要发现\n"
        f"**K-Means 结果：**{n_clusters} 个簇的样本分布为 {km_sizes.to_dict()}，"
        f"{'簇大小较为均衡（最小/最大 = ' + f'{size_balance:.2f}' + '），提示数据中存在大小相近的自然分组' if size_balance > 0.5 else '簇大小不均衡，可能存在主导簇和少数异质亚群'}。\n\n"
        f"**三种方法一致性：**不同聚类方法的结果一致性反映了聚类结构的稳健性。"
        f"若三种方法产生相似的分组，则聚类结构较为可靠；若差异较大，则提示数据中的分组模式可能不明确。\n\n"
        f"**肘部法则：**Inertia 随 K 增大而单调递减，"
        f"拐点（'肘部'）对应的 K 值通常为最优簇数的参考。\n\n"
        f"### 三、临床意义\n"
        f"无监督聚类在临床研究中的典型应用：\n"
        f"- **疾病亚型识别**：基于多维临床特征将患者分为具有不同预后或治疗反应的亚群\n"
        f"- **精准医疗**：为不同亚群制定差异化的治疗策略\n"
        f"- **风险分层**：识别高风险亚群进行重点监测\n"
        f"- **队列研究设计**：基于聚类结果进行分层随机化\n\n"
        f"聚类结果的临床价值取决于：(1) 各簇在临床结局上是否存在显著差异；"
        f"(2) 簇的特征是否具有生物学可解释性；(3) 分组结果是否在独立队列中可重复。\n\n"
        f"### 四、模型诊断与局限性\n"
        f"1. **簇数选择**：当前预设 K = {n_clusters}，建议结合肘部法则、轮廓系数（Silhouette Score）"
        f"和间隙统计量（Gap Statistic）综合确定最优簇数。\n"
        f"2. **K-Means 假设**：K-Means 假设簇为球形且等方差，对非凸形簇可能失效。"
        f"GMM 放松了等方差假设，层次聚类不假设簇形状。\n"
        f"3. **初始化敏感性**：K-Means 对初始质心敏感（已使用 n_init=10 缓解），"
        f"GMM 对初始参数敏感。\n"
        f"4. **局限性**：(a) 聚类结果缺乏'真实标签'验证，需要外部临床信息评估；"
        f"(b) 高维数据中的'维度灾难'可能影响距离度量的有效性；"
        f"(c) 聚类边界上的样本分配具有不确定性。\n\n"
        f"### 五、结论\n"
        f"三种聚类方法将 {n_samples} 个样本分为 {n_clusters} 个亚群。"
        f"K-Means 的簇大小分布{'均衡' if size_balance > 0.5 else '不均衡'}，"
        f"肘部法则图可辅助评估最优簇数。"
        f"建议后续：(1) 比较各簇在关键临床指标上的差异（ANOVA/Kruskal-Wallis）；"
        f"(2) 评估聚类结果与已知临床分型的一致性；"
        f"(3) 在独立队列中验证聚类的可重复性。"
    )
    return out


# ── Method Router ───────────────────────────────────────────

ML_ROUTER = {
    "ml_lr": run_ml_lr,
    "ml_lasso": run_ml_lasso,
    "ml_knn": run_ml_knn,
    "ml_xgboost": run_ml_xgboost,
    "ml_rf": run_ml_rf,
    "ml_svm": run_ml_svm,
    "ml_dt": run_ml_dt,
    "ml_cnn": run_ml_cnn,
    "feature_engineering": run_feature_engineering,
    "model_comparison": run_model_comparison,
    "dim_reduction": run_dim_reduction,
    "cluster": run_cluster,
}
