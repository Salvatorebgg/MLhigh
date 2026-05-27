from __future__ import annotations

import io
import json
import base64
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots as plotly_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ═══════════════════════════════════════════════════════════════
# Publication-quality palettes
# ═══════════════════════════════════════════════════════════════

CNS_PALETTE = [
    "#2E6F9E", "#D95F59", "#2A9D8F", "#E9A93A", "#6F5AA7",
    "#7C8B52", "#C776A5", "#4A5568", "#8F6B43", "#5BA4CF",
]
NATURE_PALETTE = [
    "#3C5488", "#E64B35", "#00A087", "#4DBBD5", "#F39B7F",
    "#8491B4", "#91D1C2", "#7E6148",
]
LANCET_PALETTE = [
    "#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F",
    "#FDAF91", "#AD002A", "#ADB6B6",
]
COLORMAPS = ["viridis", "plasma", "cividis", "RdYlBu_r", "Spectral_r", "coolwarm", "PuOr_r"]

# ═══════════════════════════════════════════════════════════════
# Matplotlib publication style
# ═══════════════════════════════════════════════════════════════


def set_publication_style(figsize=(8, 5.5), dpi=300):
    """Configure matplotlib for high-SCI publication quality."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Noto Sans SC", "SimHei"],
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })


def _add_axis_arrows(ax):
    """Add subtle L-shaped axis arrows for publication style."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(
        [xlim[0], xlim[1]], [ylim[0], ylim[0]],
        color="#26313D", linewidth=1.2, clip_on=False, zorder=10,
    )
    ax.plot(
        [xlim[0], xlim[0]], [ylim[0], ylim[1]],
        color="#26313D", linewidth=1.2, clip_on=False, zorder=10,
    )


def _fig_to_b64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def _fig_to_html_img(fig, width="100%") -> str:
    """Convert matplotlib figure to an HTML <img> tag."""
    b64 = _fig_to_b64(fig)
    return f'<img src="{b64}" style="width:{width};max-width:100%;border-radius:6px;" alt="chart"/>'


# ═══════════════════════════════════════════════════════════════
# Plotly helpers
# ═══════════════════════════════════════════════════════════════


def fig_to_json(fig) -> str:
    if fig is None:
        return "{}"
    return json.dumps(fig.to_dict() if hasattr(fig, "to_dict") else fig, default=str)


def _make_plotly_layout(title, xlabel, ylabel, height=520, **kwargs):
    return {
        "title": {"text": title, "x": 0.02, "xanchor": "left",
                  "font": {"family": "Arial, Helvetica, 'Noto Sans SC', sans-serif",
                           "size": 16, "color": "#111827"}},
        "xaxis": {"title": xlabel, "showline": True, "linewidth": 1.8, "linecolor": "#26313D",
                  "mirror": False, "ticks": "outside", "ticklen": 5, "showgrid": True,
                  "gridcolor": "rgba(31,41,55,0.07)", "gridwidth": 0.8},
        "yaxis": {"title": ylabel, "showline": True, "linewidth": 1.8, "linecolor": "#26313D",
                  "mirror": False, "ticks": "outside", "ticklen": 5, "showgrid": True,
                  "gridcolor": "rgba(31,41,55,0.07)", "gridwidth": 0.8},
        "plot_bgcolor": "#fafcfb", "paper_bgcolor": "#ffffff",
        "font": {"family": "Arial, Helvetica, 'Noto Sans SC', sans-serif", "color": "#111827", "size": 12},
        "legend": {"orientation": "h", "x": 0, "y": -0.18, "xanchor": "left",
                   "bgcolor": "rgba(255,255,255,0)", "borderwidth": 0,
                   "font": {"size": 11, "color": "#111827"}},
        "height": height, "margin": {"l": 65, "r": 30, "t": 55, "b": 65, "pad": 8},
        **kwargs,
    }


# ═══════════════════════════════════════════════════════════════
# High-SCI Publication Chart Generators (matplotlib)
# ═══════════════════════════════════════════════════════════════


def publication_scatter(x, y, groups=None, xlabel="X", ylabel="Y", title="",
                        palette=None, figsize=(9, 6)):
    """Publication-quality scatter plot with regression line."""
    palette = palette or CNS_PALETTE[:8]
    set_publication_style(figsize)
    fig, ax = plt.subplots()

    if groups is not None and len(np.unique(groups)) > 1:
        for i, grp in enumerate(np.unique(groups)):
            mask = groups == grp
            ax.scatter(np.array(x)[mask], np.array(y)[mask],
                       c=palette[i % len(palette)], alpha=0.75, s=28,
                       edgecolors="white", linewidth=0.3, label=str(grp), zorder=5)
        ax.legend(frameon=False, loc="best")
    else:
        ax.scatter(x, y, c=palette[0], alpha=0.70, s=32,
                   edgecolors="white", linewidth=0.3, zorder=5)

    # Add LOESS-like smooth if enough points
    if len(x) > 10:
        try:
            from scipy.interpolate import make_interp_spline
            x_arr, y_arr = np.array(x), np.array(y)
            sort_idx = np.argsort(x_arr)
            x_sorted, y_sorted = x_arr[sort_idx], y_arr[sort_idx]
            # Simple polynomial fit
            z = np.polyfit(x_sorted, y_sorted, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 200)
            ax.plot(x_smooth, p(x_smooth), color="#333333", linewidth=1.8,
                    linestyle="--", alpha=0.5, zorder=4)
        except Exception:
            pass

    ax.set_xlabel(xlabel, fontweight="500")
    ax.set_ylabel(ylabel, fontweight="500")
    ax.set_title(title, fontweight="bold", pad=12)
    sns.despine(ax=ax)
    _add_axis_arrows(ax)
    plt.tight_layout()
    return fig


def publication_line(x_data, y_data_dict, xlabel="X", ylabel="Y", title="",
                     palette=None, figsize=(9, 5.5)):
    """Publication-quality multi-line plot."""
    palette = palette or CNS_PALETTE[:8]
    set_publication_style(figsize)
    fig, ax = plt.subplots()

    for i, (label, y) in enumerate(y_data_dict.items()):
        x = x_data if len(x_data) == len(y) else np.arange(len(y))
        ax.plot(x, y, color=palette[i % len(palette)], linewidth=2.4,
                marker="o", markersize=5, markerfacecolor="white",
                markeredgewidth=1.5, markeredgecolor=palette[i % len(palette)],
                label=label, zorder=5)

    ax.set_xlabel(xlabel, fontweight="500")
    ax.set_ylabel(ylabel, fontweight="500")
    ax.set_title(title, fontweight="bold", pad=12)
    if len(y_data_dict) > 1:
        ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)
    _add_axis_arrows(ax)
    plt.tight_layout()
    return fig


def publication_bar(labels, values, groups=None, xlabel="", ylabel="", title="",
                    palette=None, figsize=(9, 5.5), horizontal=False):
    """Publication-quality bar/grouped bar chart."""
    palette = palette or CNS_PALETTE[:8]
    set_publication_style(figsize)
    fig, ax = plt.subplots()

    if groups is not None:
        df = pd.DataFrame({"label": labels, "value": values, "group": groups})
        unique_groups = df["group"].unique()
        unique_labels = df["label"].unique()
        n_groups = len(unique_groups)
        n_labels = len(unique_labels)
        bar_width = 0.8 / n_groups
        x_pos = np.arange(n_labels)

        for i, grp in enumerate(unique_groups):
            grp_data = df[df["group"] == grp]
            vals = [grp_data[grp_data["label"] == lbl]["value"].values
                    for lbl in unique_labels]
            vals = [v[0] if len(v) > 0 else 0 for v in vals]
            offset = (i - n_groups / 2 + 0.5) * bar_width
            bars = ax.bar(x_pos + offset, vals, bar_width * 0.9,
                          color=palette[i % len(palette)], alpha=0.88,
                          edgecolor="white", linewidth=0.5, label=str(grp), zorder=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(unique_labels)
        ax.legend(frameon=False, loc="best")
    else:
        bars = ax.bar(range(len(labels)), values, color=palette[:len(labels)],
                      alpha=0.88, edgecolor="white", linewidth=0.5, zorder=5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)

    ax.set_ylabel(ylabel, fontweight="500")
    ax.set_xlabel(xlabel, fontweight="500")
    ax.set_title(title, fontweight="bold", pad=12)
    sns.despine(ax=ax)
    _add_axis_arrows(ax)
    plt.tight_layout()
    return fig


def publication_box(data_dict, xlabel="", ylabel="", title="",
                    palette=None, figsize=(9, 5.5)):
    """Publication-quality box plot with overlaid swarm/strip."""
    palette = palette or CNS_PALETTE[:8]
    set_publication_style(figsize)
    fig, ax = plt.subplots()

    positions = list(range(len(data_dict)))
    for i, (label, values) in enumerate(data_dict.items()):
        bp = ax.boxplot(
            [values], positions=[i], widths=0.5, patch_artist=True,
            boxprops={"facecolor": palette[i % len(palette)], "alpha": 0.55,
                      "edgecolor": "#333333", "linewidth": 1.2},
            whiskerprops={"color": "#333333", "linewidth": 1.2},
            capprops={"color": "#333333", "linewidth": 1.2},
            medianprops={"color": "#ffffff", "linewidth": 2.0},
            flierprops={"marker": "o", "markerfacecolor": palette[i % len(palette)],
                        "markersize": 4, "alpha": 0.5},
        )

    ax.set_xticklabels(data_dict.keys())
    ax.set_ylabel(ylabel, fontweight="500")
    ax.set_xlabel(xlabel, fontweight="500")
    ax.set_title(title, fontweight="bold", pad=12)
    sns.despine(ax=ax)
    _add_axis_arrows(ax)
    plt.tight_layout()
    return fig


def publication_forest(estimates, ci_lower, ci_upper, labels,
                       title="Forest Plot", xlabel="Effect Size (95% CI)",
                       palette=None, figsize=(9, 6), ref_line=1.0):
    """Publication-quality forest plot for meta-analysis / regression."""
    palette = palette or CNS_PALETTE[:8]
    set_publication_style(figsize)
    fig, ax = plt.subplots()

    n = len(estimates)
    y_positions = list(range(n))
    colors = [palette[0] if (ci_lower[i] > ref_line or ci_upper[i] < ref_line)
              else "#999999" for i in range(n)]

    for i in range(n):
        ax.plot([ci_lower[i], ci_upper[i]], [i, i], color=colors[i], linewidth=2.5, zorder=5)
        ax.scatter(estimates[i], i, c=colors[i], s=60, zorder=6,
                   edgecolors="white", linewidth=0.8)
        # CI whiskers
        ax.plot([ci_lower[i], ci_lower[i]], [i - 0.12, i + 0.12],
                color=colors[i], linewidth=1.8)
        ax.plot([ci_upper[i], ci_upper[i]], [i - 0.12, i + 0.12],
                color=colors[i], linewidth=1.8)

    ax.axvline(x=ref_line, color="#333333", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel, fontweight="500")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True)
    _add_axis_arrows(ax)
    plt.tight_layout()
    return fig


def publication_roc(fpr_list, tpr_list, auc_list=None, labels=None,
                    title="ROC Curve", figsize=(7.5, 7)):
    """Publication-quality multi-model ROC comparison."""
    if not isinstance(fpr_list[0], (list, np.ndarray)):
        fpr_list = [fpr_list]
        tpr_list = [tpr_list]
    palette = CNS_PALETTE[:len(fpr_list)]
    set_publication_style(figsize)
    fig, ax = plt.subplots()

    for i in range(len(fpr_list)):
        lbl = (labels[i] if labels else f"Model {i + 1}")
        if auc_list and i < len(auc_list):
            lbl += f" (AUC={auc_list[i]:.3f})"
        ax.plot(fpr_list[i], tpr_list[i], color=palette[i], linewidth=2.2, label=lbl, zorder=5)
        # Fill under curve
        ax.fill_between(fpr_list[i], tpr_list[i], 0, color=palette[i], alpha=0.08)

    ax.plot([0, 1], [0, 1], color="#999999", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)
    ax.set_xlabel("1 - Specificity (FPR)", fontweight="500")
    ax.set_ylabel("Sensitivity (TPR)", fontweight="500")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.set_aspect("equal")
    sns.despine(ax=ax)
    _add_axis_arrows(ax)
    plt.tight_layout()
    return fig


def publication_calibration(prob_true, prob_pred, n_bins=10,
                            title="Calibration Curve", figsize=(7.5, 7)):
    """Publication-quality calibration curve with histogram."""
    set_publication_style(figsize)
    fig, ax = plt.subplots()

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_true = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (prob_pred >= bins[i]) & (prob_pred < bins[i + 1])
        bin_count[i] = mask.sum()
        if bin_count[i] > 0:
            bin_true[i] = prob_true[mask].mean()

    valid = bin_count > 0
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)
    ax.plot(bin_centers[valid], bin_true[valid], color=CNS_PALETTE[0],
            linewidth=2.5, marker="o", markersize=8, markerfacecolor="white",
            markeredgewidth=2.0, markeredgecolor=CNS_PALETTE[0], zorder=5)

    # Background histogram
    ax2 = ax.twinx()
    ax2.hist(prob_pred, bins=30, color="#BBBBBB", alpha=0.25, zorder=0)
    ax2.set_ylabel("Count", fontsize=9, color="#888888")
    ax2.tick_params(axis="y", colors="#888888", labelsize=8)

    ax.set_xlabel("Predicted Probability", fontweight="500")
    ax.set_ylabel("Observed Proportion", fontweight="500")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    sns.despine(ax=ax)
    sns.despine(ax=ax2)
    _add_axis_arrows(ax)
    plt.tight_layout()
    return fig


def publication_heatmap(matrix, row_labels=None, col_labels=None,
                        title="", cmap="RdYlBu_r", figsize=(10, 8),
                        annotate=True, center=None):
    """Publication-quality heatmap with annotations."""
    set_publication_style(figsize)
    fig, ax = plt.subplots()

    kwargs = {"cmap": cmap, "linewidths": 0.5, "linecolor": "#ffffff"}
    if center is not None:
        kwargs["center"] = center
    if annotate:
        kwargs["annot"] = True
        kwargs["fmt"] = ".2f"
        kwargs["annot_kws"] = {"fontsize": 7, "fontweight": "500"}

    if row_labels is not None:
        kwargs["yticklabels"] = row_labels
    if col_labels is not None:
        kwargs["xticklabels"] = col_labels

    sns.heatmap(matrix, ax=ax, robust=True, cbar_kws={"shrink": 0.78}, **kwargs)
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig


def publication_histogram(data_dict, bins=35, title="", xlabel="Value",
                          palette=None, figsize=(9, 5.5), alpha=0.55):
    """Publication-quality overlaid histograms / density plots."""
    palette = palette or CNS_PALETTE[:8]
    set_publication_style(figsize)
    fig, ax = plt.subplots()

    for i, (label, values) in enumerate(data_dict.items()):
        ax.hist(values, bins=bins, color=palette[i % len(palette)],
                alpha=alpha, edgecolor="white", linewidth=0.4,
                label=str(label), density=True, zorder=5)

    ax.set_xlabel(xlabel, fontweight="500")
    ax.set_ylabel("Density", fontweight="500")
    ax.set_title(title, fontweight="bold", pad=12)
    if len(data_dict) > 1:
        ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)
    _add_axis_arrows(ax)
    plt.tight_layout()
    return fig


def publication_confusion_matrix(cm, class_names=None, title="Confusion Matrix",
                                 figsize=(7, 6)):
    """Publication-quality confusion matrix heatmap."""
    return publication_heatmap(
        cm,
        row_labels=class_names,
        col_labels=class_names,
        title=title,
        cmap="Blues",
        figsize=figsize,
        annotate=True,
    )


# ═══════════════════════════════════════════════════════════════
# Plotly interactive chart generators
# ═══════════════════════════════════════════════════════════════


def plotly_scatter(x, y, groups=None, xlabel="X", ylabel="Y", title="",
                   palette=None, height=520):
    """Interactive Plotly scatter plot."""
    if not HAS_PLOTLY:
        return None
    palette = palette or CNS_PALETTE[:8]
    fig = go.Figure()
    if groups is not None:
        for i, grp in enumerate(sorted(set(groups))):
            mask = np.array(groups) == grp
            fig.add_trace(go.Scatter(
                x=np.array(x)[mask], y=np.array(y)[mask],
                mode="markers", name=str(grp),
                marker=dict(color=palette[i % len(palette)], size=10, opacity=0.78,
                            line=dict(color="white", width=0.6)),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(color=palette[0], size=10, opacity=0.78,
                        line=dict(color="white", width=0.6)),
        ))
    fig.update_layout(**_make_plotly_layout(title, xlabel, ylabel, height))
    return fig


def plotly_line(x, y_dict, xlabel="X", ylabel="Y", title="",
                palette=None, height=520):
    """Interactive Plotly multi-line chart."""
    if not HAS_PLOTLY:
        return None
    palette = palette or CNS_PALETTE[:8]
    fig = go.Figure()
    for i, (label, y) in enumerate(y_dict.items()):
        x_vals = x if len(x) == len(y) else list(range(len(y)))
        fig.add_trace(go.Scatter(
            x=x_vals, y=y, mode="lines+markers", name=str(label),
            line=dict(color=palette[i % len(palette)], width=2.8),
            marker=dict(color=palette[i % len(palette)], size=7,
                        line=dict(color="white", width=0.8)),
        ))
    fig.update_layout(**_make_plotly_layout(title, xlabel, ylabel, height))
    return fig


def plotly_bar(labels, values, groups=None, xlabel="", ylabel="", title="",
               palette=None, height=520):
    """Interactive Plotly bar chart."""
    if not HAS_PLOTLY:
        return None
    palette = palette or CNS_PALETTE[:8]
    fig = go.Figure()
    if groups is not None:
        for i, grp in enumerate(sorted(set(groups))):
            mask = np.array(groups) == grp
            fig.add_trace(go.Bar(
                x=np.array(labels)[mask], y=np.array(values)[mask],
                name=str(grp),
                marker_color=palette[i % len(palette)],
                marker_line=dict(color="white", width=0.6),
            ))
    else:
        fig.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=palette[:len(labels)],
            marker_line=dict(color="white", width=0.6),
        ))
    fig.update_layout(**_make_plotly_layout(title, xlabel, ylabel, height))
    return fig


def plotly_box(data_dict, xlabel="", ylabel="", title="",
               palette=None, height=520):
    """Interactive Plotly box plot."""
    if not HAS_PLOTLY:
        return None
    palette = palette or CNS_PALETTE[:8]
    fig = go.Figure()
    for i, (label, values) in enumerate(data_dict.items()):
        fig.add_trace(go.Box(
            y=values, name=str(label),
            marker_color=palette[i % len(palette)],
            line=dict(color=palette[i % len(palette)], width=2),
            fillcolor=palette[i % len(palette)].replace(")", ",0.2)").replace("rgb", "rgba"),
        ))
    fig.update_layout(**_make_plotly_layout(title, xlabel, ylabel, height))
    return fig


def plotly_violin(data_dict, xlabel="", ylabel="", title="",
                  palette=None, height=520):
    """Interactive Plotly violin plot."""
    if not HAS_PLOTLY:
        return None
    palette = palette or CNS_PALETTE[:8]
    fig = go.Figure()
    for i, (label, values) in enumerate(data_dict.items()):
        fig.add_trace(go.Violin(
            y=values, name=str(label),
            line_color=palette[i % len(palette)],
            fillcolor=palette[i % len(palette)],
            opacity=0.65, meanline_visible=True,
        ))
    fig.update_layout(**_make_plotly_layout(title, xlabel, ylabel, height))
    return fig


def plotly_heatmap(matrix, row_labels=None, col_labels=None, title="",
                   cmap="RdYlBu_r", height=560):
    """Interactive Plotly heatmap."""
    if not HAS_PLOTLY:
        return None
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=col_labels, y=row_labels,
        colorscale=cmap, hoverongaps=False,
        colorbar=dict(thickness=16, len=0.8, outlinewidth=0),
    ))
    fig.update_layout(**_make_plotly_layout(title, "", "", height))
    return fig


def plotly_histogram(data_dict, xlabel="Value", title="", bins=35,
                     palette=None, height=520):
    """Interactive Plotly histogram with overlaid traces."""
    if not HAS_PLOTLY:
        return None
    palette = palette or CNS_PALETTE[:8]
    fig = go.Figure()
    for i, (label, values) in enumerate(data_dict.items()):
        fig.add_trace(go.Histogram(
            x=values, nbinsx=bins, name=str(label),
            marker_color=palette[i % len(palette)], opacity=0.6,
            histnorm="probability density",
        ))
    fig.update_layout(**_make_plotly_layout(title, xlabel, "Density", height),
                      barmode="overlay")
    return fig


def plotly_roc(fpr_list, tpr_list, auc_list=None, labels=None,
               title="ROC Curve", height=540):
    """Interactive Plotly ROC comparison."""
    if not HAS_PLOTLY:
        return None
    if not isinstance(fpr_list[0], (list, np.ndarray)):
        fpr_list = [fpr_list]
        tpr_list = [tpr_list]
    palette = CNS_PALETTE[:len(fpr_list)]
    fig = go.Figure()
    for i in range(len(fpr_list)):
        lbl = (labels[i] if labels else f"Model {i + 1}")
        if auc_list and i < len(auc_list):
            lbl += f" (AUC={auc_list[i]:.3f})"
        fig.add_trace(go.Scatter(
            x=fpr_list[i], y=tpr_list[i], mode="lines", name=lbl,
            line=dict(color=palette[i], width=2.5),
            fill="tozeroy", fillcolor=palette[i].replace(")", ",0.08)"),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#AAAAAA", width=1, dash="dash"),
        showlegend=False,
    ))
    fig.update_layout(**_make_plotly_layout(title, "1 - Specificity", "Sensitivity", height))
    fig.update_xaxes(range=[-0.02, 1.02])
    fig.update_yaxes(range=[-0.02, 1.02], scaleanchor="x", scaleratio=1)
    return fig


def plotly_forest(estimates, ci_lower, ci_upper, labels,
                  title="Forest Plot", xlabel="Effect Size (95% CI)",
                  palette=None, height=550, ref_line=1.0):
    """Interactive Plotly forest plot."""
    if not HAS_PLOTLY:
        return None
    palette = palette or CNS_PALETTE[:8]
    n = len(estimates)
    fig = go.Figure()
    colors = [palette[0] if (ci_lower[i] > ref_line or ci_upper[i] < ref_line)
              else "#AAAAAA" for i in range(n)]

    for i in range(n):
        fig.add_trace(go.Scatter(
            x=[ci_lower[i], ci_upper[i]], y=[i, i],
            mode="lines", line=dict(color=colors[i], width=3),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[estimates[i]], y=[i],
            mode="markers", marker=dict(color=colors[i], size=12,
                                        line=dict(color="white", width=1)),
            showlegend=False,
        ))

    fig.add_vline(x=ref_line, line_dash="dash", line_color="#666666", line_width=1)
    fig.update_layout(**_make_plotly_layout(title, xlabel, "", height))
    fig.update_yaxes(tickvals=list(range(n)), ticktext=labels, automargin=True)
    fig.update_yaxes(autorange="reversed")
    return fig


# ═══════════════════════════════════════════════════════════════
# Composite chart generators for specific analysis types
# ═══════════════════════════════════════════════════════════════


def make_coefficient_plot(coef_dict, errors_dict=None, title="Coefficient Plot",
                          palette=None, height=520):
    """Plotly coefficient plot (horizontal bar) with optional error bars."""
    if not HAS_PLOTLY:
        return None
    palette = palette or CNS_PALETTE[:8]
    names = list(coef_dict.keys())
    values = list(coef_dict.values())
    colors = [palette[0] if v > 0 else palette[1] for v in values]
    fig = go.Figure()
    if errors_dict:
        err_vals = [errors_dict.get(k, 0) for k in names]
        fig.add_trace(go.Bar(
            y=names, x=values, orientation="h",
            error_x=dict(type="data", array=err_vals, visible=True, thickness=2, width=3),
            marker_color=colors, marker_line=dict(color="white", width=0.5),
            text=[f"{v:.3f}" for v in values], textposition="outside",
        ))
    else:
        fig.add_trace(go.Bar(
            y=names, x=values, orientation="h",
            marker_color=colors, marker_line=dict(color="white", width=0.5),
            text=[f"{v:.3f}" for v in values], textposition="outside",
        ))
    fig.update_layout(**_make_plotly_layout(title, "Coefficient Value", "", height))
    fig.update_yaxes(automargin=True)
    fig.add_vline(x=0, line_dash="dash", line_color="#666666", line_width=1)
    return fig


def make_reg_path(alphas, coef_paths, feature_names=None, title="Regularization Path",
                  palette=None, height=520):
    """Plotly regularization path for Lasso/Ridge."""
    if not HAS_PLOTLY:
        return None
    palette = palette or NATURE_PALETTE[:8]
    fig = go.Figure()
    n_features = coef_paths.shape[1]
    for i in range(n_features):
        label = feature_names[i] if feature_names and i < len(feature_names) else None
        color = palette[i % len(palette)] if i < 8 else None
        fig.add_trace(go.Scatter(
            x=alphas, y=coef_paths[:, i], mode="lines",
            name=label, line=dict(color=color, width=1.8),
            showlegend=bool(label and i < 10),
        ))
    fig.update_layout(**_make_plotly_layout(title, "Alpha (log scale)", "Coefficient", height))
    fig.update_xaxes(type="log")
    return fig


def make_shap_summary(shap_values, feature_names, title="SHAP Feature Importance", height=520):
    """Plotly SHAP summary bar chart."""
    if not HAS_PLOTLY:
        return None
    mean_abs = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_abs)[-20:]
    names_sorted = [feature_names[i] if feature_names and i < len(feature_names)
                    else f"F{i}" for i in indices]
    vals_sorted = mean_abs[indices]
    colors = CNS_PALETTE[:len(indices)]
    fig = go.Figure(go.Bar(
        y=names_sorted, x=vals_sorted, orientation="h",
        marker_color=colors[:len(indices)],
        marker_line=dict(color="white", width=0.5),
    ))
    fig.update_layout(**_make_plotly_layout(title, "Mean |SHAP Value|", "", height))
    fig.update_yaxes(automargin=True)
    return fig


def make_learning_curve(train_scores, val_scores, train_sizes, title="Learning Curve", height=520):
    """Plotly learning curve."""
    if not HAS_PLOTLY:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_scores.mean(axis=1), mode="lines+markers",
        name="Training", line=dict(color=CNS_PALETTE[0], width=2.5),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_scores.mean(axis=1), mode="lines+markers",
        name="Validation", line=dict(color=CNS_PALETTE[1], width=2.5),
        marker=dict(size=6),
    ))
    fig.update_layout(**_make_plotly_layout(title, "Training Samples", "Score", height))
    return fig


def make_smd_plot(vars_before, smd_before, smd_after, title="SMD Before/After Matching", height=520):
    """Plotly SMD (Standardized Mean Difference) plot for PSM."""
    if not HAS_PLOTLY:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=smd_before, y=vars_before, mode="markers",
        name="Before Matching", marker=dict(color=CNS_PALETTE[1], size=10, symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=smd_after, y=vars_before, mode="markers",
        name="After Matching", marker=dict(color=CNS_PALETTE[0], size=10, symbol="diamond"),
    ))
    fig.add_vline(x=0.1, line_dash="dash", line_color="#666666", line_width=1,
                  annotation_text="SMD=0.1")
    fig.update_layout(**_make_plotly_layout(title, "Standardized Mean Difference", "", height))
    fig.update_yaxes(automargin=True)
    return fig


def make_state_prob_plot(time_points, prob_matrix, state_labels, title="State Probabilities", height=520):
    """Plotly state probability over time (Markov model)."""
    if not HAS_PLOTLY:
        return None
    palette = CNS_PALETTE[:prob_matrix.shape[1]]
    fig = go.Figure()
    for j in range(prob_matrix.shape[1]):
        fig.add_trace(go.Scatter(
            x=time_points, y=prob_matrix[:, j], mode="lines",
            name=state_labels[j] if state_labels else f"State {j}",
            line=dict(color=palette[j % len(palette)], width=2.5),
            stackgroup="one" if j == 0 else None,
        ))
    fig.update_layout(**_make_plotly_layout(title, "Time", "Probability", height))
    return fig


def make_funnel_plot(se, effect_sizes, title="Funnel Plot", height=520):
    """Plotly funnel plot for meta-analysis (publication bias)."""
    if not HAS_PLOTLY:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=effect_sizes, y=1 / np.array(se), mode="markers",
        marker=dict(color=CNS_PALETTE[0], size=10, opacity=0.7,
                    line=dict(color="white", width=0.5)),
        showlegend=False,
    ))
    mean_eff = np.mean(effect_sizes)
    max_se = max(1 / np.array(se))
    fig.add_trace(go.Scatter(
        x=[mean_eff, mean_eff], y=[0, max_se * 1.1],
        mode="lines", line=dict(color="#666666", width=1.5, dash="dash"),
        showlegend=False,
    ))
    fig.update_layout(**_make_plotly_layout(title, "Effect Size", "1/SE (Precision)", height))
    return fig


def make_path_diagram(coef_dict, title="Path Coefficients", height=520):
    """Plotly Sankey-like path diagram for mediation analysis."""
    if not HAS_PLOTLY:
        return None
    # Simple coefficient bar chart as path representation
    names = list(coef_dict.keys())
    values = [abs(v) for v in coef_dict.values()]
    colors = [CNS_PALETTE[0] if coef_dict[k] > 0 else CNS_PALETTE[1] for k in names]
    fig = go.Figure(go.Bar(
        y=names, x=values, orientation="h",
        marker_color=colors, marker_line=dict(color="white", width=0.5),
        text=[f"{coef_dict[k]:.3f}" for k in names], textposition="outside",
    ))
    fig.update_layout(**_make_plotly_layout(title, "Absolute Coefficient", "", height))
    fig.update_yaxes(automargin=True)
    return fig


def make_descriptive_charts(df: pd.DataFrame) -> list[dict]:
    """Generate descriptive charts for data overview."""
    charts = []
    if not HAS_PLOTLY:
        return charts

    var_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
            var_types.setdefault("continuous", []).append(col)
        elif df[col].nunique() <= 15:
            var_types.setdefault("categorical", []).append(col)

    for col in var_types.get("continuous", [])[:3]:
        vals = df[col].dropna()
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=30, marker_color=CNS_PALETTE[0], opacity=0.7, name=col,
        ))
        fig.update_layout(**_make_plotly_layout(f"{col} 分布", col, "频数", 420))
        charts.append({"title": f"{col}分布", "plotly": fig_to_json(fig)})

    for col in var_types.get("categorical", [])[:1]:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=counts[col], y=counts["count"], marker_color=CNS_PALETTE[0]))
        fig.update_layout(**_make_plotly_layout(f"{col} 分布", col, "频数", 420))
        charts.append({"title": f"{col}分布", "plotly": fig_to_json(fig)})

    return charts
