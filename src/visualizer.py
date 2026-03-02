import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

# ─── Color Palette ─────────────────────────────────────────────────────────
COLORS = {
    "primary": "#00D4FF",
    "secondary": "#7B2FBE",
    "accent": "#FF6B35",
    "success": "#00C48C",
    "warning": "#FFB800",
    "danger": "#FF4757",
    "bg": "#0A0E1A",
    "card": "#111827",
    "text": "#E2E8F0",
    "muted": "#64748B",
    "reference": "#00D4FF",
    "current": "#FF6B35",
    "stable": "#00C48C",
    "monitor": "#FFB800",
    "high_risk": "#FF4757",
}

AXIS_STYLE = dict(
    gridcolor="rgba(100,116,139,0.2)",
    linecolor="rgba(100,116,139,0.3)",
)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.6)",
    font=dict(family="'Space Mono', monospace", color=COLORS["text"], size=12),
    xaxis=AXIS_STYLE,
)


def drift_status_color(drift: bool) -> str:
    return COLORS["danger"] if drift else COLORS["success"]


def feature_drift_heatmap(drift_results: Dict) -> go.Figure:
    """Create a heatmap of drift scores across all features."""
    features = []
    drift_flags = []
    psi_scores = []
    ks_pvals = []

    for col, res in drift_results.items():
        features.append(col)
        drift_flags.append(1 if res.get("drift_detected") else 0)
        if res.get("type") == "numerical":
            psi_scores.append(res.get("psi", {}).get("psi") or 0)
            ks_pvals.append(res.get("ks", {}).get("p_value") or 1.0)
        else:
            psi_scores.append(res.get("distribution", {}).get("max_shift") or 0)
            ks_pvals.append(res.get("chi2", {}).get("p_value") or 1.0)

    colors = [COLORS["danger"] if d else COLORS["success"] for d in drift_flags]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=psi_scores,
        y=features,
        orientation="h",
        marker_color=colors,
        name="Drift Score",
        text=[f"{'DRIFT' if d else 'OK'}" for d in drift_flags],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Feature Drift Scores", font=dict(size=16, color=COLORS["primary"])),
        height=max(300, len(features) * 28),
        showlegend=False,
        xaxis_title="Drift Score",
    )
    return fig


def distribution_comparison(reference: pd.Series, current: pd.Series,
                              col_name: str, col_type: str = "numerical") -> go.Figure:
    """Plot distribution comparison between reference and current data."""
    fig = go.Figure()

    if col_type == "numerical":
        # KDE-style histograms
        fig.add_trace(go.Histogram(
            x=reference.dropna(),
            name="Reference",
            histnorm="probability density",
            marker_color=COLORS["reference"],
            opacity=0.6,
            nbinsx=30,
        ))
        fig.add_trace(go.Histogram(
            x=current.dropna(),
            name="Current",
            histnorm="probability density",
            marker_color=COLORS["current"],
            opacity=0.6,
            nbinsx=30,
        ))
        fig.update_layout(barmode="overlay")
    else:
        ref_counts = reference.dropna().value_counts(normalize=True).head(15)
        cur_counts = current.dropna().value_counts(normalize=True).head(15)
        all_cats = list(set(ref_counts.index) | set(cur_counts.index))

        fig.add_trace(go.Bar(
            x=all_cats,
            y=[ref_counts.get(c, 0) for c in all_cats],
            name="Reference",
            marker_color=COLORS["reference"],
            opacity=0.8,
        ))
        fig.add_trace(go.Bar(
            x=all_cats,
            y=[cur_counts.get(c, 0) for c in all_cats],
            name="Current",
            marker_color=COLORS["current"],
            opacity=0.8,
        ))
        fig.update_layout(barmode="group")

    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=f"Distribution: {col_name}", font=dict(size=14, color=COLORS["primary"])),
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def model_performance_gauge(metric_value: float, metric_name: str,
                              threshold: float = 0.8) -> go.Figure:
    """Gauge chart for a single performance metric."""
    color = COLORS["success"] if metric_value >= threshold else (
        COLORS["warning"] if metric_value >= threshold * 0.8 else COLORS["danger"]
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=metric_value * 100,
        title={"text": metric_name, "font": {"color": COLORS["text"], "size": 14}},
        delta={"reference": threshold * 100, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": COLORS["muted"]}},
            "bar": {"color": color, "thickness": 0.8},
            "bgcolor": COLORS["card"],
            "bordercolor": COLORS["muted"],
            "steps": [
                {"range": [0, threshold * 80], "color": "rgba(255,71,87,0.15)"},
                {"range": [threshold * 80, threshold * 100], "color": "rgba(255,184,0,0.15)"},
                {"range": [threshold * 100, 100], "color": "rgba(0,196,140,0.15)"},
            ],
        },
        number={"suffix": "%", "font": {"color": color, "size": 24}},
    ))
    fig.update_layout(height=220)
    return fig


def drift_timeline(drift_history: List[Dict]) -> go.Figure:
    """Line chart of drift score over time (sessions)."""
    if not drift_history:
        return go.Figure()

    sessions = [f"Session {i+1}" for i in range(len(drift_history))]
    scores = [d.get("drift_ratio", 0) for d in drift_history]
    statuses = [d.get("status", "STABLE") for d in drift_history]
    colors = [
        COLORS["success"] if s == "STABLE" else
        COLORS["warning"] if s == "MONITOR" else COLORS["danger"]
        for s in statuses
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sessions, y=scores,
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(color=colors, size=12, line=dict(color="white", width=2)),
        hovertemplate="<b>%{x}</b><br>Drift: %{y:.1f}%<extra></extra>",
        name="Drift %"
    ))
    # Threshold lines
    fig.add_hline(y=20, line_dash="dash", line_color=COLORS["warning"],
                   annotation_text="Monitor Threshold (20%)", annotation_font_color=COLORS["warning"])
    fig.add_hline(y=50, line_dash="dash", line_color=COLORS["danger"],
                   annotation_text="High Risk Threshold (50%)", annotation_font_color=COLORS["danger"])

    fig.update_layout(title=dict(text="Drift Score Over Time", font=dict(size=16, color=COLORS["primary"])),
                      height=300,
                      yaxis_title="Drifted Features (%)")
    fig.update_yaxes(range=[0, 105], **AXIS_STYLE)
    return fig


def feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    top = importance_df.head(15)

    fig = go.Figure(go.Bar(
        x=top["importance"],
        y=top["feature"],
        orientation="h",
        marker=dict(
            color=top["importance"],
            colorscale=[[0, COLORS["secondary"]], [0.5, COLORS["primary"]], [1, COLORS["accent"]]],
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Feature Importances", font=dict(size=16, color=COLORS["primary"])),
        height=max(300, len(top) * 28),
        xaxis_title="Importance Score",
    )
    fig.update_yaxes(autorange="reversed", **AXIS_STYLE)
    return fig


def prediction_drift_chart(ref_preds: np.ndarray, cur_preds: np.ndarray,
                             task_type: str = "classification") -> go.Figure:
    """Compare prediction distributions."""
    fig = go.Figure()
    if task_type == "classification":
        # Bar chart of class distributions
        ref_classes, ref_counts = np.unique(ref_preds, return_counts=True)
        cur_classes, cur_counts = np.unique(cur_preds, return_counts=True)
        all_classes = list(set(ref_classes) | set(cur_classes))

        fig.add_trace(go.Bar(
            x=[str(c) for c in all_classes],
            y=[ref_counts[list(ref_classes).index(c)] / len(ref_preds) if c in ref_classes else 0 for c in all_classes],
            name="Reference", marker_color=COLORS["reference"], opacity=0.8,
        ))
        fig.add_trace(go.Bar(
            x=[str(c) for c in all_classes],
            y=[cur_counts[list(cur_classes).index(c)] / len(cur_preds) if c in cur_classes else 0 for c in all_classes],
            name="Current", marker_color=COLORS["current"], opacity=0.8,
        ))
        fig.update_layout(barmode="group")
    else:
        fig.add_trace(go.Histogram(
            x=ref_preds, name="Reference",
            marker_color=COLORS["reference"], opacity=0.6, histnorm="probability density",
        ))
        fig.add_trace(go.Histogram(
            x=cur_preds, name="Current",
            marker_color=COLORS["current"], opacity=0.6, histnorm="probability density",
        ))
        fig.update_layout(barmode="overlay")

    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Prediction Distribution Shift", font=dict(size=16, color=COLORS["primary"])),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def dataset_overview_chart(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> go.Figure:
    """Overview comparison: row count, missing values, unique counts."""
    metrics = ["Row Count", "Missing %", "Duplicate Rows"]
    ref_vals = [
        len(ref_df),
        round(ref_df.isnull().mean().mean() * 100, 2),
        ref_df.duplicated().sum(),
    ]
    cur_vals = [
        len(cur_df),
        round(cur_df.isnull().mean().mean() * 100, 2),
        cur_df.duplicated().sum(),
    ]

    fig = go.Figure(data=[
        go.Bar(name="Reference", x=metrics, y=ref_vals, marker_color=COLORS["reference"]),
        go.Bar(name="Current", x=metrics, y=cur_vals, marker_color=COLORS["current"]),
    ])
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Dataset Comparison Overview", font=dict(size=16, color=COLORS["primary"])),
        barmode="group",
        height=300,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, numerical_cols: List[str]) -> go.Figure:
    """Correlation heatmap for numerical features."""
    cols = [c for c in numerical_cols if c in df.columns]
    if len(cols) < 2:
        return go.Figure()

    corr = df[cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="<b>%{x} × %{y}</b><br>Corr: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Feature Correlations", font=dict(size=16, color=COLORS["primary"])),
        height=max(350, len(cols) * 40),
    )
    return fig
