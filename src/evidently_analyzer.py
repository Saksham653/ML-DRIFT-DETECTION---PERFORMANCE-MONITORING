import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "float"):
    np.float = np.float64
import time
import plotly.io as pio
from src.visualizer import feature_drift_heatmap, COLORS
from src.drift_detector import DriftDetector
from typing import Any
from typing import Optional, List
import os


def _evidently_imports():
    import importlib
    import numpy as _np
    if not hasattr(_np, "float_"):
        _np.float_ = _np.float64
    if not hasattr(_np, "float"):
        _np.float = _np.float64
    R = importlib.import_module("evidently.report").Report
    mp = importlib.import_module("evidently.metric_preset")
    DDP = mp.DataDriftPreset
    TDP = mp.TargetDriftPreset
    CP = mp.ClassificationPreset
    DQP = mp.DataQualityPreset
    CM = importlib.import_module("evidently").ColumnMapping
    return R, DDP, TDP, CP, DQP, CM


class EvidentlyAnalyzer:
    """Wrapper for Evidently AI drift reports with universal dataset support."""

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    def build_column_mapping(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str],
        target_col: Optional[str] = None,
        prediction_col: Optional[str] = None,
        id_col: Optional[str] = None,
    ) -> "ColumnMapping":
        """Build Evidently ColumnMapping from detected column types."""
        try:
            _, _, _, _, _, ColumnMapping = _evidently_imports()
            cm = ColumnMapping()
            cm.numerical_features = [c for c in numerical_cols if c in df.columns]
            cm.categorical_features = [c for c in categorical_cols if c in df.columns]
            if target_col and target_col in df.columns:
                cm.target = target_col
            if prediction_col and prediction_col in df.columns:
                cm.prediction = prediction_col
            if id_col:
                cm.id = id_col
            return cm
        except Exception:
            return {
                "numerical_features": [c for c in numerical_cols if c in df.columns],
                "categorical_features": [c for c in categorical_cols if c in df.columns],
                "target": target_col if target_col and target_col in df.columns else None,
                "prediction": prediction_col if prediction_col and prediction_col in df.columns else None,
                "id": id_col,
            }

    def _load_html(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _render_html(self, html_data: str, height: int = 2000):
        components.html(html_data, height=height, scrolling=True)

    def run_data_drift_report(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column_mapping: Optional[ColumnMapping] = None,
        save: bool = True,
    ) -> str:
        """Run and optionally save data drift report."""
        path = os.path.join(self.reports_dir, "data_drift.html")
        try:
            Report, DataDriftPreset, _, _, _, _ = _evidently_imports()
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
            if save:
                report.save_html(path)
            return path
        except Exception:
            results = st.session_state.get("drift_results")
            if results is None:
                try:
                    detector = DriftDetector()
                    if isinstance(column_mapping, dict):
                        num = column_mapping.get("numerical_features", [])
                        cat = column_mapping.get("categorical_features", [])
                    else:
                        num = getattr(column_mapping, "numerical_features", []) or []
                        cat = getattr(column_mapping, "categorical_features", []) or []
                    results = detector.compute_full_drift(reference, current, num, cat)
                except Exception:
                    results = None
            if results is None:
                raise RuntimeError("Unable to compute drift results for fallback report")
            fig = feature_drift_heatmap(results)
            fig.update_layout(annotations=[dict(
                text=f"© {time.strftime('%Y')} Saksham Srivastava · sakshamsrivastava7000@gmail.com",
                x=1, y=-0.15, xref="paper", yref="paper", showarrow=False,
                font=dict(color=COLORS.get('muted', '#64748B'), size=12)
            )])
            pio.write_html(fig, file=path, include_plotlyjs="cdn", full_html=True)
            return path

    def run_data_quality_report(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column_mapping: Optional[ColumnMapping] = None,
    ) -> str:
        path = os.path.join(self.reports_dir, "data_quality.html")
        try:
            Report, _, _, _, DataQualityPreset, _ = _evidently_imports()
            report = Report(metrics=[DataQualityPreset()])
            report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
            report.save_html(path)
            return path
        except Exception:
            df = reference.copy()
            nulls = df.isnull().mean().sort_values(ascending=False).head(20)
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(x=nulls.index.tolist(), y=(nulls.values * 100).tolist(), marker_color=COLORS["warning"]))
            fig.update_layout(title=dict(text="Data Quality: Top Missing Columns"), xaxis_title="Column", yaxis_title="Missing %")
            fig.update_layout(annotations=[dict(
                text=f"© {time.strftime('%Y')} Saksham Srivastava · sakshamsrivastava7000@gmail.com",
                x=1, y=-0.15, xref="paper", yref="paper", showarrow=False,
                font=dict(color=COLORS.get('muted', '#64748B'), size=12)
            )])
            pio.write_html(fig, file=path, include_plotlyjs="cdn", full_html=True)
            return path

    def run_target_drift_report(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column_mapping: Optional[ColumnMapping] = None,
    ) -> str:
        path = os.path.join(self.reports_dir, "target_drift.html")
        try:
            Report, _, TargetDriftPreset, _, _, _ = _evidently_imports()
            report = Report(metrics=[TargetDriftPreset()])
            report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
            report.save_html(path)
            return path
        except Exception:
            target = (column_mapping or {}).get("target")
            if target and target in reference.columns and target in current.columns:
                import plotly.graph_objects as go
                ref = reference[target].value_counts(normalize=True)
                cur = current[target].value_counts(normalize=True)
                cats = list(set(ref.index) | set(cur.index))
                fig = go.Figure()
                fig.add_trace(go.Bar(x=cats, y=[ref.get(c, 0) for c in cats], name="Reference", marker_color=COLORS["reference"]))
                fig.add_trace(go.Bar(x=cats, y=[cur.get(c, 0) for c in cats], name="Current", marker_color=COLORS["current"]))
                fig.update_layout(title=dict(text="Target Drift"), barmode="group", xaxis_title="Class", yaxis_title="Share")
                fig.update_layout(annotations=[dict(
                    text=f"© {time.strftime('%Y')} Saksham Srivastava · sakshamsrivastava7000@gmail.com",
                    x=1, y=-0.15, xref="paper", yref="paper", showarrow=False,
                    font=dict(color=COLORS.get('muted', '#64748B'), size=12)
                )])
                pio.write_html(fig, file=path, include_plotlyjs="cdn", full_html=True)
                return path
            raise RuntimeError("Target column not available for fallback target drift report")

    def run_classification_report(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column_mapping: Optional[ColumnMapping] = None,
    ) -> str:
        path = os.path.join(self.reports_dir, "classification.html")
        try:
            Report, _, _, ClassificationPreset, _, _ = _evidently_imports()
            report = Report(metrics=[ClassificationPreset()])
            report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
            report.save_html(path)
            return path
        except Exception:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.update_layout(title=dict(text="Classification Report Unavailable"), annotations=[
                dict(text="Fallback: train model and generate prediction-based charts", showarrow=False, x=0.5, y=0.55),
                dict(text=f"© {time.strftime('%Y')} Saksham Srivastava · sakshamsrivastava7000@gmail.com", showarrow=False, x=0.5, y=0.45, font=dict(color=COLORS.get('muted', '#64748B'), size=12))
            ])  
            pio.write_html(fig, file=path, include_plotlyjs="cdn", full_html=True)
            return path

    def display_report(self, path: str, height: int = 2000):
        """Display an existing HTML report in Streamlit."""
        if os.path.exists(path):
            html_data = self._load_html(path)
            self._render_html(html_data, height)
        else:
            st.error(f"Report not found: {path}")

    def get_report_bytes(self, path: str) -> bytes:
        """Read report as bytes for download."""
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        return b""
