"""
╔══════════════════════════════════════════════════════════════════╗
║   DRIFTWATCH  ·  Production ML Drift Detection & Monitoring     ║
║   Universal · Real-time · Multi-method · Evidently-powered      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO, BytesIO
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "float"):
    np.float = np.float64
OWNER_NAME = "Saksham Srivastava"
OWNER_EMAIL = "sakshamsrivastava7000@gmail.com"

# ─── Page Config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="DriftWatch · ML Monitoring",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Source Modules ───────────────────────────────────────────────────────────
from src.data_processor import DataProcessor
from src.drift_detector import DriftDetector
from src.model_manager import ModelManager
from src.visualizer import (
    feature_drift_heatmap, distribution_comparison, model_performance_gauge,
    drift_timeline, feature_importance_chart, prediction_drift_chart,
    dataset_overview_chart, correlation_heatmap, COLORS
)
from src.evidently_analyzer import EvidentlyAnalyzer

# ─── CSS Injection ────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg: #060B17;
        --card: #0D1526;
        --card2: #111E35;
        --border: rgba(0,212,255,0.12);
        --primary: #00D4FF;
        --accent: #FF6B35;
        --success: #00C48C;
        --warning: #FFB800;
        --danger: #FF4757;
        --text: #E2E8F0;
        --muted: #64748B;
        --font-mono: 'Space Mono', monospace;
        --font-sans: 'Syne', sans-serif;
    }

    /* ── Global Reset ── */
    html, body, [class*="css"] { font-family: var(--font-sans) !important; }
    .stApp { background: var(--bg) !important; }
    .main .block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1400px !important; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1020 0%, #060B17 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }

    /* ── Header / Logo ── */
    .dw-header {
        background: linear-gradient(135deg, rgba(0,212,255,0.06) 0%, rgba(123,47,190,0.06) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.4rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        backdrop-filter: blur(10px);
    }
    .dw-logo { font-size: 2.2rem; }
    .dw-title {
        font-family: var(--font-sans) !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, var(--primary), #7B2FBE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 !important;
        line-height: 1 !important;
    }
    .dw-subtitle { color: var(--muted); font-size: 0.78rem; font-family: var(--font-mono); margin-top: 4px; }

    /* ── Status Badge ── */
    .status-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 14px; border-radius: 20px;
        font-family: var(--font-mono) !important;
        font-size: 0.75rem; font-weight: 700; letter-spacing: 1px;
    }
    .status-stable { background: rgba(0,196,140,0.15); color: var(--success); border: 1px solid rgba(0,196,140,0.35); }
    .status-monitor { background: rgba(255,184,0,0.15); color: var(--warning); border: 1px solid rgba(255,184,0,0.35); }
    .status-high_risk, .status-high-risk { background: rgba(255,71,87,0.15); color: var(--danger); border: 1px solid rgba(255,71,87,0.35); }
    .status-none { background: rgba(100,116,139,0.15); color: var(--muted); border: 1px solid rgba(100,116,139,0.25); }

    /* ── Metric Cards ── */
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 1rem 0; }
    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); border-color: rgba(0,212,255,0.3); }
    .metric-value { font-family: var(--font-mono) !important; font-size: 1.6rem; font-weight: 700; color: var(--primary); }
    .metric-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

    /* ── Feature Drift Table ── */
    .drift-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    .drift-table th {
        background: rgba(0,212,255,0.08);
        color: var(--primary);
        font-family: var(--font-mono) !important;
        font-size: 0.7rem;
        letter-spacing: 0.5px;
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }
    .drift-table td {
        padding: 9px 14px;
        border-bottom: 1px solid rgba(100,116,139,0.1);
        color: var(--text);
        font-family: var(--font-mono) !important;
        font-size: 0.75rem;
    }
    .drift-table tr:hover td { background: rgba(0,212,255,0.04); }
    .tag-drift { background: rgba(255,71,87,0.2); color: #FF6B7A; border-radius: 4px; padding: 2px 8px; }
    .tag-ok { background: rgba(0,196,140,0.2); color: #00E6A8; border-radius: 4px; padding: 2px 8px; }
    .tag-num { background: rgba(0,212,255,0.15); color: var(--primary); border-radius: 4px; padding: 2px 6px; font-size: 0.65rem; }
    .tag-cat { background: rgba(123,47,190,0.2); color: #A06FE2; border-radius: 4px; padding: 2px 6px; font-size: 0.65rem; }

    /* ── Section Headers ── */
    .section-header {
        display: flex; align-items: center; gap: 10px;
        margin: 1.5rem 0 0.8rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 8px;
    }
    .section-title {
        font-family: var(--font-sans) !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: var(--primary) !important;
        margin: 0 !important;
    }
    .section-icon { font-size: 1.1rem; }

    /* ── Upload Zone ── */
    .upload-zone {
        border: 2px dashed rgba(0,212,255,0.25);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        background: rgba(0,212,255,0.03);
        transition: border-color 0.2s;
        margin-bottom: 0.5rem;
    }
    .upload-zone:hover { border-color: rgba(0,212,255,0.5); }
    .upload-icon { font-size: 2rem; margin-bottom: 6px; }
    .upload-label { color: var(--muted); font-size: 0.8rem; font-family: var(--font-mono); }

    /* ── Alert Boxes ── */
    .alert { padding: 12px 16px; border-radius: 10px; font-size: 0.82rem; margin: 8px 0; }
    .alert-info { background: rgba(0,212,255,0.08); border-left: 3px solid var(--primary); color: var(--text); }
    .alert-warn { background: rgba(255,184,0,0.08); border-left: 3px solid var(--warning); color: var(--text); }
    .alert-danger { background: rgba(255,71,87,0.08); border-left: 3px solid var(--danger); color: var(--text); }
    .alert-success { background: rgba(0,196,140,0.08); border-left: 3px solid var(--success); color: var(--text); }

    /* ── Streamlit overrides ── */
    .stButton > button {
        background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(123,47,190,0.15)) !important;
        color: var(--primary) !important;
        border: 1px solid rgba(0,212,255,0.4) !important;
        border-radius: 8px !important;
        font-family: var(--font-mono) !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0,212,255,0.25), rgba(123,47,190,0.25)) !important;
        border-color: var(--primary) !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stMultiSelect"] label,
    div[data-testid="stFileUploader"] label {
        color: var(--muted) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stMultiSelect"] > div > div {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: var(--card) !important;
        border-radius: 10px !important;
        padding: 4px !important;
        border: 1px solid var(--border) !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--muted) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.78rem !important;
        border-radius: 6px !important;
        padding: 6px 14px !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0,212,255,0.15) !important;
        color: var(--primary) !important;
    }
    .stSpinner > div { border-top-color: var(--primary) !important; }
    .stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; }
    [data-testid="stMetricValue"] { color: var(--primary) !important; font-family: var(--font-mono) !important; }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; }
    div[data-testid="stExpander"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    .stProgress > div > div { background: linear-gradient(90deg, var(--primary), #7B2FBE) !important; }
    hr { border-color: var(--border) !important; }
    p, li { color: var(--text) !important; }
    h1, h2, h3, h4 { color: var(--text) !important; font-family: var(--font-sans) !important; }
    .dw-footer {
        margin-top: 1.5rem;
        padding: 10px 14px;
        border-top: 1px solid var(--border);
        color: var(--muted);
        font-family: var(--font-mono);
        font-size: 0.75rem;
        display: flex;
        justify-content: flex-end;
        gap: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────────────────
def init_session():
    defaults = {
        "ref_df": None,
        "cur_df": None,
        "col_info": None,
        "model_mgr": None,
        "model_trained": False,
        "model_metrics": None,
        "drift_results": None,
        "drift_summary": None,
        "pred_drift": None,
        "drift_history": [],
        "active_page": "setup",
        "target_col": None,
        "ref_preds": None,
        "cur_preds": None,
        "evidently_analyzer": EvidentlyAnalyzer(),
        "data_processor": DataProcessor(),
        "drift_detector": DriftDetector(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Helpers ─────────────────────────────────────────────────────────────────
def status_badge(status: str) -> str:
    icons = {"STABLE": "🟢", "MONITOR": "🟡", "HIGH_RISK": "🔴", None: "⚪"}
    icon = icons.get(status, "⚪")
    css_cls = f"status-{status.lower().replace('_', '-') if status else 'none'}"
    label = status.replace("_", " ") if status else "NO DATA"
    return f'<span class="status-badge {css_cls}">{icon} {label}</span>'


def metric_card(value, label, unit="", color=None):
    color = color or COLORS["primary"]
    return f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{value}{unit}</div>
        <div class="metric-label">{label}</div>
    </div>"""


def section_header(icon, title):
    st.markdown(f"""
    <div class="section-header">
        <span class="section-icon">{icon}</span>
        <span class="section-title">{title}</span>
    </div>""", unsafe_allow_html=True)


def alert(msg, kind="info"):
    st.markdown(f'<div class="alert alert-{kind}">{msg}</div>', unsafe_allow_html=True)

def render_footer():
    year = time.strftime("%Y")
    st.markdown(
        f'<div class="dw-footer">© {year} {OWNER_NAME} · <a href="mailto:{OWNER_EMAIL}">{OWNER_EMAIL}</a></div>',
        unsafe_allow_html=True,
    )

# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:12px 0 18px;">
            <div style="font-size:2.2rem">🔭</div>
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.2rem;
                        background:linear-gradient(135deg,#00D4FF,#7B2FBE);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                DRIFTWATCH
            </div>
            <div style="color:#64748B;font-size:0.65rem;font-family:'Space Mono',monospace;margin-top:2px;">
                ML DRIFT DETECTION v2.0
            </div>
        </div>""", unsafe_allow_html=True)

        # System status
        model_ok = st.session_state.model_trained
        data_ok = st.session_state.ref_df is not None and st.session_state.cur_df is not None
        drift_ok = st.session_state.drift_results is not None
        status = st.session_state.drift_summary.get("status") if st.session_state.drift_summary else None

        st.markdown(f"""
        <div style="background:#0D1526;border:1px solid rgba(0,212,255,0.12);border-radius:10px;padding:12px;margin-bottom:14px;">
            <div style="font-size:0.65rem;color:#64748B;font-family:'Space Mono',monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">SYSTEM STATUS</div>
            <div style="display:flex;align-items:center;gap:8px;font-size:0.75rem;color:#E2E8F0;margin:4px 0;font-family:'Space Mono',monospace;">
                {'🟢' if data_ok else '⚫'} Data Loaded
            </div>
            <div style="display:flex;align-items:center;gap:8px;font-size:0.75rem;color:#E2E8F0;margin:4px 0;font-family:'Space Mono',monospace;">
                {'🟢' if model_ok else '⚫'} Model Ready
            </div>
            <div style="display:flex;align-items:center;gap:8px;font-size:0.75rem;color:#E2E8F0;margin:4px 0;font-family:'Space Mono',monospace;">
                {'🟢' if drift_ok else '⚫'} Analysis Run
            </div>
            <div style="margin-top:10px;">
                {status_badge(status)}
            </div>
        </div>""", unsafe_allow_html=True)

        # Navigation
        st.markdown('<div style="color:#64748B;font-size:0.65rem;font-family:\'Space Mono\',monospace;text-transform:uppercase;letter-spacing:1px;margin:6px 0 8px;">NAVIGATION</div>', unsafe_allow_html=True)
        pages = [
            ("📂", "setup", "Data Setup"),
            ("🔬", "feature_drift", "Feature Drift"),
            ("🤖", "model", "Model Performance"),
            ("📊", "evidently", "Evidently Reports"),
            ("📈", "monitoring", "Live Monitoring"),
            ("📥", "export", "Export Reports"),
        ]
        for icon, key, label in pages:
            active = st.session_state.active_page == key
            if st.button(f"{icon}  {label}", key=f"nav_{key}",
                          use_container_width=True,
                          type="primary" if active else "secondary"):
                st.session_state.active_page = key
                st.rerun()

        # Drift settings
        with st.expander("⚙️ Drift Thresholds", expanded=False):
            st.session_state.drift_detector.ks_threshold = st.slider(
                "KS p-value threshold", 0.01, 0.20, 0.05, 0.01,
                help="Lower = stricter drift detection"
            )
            st.session_state.drift_detector.psi_threshold = st.slider(
                "PSI threshold", 0.05, 0.50, 0.20, 0.05,
                help="<0.1 stable, 0.1–0.25 monitor, >0.25 high risk"
            )

        if st.session_state.drift_results:
            st.divider()
            if st.button("🔄  Reset Analysis", use_container_width=True):
                for k in ["drift_results", "drift_summary", "pred_drift", "ref_preds", "cur_preds"]:
                    st.session_state[k] = None
                st.rerun()


# ─── Pages ───────────────────────────────────────────────────────────────────

def page_setup():
    """Data upload and configuration page."""
    section_header("📂", "Data Setup & Configuration")

    alert("Upload your <b>Reference</b> (training/baseline) and <b>Current</b> (production/new) datasets. "
          "The system auto-detects column types and supports any tabular CSV dataset.", "info")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="upload-zone"><div class="upload-icon">📋</div><div class="upload-label">Reference Dataset</div></div>', unsafe_allow_html=True)
        ref_file = st.file_uploader("Reference CSV", type=["csv"], key="ref_upload", label_visibility="collapsed")
        use_demo = st.checkbox("Use built-in demo dataset (Telco Churn)", value=st.session_state.ref_df is None)

        if use_demo:
            demo_path = "data/Telco_Customer_churn.csv"
            if os.path.exists(demo_path):
                st.session_state.ref_df = pd.read_csv(demo_path)
                alert("✅ Demo dataset loaded: Telco Customer Churn (7043 rows)", "success")
        elif ref_file is not None:
            st.session_state.ref_df = st.session_state.data_processor.load_csv(ref_file)
            alert(f"✅ Reference loaded: {len(st.session_state.ref_df)} rows × {len(st.session_state.ref_df.columns)} cols", "success")

    with col2:
        st.markdown('<div class="upload-zone"><div class="upload-icon">🔄</div><div class="upload-label">Current Dataset (New/Production)</div></div>', unsafe_allow_html=True)
        cur_file = st.file_uploader("Current CSV", type=["csv"], key="cur_upload", label_visibility="collapsed")
        simulate_drift = st.checkbox("Simulate drift from reference data", value=st.session_state.cur_df is None)

        if simulate_drift and st.session_state.ref_df is not None:
            cur = st.session_state.ref_df.copy()
            num_cols = cur.select_dtypes(include=["number"]).columns
            # Apply synthetic drift
            for col in num_cols[:min(4, len(num_cols))]:
                cur[col] = cur[col] * np.random.uniform(1.15, 1.4)
            st.session_state.cur_df = cur
            alert("⚠️ Synthetic drift applied to 4 numerical features for demo.", "warn")
        elif cur_file is not None:
            st.session_state.cur_df = st.session_state.data_processor.load_csv(cur_file)
            alert(f"✅ Current loaded: {len(st.session_state.cur_df)} rows × {len(st.session_state.cur_df.columns)} cols", "success")

    # Column configuration (only when both datasets loaded)
    if st.session_state.ref_df is not None:
        st.markdown("---")
        section_header("⚙️", "Column Configuration")

        ref_df = st.session_state.ref_df
        all_cols = list(ref_df.columns)

        c1, c2, c3 = st.columns(3)
        with c1:
            target_col = st.selectbox(
                "Target Column",
                options=["(None)"] + all_cols,
                index=0,
                help="The column your model predicts"
            )
            target_col = None if target_col == "(None)" else target_col
            st.session_state.target_col = target_col

        with c2:
            col_info = st.session_state.data_processor.infer_column_types(ref_df, target_col)
            num_default = col_info["numerical"]
            num_cols = st.multiselect(
                "Numerical Features",
                options=[c for c in all_cols if c != target_col],
                default=num_default[:min(20, len(num_default))],
            )

        with c3:
            cat_default = col_info["categorical"]
            cat_cols = st.multiselect(
                "Categorical Features",
                options=[c for c in all_cols if c != target_col and c not in num_cols],
                default=cat_default[:min(20, len(cat_default))],
            )

        # Update col_info
        st.session_state.col_info = {
            "numerical": num_cols,
            "categorical": cat_cols,
            "id": col_info.get("id", []),
            "target": target_col,
        }

        # Dataset Preview
        with st.expander("📋 Reference Data Preview", expanded=False):
            st.dataframe(ref_df.head(10), use_container_width=True)
            summary = st.session_state.data_processor.get_dataset_summary(ref_df)
            st.markdown(f"""
            <div class="metric-grid">
                {metric_card(summary['rows'], 'ROWS')}
                {metric_card(summary['cols'], 'COLUMNS')}
                {metric_card(f"{summary['missing_pct']:.1f}", 'MISSING', '%', '#FFB800')}
                {metric_card(summary['duplicate_rows'], 'DUPLICATES', '', '#FF4757')}
                {metric_card(f"{summary['memory_mb']:.1f}", 'MEMORY', 'MB')}
            </div>""", unsafe_allow_html=True)

        if st.session_state.cur_df is not None:
            with st.expander("🔄 Current Data Preview", expanded=False):
                st.dataframe(st.session_state.cur_df.head(10), use_container_width=True)

        st.markdown("---")
        section_header("🤖", "Model Training")
        col_info = st.session_state.col_info
        can_train = target_col and target_col in ref_df.columns

        if not can_train:
            alert("Select a target column above to enable model training.", "warn")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                train_mode = st.radio(
                    "Training mode",
                    ["Auto-train on reference data", "Load pre-saved model (Telco Churn demo)"],
                    horizontal=True,
                )
            with c2:
                train_btn = st.button("🚀  Train / Load Model", use_container_width=True, type="primary")

            if train_btn:
                mm = ModelManager()
                X = ref_df.drop(columns=[c for c in [target_col] + col_info.get("id", []) + ["predictions", "customerID"] if c in ref_df.columns], errors="ignore")
                # Keep only selected feature columns
                sel_cols = col_info["numerical"] + col_info["categorical"]
                X = X[[c for c in sel_cols if c in X.columns]]
                y = ref_df[target_col]

                if train_mode == "Load pre-saved model (Telco Churn demo)":
                    with st.spinner("Loading pre-trained pipeline..."):
                        loaded = mm.load("artifacts/model_pipeline.pkl", "artifacts/model_meta.pkl")
                        if not loaded:
                            loaded = mm.load("artifacts/model_pipeline.pkl")
                    if loaded:
                        mm.numerical_cols = col_info["numerical"]
                        mm.categorical_cols = col_info["categorical"]
                        mm.task_type = "classification"
                        st.session_state.model_mgr = mm
                        st.session_state.model_trained = True
                        st.session_state.model_metrics = {"note": "Pre-trained model loaded."}
                        alert("✅ Pre-trained model loaded.", "success")
                    else:
                        alert("❌ Failed to load model. Switching to auto-train.", "danger")
                        train_mode = "Auto-train on reference data"

                if train_mode == "Auto-train on reference data":
                    with st.spinner(f"Training RandomForest on {len(X)} samples..."):
                        t0 = time.time()
                        metrics = mm.train(X, y, col_info["numerical"], col_info["categorical"])
                        elapsed = time.time() - t0
                        mm.save()
                    st.session_state.model_mgr = mm
                    st.session_state.model_trained = True
                    st.session_state.model_metrics = metrics
                    st.success(f"✅ Model trained in {elapsed:.1f}s · Task: {mm.task_type.upper()}")

                # Generate predictions on both datasets
                if st.session_state.model_trained:
                    mm = st.session_state.model_mgr
                    try:
                        ref_X = ref_df[[c for c in (col_info["numerical"] + col_info["categorical"]) if c in ref_df.columns]]
                        st.session_state.ref_preds = mm.predict(ref_X)
                    except Exception as e:
                        alert(f"Warning: Could not predict on reference data: {e}", "warn")

                    if st.session_state.cur_df is not None:
                        try:
                            cur_X = st.session_state.cur_df[[c for c in (col_info["numerical"] + col_info["categorical"]) if c in st.session_state.cur_df.columns]]
                            st.session_state.cur_preds = mm.predict(cur_X)
                        except Exception as e:
                            alert(f"Warning: Could not predict on current data: {e}", "warn")

        # Show metrics if available
        if st.session_state.model_metrics:
            m = st.session_state.model_metrics
            if "accuracy" in m:
                cols = st.columns(min(4, len([k for k in m if k not in ["task_type", "note"]])))
                keys = [(k, v) for k, v in m.items() if k not in ["task_type", "note"]]
                for i, (k, v) in enumerate(keys[:4]):
                    with cols[i]:
                        st.metric(k.upper(), f"{v:.4f}" if isinstance(v, float) else v)


def page_feature_drift():
    """Feature drift analysis page."""
    if st.session_state.ref_df is None or st.session_state.cur_df is None:
        alert("Please upload both Reference and Current datasets in Data Setup.", "warn")
        return
    if st.session_state.col_info is None:
        alert("Configure columns in Data Setup.", "warn")
        return

    section_header("🔬", "Feature Drift Analysis")

    col_info = st.session_state.col_info
    ref_df = st.session_state.ref_df
    cur_df = st.session_state.cur_df

    c1, c2 = st.columns([3, 1])
    with c2:
        run_btn = st.button("▶  Run Drift Analysis", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Running multi-method drift detection..."):
            detector = st.session_state.drift_detector
            results = detector.compute_full_drift(
                ref_df, cur_df,
                col_info["numerical"], col_info["categorical"]
            )
            summary = detector.compute_dataset_drift_score(results)

            # Prediction drift
            pred_drift = None
            if st.session_state.ref_preds is not None and st.session_state.cur_preds is not None:
                pred_drift = detector.compute_prediction_drift(
                    st.session_state.ref_preds, st.session_state.cur_preds,
                    st.session_state.model_mgr.task_type if st.session_state.model_mgr else "classification"
                )

            st.session_state.drift_results = results
            st.session_state.drift_summary = summary
            st.session_state.pred_drift = pred_drift
            st.session_state.drift_history.append(summary)

    if st.session_state.drift_results is None:
        alert("Click <b>Run Drift Analysis</b> to start.", "info")
        return

    results = st.session_state.drift_results
    summary = st.session_state.drift_summary
    pred_drift = st.session_state.pred_drift

    # ── Overall Status ──
    status_color = {
        "STABLE": COLORS["success"], "MONITOR": COLORS["warning"], "HIGH_RISK": COLORS["danger"]
    }.get(summary["status"], COLORS["muted"])

    st.markdown(f"""
    <div style="background:var(--card);border:1px solid {status_color}33;border-radius:14px;padding:1.2rem 1.6rem;margin:1rem 0;
                display:flex;align-items:center;gap:2rem;flex-wrap:wrap;">
        <div>{status_badge(summary['status'])}</div>
        <div class="metric-grid" style="flex:1;margin:0;">
            {metric_card(f"{summary['drift_ratio']:.1f}", 'DRIFT RATIO', '%', status_color)}
            {metric_card(summary['drifted_features'], 'DRIFTED FEATURES', f"/{summary['total_features']}", status_color)}
            {metric_card(summary['total_features'], 'TOTAL FEATURES')}
            {metric_card(len(st.session_state.drift_history), 'ANALYSIS SESSIONS')}
        </div>
    </div>""", unsafe_allow_html=True)

    # Prediction drift
    if pred_drift:
        pd_color = COLORS["danger"] if pred_drift["drift_detected"] else COLORS["success"]
        if "current_positive_rate" in pred_drift:
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid {pd_color}33;border-radius:10px;padding:12px 16px;margin-bottom:12px;">
                <span style="font-family:'Space Mono',monospace;font-size:0.75rem;color:var(--muted);">PREDICTION DRIFT · </span>
                {'<span class="tag-drift">DRIFT DETECTED</span>' if pred_drift["drift_detected"] else '<span class="tag-ok">STABLE</span>'}
                <span style="font-family:'Space Mono',monospace;font-size:0.75rem;color:var(--text);margin-left:16px;">
                    Reference Rate: {pred_drift['reference_positive_rate']:.3f} &nbsp;→&nbsp; Current Rate: {pred_drift['current_positive_rate']:.3f}
                    &nbsp;&nbsp;Shift: {pred_drift['shift']:.3f}
                </span>
            </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["📊 Heatmap", "📋 Feature Table", "🔍 Distributions", "📈 Predictions"])

    with tabs[0]:
        if results:
            st.plotly_chart(feature_drift_heatmap(results), use_container_width=True, key="feat_heatmap")

    with tabs[1]:
        # Build table
        rows = []
        for col, res in results.items():
            row = {"Feature": col, "Type": res["type"]}
            row["Drift"] = "⚠️ DRIFT" if res.get("drift_detected") else "✅ OK"
            if res["type"] == "numerical":
                row["KS p-value"] = res.get("ks", {}).get("p_value", "—")
                row["PSI"] = res.get("psi", {}).get("psi", "—")
                ms = res.get("mean_shift", {})
                row["Mean Change %"] = ms.get("mean_change_pct", "—")
                row["Z-Score"] = ms.get("z_score", "—")
            else:
                row["Chi2 p-value"] = res.get("chi2", {}).get("p_value", "—")
                row["Max Dist. Shift"] = res.get("distribution", {}).get("max_shift", "—")
                row["PSI"] = "—"
                row["Mean Change %"] = "—"
                row["Z-Score"] = "—"
            rows.append(row)

        df_table = pd.DataFrame(rows)
        # Highlight drifted rows
        drifted_idx = df_table["Drift"].str.contains("DRIFT")

        table_html = '<table class="drift-table"><thead><tr>'
        for col in df_table.columns:
            table_html += f'<th>{col}</th>'
        table_html += '</tr></thead><tbody>'
        for _, row in df_table.iterrows():
            row_style = 'style="background:rgba(255,71,87,0.05);"' if "DRIFT" in str(row["Drift"]) else ''
            table_html += f'<tr {row_style}>'
            for col_n, val in row.items():
                if col_n == "Drift":
                    cls = "tag-drift" if "DRIFT" in str(val) else "tag-ok"
                    table_html += f'<td><span class="{cls}">{val}</span></td>'
                elif col_n == "Type":
                    cls = "tag-num" if val == "numerical" else "tag-cat"
                    table_html += f'<td><span class="{cls}">{val}</span></td>'
                else:
                    if isinstance(val, float):
                        val = f"{val:.4f}"
                    table_html += f'<td>{val}</td>'
            table_html += '</tr>'
        table_html += '</tbody></table>'
        st.markdown(table_html, unsafe_allow_html=True)

    with tabs[2]:
        all_feat_cols = col_info["numerical"] + col_info["categorical"]
        sel_feature = st.selectbox(
            "Select feature to compare",
            options=[c for c in all_feat_cols if c in ref_df.columns and c in cur_df.columns],
        )
        if sel_feature:
            ftype = "numerical" if sel_feature in col_info["numerical"] else "categorical"
            st.plotly_chart(
                distribution_comparison(ref_df[sel_feature], cur_df[sel_feature], sel_feature, ftype),
                use_container_width=True,
                key=f"dist_{sel_feature}"
            )
            # Show stats
            if ftype == "numerical":
                c1, c2, c3 = st.columns(3)
                c1.metric("Ref Mean", f"{ref_df[sel_feature].mean():.3f}")
                c2.metric("Cur Mean", f"{cur_df[sel_feature].mean():.3f}", f"{((cur_df[sel_feature].mean()-ref_df[sel_feature].mean())/abs(ref_df[sel_feature].mean())*100):.1f}%")
                c3.metric("Ref Std", f"{ref_df[sel_feature].std():.3f}")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Reference unique values", ref_df[sel_feature].nunique())
                c2.metric("Current unique values", cur_df[sel_feature].nunique())

        # All drifted features quick view
        drifted_feats = [c for c, r in results.items() if r.get("drift_detected")]
        if drifted_feats:
            st.markdown("---")
            st.markdown(f'<div class="section-title">⚠️ Drifted Features Quick View ({len(drifted_feats)})</div>', unsafe_allow_html=True)
            for feat in drifted_feats[:6]:
                if feat in ref_df.columns and feat in cur_df.columns:
                    ftype = "numerical" if feat in col_info["numerical"] else "categorical"
                    with st.expander(f"{'📊' if ftype=='numerical' else '📋'} {feat}", expanded=False):
                        st.plotly_chart(
                            distribution_comparison(ref_df[feat], cur_df[feat], feat, ftype),
                            use_container_width=True,
                            key=f"quick_{feat}"
                        )

    with tabs[3]:
        if st.session_state.ref_preds is not None and st.session_state.cur_preds is not None:
            mm = st.session_state.model_mgr
            task = mm.task_type if mm else "classification"
            st.plotly_chart(
                prediction_drift_chart(st.session_state.ref_preds, st.session_state.cur_preds, task),
                use_container_width=True,
                key="pred_drift_chart"
            )
        else:
            alert("Train a model first to see prediction drift.", "warn")


def page_model():
    """Model performance and evaluation page."""
    section_header("🤖", "Model Performance Monitoring")

    if not st.session_state.model_trained:
        alert("Train a model in Data Setup first.", "warn")
        return

    mm = st.session_state.model_mgr
    metrics = st.session_state.model_metrics or {}

    # Metric gauges
    if "accuracy" in metrics:
        cols = st.columns(min(4, len([k for k in metrics if isinstance(metrics[k], float)])))
        float_metrics = [(k, v) for k, v in metrics.items() if isinstance(v, float)]
        thresholds = {"accuracy": 0.8, "f1_score": 0.75, "roc_auc": 0.80, "r2": 0.75}
        for i, (k, v) in enumerate(float_metrics[:4]):
            with cols[i]:
                thresh = thresholds.get(k, 0.75)
                st.plotly_chart(model_performance_gauge(v, k.upper(), thresh), use_container_width=True, key=f"gauge_{k}")

    # Model info
    col1, col2 = st.columns(2)
    with col1:
        section_header("📋", "Model Information")
        info_data = {
            "Task Type": mm.task_type.upper(),
            "Algorithm": type(mm.pipeline.named_steps["model"]).__name__ if mm.pipeline else "Unknown",
            "Numerical Features": len(mm.numerical_cols),
            "Categorical Features": len(mm.categorical_cols),
            "Total Features": len(mm.feature_names),
        }
        for k, v in info_data.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:8px 12px;border-bottom:1px solid rgba(100,116,139,0.1);">
                <span style="color:var(--muted);font-size:0.78rem;font-family:'Space Mono',monospace;">{k}</span>
                <span style="color:var(--text);font-weight:600;font-size:0.82rem;font-family:'Space Mono',monospace;">{v}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        section_header("⭐", "Feature Importance")
        importance_df = mm.get_feature_importance()
        if importance_df is not None and len(importance_df) > 0:
            st.plotly_chart(feature_importance_chart(importance_df), use_container_width=True, key="feature_importance")
        else:
            alert("Feature importance not available for this model.", "info")

    # Evaluate on current data
    section_header("📊", "Evaluate on Current Data")
    if st.session_state.cur_df is not None and st.session_state.target_col:
        col_info = st.session_state.col_info
        cur_df = st.session_state.cur_df
        if st.session_state.target_col in cur_df.columns:
            if st.button("📊  Evaluate on Current Data", type="primary"):
                try:
                    cur_X = cur_df[[c for c in (col_info["numerical"] + col_info["categorical"]) if c in cur_df.columns]]
                    cur_metrics = mm.evaluate_on_new_data(cur_X, cur_df[st.session_state.target_col])
                    st.subheader("Current Data Performance")
                    for k, v in cur_metrics.items():
                        if isinstance(v, float):
                            st.metric(k.upper(), f"{v:.4f}")
                except Exception as e:
                    alert(f"Evaluation error: {e}", "danger")
        else:
            alert(f"Target column '{st.session_state.target_col}' not found in current data.", "warn")
    else:
        alert("Upload current data and select a target column to evaluate.", "info")

    # Correlation heatmap
    if st.session_state.ref_df is not None and st.session_state.col_info:
        section_header("🔗", "Feature Correlations (Reference)")
        num_cols = st.session_state.col_info.get("numerical", [])
        if len(num_cols) >= 2:
            fig = correlation_heatmap(st.session_state.ref_df, num_cols)
            st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")


def page_evidently():
    """Evidently AI reports page."""
    section_header("📊", "Evidently AI Deep Reports")

    if st.session_state.ref_df is None or st.session_state.cur_df is None:
        alert("Upload both datasets in Data Setup.", "warn")
        return

    analyzer = st.session_state.evidently_analyzer
    col_info = st.session_state.col_info or {}
    ref_df = st.session_state.ref_df
    cur_df = st.session_state.cur_df
    target_col = st.session_state.target_col

    # Setup column mapping
    mm = st.session_state.model_mgr
    pred_col = None
    ref_for_ev = ref_df.copy()
    cur_for_ev = cur_df.copy()

    # Add predictions if model is available
    if mm and st.session_state.ref_preds is not None:
        ref_for_ev["predictions"] = st.session_state.ref_preds
        pred_col = "predictions"
    if mm and st.session_state.cur_preds is not None:
        cur_for_ev["predictions"] = st.session_state.cur_preds

    try:
        column_mapping = analyzer.build_column_mapping(
            ref_for_ev,
            col_info.get("numerical", []),
            col_info.get("categorical", []),
            target_col=target_col,
            prediction_col=pred_col,
        )
    except Exception as e:
        alert(f"Evidently unavailable: {e}", "danger")
        return

    report_tabs = st.tabs([
        "🌊 Data Drift",
        "🎯 Target Drift",
        "📉 Data Quality",
        "🏆 Classification"
    ])

    with report_tabs[0]:
        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button("▶  Generate Data Drift", use_container_width=True, type="primary"):
                with st.spinner("Generating Evidently Data Drift report..."):
                    try:
                        path = analyzer.run_data_drift_report(ref_for_ev, cur_for_ev, column_mapping)
                        st.session_state["ev_data_drift_path"] = path
                        st.success("Report generated!")
                        # Download button
                        data = analyzer.get_report_bytes(path)
                        st.download_button("📥 Download HTML", data, "data_drift.html", "text/html")
                    except Exception as e:
                        alert(f"Error: {e}", "danger")
        if "ev_data_drift_path" in st.session_state:
            analyzer.display_report(st.session_state["ev_data_drift_path"], 2200)

    with report_tabs[1]:
        if not target_col:
            alert("Select a target column in Data Setup to generate Target Drift report.", "warn")
        else:
            c1, c2 = st.columns([3, 1])
            with c2:
                if st.button("▶  Generate Target Drift", use_container_width=True, type="primary"):
                    with st.spinner("Generating Evidently Target Drift report..."):
                        try:
                            path = analyzer.run_target_drift_report(ref_for_ev, cur_for_ev, column_mapping)
                            st.session_state["ev_target_drift_path"] = path
                            st.success("Report generated!")
                            data = analyzer.get_report_bytes(path)
                            st.download_button("📥 Download HTML", data, "target_drift.html", "text/html")
                        except Exception as e:
                            alert(f"Error: {e}", "danger")
            if "ev_target_drift_path" in st.session_state:
                analyzer.display_report(st.session_state["ev_target_drift_path"], 2000)

    with report_tabs[2]:
        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button("▶  Generate Quality Report", use_container_width=True, type="primary"):
                with st.spinner("Generating Evidently Data Quality report..."):
                    try:
                        path = analyzer.run_data_quality_report(ref_for_ev, cur_for_ev, column_mapping)
                        st.session_state["ev_quality_path"] = path
                        st.success("Report generated!")
                        data = analyzer.get_report_bytes(path)
                        st.download_button("📥 Download HTML", data, "data_quality.html", "text/html")
                    except Exception as e:
                        alert(f"Error: {e}", "danger")
        if "ev_quality_path" in st.session_state:
            analyzer.display_report(st.session_state["ev_quality_path"], 2000)

    with report_tabs[3]:
        if not (mm and pred_col):
            alert("Train a model first to generate Classification report.", "warn")
        else:
            c1, c2 = st.columns([3, 1])
            with c2:
                if st.button("▶  Generate Classification Report", use_container_width=True, type="primary"):
                    with st.spinner("Generating Evidently Classification report..."):
                        try:
                            path = analyzer.run_classification_report(ref_for_ev, cur_for_ev, column_mapping)
                            st.session_state["ev_class_path"] = path
                            st.success("Report generated!")
                            data = analyzer.get_report_bytes(path)
                            st.download_button("📥 Download HTML", data, "classification.html", "text/html")
                        except Exception as e:
                            alert(f"Error: {e}", "danger")
            if "ev_class_path" in st.session_state:
                analyzer.display_report(st.session_state["ev_class_path"], 2000)


def page_monitoring():
    """Live monitoring / timeline page."""
    section_header("📈", "Live Monitoring Dashboard")

    if not st.session_state.drift_history:
        alert("Run at least one Drift Analysis to see monitoring data.", "info")
        return

    summary = st.session_state.drift_summary
    history = st.session_state.drift_history

    # Current status banner
    status = summary["status"]
    color_map = {"STABLE": "#00C48C", "MONITOR": "#FFB800", "HIGH_RISK": "#FF4757"}
    color = color_map.get(status, "#64748B")
    icons = {"STABLE": "🟢", "MONITOR": "🟡", "HIGH_RISK": "🔴"}

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{color}15,{color}05);
                border:1px solid {color}40;border-radius:16px;
                padding:1.8rem 2.2rem;text-align:center;margin:1rem 0 1.5rem;">
        <div style="font-size:3rem;margin-bottom:8px;">{icons.get(status,'⚪')}</div>
        <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{color};">
            MODEL STATUS: {status.replace('_', ' ')}
        </div>
        <div style="color:#64748B;font-family:'Space Mono',monospace;font-size:0.78rem;margin-top:8px;">
            {summary['drifted_features']} of {summary['total_features']} features drifted &nbsp;·&nbsp;
            {summary['drift_ratio']:.1f}% drift ratio &nbsp;·&nbsp;
            Session {len(history)} of {len(history)}
        </div>
    </div>""", unsafe_allow_html=True)

    # Timeline
    if len(history) > 1:
        st.plotly_chart(drift_timeline(history), use_container_width=True, key="drift_timeline")
    else:
        alert("Run multiple analyses to see the drift timeline.", "info")

    # Dataset overview comparison
    if st.session_state.ref_df is not None and st.session_state.cur_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                dataset_overview_chart(st.session_state.ref_df, st.session_state.cur_df),
                use_container_width=True,
                key="dataset_overview"
            )
        with col2:
            if st.session_state.drift_results:
                results = st.session_state.drift_results
                drifted = [c for c, r in results.items() if r.get("drift_detected")]
                stable = [c for c, r in results.items() if not r.get("drift_detected")]

                st.markdown(f"""
                <div style="background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.2rem;">
                    <div class="section-title" style="margin-bottom:12px;">Feature Status Breakdown</div>
                    <div style="display:flex;gap:16px;margin-bottom:12px;">
                        <div style="flex:1;background:rgba(255,71,87,0.1);border:1px solid rgba(255,71,87,0.3);border-radius:8px;padding:12px;text-align:center;">
                            <div style="font-size:1.8rem;font-weight:800;color:#FF4757;font-family:'Space Mono',monospace;">{len(drifted)}</div>
                            <div style="font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;">DRIFTED</div>
                        </div>
                        <div style="flex:1;background:rgba(0,196,140,0.1);border:1px solid rgba(0,196,140,0.3);border-radius:8px;padding:12px;text-align:center;">
                            <div style="font-size:1.8rem;font-weight:800;color:#00C48C;font-family:'Space Mono',monospace;">{len(stable)}</div>
                            <div style="font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;">STABLE</div>
                        </div>
                    </div>
                    <div style="font-size:0.75rem;color:var(--muted);font-family:'Space Mono',monospace;">
                        <b style="color:#FF4757;">Drifted:</b> {", ".join(drifted[:10]) if drifted else "None"}<br>
                        <b style="color:#00C48C;">Stable:</b> {", ".join(stable[:10]) if stable else "None"}
                    </div>
                </div>""", unsafe_allow_html=True)

    # Action recommendations
    section_header("💡", "Recommendations")
    status_recs = {
        "STABLE": [
            ("✅", "Model is performing as expected", "Continue monitoring on regular schedule."),
            ("📊", "Maintain baseline", "Update baseline data periodically to account for natural distribution shifts."),
        ],
        "MONITOR": [
            ("⚠️", "Significant feature drift detected", "Investigate drifted features and understand root causes."),
            ("🔍", "Consider model retraining", "Collect more recent labeled data and plan a retraining cycle."),
            ("📈", "Increase monitoring frequency", "Switch from weekly to daily monitoring."),
        ],
        "HIGH_RISK": [
            ("🚨", "Critical drift detected — immediate action required", "Model predictions may be unreliable."),
            ("🔄", "Retrain immediately", "Collect fresh labeled data and retrain the model."),
            ("🛑", "Consider model rollback", "If a previous stable version exists, consider rolling back."),
            ("📧", "Alert stakeholders", "Notify data science team and business stakeholders."),
        ],
    }
    for icon, title, desc in status_recs.get(status, []):
        st.markdown(f"""
        <div style="display:flex;gap:12px;padding:10px 0;border-bottom:1px solid rgba(100,116,139,0.1);">
            <span style="font-size:1.1rem">{icon}</span>
            <div>
                <div style="font-weight:600;color:var(--text);font-size:0.85rem;">{title}</div>
                <div style="color:var(--muted);font-size:0.78rem;margin-top:2px;">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)


def page_export():
    """Export reports page."""
    section_header("📥", "Export Reports")

    if st.session_state.drift_results is None:
        alert("Run drift analysis first to export reports.", "warn")
        return

    results = st.session_state.drift_results
    summary = st.session_state.drift_summary
    col_info = st.session_state.col_info or {}

    # ── JSON Report ──────────────────────────────────────────────────────────
    report_data = {
        "summary": summary,
        "model_metrics": st.session_state.model_metrics,
        "copyright": {"owner": OWNER_NAME, "email": OWNER_EMAIL, "year": time.strftime("%Y")},
        "feature_drift": {
            col: {
                "type": res["type"],
                "drift_detected": res["drift_detected"],
                "ks_p_value": res.get("ks", {}).get("p_value"),
                "psi": res.get("psi", {}).get("psi"),
                "mean_change_pct": res.get("mean_shift", {}).get("mean_change_pct"),
            }
            for col, res in results.items()
        },
        "prediction_drift": st.session_state.pred_drift,
        "sessions": st.session_state.drift_history,
    }

    col1, col2 = st.columns(2)
    with col1:
        section_header("📄", "Summary Report (JSON)")
        json_str = json.dumps(report_data, indent=2, default=str)
        st.download_button(
            "📥  Download JSON Report",
            json_str,
            "drift_report.json",
            "application/json",
            use_container_width=True,
        )
        with st.expander("Preview JSON", expanded=False):
            st.code(json_str[:2000] + "..." if len(json_str) > 2000 else json_str, language="json")

    with col2:
        section_header("📊", "Summary Report (CSV)")
        rows = []
        for col, res in results.items():
            rows.append({
                "Feature": col,
                "Type": res["type"],
                "Drift Detected": res["drift_detected"],
                "KS p-value": res.get("ks", {}).get("p_value"),
                "PSI": res.get("psi", {}).get("psi"),
                "Mean Change %": res.get("mean_shift", {}).get("mean_change_pct"),
                "Chi2 p-value": res.get("chi2", {}).get("p_value"),
            })
        df_export = pd.DataFrame(rows)
        csv_str = df_export.to_csv(index=False)
        st.download_button(
            "📥  Download CSV Report",
            csv_str,
            "drift_report.csv",
            "text/csv",
            use_container_width=True,
        )

    # ── Evidently HTML Reports ───────────────────────────────────────────────
    section_header("🌐", "Evidently HTML Reports")
    report_files = {
        "Data Drift": "reports/data_drift.html",
        "Target Drift": "reports/target_drift.html",
        "Data Quality": "reports/data_quality.html",
        "Classification": "reports/classification.html",
    }
    analyzer = st.session_state.evidently_analyzer
    cols = st.columns(2)
    for i, (name, path) in enumerate(report_files.items()):
        with cols[i % 2]:
            if os.path.exists(path):
                data = analyzer.get_report_bytes(path)
                st.download_button(
                    f"📥  {name} Report",
                    data,
                    os.path.basename(path),
                    "text/html",
                    use_container_width=True,
                )
            else:
                st.markdown(f'<div class="alert alert-warn">⚠️ {name} not yet generated</div>', unsafe_allow_html=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    init_session()

    # Header
    status = st.session_state.drift_summary.get("status") if st.session_state.drift_summary else None
    st.markdown(f"""
    <div class="dw-header">
        <div class="dw-logo">🔭</div>
        <div>
            <div class="dw-title">DRIFTWATCH</div>
            <div class="dw-subtitle">ML DRIFT DETECTION & PERFORMANCE MONITORING · PRODUCTION v2.0</div>
        </div>
        <div style="margin-left:auto;">
            {status_badge(status)}
        </div>
    </div>""", unsafe_allow_html=True)

    render_sidebar()

    # Route to page
    page = st.session_state.active_page
    if page == "setup":
        page_setup()
    elif page == "feature_drift":
        page_feature_drift()
    elif page == "model":
        page_model()
    elif page == "evidently":
        page_evidently()
    elif page == "monitoring":
        page_monitoring()
    elif page == "export":
        page_export()
    render_footer()


if __name__ == "__main__":
    main()
