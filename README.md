<div align="center">

<img src="https://img.shields.io/badge/🔭_DriftWatch-v2.0-00D4FF?style=for-the-badge&labelColor=0A0E1A" alt="DriftWatch"/>

# 🔭 DriftWatch
### Production ML Drift Detection & Performance Monitoring

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Evidently_AI-0.7+-7B2FBE?style=flat-square"/>
  <img src="https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=flat-square&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-00C48C?style=flat-square"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Deploy-Streamlit_Cloud-FF4B4B?style=flat-square&logo=streamlit"/>
  <img src="https://img.shields.io/badge/Deploy-HuggingFace_Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Deploy-Railway-0B0D0E?style=flat-square&logo=railway"/>
  <img src="https://img.shields.io/badge/🚀_Live_App-Visit_Now-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
</p>

<br/>

**A complete, production-ready system for detecting machine learning model drift and monitoring performance — compatible with *any* tabular CSV dataset.**

[🌐 Live Demo](#-live-demo) · [🚀 Quick Start](#-quick-start) · [📐 Architecture](#-architecture) · [✨ Features](#-features) · [📊 Detection Methods](#-drift-detection-methods) · [📂 Project Structure](#-project-structure) · [🛠️ Tech Stack](#%EF%B8%8F-tech-stack)

</div>

---

## 🌐 Live Demo

> **The app is live and fully interactive — no setup required.**

<div align="center">

### 👉 [https://ml-drift-detection---performance-monitoring-nm3yyacwvxecxrjgkg.streamlit.app/](https://ml-drift-detection---performance-monitoring-nm3yyacwvxecxrjgkg.streamlit.app/)

</div>

Open the link, select the built-in **Telco Customer Churn** demo, enable **"Simulate Drift"**, and explore all 6 monitoring pages — no data upload needed.

---

## 🌟 What is DriftWatch?

When ML models are deployed to production, the real world changes — but the model doesn't. **Data drift** occurs when the statistical properties of the inputs shift over time, causing predictions to degrade silently. **DriftWatch** continuously monitors your model's data and predictions, alerting you before users notice performance degradation.
```
Training Data                Production Data (Weeks Later)
─────────────────            ───────────────────────────────
μ = 47.3, σ = 12.1    →     μ = 61.8, σ = 19.4  ← DRIFT!
Cat: [A:40%, B:60%]   →     Cat: [A:82%, B:18%]  ← DRIFT!
```

DriftWatch gives you **6 statistical methods**, **Evidently AI-powered reports**, **interactive Plotly visualizations**, and **real-time monitoring dashboards** — all in a slick dark-mode Streamlit UI.

---

## ✨ Features

### 🧠 Core Capabilities

| Feature | Description |
|---|---|
| **Universal Dataset Support** | Upload any CSV — column types are auto-detected (numerical, categorical, ID) |
| **Multi-Method Drift Detection** | 5 statistical tests run simultaneously per feature |
| **Auto ML Pipeline** | Trains a RandomForest pipeline on your reference data automatically |
| **Evidently AI Reports** | Deep-dive HTML reports for Data Drift, Target Drift, Quality & Classification |
| **Interactive Visualizations** | Plotly heatmaps, gauge charts, distribution overlays, drift timelines |
| **Model Performance Monitoring** | Live accuracy, F1, ROC-AUC gauges with threshold alerts |
| **Session History** | Tracks drift across multiple analysis runs as a timeline |
| **Export Reports** | Download results as JSON, CSV, or full Evidently HTML |

### 🎨 UI/UX Highlights

- **Dark cyberpunk aesthetic** — `#060B17` background, cyan/purple gradient accents
- **Space Mono + Syne typography** — monospaced metrics, clean sans-serif prose  
- **Responsive metric cards** with hover animations  
- **Tabbed navigation** — 6 pages, each purpose-built  
- **Live system status panel** in sidebar showing Data / Model / Analysis state  
- **Configurable drift thresholds** via interactive sliders  

---

## 📐 Architecture

### System Overview
```
╔══════════════════════════════════════════════════════════════╗
║                       DRIFTWATCH v2.0                        ║
║            ML Drift Detection & Performance Monitoring       ║
╚══════════════════════════════════════════════════════════════╝
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   ┌───────────┐       ┌────────────┐       ┌───────────────┐
   │  Reference │       │  Current   │       │  Pre-trained  │
   │  Dataset   │       │  Dataset   │       │    Model      │
   │  (CSV)     │       │  (CSV)     │       │  (optional)   │
   └─────┬──────┘       └─────┬──────┘       └───────┬───────┘
         └──────────┬──────────┘                     │
                    ▼                                 │
          ┌─────────────────┐                         │
          │  DataProcessor  │                         │
          │ • Auto type     │                         │
          │   detection     │                         │
          │ • Preprocessing │                         │
          │ • Missing vals  │                         │
          └────────┬────────┘                         │
          ┌────────┴────────────────────┐             │
          ▼                             ▼             ▼
  ┌──────────────────┐         ┌──────────────────────┐
  │   DriftDetector  │         │     ModelManager      │
  │                  │         │                       │
  │ ▸ KS Test        │         │ ▸ Task detection      │
  │ ▸ Chi-Squared    │         │ ▸ sklearn Pipeline    │
  │ ▸ PSI            │         │ ▸ RandomForest        │
  │ ▸ Wasserstein    │         │ ▸ Preprocessing       │
  │ ▸ Z-Score        │         │ ▸ Metrics eval        │
  │ ▸ JS Divergence  │         │ ▸ Save/Load model     │
  └──────────┬───────┘         └──────────┬────────────┘
             │                            │
             └─────────────┬──────────────┘
                           ▼
              ┌────────────────────────┐
              │     Evidently AI       │
              │                        │
              │ ▸ Data Drift Report    │
              │ ▸ Target Drift Report  │
              │ ▸ Data Quality Report  │
              │ ▸ Classification Rpt   │
              └────────────┬───────────┘
                           │
              ┌────────────▼───────────┐
              │       Visualizer        │
              │                        │
              │ ▸ Feature Heatmap      │
              │ ▸ Distribution Charts  │
              │ ▸ Performance Gauges   │
              │ ▸ Drift Timeline       │
              │ ▸ Importance Chart     │
              │ ▸ Correlation Heatmap  │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │     Streamlit UI       │
              │    (6-page app)        │
              └────────────────────────┘
```

### Application Flow
```
User Opens App
     │
     ▼
📂 Data Setup ──────────────────────────────────────────────────┐
│  Upload Reference CSV (or use Telco demo)                     │
│  Upload Current CSV  (or simulate drift)                      │
│  Auto-detect: numerical cols / categorical cols / ID cols     │
│  Select target column                                         │
│  Train RandomForest OR load pre-trained model                 │
│  → Generates ref_preds & cur_preds                            │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
🔬 Feature Drift Analysis
│  Run multi-method drift detection on all features
│  ┌─────────────────────────────────────────────────┐
│  │ Numerical Features → KS + PSI + Wasserstein     │
│  │                    + Z-Score + Jensen-Shannon    │
│  │ Categorical Feats  → Chi-Squared + PSI          │
│  │                    + Distribution Shift %        │
│  └─────────────────────────────────────────────────┘
│  → Drift Heatmap | Feature Table | Distributions | Pred Drift
│
├──▶ 🤖 Model Performance Monitoring
│     Accuracy / F1 / ROC-AUC gauges
│     Feature Importance chart
│     Correlation heatmap
│     Evaluate on current data
│
├──▶ 📊 Evidently AI Reports
│     Data Drift | Target Drift | Data Quality | Classification
│     → Inline HTML rendering + Download
│
├──▶ 📈 Live Monitoring Dashboard
│     Status banner (STABLE / MONITOR / HIGH_RISK)
│     Multi-session drift timeline
│     Feature status breakdown
│     Actionable recommendations
│
└──▶ 📥 Export Reports
      JSON | CSV | Evidently HTML
```

---

## 🚀 Quick Start

### Local Development
```bash
# 1. Clone the repository
git clone https://github.com/Saksham653/ML-DRIFT-DETECTION---PERFORMANCE-MONITORING.git
cd ML-DRIFT-DETECTION---PERFORMANCE-MONITORING

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. The built-in **Telco Customer Churn** dataset loads automatically with synthetic drift applied — no uploads needed to explore.

### ☁️ Deploy to Streamlit Cloud *(Free)*
```
1. Push to a public GitHub repository
2. Visit https://share.streamlit.io
3. Connect repo → set app.py as main file
4. Click Deploy
```

### 🤗 Deploy to Hugging Face Spaces *(Free)*
```
1. Create new Space at https://huggingface.co/spaces
2. Select Streamlit as the SDK
3. Upload all project files
4. App auto-deploys!
```

### 🚂 Deploy to Railway / Render
```bash
# Add a Procfile to the project root:
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
```

---

## 📂 Project Structure
```
driftwatch/
│
├── 🖥️  app.py                       # Main Streamlit app (6 pages, CSS injection, routing)
├── 📦  requirements.txt             # All Python dependencies with version pins
├── 🚫  .gitignore
│
├── src/                             # Core source modules
│   ├── __init__.py
│   ├── 📊  data_processor.py        # Universal CSV loader + auto column-type inference
│   ├── 🔬  drift_detector.py        # 6 statistical drift tests (KS, Chi2, PSI, Wasserstein…)
│   ├── 🤖  model_manager.py         # sklearn Pipeline builder, trainer, evaluator, save/load
│   ├── 📈  visualizer.py            # Plotly charts: heatmap, gauges, distributions, timelines
│   └── 📋  evidently_analyzer.py    # Evidently AI report wrapper (Data/Target/Quality/Class)
│
├── artifacts/                       # Saved model pipelines (auto-created)
│   ├── model_pipeline.pkl
│   └── model_meta.pkl
│
├── reports/                         # Generated Evidently HTML reports (auto-created)
│   ├── data_drift.html
│   ├── target_drift.html
│   ├── data_quality.html
│   └── classification.html
│
└── data/
    └── 📂  Telco_Customer_churn.csv  # Built-in demo dataset (7 043 rows, 21 columns)
```

---

## 📊 Drift Detection Methods

DriftWatch runs **multiple statistical tests simultaneously** so no drift goes unnoticed.

### Numerical Features

| Method | What It Measures | Drift Threshold | When to Use |
|--------|-----------------|-----------------|-------------|
| **Kolmogorov-Smirnov (KS)** | Max difference between CDFs | `p-value < 0.05` | Gold standard for distribution shift |
| **Population Stability Index (PSI)** | Magnitude of distribution shift across bins | `PSI > 0.20` | Industry standard in finance & banking |
| **Wasserstein Distance** | "Earth mover's" cost to transform one dist to another | `normalized > 0.2` | Sensitive to small mean shifts |
| **Z-Score / Mean Shift** | How many σ the mean has shifted | `\|z\| > 2.0` | Fast sanity check on average behavior |
| **Jensen-Shannon Divergence** | Symmetric KL-divergence between distributions | Internal | Bounded [0,1] — great for ranking drift severity |

### Categorical Features

| Method | What It Measures | Drift Threshold |
|--------|-----------------|-----------------|
| **Chi-Squared Test** | Statistical independence of category frequencies | `p-value < 0.05` |
| **Max Distribution Shift** | Largest proportion change across categories | `> 10%` shift |
| **PSI (categorical)** | Category frequency stability over time | `PSI > 0.20` |

### Drift Score Interpretation
```
PSI Score    │ Interpretation             │ Status
─────────────┼────────────────────────────┼──────────────
< 0.10       │ No significant change       │ 🟢 STABLE
0.10 – 0.20  │ Moderate shift — monitor    │ 🟡 MONITOR
> 0.20       │ Significant drift           │ 🔴 HIGH RISK

KS p-value   │ Interpretation
─────────────┼──────────────────────────
> 0.05       │ Distributions not significantly different
< 0.05       │ Reject null — distributions differ (DRIFT)
< 0.01       │ Strong evidence of drift
```

### Dataset-level Status
```
Drift Ratio   │ System Status
──────────────┼──────────────────────
0% – 25%      │ 🟢 STABLE     — normal operation
25% – 50%     │ 🟡 MONITOR    — investigate features
> 50%         │ 🔴 HIGH_RISK  — immediate action required
```

---

## 🖥️ App Pages

### 📂 Page 1 — Data Setup
```
┌──────────────────────────────────────────┐
│  📋 Reference Dataset    🔄 Current Dataset │
│  ─────────────────────   ──────────────── │
│  [ Upload CSV ]          [ Upload CSV ]   │
│  ☑ Use Telco demo        ☑ Simulate drift │
│                                          │
│  ⚙️ Column Configuration                  │
│  Target: [Churn ▼]                       │
│  Numerical: [tenure, MonthlyCharges...]  │
│  Categorical: [gender, Contract...]      │
│                                          │
│  🤖 Model Training                        │
│  ○ Auto-train  ○ Load pre-saved          │
│  [🚀 Train / Load Model]                 │
└──────────────────────────────────────────┘
```

- Loads demo dataset (Telco Churn — 7 043 rows, 21 cols) instantly  
- Simulates drift by scaling 4 numerical features by 1.15–1.40×  
- Auto-infers column types using dtype + cardinality heuristics  
- Trains or loads RandomForest pipeline in seconds  

---

### 🔬 Page 2 — Feature Drift Analysis

Four sub-tabs in one powerful view:
```
[ 📊 Heatmap ] [ 📋 Feature Table ] [ 🔍 Distributions ] [ 📈 Predictions ]
```

**Heatmap Tab** — Horizontal bar chart showing drift score per feature, red = drifted  
**Feature Table** — Full table with KS p-value, PSI, Mean Change %, Z-Score, Chi2  
**Distributions Tab** — Overlay reference vs current distributions per feature  
**Predictions Tab** — Prediction probability shift between reference and current  

---

### 🤖 Page 3 — Model Performance
```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ ACCURACY │  │   F1     │  │ ROC-AUC  │  │    R²    │
│  [████]  │  │  [███]   │  │  [████]  │  │  [████]  │
│   0.87   │  │  0.82    │  │   0.91   │  │    —     │
└──────────┘  └──────────┘  └──────────┘  └──────────┘

Model Info               Feature Importance
─────────────────────    ──────────────────────────────
Task: CLASSIFICATION      tenure         ████████ 0.21
Algorithm: RandomForest   MonthlyCharges ██████   0.18
Numerical Features: 14    Contract       █████    0.14
Categorical Features: 6   TotalCharges   ████     0.12
Total Features: 20        ...
```

---

### 📊 Page 4 — Evidently AI Reports

Full HTML reports rendered inline — each can be downloaded as a standalone HTML file:

| Report | What It Shows |
|--------|--------------|
| **Data Drift** | Column-by-column drift for entire dataset with statistical tests |
| **Target Drift** | How the label distribution has shifted |
| **Data Quality** | Missing values, duplicates, out-of-range values per column |
| **Classification** | Precision, Recall, F1, Confusion Matrix, Prediction Drift |

---

### 📈 Page 5 — Live Monitoring Dashboard
```
        🔴  MODEL STATUS: HIGH RISK
    4 of 20 features drifted · 20.0% drift ratio · Session 3

 ┌────────────────────────────────────────────────────┐
 │          Drift Ratio Timeline                      │
 │  50% ┤                              ·              │
 │  40% ┤                         ·                  │
 │  30% ┤                    ·                       │
 │  20% ┤  ·──────·                                  │
 │  10% ┤                                            │
 │      └─────────────────────────────────────────── │
 │          Session 1    Session 2    Session 3       │
 └────────────────────────────────────────────────────┘

💡 Recommendations:
🚨 Critical drift detected — model predictions may be unreliable
🔄 Retrain immediately — collect fresh labeled data
🛑 Consider model rollback to last stable version
📧 Alert stakeholders — notify data science team
```

---

### 📥 Page 6 — Export Reports
```
┌─────────────────────────┐  ┌─────────────────────────┐
│  📄 JSON Report          │  │  📊 CSV Report           │
│  Full drift results      │  │  Feature-level table    │
│  + model metrics         │  │  ready for Excel/BI     │
│  + session history       │  │                         │
│  [📥 Download JSON]      │  │  [📥 Download CSV]       │
└─────────────────────────┘  └─────────────────────────┘

📥 Evidently HTML Reports
[Data Drift] [Target Drift] [Data Quality] [Classification]
```

---

## 🛠️ Tech Stack
```
┌─────────────────────────────────────────────────────────────┐
│  Layer              Library              Purpose             │
├─────────────────────────────────────────────────────────────┤
│  UI Framework       Streamlit ≥ 1.32     Web app & routing   │
│  ML Pipeline        scikit-learn ≥ 1.3   Train & evaluate    │
│  Drift Reports      Evidently AI ≥ 0.7   Deep HTML reports   │
│  Visualizations     Plotly ≥ 5.18        Interactive charts  │
│  Statistical Tests  SciPy ≥ 1.11         KS, Chi2, etc.      │
│  Data Processing    Pandas ≥ 2.0          DataFrames          │
│  Numerics           NumPy ≥ 1.24         Arrays & math        │
│  Model Persistence  joblib ≥ 1.3          Save/load pipelines │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

Drift thresholds can be adjusted live from the **sidebar** without restarting:
```python
# Default thresholds (configurable via sidebar sliders)
DriftDetector(
    ks_threshold   = 0.05,   # KS p-value → lower = stricter
    psi_threshold  = 0.20,   # PSI score  → lower = stricter
    psi_buckets    = 10,     # Histogram bins for PSI calculation
    chi2_threshold = 0.05,   # Chi-squared p-value for categorical
)
```

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `ks_threshold` | `0.05` | `0.01–0.20` | KS p-value cutoff for numerical drift |
| `psi_threshold` | `0.20` | `0.05–0.50` | PSI score cutoff (0.1=stable, 0.25=high risk) |

---

## 📦 Dependencies
```txt
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
evidently>=0.7.0
scipy>=1.11.0
plotly>=5.18.0,<6
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🧩 Module Reference

### `DataProcessor` — `src/data_processor.py`

| Method | Description |
|--------|-------------|
| `load_csv(file)` | Load CSV from file upload or path |
| `infer_column_types(df, target)` | Auto-detect numerical / categorical / ID columns |
| `preprocess(df, col_info)` | Fill missing values, return feature matrix |
| `get_dataset_summary(df)` | Rows, cols, missing %, duplicates, memory MB |
| `compute_statistics(df, cols)` | Mean, std, percentiles per numerical column |

### `DriftDetector` — `src/drift_detector.py`

| Method | Description |
|--------|-------------|
| `ks_test(ref, cur)` | Kolmogorov-Smirnov 2-sample test |
| `wasserstein_distance(ref, cur)` | Earth Mover's Distance (normalized) |
| `psi(ref, cur)` | Population Stability Index |
| `mean_shift_zscore(ref, cur)` | Z-score of mean change |
| `chi2_test(ref, cur)` | Chi-squared categorical test |
| `compute_full_drift(ref, cur, num, cat)` | All tests across all features |
| `compute_dataset_drift_score(results)` | Overall status + drift ratio |
| `compute_prediction_drift(ref_p, cur_p)` | Model output distribution shift |

### `ModelManager` — `src/model_manager.py`

| Method | Description |
|--------|-------------|
| `detect_task_type(y)` | Auto-detect classification vs regression |
| `build_pipeline(num, cat, task)` | sklearn Pipeline with preprocessing + RandomForest |
| `train(X, y, num, cat)` | Fit pipeline, return metrics dict |
| `predict(X)` | Generate predictions or probabilities |
| `evaluate_on_new_data(X, y)` | Score model on current data |
| `get_feature_importance()` | Return importance DataFrame |
| `save()` / `load()` | Persist/restore pipeline with joblib |

### `Visualizer` — `src/visualizer.py`

| Function | Chart Type |
|----------|-----------|
| `feature_drift_heatmap(results)` | Horizontal bar chart — drift scores |
| `distribution_comparison(ref, cur, col, type)` | Histogram / bar overlay |
| `model_performance_gauge(val, label, threshold)` | Gauge chart with threshold arc |
| `drift_timeline(history)` | Line chart — drift ratio over sessions |
| `feature_importance_chart(df)` | Horizontal bar — top features |
| `prediction_drift_chart(ref_p, cur_p, task)` | Distribution overlay of predictions |
| `dataset_overview_chart(ref, cur)` | Side-by-side dataset statistics |
| `correlation_heatmap(df, cols)` | Feature correlation matrix |

### `EvidentlyAnalyzer` — `src/evidently_analyzer.py`

| Method | Description |
|--------|-------------|
| `build_column_mapping(df, num, cat, target, pred)` | Configure Evidently ColumnMapping |
| `run_data_drift_report(ref, cur, mapping)` | Generate & save Data Drift HTML |
| `run_target_drift_report(ref, cur, mapping)` | Generate & save Target Drift HTML |
| `run_data_quality_report(ref, cur, mapping)` | Generate & save Quality HTML |
| `run_classification_report(ref, cur, mapping)` | Generate & save Classification HTML |
| `display_report(path, height)` | Render HTML inline via `st.components` |
| `get_report_bytes(path)` | Read report as bytes for download |

---

## 🗺️ Usage Walkthrough
```
Step 1 ─── Open the app → Data Setup tab
            ☑ Use built-in Telco Churn demo
            ☑ Simulate drift from reference data
            Select Target: "Churn"
            Click: 🚀 Train / Load Model

Step 2 ─── Navigate to: Feature Drift
            Click: ▶ Run Drift Analysis
            View heatmap → see 4 red bars (drifted features)
            Check Feature Table for KS p-values and PSI scores
            Click Distributions tab → compare overlaid histograms

Step 3 ─── Navigate to: Model Performance
            Review Accuracy / F1 / ROC-AUC gauges
            Check Feature Importance chart
            Click: 📊 Evaluate on Current Data

Step 4 ─── Navigate to: Evidently Reports
            Click: ▶ Generate Data Drift
            Explore full interactive HTML report
            Download if needed

Step 5 ─── Navigate to: Live Monitoring
            See overall status banner: 🔴 HIGH RISK
            Read actionable recommendations

Step 6 ─── Navigate to: Export Reports
            📥 Download JSON  — full drift results + metrics
            📥 Download CSV   — feature-level table
            📥 Download HTML  — Evidently reports
```

---

## 🔄 Simulation Mode

DriftWatch includes a built-in **drift simulator** for demo and testing purposes:
```python
# Synthetic drift applied automatically in simulation mode
for col in num_cols[:4]:
    current[col] = current[col] * np.random.uniform(1.15, 1.40)
    #                                                ^^^^^^^^^^^^
    #                              Scale factor: +15% to +40% shift
```

This triggers **KS drift** and **PSI alerts** on 4 numerical features, pushing the overall status to `HIGH_RISK` — perfect for exploring the full monitoring workflow without real production data.

---

## 📤 Export Format

### JSON Report Structure
```json
{
  "summary": {
    "status": "HIGH_RISK",
    "drift_ratio": 20.0,
    "drifted_features": 4,
    "total_features": 20
  },
  "model_metrics": {
    "accuracy": 0.8714,
    "f1_score": 0.8231,
    "roc_auc": 0.9102
  },
  "feature_drift": {
    "tenure": {
      "type": "numerical",
      "drift_detected": true,
      "ks_p_value": 0.0001,
      "psi": 0.3412,
      "mean_change_pct": 32.4
    }
  },
  "sessions": [...],
  "copyright": {
    "owner": "Saksham Srivastava",
    "email": "sakshamsrivastava7000@gmail.com"
  }
}
```

---

## 🤝 Contributing
```bash
# Fork → Clone → Create branch
git checkout -b feature/new-drift-method

# Make changes, add tests
# Submit PR with description of the drift method added
```

**Areas where contributions are welcome:**
- Additional drift detection methods (e.g., Maximum Mean Discrepancy)  
- Support for time-series data  
- Slack / email alerting integrations  
- Scheduled monitoring (APScheduler / cron)  
- Docker deployment configuration  

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

**Built with ❤️ by [Saksham Srivastava](mailto:sakshamsrivastava7000@gmail.com)**

*DriftWatch — Know before your users do.*

[![Live App](https://img.shields.io/badge/🚀_Try_It_Live-DriftWatch-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ml-drift-detection---performance-monitoring-nm3yyacwvxecxrjgkg.streamlit.app/)

</div>
