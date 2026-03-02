# 🔭 DriftWatch — Production ML Drift Detection & Monitoring

A complete, production-ready ML model drift detection system that works with **any tabular dataset**.

## ✨ Features

- **Universal Dataset Support** — Upload any CSV dataset, auto-detects column types
- **Multi-Method Drift Detection:**
  - Kolmogorov-Smirnov Test (numerical)
  - Chi-Squared Test (categorical)
  - Population Stability Index (PSI)
  - Wasserstein Distance
  - Z-Score / Mean Shift
- **Auto ML Pipeline** — Auto-trains RandomForest on your reference data
- **Evidently AI Reports** — Data Drift, Target Drift, Data Quality, Classification
- **Interactive Visualizations** — Plotly-powered charts & heatmaps
- **Model Performance Monitoring** — Accuracy, F1, ROC-AUC gauges
- **Export Reports** — JSON, CSV, HTML formats
- **Session History** — Tracks drift across multiple analysis runs

## 🚀 Quick Start

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy to Streamlit Cloud (Free)
1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Connect your repo → set `app.py` as the main file
4. Click **Deploy**

### Deploy to Hugging Face Spaces
1. Create a new Space at https://huggingface.co/spaces
2. Select **Streamlit** as the SDK
3. Upload all files
4. The app auto-deploys!

### Deploy to Railway / Render
```bash
# Add a Procfile:
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
```

## 📂 Project Structure

```
drift_app/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Dependencies
├── src/
│   ├── __init__.py
│   ├── data_processor.py       # Universal data handling & type inference
│   ├── drift_detector.py       # Statistical drift tests (KS, Chi2, PSI, etc.)
│   ├── model_manager.py        # Auto-train & evaluate ML models
│   ├── visualizer.py           # Plotly interactive charts
│   └── evidently_analyzer.py   # Evidently AI report wrapper
├── artifacts/                  # Saved model pipelines
├── reports/                    # Generated HTML reports
└── data/
    └── Telco_Customer_churn.csv # Demo dataset
```

## 🔧 Usage

1. **Data Setup** — Upload reference and current datasets (or use demo data)
2. **Configure** — Select target column and feature types
3. **Train Model** — Auto-train or load pre-trained pipeline
4. **Feature Drift** — Run multi-method drift analysis
5. **Model Performance** — View gauges and feature importances
6. **Evidently Reports** — Deep-dive HTML reports
7. **Monitoring** — Timeline and recommendations dashboard
8. **Export** — Download JSON/CSV/HTML reports

## 📊 Drift Thresholds

| Method | Default Threshold | Interpretation |
|--------|------------------|----------------|
| KS p-value | < 0.05 | Reject null (distributions differ) |
| PSI | > 0.20 | Moderate shift |
| Z-Score | > 2.0 | 2 standard deviations from mean |
| Max Cat Shift | > 0.10 | 10% proportion shift |

## 🛠️ Tech Stack

- **Streamlit** — Web UI framework
- **scikit-learn** — ML pipeline & model training
- **Evidently AI** — Statistical drift reports
- **Plotly** — Interactive charts
- **SciPy** — Statistical tests (KS, Chi-Squared)
- **Pandas / NumPy** — Data processing
