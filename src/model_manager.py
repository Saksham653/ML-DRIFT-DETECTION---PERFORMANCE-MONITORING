import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Optional, Tuple, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import train_test_split


class ModelManager:
    """Auto-train and manage ML models for any dataset."""

    DEFAULT_MODEL_PATH = "artifacts/model_pipeline.pkl"
    DEFAULT_META_PATH = "artifacts/model_meta.pkl"

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.task_type: str = "classification"  # or "regression"
        self.target_encoder: Optional[LabelEncoder] = None
        self.feature_names: list = []
        self.numerical_cols: list = []
        self.categorical_cols: list = []

    def detect_task_type(self, y: pd.Series) -> str:
        """Auto-detect if target is for classification or regression."""
        if y.dtype in ["object", "bool", "category"]:
            return "classification"
        unique_ratio = y.nunique() / len(y)
        if y.nunique() <= 20 or unique_ratio < 0.05:
            return "classification"
        return "regression"

    def build_pipeline(self, numerical_cols: list, categorical_cols: list,
                        task_type: str = "classification") -> Pipeline:
        """Build a sklearn pipeline with preprocessing + model."""
        # Preprocessing
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        steps = []
        if numerical_cols:
            steps.append(("num", num_transformer, numerical_cols))
        if categorical_cols:
            steps.append(("cat", cat_transformer, categorical_cols))

        if steps:
            preprocessor = ColumnTransformer(transformers=steps, remainder="drop")
        else:
            preprocessor = ColumnTransformer(transformers=[], remainder="passthrough")

        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    def train(self, X: pd.DataFrame, y: pd.Series, numerical_cols: list,
               categorical_cols: list) -> Dict:
        """Train a model and return performance metrics."""
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.feature_names = list(X.columns)

        # Detect task type
        self.task_type = self.detect_task_type(y)

        # Encode target for classification
        y_encoded = y.copy()
        if self.task_type == "classification":
            self.target_encoder = LabelEncoder()
            y_encoded = self.target_encoder.fit_transform(y.astype(str))

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        # Build and train pipeline
        self.pipeline = self.build_pipeline(
            [c for c in numerical_cols if c in X.columns],
            [c for c in categorical_cols if c in X.columns],
            self.task_type
        )
        self.pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        metrics = self._compute_metrics(y_test, y_pred, X_test)

        return metrics

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                           X_test: Optional[pd.DataFrame] = None) -> Dict:
        """Compute performance metrics."""
        if self.task_type == "classification":
            metrics = {
                "task_type": "classification",
                "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                "f1_score": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            }
            # Try ROC AUC
            try:
                if X_test is not None and hasattr(self.pipeline, "predict_proba"):
                    y_prob = self.pipeline.predict_proba(X_test)
                    if y_prob.shape[1] == 2:
                        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob[:, 1])), 4)
                    else:
                        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")), 4)
            except Exception:
                pass
        else:
            metrics = {
                "task_type": "regression",
                "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
                "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
                "r2": round(float(r2_score(y_true, y_pred)), 4),
            }
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self.pipeline is None:
            raise ValueError("Model not trained or loaded.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate probability predictions."""
        if self.pipeline is None or not hasattr(self.pipeline, "predict_proba"):
            return None
        try:
            return self.pipeline.predict_proba(X)
        except Exception:
            return None

    def save(self, path: str = DEFAULT_MODEL_PATH, meta_path: str = DEFAULT_META_PATH):
        """Save model pipeline and metadata."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self.pipeline, path)
        meta = {
            "task_type": self.task_type,
            "feature_names": self.feature_names,
            "numerical_cols": self.numerical_cols,
            "categorical_cols": self.categorical_cols,
            "target_encoder": self.target_encoder,
        }
        joblib.dump(meta, meta_path)

    def load(self, path: str = DEFAULT_MODEL_PATH, meta_path: str = DEFAULT_META_PATH) -> bool:
        """Load a saved model pipeline."""
        try:
            self.pipeline = joblib.load(path)
            if os.path.exists(meta_path):
                meta = joblib.load(meta_path)
                self.task_type = meta.get("task_type", "classification")
                self.feature_names = meta.get("feature_names", [])
                self.numerical_cols = meta.get("numerical_cols", [])
                self.categorical_cols = meta.get("categorical_cols", [])
                self.target_encoder = meta.get("target_encoder")
            return True
        except Exception:
            return False

    def evaluate_on_new_data(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate model on new data with ground truth labels."""
        if self.pipeline is None:
            return {}
        y_pred = self.predict(X)
        y_true = y.copy()
        if self.task_type == "classification" and self.target_encoder is not None:
            try:
                y_true = self.target_encoder.transform(y.astype(str))
            except Exception:
                pass
        return self._compute_metrics(y_true, y_pred, X)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Extract feature importances from the trained model."""
        if self.pipeline is None:
            return None
        try:
            model = self.pipeline.named_steps["model"]
            preprocessor = self.pipeline.named_steps["preprocessor"]

            # Get feature names after preprocessing
            feature_names = []
            for name, transformer, cols in preprocessor.transformers_:
                if name == "num":
                    feature_names.extend(cols)
                elif name == "cat":
                    enc = transformer.named_steps["encoder"]
                    feature_names.extend([f"{c}_{v}" for c, vals in zip(cols, enc.categories_) for v in vals])

            importances = model.feature_importances_
            if len(importances) == len(feature_names):
                df = pd.DataFrame({"feature": feature_names, "importance": importances})
                return df.sort_values("importance", ascending=False).head(20)
        except Exception:
            pass
        return None
