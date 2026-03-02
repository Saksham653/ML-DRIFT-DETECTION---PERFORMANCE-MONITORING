import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional


class DataProcessor:
    """Universal data processor for any dataset."""

    def __init__(self):
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.target_col: Optional[str] = None
        self.id_cols: List[str] = []

    def load_csv(self, file) -> pd.DataFrame:
        """Load CSV from file upload or path."""
        try:
            if hasattr(file, "read"):
                df = pd.read_csv(file)
            else:
                df = pd.read_csv(file)
            return df
        except Exception as e:
            raise ValueError(f"Error loading CSV: {e}")

    def infer_column_types(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict:
        """Auto-detect column types."""
        id_patterns = ["id", "ID", "Id", "uuid", "index", "key", "customer", "user", "account"]
        id_cols = []
        num_cols = []
        cat_cols = []

        for col in df.columns:
            if col == target_col:
                continue
            # Check if it looks like an ID column
            if any(pat.lower() in col.lower() for pat in id_patterns) and df[col].nunique() > 0.9 * len(df):
                id_cols.append(col)
                continue
            # Check dtype
            if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                # If low cardinality int, might be categorical
                if df[col].dtype in ["int64", "int32"] and df[col].nunique() <= 15:
                    cat_cols.append(col)
                else:
                    num_cols.append(col)
            else:
                cat_cols.append(col)

        return {
            "numerical": num_cols,
            "categorical": cat_cols,
            "id": id_cols,
            "target": target_col
        }

    def preprocess(self, df: pd.DataFrame, col_info: Dict, fit: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess data for modeling - returns (features_df, processed_df)."""
        drop_cols = col_info.get("id", [])
        target = col_info.get("target")

        # Drop ID columns and target
        feature_cols = [c for c in df.columns if c not in drop_cols and c != target and c != "predictions"]
        X = df[feature_cols].copy()

        # Fill missing values
        for col in col_info.get("numerical", []):
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())

        for col in col_info.get("categorical", []):
            if col in X.columns:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "unknown")

        return X, df[feature_cols + ([target] if target and target in df.columns else [])]

    def compute_statistics(self, df: pd.DataFrame, numerical_cols: List[str]) -> Dict:
        """Compute baseline statistics for numerical columns."""
        stats = {}
        for col in numerical_cols:
            if col in df.columns:
                stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "median": float(df[col].median()),
                    "q25": float(df[col].quantile(0.25)),
                    "q75": float(df[col].quantile(0.75)),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "missing_rate": float(df[col].isnull().mean()),
                }
        return stats

    def compute_cat_statistics(self, df: pd.DataFrame, categorical_cols: List[str]) -> Dict:
        """Compute baseline statistics for categorical columns."""
        stats = {}
        for col in categorical_cols:
            if col in df.columns:
                val_counts = df[col].value_counts(normalize=True)
                stats[col] = val_counts.to_dict()
        return stats

    def get_dataset_summary(self, df: pd.DataFrame) -> Dict:
        """Get a high-level summary of the dataset."""
        return {
            "rows": len(df),
            "cols": len(df.columns),
            "missing_pct": float(df.isnull().mean().mean() * 100),
            "duplicate_rows": int(df.duplicated().sum()),
            "dtypes": df.dtypes.value_counts().to_dict(),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        }
