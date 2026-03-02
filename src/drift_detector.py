import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple


class DriftDetector:
    """
    Production-grade drift detector with multiple statistical methods:
    - KS Test (numerical)
    - Chi-Squared Test (categorical)
    - Population Stability Index (PSI)
    - Jensen-Shannon Divergence
    - Wasserstein Distance
    - Z-Score (mean shift)
    """

    def __init__(self, psi_buckets: int = 10, psi_threshold: float = 0.2,
                 ks_threshold: float = 0.05, chi2_threshold: float = 0.05):
        self.psi_buckets = psi_buckets
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.chi2_threshold = chi2_threshold

    # ─── Numerical Drift ────────────────────────────────────────────────────

    def ks_test(self, reference: pd.Series, current: pd.Series) -> Dict:
        """Kolmogorov-Smirnov test for numerical columns."""
        ref_clean = reference.dropna()
        cur_clean = current.dropna()
        if len(ref_clean) < 5 or len(cur_clean) < 5:
            return {"statistic": None, "p_value": None, "drift_detected": False, "method": "ks"}
        stat, p_value = stats.ks_2samp(ref_clean, cur_clean)
        return {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": bool(p_value < self.ks_threshold),
            "method": "ks",
        }

    def wasserstein_distance(self, reference: pd.Series, current: pd.Series) -> Dict:
        """Earth Mover's Distance (Wasserstein-1)."""
        ref_clean = reference.dropna()
        cur_clean = current.dropna()
        if len(ref_clean) < 5 or len(cur_clean) < 5:
            return {"distance": None, "drift_detected": False, "method": "wasserstein"}
        distance = float(stats.wasserstein_distance(ref_clean, cur_clean))
        # Normalize by reference std to get relative distance
        ref_std = float(ref_clean.std()) or 1.0
        normalized = distance / ref_std
        return {
            "distance": round(distance, 4),
            "normalized_distance": round(normalized, 4),
            "drift_detected": bool(normalized > 0.2),
            "method": "wasserstein",
        }

    def psi(self, reference: pd.Series, current: pd.Series) -> Dict:
        """Population Stability Index."""
        ref_clean = reference.dropna()
        cur_clean = current.dropna()
        if len(ref_clean) < 10 or len(cur_clean) < 10:
            return {"psi": None, "drift_detected": False, "method": "psi"}

        # Create bins based on reference quantiles
        breakpoints = np.percentile(ref_clean, np.linspace(0, 100, self.psi_buckets + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return {"psi": None, "drift_detected": False, "method": "psi"}

        ref_counts = np.histogram(ref_clean, bins=breakpoints)[0]
        cur_counts = np.histogram(cur_clean, bins=breakpoints)[0]

        # Avoid division by zero
        ref_dist = (ref_counts + 0.001) / (len(ref_clean) + 0.001 * len(breakpoints))
        cur_dist = (cur_counts + 0.001) / (len(cur_clean) + 0.001 * len(breakpoints))

        psi_value = float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))

        return {
            "psi": round(psi_value, 4),
            "drift_detected": bool(psi_value > self.psi_threshold),
            "severity": "stable" if psi_value < 0.1 else "monitor" if psi_value < 0.25 else "high_risk",
            "method": "psi",
        }

    def mean_shift(self, reference: pd.Series, current: pd.Series) -> Dict:
        """Z-score based mean shift detection."""
        ref_clean = reference.dropna()
        cur_clean = current.dropna()
        if len(ref_clean) < 5 or len(cur_clean) < 5:
            return {"z_score": None, "drift_detected": False, "method": "mean_shift"}

        ref_mean = float(ref_clean.mean())
        ref_std = float(ref_clean.std()) or 1.0
        cur_mean = float(cur_clean.mean())

        z_score = abs(cur_mean - ref_mean) / ref_std
        return {
            "reference_mean": round(ref_mean, 4),
            "current_mean": round(cur_mean, 4),
            "mean_change_pct": round((cur_mean - ref_mean) / abs(ref_mean) * 100 if ref_mean != 0 else 0, 2),
            "z_score": round(z_score, 4),
            "drift_detected": bool(z_score > 2.0),
            "method": "mean_shift",
        }

    # ─── Categorical Drift ───────────────────────────────────────────────────

    def chi2_test(self, reference: pd.Series, current: pd.Series) -> Dict:
        """Chi-squared test for categorical columns."""
        ref_clean = reference.dropna().astype(str)
        cur_clean = current.dropna().astype(str)
        if len(ref_clean) < 5 or len(cur_clean) < 5:
            return {"statistic": None, "p_value": None, "drift_detected": False, "method": "chi2"}

        all_cats = list(set(ref_clean.unique()) | set(cur_clean.unique()))
        ref_counts = [ref_clean.value_counts().get(c, 0) for c in all_cats]
        cur_counts = [cur_clean.value_counts().get(c, 0) for c in all_cats]

        # Normalize to same size
        ref_expected = np.array(ref_counts) * len(cur_clean) / max(len(ref_clean), 1)

        # Filter zero expected
        mask = np.array(ref_expected) > 0
        if mask.sum() < 2:
            return {"statistic": None, "p_value": None, "drift_detected": False, "method": "chi2"}

        try:
            stat, p_value = stats.chisquare(np.array(cur_counts)[mask], f_exp=ref_expected[mask])
            return {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": bool(p_value < self.chi2_threshold),
                "method": "chi2",
            }
        except Exception:
            return {"statistic": None, "p_value": None, "drift_detected": False, "method": "chi2"}

    def distribution_shift(self, reference: pd.Series, current: pd.Series) -> Dict:
        """Distribution shift for categorical (top-N proportions)."""
        ref_dist = reference.dropna().value_counts(normalize=True)
        cur_dist = current.dropna().value_counts(normalize=True)
        all_cats = set(ref_dist.index) | set(cur_dist.index)

        max_shift = 0.0
        shifts = {}
        for cat in all_cats:
            ref_p = float(ref_dist.get(cat, 0))
            cur_p = float(cur_dist.get(cat, 0))
            shift = abs(cur_p - ref_p)
            max_shift = max(max_shift, shift)
            shifts[str(cat)] = {"reference": round(ref_p, 4), "current": round(cur_p, 4), "shift": round(shift, 4)}

        return {
            "max_shift": round(max_shift, 4),
            "drift_detected": bool(max_shift > 0.1),
            "category_shifts": shifts,
            "method": "distribution_shift",
        }

    # ─── Full Dataset Drift ──────────────────────────────────────────────────

    def compute_full_drift(self, reference: pd.DataFrame, current: pd.DataFrame,
                            numerical_cols: List[str], categorical_cols: List[str]) -> Dict:
        """Run full drift analysis on all columns."""
        results = {}

        for col in numerical_cols:
            if col not in reference.columns or col not in current.columns:
                continue
            results[col] = {
                "type": "numerical",
                "ks": self.ks_test(reference[col], current[col]),
                "psi": self.psi(reference[col], current[col]),
                "mean_shift": self.mean_shift(reference[col], current[col]),
                "wasserstein": self.wasserstein_distance(reference[col], current[col]),
            }
            # Aggregate drift decision (any method detects drift)
            results[col]["drift_detected"] = (
                results[col]["ks"]["drift_detected"] or
                results[col]["psi"]["drift_detected"] or
                results[col]["mean_shift"]["drift_detected"]
            )

        for col in categorical_cols:
            if col not in reference.columns or col not in current.columns:
                continue
            results[col] = {
                "type": "categorical",
                "chi2": self.chi2_test(reference[col], current[col]),
                "distribution": self.distribution_shift(reference[col], current[col]),
            }
            results[col]["drift_detected"] = (
                results[col]["chi2"]["drift_detected"] or
                results[col]["distribution"]["drift_detected"]
            )

        return results

    def compute_dataset_drift_score(self, drift_results: Dict) -> Dict:
        """Compute an overall dataset drift score."""
        total = len(drift_results)
        if total == 0:
            return {"score": 0.0, "drifted_features": 0, "total_features": 0, "status": "STABLE"}

        drifted = sum(1 for v in drift_results.values() if v.get("drift_detected", False))
        score = drifted / total

        if score < 0.2:
            status = "STABLE"
        elif score < 0.5:
            status = "MONITOR"
        else:
            status = "HIGH_RISK"

        return {
            "score": round(score, 4),
            "drift_ratio": round(score * 100, 1),
            "drifted_features": drifted,
            "total_features": total,
            "status": status,
        }

    def compute_prediction_drift(self, ref_preds: np.ndarray, cur_preds: np.ndarray,
                                  task_type: str = "classification") -> Dict:
        """Detect drift in model predictions."""
        if task_type == "classification":
            ref_rate = float(np.mean(ref_preds))
            cur_rate = float(np.mean(cur_preds))
            shift = abs(cur_rate - ref_rate)
            return {
                "reference_positive_rate": round(ref_rate, 4),
                "current_positive_rate": round(cur_rate, 4),
                "shift": round(shift, 4),
                "drift_detected": bool(shift > 0.1),
                "ks": self.ks_test(pd.Series(ref_preds.astype(float)), pd.Series(cur_preds.astype(float))),
            }
        else:
            return {
                "mean_shift": self.mean_shift(pd.Series(ref_preds), pd.Series(cur_preds)),
                "ks": self.ks_test(pd.Series(ref_preds), pd.Series(cur_preds)),
                "drift_detected": self.ks_test(pd.Series(ref_preds), pd.Series(cur_preds))["drift_detected"],
            }
