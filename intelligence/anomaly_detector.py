"""
Anomaly Detection and Intelligent Monitoring
Author: Member 2 
Provides:
 - AnomalyDetector: statistical + threshold + optional ML-based anomaly detection
 - PredictiveMonitor: simple trend fitting + short-term forecasting
"""
import time
import logging
import json
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
import joblib

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Try to import sklearn components; degrade gracefully if unavailable
_SKLEARN_AVAILABLE = True
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
except Exception:
    _SKLEARN_AVAILABLE = False
    logger.info("scikit-learn not available; ML-based anomaly detection disabled.")


class AnomalyDetector:
    """Intelligent anomaly detection for system metrics"""

    def __init__(self, window_size: int = 100, contamination: float = 0.1):
        # Configuration
        self.window_size = window_size
        self.contamination = contamination

        # Data storage
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_stats: Dict[str, Dict] = defaultdict(dict)
        self.anomaly_history: deque = deque(maxlen=1000)

        # Models (optional)
        self.isolation_forests: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}

        # Thresholds (example defaults; customize as needed)
        self.thresholds = {
            "cpu_percent": {"high": 85, "critical": 95},
            "memory_percent": {"high": 80, "critical": 90},
            "disk_percent": {"high": 85, "critical": 95},
            "temperature": {"high": 70, "critical": 85},
            "network_errors": {"high": 10, "critical": 100},
        }

        # State
        self.is_learning = True
        self.learning_samples = 0
        self.min_samples_for_detection = 50

        logger.info("Anomaly detector initialized")

    # ---------------------------
    # Data ingestion & detection
    # ---------------------------
    def add_sample(self, metric_name: str, value: float, timestamp: Optional[str] = None) -> None:
        """Add a new metric sample for anomaly detection"""
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat() + "Z"

        sample = {"value": float(value), "timestamp": timestamp}
        self.metric_history[metric_name].append(sample)
        # Update baseline stats (if enough samples)
        self._update_baseline_stats(metric_name)

        # Check for anomalies if we have enough data
        if len(self.metric_history[metric_name]) >= self.min_samples_for_detection:
            score = self._detect_anomaly(metric_name, value)
            if score > 0.5:  # anomaly threshold (tunable)
                self._record_anomaly(metric_name, float(value), float(score), timestamp)

    def _update_baseline_stats(self, metric_name: str) -> None:
        """Update baseline statistics for a metric"""
        values = [sample["value"] for sample in self.metric_history[metric_name]]
        if len(values) >= 3:
            self.baseline_stats[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "median": float(np.median(values)),
                "q1": float(np.percentile(values, 25)),
                "q3": float(np.percentile(values, 75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "samples": len(values),
                "last_updated": datetime.utcnow().isoformat() + "Z",
            }

    def _detect_anomaly(self, metric_name: str, value: float) -> float:
        """Detect if a value is anomalous and return a score in [0,1]"""
        try:
            stats_ = self.baseline_stats.get(metric_name)
            # Statistical z-score component
            statistical_anomaly = 0.0
            if stats_ and stats_.get("std", 0) != 0:
                z_score = abs((value - stats_["mean"]) / stats_["std"])
                # Normalize z-score to 0..1 (3-sigma -> ~1.0)
                statistical_anomaly = float(min(z_score / 3.0, 1.0))

            # Threshold-based component
            threshold_anomaly = self._check_threshold_anomaly(metric_name, value)

            # ML-based component (if model available)
            ml_anomaly = self._ml_anomaly_detection(metric_name, value)

            # Combine — use maximum to be conservative
            combined_score = float(max(statistical_anomaly, threshold_anomaly, ml_anomaly))
            return combined_score
        except Exception as e:
            logger.error(f"Anomaly detection error for {metric_name}: {e}", exc_info=True)
            return 0.0

    def _check_threshold_anomaly(self, metric_name: str, value: float) -> float:
        """Check for threshold-based anomalies (returns 0..1)"""
        thresholds = self.thresholds.get(metric_name, {})
        try:
            if "critical" in thresholds and value >= thresholds["critical"]:
                return 1.0
            elif "high" in thresholds and value >= thresholds["high"]:
                return 0.7
        except Exception:
            pass
        return 0.0

    def _ml_anomaly_detection(self, metric_name: str, value: float) -> float:
        """Machine learning based anomaly detection (IsolationForest). Returns 0..1."""
        if not _SKLEARN_AVAILABLE:
            return 0.0

        try:
            if metric_name not in self.isolation_forests or metric_name not in self.scalers:
                return 0.0

            X = np.array([[float(value)]])
            X_scaled = self.scalers[metric_name].transform(X)

            # decision_function returns shape (n_samples,)
            score_arr = self.isolation_forests[metric_name].decision_function(X_scaled)
            score_val = float(score_arr[0])

            # Convert decision_function output to 0..1 anomaly score:
            # decision_function: larger -> more normal. We'll map roughly:
            # normalized = (score_val + 0.5) clipped to [0,1], then invert for anomaly.
            normalized = max(0.0, min(1.0, (score_val + 0.5)))
            ml_score = 1.0 - normalized
            return float(max(0.0, min(1.0, ml_score)))
        except Exception as e:
            logger.error(f"ML anomaly detection error for {metric_name}: {e}", exc_info=True)
            return 0.0

    def _record_anomaly(self, metric_name: str, value: float, score: float, timestamp: str) -> None:
        """Record an anomaly event"""
        anomaly = {
            "metric_name": metric_name,
            "value": float(value),
            "anomaly_score": float(score),
            "timestamp": timestamp,
            "severity": self._get_severity(score),
            "baseline_stats": self.baseline_stats.get(metric_name, {}),
            "threshold_info": self.thresholds.get(metric_name, {}),
        }
        self.anomaly_history.append(anomaly)
        logger.warning(f"Anomaly detected: {metric_name}={value} (score={score:.3f})")

    def _get_severity(self, score: float) -> str:
        """Get severity level based on anomaly score"""
        if score >= 0.9:
            return "critical"
        elif score >= 0.7:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"

    # ---------------------------
    # ML model training / persistence
    # ---------------------------
    def train_ml_models(self) -> None:
        """Train IsolationForest models on collected history for each metric"""
        if not _SKLEARN_AVAILABLE:
            logger.info("Skipping ML training: scikit-learn not available.")
            return

        for metric_name in list(self.metric_history.keys()):
            samples = self.metric_history[metric_name]
            if len(samples) < self.min_samples_for_detection:
                continue
            try:
                values = np.array([float(s["value"]) for s in samples]).reshape(-1, 1)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(values)

                iso_forest = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100,
                )
                iso_forest.fit(X_scaled)

                self.scalers[metric_name] = scaler
                self.isolation_forests[metric_name] = iso_forest
                logger.info(f"ML model trained for {metric_name} with {len(values)} samples")
            except Exception as e:
                logger.error(f"ML model training error for {metric_name}: {e}", exc_info=True)

    def save_models(self, filepath: str) -> None:
        """Save trained models and metadata to disk"""
        try:
            model_data = {
                "isolation_forests": self.isolation_forests,
                "scalers": self.scalers,
                "baseline_stats": dict(self.baseline_stats),
                "thresholds": self.thresholds,
                "metadata": {
                    "saved_at": datetime.utcnow().isoformat() + "Z",
                    "window_size": self.window_size,
                    "contamination": self.contamination,
                },
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
        except Exception as e:
            logger.error(f"Model saving error: {e}", exc_info=True)

    def load_models(self, filepath: str) -> None:
        """Load trained models and metadata from disk"""
        try:
            model_data = joblib.load(filepath)
            self.isolation_forests = model_data.get("isolation_forests", {})
            self.scalers = model_data.get("scalers", {})
            self.baseline_stats = defaultdict(dict, model_data.get("baseline_stats", {}))
            self.thresholds = model_data.get("thresholds", self.thresholds)
            logger.info(f"Models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Model loading error: {e}", exc_info=True)

    # ---------------------------
    # Reporting
    # ---------------------------
    def get_anomaly_summary(self, hours: int = 24) -> Dict:
        """Get summary of anomalies in the specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_anomalies = []

        for a in self.anomaly_history:
            ts = a.get("timestamp")
            try:
                # Accept both Z and no-Z ISO strings
                anomaly_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                try:
                    anomaly_time = datetime.fromisoformat(ts)
                except Exception:
                    continue
            if anomaly_time >= cutoff_time:
                recent_anomalies.append(a)

        severity_counts = defaultdict(int)
        metric_counts = defaultdict(int)
        for anomaly in recent_anomalies:
            severity_counts[anomaly["severity"]] += 1
            metric_counts[anomaly["metric_name"]] += 1

        return {
            "total_anomalies": len(recent_anomalies),
            "by_severity": dict(severity_counts),
            "by_metric": dict(metric_counts),
            "recent_anomalies": recent_anomalies[-10:],
            "time_period_hours": hours,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def get_metric_health_score(self, metric_name: str) -> Dict:
        """Get health score for a specific metric (0-100, higher is better)"""
        if metric_name not in self.metric_history:
            return {"error": "Metric not found"}

        recent_anomalies = [a for a in list(self.anomaly_history) if a["metric_name"] == metric_name][-10:]
        if not recent_anomalies:
            health_score = 100
        else:
            avg_anomaly_score = float(np.mean([a["anomaly_score"] for a in recent_anomalies]))
            health_score = max(0, int(round(100 - (avg_anomaly_score * 100))))

        return {
            "metric_name": metric_name,
            "health_score": health_score,
            "recent_anomaly_count": len(recent_anomalies),
            "baseline_stats": self.baseline_stats.get(metric_name, {}),
            "samples_collected": len(self.metric_history[metric_name]),
            "ml_model_available": metric_name in self.isolation_forests,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


class PredictiveMonitor:
    """Predictive monitoring for system resources using simple linear trend fitting"""

    def __init__(self):
        self.metric_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.predictions: Dict[str, Dict] = {}
        self.prediction_accuracy: Dict[str, List[float]] = defaultdict(list)

    def add_metric_sample(self, metric_name: str, value: float, timestamp: Optional[str] = None) -> None:
        """Add metric sample for trend analysis"""
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat() + "Z"

        sample = {"value": float(value), "timestamp": timestamp, "time": time.time()}
        self.metric_trends[metric_name].append(sample)

        # Update predictions if we have enough data
        if len(self.metric_trends[metric_name]) >= 20:
            self._update_predictions(metric_name)

    def _update_predictions(self, metric_name: str) -> None:
        """Update predictions for a metric using linear regression"""
        try:
            samples = list(self.metric_trends[metric_name])
            times = np.array([s["time"] for s in samples], dtype=float)
            values = np.array([s["value"] for s in samples], dtype=float)

            # Normalize times so regression intercept is interpretable
            times_rel = times - times[0]
            if len(times_rel) < 2 or np.allclose(values, values[0]):
                return

            slope, intercept, r_value, p_value, std_err = stats.linregress(times_rel, values)

            # Predict next 5 time points (1..5 minutes ahead)
            future_seconds = np.arange(1, 6) * 60.0
            last_time_rel = times_rel[-1]
            future_times = last_time_rel + future_seconds
            preds = slope * future_times + intercept

            trend_label = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
            r2 = float(r_value ** 2)
            confidence = "high" if r2 > 0.7 else "medium" if r2 > 0.4 else "low"

            self.predictions[metric_name] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": r2,
                "p_value": float(p_value),
                "predictions": {
                    "1_min": float(preds[0]),
                    "2_min": float(preds[1]),
                    "3_min": float(preds[2]),
                    "4_min": float(preds[3]),
                    "5_min": float(preds[4]),
                },
                "trend": trend_label,
                "confidence": confidence,
                "last_updated": datetime.utcnow().isoformat() + "Z",
            }
        except Exception as e:
            logger.error(f"Prediction update error for {metric_name}: {e}", exc_info=True)

    def get_predictions(self, metric_name: str) -> Dict:
        """Get predictions for a specific metric"""
        return self.predictions.get(metric_name, {"error": "No predictions available"})

    def get_all_predictions(self) -> Dict:
        """Get predictions for all metrics"""
        return {"predictions": dict(self.predictions), "timestamp": datetime.utcnow().isoformat() + "Z"}


# Global instances
anomaly_detector = AnomalyDetector()
predictive_monitor = PredictiveMonitor()
