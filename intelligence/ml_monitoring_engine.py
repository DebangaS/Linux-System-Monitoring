"""
Machine Learning Monitoring Engine
Author: Member 2 
"""
import os
import time
import threading
import asyncio
import joblib
import json
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Local imports (assumed to exist in project)
from database.advanced_models import get_advanced_db_manager
from monitors.system_monitor import system_monitor

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    """Machine learning prediction result"""
    metric_name: str
    predicted_value: float
    confidence_score: float
    prediction_horizon_minutes: int
    model_accuracy: float
    timestamp: str
    features_used: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnomalyDetectionResult:
    """Anomaly detection result"""
    metric_name: str
    current_value: float
    anomaly_score: float
    is_anomaly: bool
    severity: str
    expected_range: Tuple[float, float]
    contributing_factors: List[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MLMonitoringEngine:
    """Advanced machine learning monitoring engine"""

    def __init__(self):
        self.db_manager = get_advanced_db_manager()
        # Model storage
        self.prediction_models: Dict[str, Any] = {}
        self.anomaly_detectors: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        # Configuration
        self.config = {
            "prediction_horizon_minutes": [5, 15, 30, 60],
            "retrain_interval_hours": 24,
            "min_training_samples": 100,
            "anomaly_contamination": 0.01,
            "prediction_confidence_threshold": 0.7,
        }
        # Data buffers
        self.feature_buffer = defaultdict(lambda: deque(maxlen=2000))
        self.prediction_history = defaultdict(list)
        self.model_performance = defaultdict(dict)
        # Training status
        self.last_training_time: Dict[str, datetime] = {}
        self.training_in_progress = set()
        # Real-time processing
        self.processing_active = False
        self.processing_thread: Optional[threading.Thread] = None
        logger.info("ML Monitoring Engine initialized")

    def start_ml_engine(self):
        """Start the ML monitoring engine"""
        try:
            self.processing_active = True
            # Load existing models
            self._load_models()
            # Start background processing thread
            self.processing_thread = threading.Thread(target=self._ml_processing_loop, daemon=True)
            self.processing_thread.start()
            # Schedule model training using asyncio in a background thread-safe way
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._schedule_model_training())
            except RuntimeError:
                # No running loop; create a separate event loop in a thread
                threading.Thread(target=self._start_background_event_loop, daemon=True).start()

            logger.info("ML Monitoring Engine started")
        except Exception as e:
            logger.error(f"Error starting ML engine: {e}")
            raise

    def _start_background_event_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.create_task(self._schedule_model_training())
        loop.run_forever()

    def stop_ml_engine(self):
        """Stop the ML monitoring engine"""
        try:
            self.processing_active = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)
            # Save models
            self._save_models()
            logger.info("ML Monitoring Engine stopped")
        except Exception as e:
            logger.error(f"Error stopping ML engine: {e}")

    def _ml_processing_loop(self):
        """Main ML processing loop"""
        while self.processing_active:
            try:
                # Collect current metrics
                self._collect_features_for_ml()
                # Generate predictions
                self._generate_predictions()
                # Detect anomalies
                self._detect_anomalies()
                # Update model performance
                self._update_model_performance()
                # Sleep between processing cycles
                time.sleep(30)  # Process every 30 seconds
            except Exception as e:
                logger.error(f"Error in ML processing loop: {e}")
                time.sleep(30)

    def _collect_features_for_ml(self):
        """Collect features for machine learning models"""
        try:
            current_time = datetime.utcnow()
            # CPU features
            cpu_data = system_monitor.get_cpu_data()
            if cpu_data:
                cpu_times = cpu_data.get("cpu_times", {}) or {}
                freq = cpu_data.get("frequency", {}) or {}
                load_avg = cpu_data.get("load_average", [])
                cpu_features = {
                    "cpu_percent": cpu_data.get("usage_percent", 0),
                    "cpu_user_time": cpu_times.get("user", 0),
                    "cpu_system_time": cpu_times.get("system", 0),
                    "cpu_idle_time": cpu_times.get("idle", 0),
                    "cpu_frequency": freq.get("current", 0),
                    "load_avg_1m": load_avg[0] if len(load_avg) > 0 else 0,
                    "timestamp": current_time,
                }
                self.feature_buffer["cpu"].append(cpu_features)

            # Memory features
            memory_data = system_monitor.get_memory_data()
            if memory_data:
                swap = memory_data.get("swap", {}) or {}
                memory_features = {
                    "memory_percent": memory_data.get("usage_percent", 0),
                    "memory_available": memory_data.get("available", 0),
                    "memory_cached": memory_data.get("cached", 0),
                    "memory_buffers": memory_data.get("buffers", 0),
                    "swap_percent": swap.get("usage_percent", 0),
                    "timestamp": current_time,
                }
                self.feature_buffer["memory"].append(memory_features)

            # Disk features
            disk_data = system_monitor.get_disk_data()
            if disk_data and disk_data.get("partitions"):
                partitions = disk_data.get("partitions", [])
                total_usage = sum(p.get("usage_percent", 0) for p in partitions)
                avg_usage = total_usage / len(partitions) if partitions else 0
                io_stats = disk_data.get("io_stats", {}) or {}
                disk_features = {
                    "disk_avg_usage_percent": avg_usage,
                    "disk_total_read_bytes": io_stats.get("read_bytes", 0),
                    "disk_total_write_bytes": io_stats.get("write_bytes", 0),
                    "disk_read_time": io_stats.get("read_time", 0),
                    "disk_write_time": io_stats.get("write_time", 0),
                    "timestamp": current_time,
                }
                self.feature_buffer["disk"].append(disk_features)

            # Network features
            network_data = system_monitor.get_network_data()
            if network_data:
                network_features = {
                    "network_bytes_sent": network_data.get("bytes_sent", 0),
                    "network_bytes_recv": network_data.get("bytes_recv", 0),
                    "network_packets_sent": network_data.get("packets_sent", 0),
                    "network_packets_recv": network_data.get("packets_recv", 0),
                    "network_errors": (network_data.get("errors_in", 0) + network_data.get("errors_out", 0)),
                    "network_drops": (network_data.get("drops_in", 0) + network_data.get("drops_out", 0)),
                    "timestamp": current_time,
                }
                self.feature_buffer["network"].append(network_features)

            # Process features
            processes_data = system_monitor.get_top_processes(limit=10)
            if processes_data:
                total_cpu = sum(p.get("cpu_percent", 0) for p in processes_data)
                total_memory = sum(p.get("memory_percent", 0) for p in processes_data)
                process_features = {
                    "top_processes_cpu": total_cpu,
                    "top_processes_memory": total_memory,
                    "process_count": len(processes_data),
                    "timestamp": current_time,
                }
                self.feature_buffer["processes"].append(process_features)

        except Exception as e:
            logger.error(f"Error collecting ML features: {e}")

    def _generate_predictions(self):
        """Generate predictions using ML models"""
        try:
            for metric_type in ["cpu", "memory", "disk", "network"]:
                # Check if we have any models for this metric
                available_models = [k for k in self.prediction_models.keys() if k.startswith(metric_type + "_")]
                if not available_models:
                    continue
                # Get recent feature data
                features = list(self.feature_buffer[metric_type])
                if len(features) < 10:  # Need minimum data points
                    continue
                # Generate predictions for different horizons
                for horizon in self.config["prediction_horizon_minutes"]:
                    model_key = f"{metric_type}_{horizon}min"
                    if model_key in self.prediction_models:
                        prediction = self._predict_with_model(model_key, features, horizon)
                        if prediction:
                            self.prediction_history[model_key].append(prediction)
                            # Store prediction in database
                            try:
                                self.db_manager.store_advanced_metric(
                                    metric_type="ml_prediction",
                                    metric_name=f"{metric_type}_prediction_{horizon}m",
                                    value=prediction.predicted_value,
                                    metadata=prediction.to_dict(),
                                )
                            except Exception:
                                # DB failures should not stop processing
                                logger.exception("Failed to store ML prediction in DB")

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")

    def _predict_with_model(self, model_key: str, features: List[Dict], horizon: int) -> Optional[MLPrediction]:
        """Generate prediction with a specific model"""
        try:
            if model_key not in self.prediction_models:
                return None
            model = self.prediction_models[model_key]
            scaler = self.scalers.get(model_key)
            if not scaler:
                return None
            # Prepare features for prediction
            feature_matrix = self._prepare_feature_matrix(features, model_key)
            if feature_matrix.size == 0:
                return None
            # Scale features
            scaled_features = scaler.transform(feature_matrix)
            # Generate prediction using last row
            prediction = model.predict(scaled_features[-1:].reshape(1, -1))
            predicted_value = float(prediction[0])
            # Calculate confidence (simplified)
            perf = self.model_performance.get(model_key, {})
            confidence_score = min(1.0, float(perf.get("accuracy", 0.0)))
            # Determine features used
            features_used = self._get_feature_names(model_key)
            # Get model accuracy
            model_accuracy = float(perf.get("accuracy", 0.0))
            return MLPrediction(
                metric_name=model_key,
                predicted_value=predicted_value,
                confidence_score=confidence_score,
                prediction_horizon_minutes=horizon,
                model_accuracy=model_accuracy,
                timestamp=datetime.utcnow().isoformat(),
                features_used=features_used,
            )
        except Exception as e:
            logger.error(f"Error predicting with model {model_key}: {e}")
            return None

    def _detect_anomalies(self):
        """Detect anomalies using ML models"""
        try:
            for metric_type in ["cpu", "memory", "disk", "network"]:
                if metric_type not in self.anomaly_detectors:
                    continue
                features = list(self.feature_buffer[metric_type])
                if len(features) < 5:
                    continue
                current_features = features[-1]
                detector = self.anomaly_detectors[metric_type]
                feature_matrix = self._prepare_feature_matrix([current_features], f"{metric_type}_anomaly")
                if feature_matrix.size == 0:
                    continue
                scaler = self.scalers.get(f"{metric_type}_anomaly")
                if not scaler:
                    continue
                scaled_features = scaler.transform(feature_matrix)
                anomaly_score = float(detector.decision_function(scaled_features)[0])
                is_anomaly = int(detector.predict(scaled_features)[0]) == -1
                severity = self._calculate_anomaly_severity(anomaly_score)
                expected_range = self._get_expected_range(metric_type, features)
                contributing_factors = self._identify_contributing_factors(metric_type, current_features)
                current_value = self._extract_primary_value(metric_type, current_features)
                # Store anomaly detection result
                anomaly_result = AnomalyDetectionResult(
                    metric_name=metric_type,
                    current_value=current_value,
                    anomaly_score=anomaly_score,
                    is_anomaly=is_anomaly,
                    severity=severity,
                    expected_range=expected_range,
                    contributing_factors=contributing_factors,
                    timestamp=datetime.utcnow().isoformat(),
                )
                try:
                    self.db_manager.store_advanced_metric(
                        metric_type="anomaly_detection",
                        metric_name=f"{metric_type}_anomaly",
                        value=anomaly_score,
                        metadata=anomaly_result.to_dict(),
                    )
                except Exception:
                    logger.exception("Failed to store anomaly detection result in DB")

                if is_anomaly:
                    logger.warning(f"Anomaly detected in {metric_type}: score={anomaly_score}")

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")

    @staticmethod
    def _calculate_anomaly_severity(anomaly_score: float) -> str:
        """Calculate anomaly severity based on score"""
        abs_score = abs(anomaly_score)
        if abs_score > 0.8:
            return "critical"
        elif abs_score > 0.5:
            return "high"
        elif abs_score > 0.2:
            return "medium"
        else:
            return "low"

    def _get_expected_range(self, metric_type: str, features: List[Dict]) -> Tuple[float, float]:
        """Get expected range for metric"""
        try:
            recent_values = []
            for feature in features[-20:]:
                value = self._extract_primary_value(metric_type, feature)
                if value is not None:
                    recent_values.append(float(value))
            if not recent_values:
                return (0.0, 100.0)
            p25 = float(np.percentile(recent_values, 25))
            p75 = float(np.percentile(recent_values, 75))
            return (p25, p75)
        except Exception as e:
            logger.error(f"Error getting expected range: {e}")
            return (0.0, 100.0)

    def _identify_contributing_factors(self, metric_type: str, current_features: Dict) -> List[str]:
        """Identify contributing factors to anomalies"""
        factors: List[str] = []
        try:
            if metric_type == "cpu":
                cpu_percent = current_features.get("cpu_percent", 0)
                load_avg = current_features.get("load_avg_1m", 0)
                if cpu_percent > 90:
                    factors.append("high_cpu_usage")
                if load_avg > 5:
                    factors.append("high_system_load")
            elif metric_type == "memory":
                memory_percent = current_features.get("memory_percent", 0)
                swap_percent = current_features.get("swap_percent", 0)
                if memory_percent > 90:
                    factors.append("high_memory_usage")
                if swap_percent > 50:
                    factors.append("high_swap_usage")
            elif metric_type == "disk":
                disk_usage = current_features.get("disk_avg_usage_percent", 0)
                if disk_usage > 90:
                    factors.append("high_disk_usage")
            elif metric_type == "network":
                errors = current_features.get("network_errors", 0)
                drops = current_features.get("network_drops", 0)
                if errors > 100:
                    factors.append("network_errors")
                if drops > 100:
                    factors.append("network_packet_drops")
        except Exception as e:
            logger.error(f"Error identifying contributing factors: {e}")
        return factors

    def _extract_primary_value(self, metric_type: str, features: Dict) -> float:
        """Extract primary value for metric type"""
        try:
            if metric_type == "cpu":
                return float(features.get("cpu_percent", 0))
            elif metric_type == "memory":
                return float(features.get("memory_percent", 0))
            elif metric_type == "disk":
                return float(features.get("disk_avg_usage_percent", 0))
            elif metric_type == "network":
                return float(features.get("network_bytes_sent", 0) + features.get("network_bytes_recv", 0))
            else:
                return 0.0
        except Exception:
            return 0.0

    def _update_model_performance(self):
        """Update model performance metrics"""
        try:
            # This would compare predictions with actual values (placeholder)
            for model_key in list(self.prediction_models.keys()):
                if model_key not in self.model_performance:
                    self.model_performance[model_key] = {
                        "accuracy": 0.75,  # Initial accuracy
                        "mae": 0.0,
                        "mse": 0.0,
                        "last_updated": datetime.utcnow(),
                    }
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")

    async def _schedule_model_training(self):
        """Schedule periodic model training"""
        while self.processing_active:
            try:
                await asyncio.sleep(3600)  # Check every hour
                for metric_type in ["cpu", "memory", "disk", "network"]:
                    last_training = self.last_training_time.get(metric_type, datetime.utcfromtimestamp(0))
                    hours_since_training = (datetime.utcnow() - last_training).total_seconds() / 3600.0
                    if hours_since_training >= self.config["retrain_interval_hours"]:
                        await self._train_models_for_metric(metric_type)
            except Exception as e:
                logger.error(f"Error in model training scheduler: {e}")
                await asyncio.sleep(1800)  # Retry in 30 minutes

    async def _train_models_for_metric(self, metric_type: str):
        """Train ML models for a specific metric type"""
        if metric_type in self.training_in_progress:
            return
        try:
            self.training_in_progress.add(metric_type)
            logger.info(f"Starting model training for {metric_type}")
            training_data = await self._get_training_data(metric_type)
            if len(training_data) < self.config["min_training_samples"]:
                logger.warning(f"Insufficient training data for {metric_type}: {len(training_data)} samples")
                return
            await self._train_prediction_models(metric_type, training_data)
            await self._train_anomaly_model(metric_type, training_data)
            self.last_training_time[metric_type] = datetime.utcnow()
            logger.info(f"Model training completed for {metric_type}")
        except Exception as e:
            logger.error(f"Error training models for {metric_type}: {e}")
        finally:
            self.training_in_progress.discard(metric_type)

    async def _get_training_data(self, metric_type: str) -> pd.DataFrame:
        """Get training data from database"""
        try:
            # Get historical data for the last 30 days
            historical_data = self.db_manager.get_historical_metrics(metric_type, days=30)
            if not historical_data:
                return pd.DataFrame()
            df = pd.DataFrame(historical_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.to_datetime(df.index)
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            df = df.sort_values("timestamp")
            return df
        except Exception as e:
            logger.error(f"Error getting training data for {metric_type}: {e}")
            return pd.DataFrame()

    async def _train_prediction_models(self, metric_type: str, training_data: pd.DataFrame):
        """Train prediction models for different time horizons"""
        try:
            for horizon in self.config["prediction_horizon_minutes"]:
                model_key = f"{metric_type}_{horizon}min"
                X, y = self._prepare_training_data(training_data.copy(), metric_type, horizon)
                if X.size == 0 or len(X) < 50:
                    continue
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                accuracy = 0.0
                try:
                    accuracy = 1 - (mae / (float(y_test.max()) - float(y_test.min()))) if (y_test.max() - y_test.min()) != 0 else 0.0
                except Exception:
                    accuracy = 0.0
                self.prediction_models[model_key] = model
                self.scalers[model_key] = scaler
                self.model_performance[model_key] = {
                    "accuracy": float(max(0, min(1, accuracy))),
                    "mae": float(mae),
                    "mse": float(mse),
                    "last_updated": datetime.utcnow(),
                }
                logger.info(f"Trained prediction model {model_key}: accuracy={self.model_performance[model_key]['accuracy']}")
        except Exception as e:
            logger.error(f"Error training prediction models for {metric_type}: {e}")

    async def _train_anomaly_model(self, metric_type: str, training_data: pd.DataFrame):
        """Train anomaly detection model"""
        try:
            model_key = f"{metric_type}_anomaly"
            features = self._prepare_anomaly_features(training_data.copy(), metric_type)
            if features.size == 0 or len(features) < 50:
                return
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            iso_forest = IsolationForest(contamination=self.config.get("anomaly_contamination", 0.01), random_state=42, n_jobs=-1)
            iso_forest.fit(features_scaled)
            self.anomaly_detectors[metric_type] = iso_forest
            self.scalers[model_key] = scaler
            logger.info(f"Trained anomaly detection model for {metric_type}")
        except Exception as e:
            logger.error(f"Error training anomaly model for {metric_type}: {e}")

    def _prepare_training_data(self, df: pd.DataFrame, metric_type: str, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for prediction models"""
        try:
            feature_columns: List[str] = []
            primary_col = self._get_primary_column(metric_type)
            if primary_col in df.columns:
                feature_columns.append(primary_col)
            if "hour" in df.columns:
                feature_columns.append("hour")
            if "day_of_week" in df.columns:
                feature_columns.append("day_of_week")
            if "is_weekend" in df.columns:
                feature_columns.append("is_weekend")
            for lag in [1, 2, 3, 5, 10]:
                if primary_col in df.columns:
                    lag_col = f"{primary_col}_lag_{lag}"
                    df[lag_col] = df[primary_col].shift(lag)
                    feature_columns.append(lag_col)
            if primary_col in df.columns:
                target_col = f"{primary_col}_future_{horizon}m"
                df[target_col] = df[primary_col].shift(-horizon)
            else:
                return np.array([]), np.array([])
            df_clean = df[feature_columns + [target_col]].dropna()
            X = df_clean[feature_columns].values
            y = df_clean[target_col].values
            return X, y
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])

    def _prepare_anomaly_features(self, df: pd.DataFrame, metric_type: str) -> np.ndarray:
        """Prepare features for anomaly detection"""
        try:
            feature_columns: List[str] = []
            primary_col = self._get_primary_column(metric_type)
            if primary_col in df.columns:
                feature_columns.append(primary_col)
                window_size = 10
                df[f"{primary_col}_rolling_mean"] = df[primary_col].rolling(window=window_size).mean()
                df[f"{primary_col}_rolling_std"] = df[primary_col].rolling(window=window_size).std()
                feature_columns.extend([f"{primary_col}_rolling_mean", f"{primary_col}_rolling_std"])
            if "hour" in df.columns:
                feature_columns.append("hour")
            if "day_of_week" in df.columns:
                feature_columns.append("day_of_week")
            df_clean = df[feature_columns].dropna()
            return df_clean.values
        except Exception as e:
            logger.error(f"Error preparing anomaly features: {e}")
            return np.array([])

    @staticmethod
    def _get_primary_column(metric_type: str) -> str:
        """Get primary column name for metric type"""
        mapping = {
            "cpu": "usage_percent",
            "memory": "usage_percent",
            "disk": "usage_percent",
            "network": "bytes_sent",
        }
        return mapping.get(metric_type, "value")

    def _prepare_feature_matrix(self, features: List[Dict], model_key: str) -> np.ndarray:
        """Prepare feature matrix from feature dictionaries"""
        try:
            if not features:
                return np.array([])
            feature_names = self._get_feature_names(model_key)
            feature_matrix: List[List[float]] = []
            for feature_dict in features:
                feature_row: List[float] = []
                for feature_name in feature_names:
                    value = feature_dict.get(feature_name, 0)
                    if isinstance(value, (int, float, np.floating, np.integer)):
                        feature_row.append(float(value))
                    else:
                        feature_row.append(0.0)
                if len(feature_row) == len(feature_names):
                    feature_matrix.append(feature_row)
            return np.array(feature_matrix)
        except Exception as e:
            logger.error(f"Error preparing feature matrix: {e}")
            return np.array([])

    def _get_feature_names(self, model_key: str) -> List[str]:
        """Get feature names for a model"""
        if "cpu" in model_key:
            return ["cpu_percent", "cpu_user_time", "cpu_system_time", "load_avg_1m"]
        elif "memory" in model_key:
            return ["memory_percent", "memory_available", "swap_percent"]
        elif "disk" in model_key:
            return ["disk_avg_usage_percent", "disk_total_read_bytes", "disk_total_write_bytes"]
        elif "network" in model_key:
            return ["network_bytes_sent", "network_bytes_recv", "network_errors"]
        else:
            return ["value"]

    def _load_models(self):
        """Load existing models from disk"""
        try:
            model_file = "data/models/ml_monitoring_models.pkl"
            if os.path.exists(model_file):
                with open(model_file, "rb") as f:
                    model_data = joblib.load(f)
                self.prediction_models = model_data.get("prediction_models", {})
                self.anomaly_detectors = model_data.get("anomaly_detectors", {})
                self.scalers = model_data.get("scalers", {})
                self.model_performance = model_data.get("model_performance", {})
                logger.info(f"Loaded {len(self.prediction_models)} prediction models and {len(self.anomaly_detectors)} anomaly detectors")
            else:
                logger.info("No existing models found")
        except Exception as e:
            logger.exception(f"No existing models found or error loading: {e}")

    def _save_models(self):
        """Save models to disk"""
        try:
            os.makedirs("data/models", exist_ok=True)
            model_file = "data/models/ml_monitoring_models.pkl"
            model_data = {
                "prediction_models": self.prediction_models,
                "anomaly_detectors": self.anomaly_detectors,
                "scalers": self.scalers,
                "model_performance": self.model_performance,
                "saved_at": datetime.utcnow().isoformat(),
            }
            with open(model_file, "wb") as f:
                joblib.dump(model_data, f)
            logger.info(f"Saved {len(self.prediction_models)} prediction models and {len(self.anomaly_detectors)} anomaly detectors")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def get_prediction(self, metric_type: str, horizon_minutes: int) -> Optional[MLPrediction]:
        """Get latest prediction for metric and horizon"""
        try:
            model_key = f"{metric_type}_{horizon_minutes}min"
            predictions = self.prediction_history.get(model_key, [])
            if predictions:
                return predictions[-1]
            return None
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            return None

    def get_anomaly_status(self, metric_type: str) -> Dict[str, Any]:
        """Get current anomaly status for metric type"""
        try:
            return {
                "metric_type": metric_type,
                "has_detector": metric_type in self.anomaly_detectors,
                "last_check": datetime.utcnow().isoformat(),
                "status": "monitoring",
            }
        except Exception as e:
            logger.error(f"Error getting anomaly status: {e}")
            return {"error": str(e)}

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all model performance"""
        try:
            summary: Dict[str, Any] = {
                "prediction_models": len(self.prediction_models),
                "anomaly_detectors": len(self.anomaly_detectors),
                "model_performance": {},
                "last_training_times": {k: v.isoformat() for k, v in self.last_training_time.items()},
                "training_in_progress": list(self.training_in_progress),
                "timestamp": datetime.utcnow().isoformat(),
            }
            for model_key, perf in self.model_performance.items():
                summary["model_performance"][model_key] = {
                    "accuracy": perf.get("accuracy", 0.0),
                    "mae": perf.get("mae", 0.0),
                    "last_updated": perf.get("last_updated", datetime.utcnow()).isoformat(),
                }
            return summary
        except Exception as e:
            logger.error(f"Error getting model performance summary: {e}")
            return {"error": str(e)}


# Global ML monitoring engine instance
ml_engine: Optional[MLMonitoringEngine] = None


def get_ml_monitoring_engine() -> MLMonitoringEngine:
    """Get global ML monitoring engine instance"""
    global ml_engine
    if ml_engine is None:
        ml_engine = MLMonitoringEngine()
    return ml_engine
