"""
Advanced Data Analytics Engine
Author: Member 3 
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class AnalyticsConfig:
    """Analytics engine configuration"""
    lookback_days: int = 30
    min_samples_for_analysis: int = 100
    confidence_level: float = 0.95
    anomaly_threshold: float = 2.0
    forecast_periods: int = 24  # hours
    seasonal_period: int = 24  # hours for daily seasonality


class StatisticalAnalyzer:
    """Statistical analysis engine"""

    def __init__(self, config: AnalyticsConfig):
        self.config = config

    def calculate_descriptive_stats(self, data: List[float]) -> Dict:
        """Calculate comprehensive descriptive statistics"""
        if not data or len(data) < 2:
            return {'error': 'Insufficient data'}

        data_array = np.array(data, dtype=float)
        mean = float(np.mean(data_array))
        std = float(np.std(data_array, ddof=0))

        # Mode may return multiple values; pick the first
        try:
            mode_res = stats.mode(data_array, keepdims=True)
            mode_val = float(mode_res.mode[0]) if mode_res.count.size and mode_res.count[0] > 0 else None
        except Exception:
            mode_val = None

        percentiles = {
            'p5': float(np.percentile(data_array, 5)),
            'p10': float(np.percentile(data_array, 10)),
            'p25': float(np.percentile(data_array, 25)),
            'p50': float(np.percentile(data_array, 50)),
            'p75': float(np.percentile(data_array, 75)),
            'p90': float(np.percentile(data_array, 90)),
            'p95': float(np.percentile(data_array, 95)),
            'p99': float(np.percentile(data_array, 99)),
        }

        cv = float(std / mean) if mean != 0 else None

        return {
            'count': len(data_array),
            'mean': mean,
            'median': float(np.median(data_array)),
            'mode': mode_val,
            'std': std,
            'variance': float(np.var(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array)),
            'range': float(np.max(data_array) - np.min(data_array)),
            'q1': percentiles['p25'],
            'q3': percentiles['p75'],
            'iqr': percentiles['p75'] - percentiles['p25'],
            'skewness': float(stats.skew(data_array)),
            'kurtosis': float(stats.kurtosis(data_array)),
            'cv': cv,
            'percentiles': percentiles
        }

    def detect_outliers(self, data: List[float], method: str = 'iqr') -> Dict:
        """Detect outliers using various methods"""
        if not data or len(data) < 4:
            return {'outliers': [], 'method': method}

        data_array = np.array(data, dtype=float)
        outliers = []

        if method == 'iqr':
            q1, q3 = np.percentile(data_array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = [float(x) for x in data_array if x < lower_bound or x > upper_bound]

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data_array, nan_policy='omit'))
            threshold = self.config.anomaly_threshold
            outliers = [float(data_array[i]) for i in range(len(data_array)) if z_scores[i] > threshold]

        elif method == 'modified_zscore':
            median = np.median(data_array)
            mad = np.median(np.abs(data_array - median))
            if mad == 0:
                outliers = []
            else:
                modified_z_scores = 0.6745 * (data_array - median) / mad
                threshold = self.config.anomaly_threshold
                outliers = [float(data_array[i]) for i in range(len(data_array)) if abs(modified_z_scores[i]) > threshold]

        return {
            'outliers': outliers,
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(data_array)) * 100,
            'method': method
        }

    def trend_analysis(self, timestamps: List[str], values: List[float]) -> Dict:
        """Analyze trends in time series data"""
        if len(timestamps) != len(values) or len(values) < 10:
            return {'error': 'Insufficient data for trend analysis'}

        # Convert timestamps to numeric values
        time_numeric = []
        for ts in timestamps:
            try:
                # Accept ISO strings with or without timezone
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                time_numeric.append(dt.timestamp())
            except Exception:
                # try parsing as pandas
                try:
                    dt = pd.to_datetime(ts)
                    time_numeric.append(dt.timestamp())
                except Exception:
                    continue

        if len(time_numeric) != len(values):
            return {'error': 'Timestamp conversion failed or inconsistent lengths'}

        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)

        # Seasonal decomposition (simplified)
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'value': values
        })
        df.set_index('timestamp', inplace=True)

        seasonal_strength = 0.0
        if len(df) > 24:
            hourly = df.resample('H').mean().dropna()
            if len(hourly) > 24:
                hourly_pattern = hourly.groupby(hourly.index.hour)['value'].mean()
                if hourly_pattern.mean() != 0:
                    seasonal_strength = float(hourly_pattern.std() / hourly_pattern.mean())

        direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'flat'
        strength = 'strong' if abs(r_value) > 0.7 else 'moderate' if abs(r_value) > 0.4 else 'weak'

        return {
            'trend': {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'standard_error': float(std_err),
                'direction': direction,
                'strength': strength
            },
            'seasonality': {
                'strength': float(seasonal_strength),
                'present': seasonal_strength > 0.1
            },
            'stationarity': self._test_stationarity(values)
        }

    def _test_stationarity(self, data: List[float]) -> Dict:
        """Test for stationarity using augmented Dickey-Fuller test"""
        try:
            from statsmodels.tsa.stattools import adfuller

            result = adfuller(data)

            return {
                'adf_statistic': float(result[0]),
                'p_value': float(result[1]),
                'is_stationary': result[1] < 0.05,
                'critical_values': {
                    '1%': float(result[4]['1%']),
                    '5%': float(result[4]['5%']),
                    '10%': float(result[4]['10%'])
                }
            }
        except Exception:
            # Fallback simple test
            series = pd.Series(data)
            rolling_mean = series.rolling(window=min(12, max(3, len(series)//4))).mean()
            rolling_std = series.rolling(window=min(12, max(3, len(series)//4))).std()

            mean_stability = float(rolling_mean.std() / rolling_mean.mean()) if rolling_mean.mean() != 0 else float('inf')
            std_stability = float(rolling_std.std() / rolling_std.mean()) if rolling_std.mean() != 0 else float('inf')

            return {
                'is_stationary': (mean_stability < 0.1 and std_stability < 0.1),
                'mean_stability': mean_stability,
                'std_stability': std_stability,
                'method': 'simplified'
            }

    def correlation_analysis(self, data_dict: Dict[str, List[float]]) -> Dict:
        """Analyze correlations between multiple metrics"""
        if len(data_dict) < 2:
            return {'error': 'Need at least 2 metrics for correlation analysis'}

        # Create DataFrame
        df = pd.DataFrame(data_dict)

        # Calculate correlation matrix
        corr_matrix = df.corr()

        # Find significant correlations
        significant_correlations = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Significant correlation threshold
                    significant_correlations.append({
                        'metric1': cols[i],
                        'metric2': cols[j],
                        'correlation': float(corr_value),
                        'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                    })

        summary = {
            'total_pairs': len(significant_correlations),
            'strong_correlations': len([c for c in significant_correlations if abs(c['correlation']) > 0.7]),
            'moderate_correlations': len([c for c in significant_correlations if 0.5 < abs(c['correlation']) <= 0.7])
        }

        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'significant_correlations': significant_correlations,
            'summary': summary
        }


class ForecastingEngine:
    """Time series forecasting engine"""

    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.models = {}

    def linear_forecast(self, timestamps: List[str], values: List[float], periods: int = None) -> Dict:
        """Simple linear regression forecast"""
        if periods is None:
            periods = self.config.forecast_periods

        if len(timestamps) != len(values) or len(values) < 10:
            return {'error': 'Insufficient data for forecasting'}

        # Convert timestamps to numeric
        time_numeric = []
        for ts in timestamps:
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                time_numeric.append(dt.timestamp())
            except Exception:
                try:
                    dt = pd.to_datetime(ts)
                    time_numeric.append(dt.timestamp())
                except Exception:
                    continue

        if len(time_numeric) != len(values):
            return {'error': 'Timestamp conversion failed or inconsistent lengths'}

        X = np.array(time_numeric).reshape(-1, 1)
        y = np.array(values, dtype=float)

        model = LinearRegression()
        model.fit(X, y)

        # Generate future timestamps
        last_timestamp = max(time_numeric)
        time_interval = (max(time_numeric) - min(time_numeric)) / (len(time_numeric) - 1)

        future_timestamps = []
        future_values = []

        for i in range(1, periods + 1):
            future_time = last_timestamp + (i * time_interval)
            future_value = model.predict(np.array([[future_time]]))[0]

            future_timestamps.append(datetime.fromtimestamp(future_time).isoformat())
            future_values.append(float(future_value))

        # Calculate confidence intervals (simplified)
        residuals = y - model.predict(X)
        mse = float(np.mean(residuals ** 2))
        std_error = float(np.sqrt(mse))

        confidence_intervals = []
        for value in future_values:
            ci_lower = value - 1.96 * std_error
            ci_upper = value + 1.96 * std_error
            confidence_intervals.append({'lower': float(ci_lower), 'upper': float(ci_upper)})

        return {
            'forecast': {
                'timestamps': future_timestamps,
                'values': future_values,
                'confidence_intervals': confidence_intervals
            },
            'model_info': {
                'type': 'linear_regression',
                'r_squared': float(model.score(X, y)),
                'mse': mse,
                'mae': float(np.mean(np.abs(residuals)))
            },
            'forecast_horizon_hours': periods
        }

    def exponential_smoothing_forecast(self, values: List[float], periods: int = None) -> Dict:
        """Exponential smoothing forecast"""
        if periods is None:
            periods = self.config.forecast_periods

        if len(values) < 10:
            return {'error': 'Insufficient data for forecasting'}

        # Simple exponential smoothing
        alpha = 0.3  # Smoothing parameter

        smoothed = [float(values[0])]
        for i in range(1, len(values)):
            smoothed.append(alpha * float(values[i]) + (1 - alpha) * smoothed[i - 1])

        # Forecast
        last_smoothed = smoothed[-1]
        forecast_values = [last_smoothed] * periods

        # Calculate prediction intervals
        residuals = np.array(values[1:]) - np.array(smoothed[1:])
        residual_std = float(np.std(residuals)) if residuals.size > 0 else 0.0

        confidence_intervals = []
        for i in range(periods):
            # Prediction interval widens with forecast horizon
            interval_width = 1.96 * residual_std * np.sqrt(1 + (i + 1) * 0.1)
            ci_lower = last_smoothed - interval_width
            ci_upper = last_smoothed + interval_width
            confidence_intervals.append({'lower': float(ci_lower), 'upper': float(ci_upper)})

        return {
            'forecast': {
                'values': [float(v) for v in forecast_values],
                'confidence_intervals': confidence_intervals
            },
            'model_info': {
                'type': 'exponential_smoothing',
                'alpha': alpha,
                'last_smoothed_value': float(last_smoothed),
                'residual_std': residual_std
            },
            'forecast_horizon_hours': periods
        }

    def seasonal_naive_forecast(self, values: List[float], seasonal_period: int = None, periods: int = None) -> Dict:
        """Seasonal naive forecasting"""
        if seasonal_period is None:
            seasonal_period = self.config.seasonal_period
        if periods is None:
            periods = self.config.forecast_periods

        if len(values) < seasonal_period * 2:
            return {'error': f'Need at least {seasonal_period * 2} data points for seasonal naive forecast'}

        # Use last seasonal period for forecasting
        seasonal_pattern = values[-seasonal_period:]

        # Repeat pattern for forecast horizon
        forecast_values = [seasonal_pattern[i % seasonal_period] for i in range(periods)]

        # Calculate seasonal indices
        seasonal_indices = []
        for i in range(seasonal_period):
            season_values = [values[j] for j in range(i, len(values), seasonal_period)]
            seasonal_indices.append(float(np.mean(season_values)))

        return {
            'forecast': {
                'values': [float(v) for v in forecast_values],
                'seasonal_pattern': [float(v) for v in seasonal_pattern]
            },
            'model_info': {
                'type': 'seasonal_naive',
                'seasonal_period': seasonal_period,
                'seasonal_indices': seasonal_indices
            },
            'forecast_horizon_hours': periods
        }


class ClusterAnalyzer:
    """Cluster analysis for pattern detection"""

    def __init__(self, config: AnalyticsConfig):
        self.config = config

    def performance_clustering(self, metrics_data: Dict[str, List[float]]) -> Dict:
        """Cluster system performance states"""
        if len(metrics_data) < 2:
            return {'error': 'Need at least 2 metrics for clustering'}

        # Create DataFrame
        df = pd.DataFrame(metrics_data)

        if len(df) < 10:
            return {'error': 'Insufficient data for clustering'}

        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.fillna(0))

        # Determine optimal number of clusters using elbow method
        max_clusters = min(10, max(2, len(df) // 3))
        if max_clusters < 2:
            return {'error': 'Too few data points for clustering'}

        inertias = []
        K_range = list(range(2, max_clusters + 1))

        for k in K_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)
            except Exception:
                inertias.append(np.inf)

        # Use elbow method to find optimal k
        if len(inertias) >= 2 and all(np.isfinite(inertias)):
            deltas = [inertias[i - 1] - inertias[i] for i in range(1, len(inertias))]
            optimal_k = K_range[int(np.argmax(deltas))]
        else:
            optimal_k = 2

        # Perform clustering with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)

        # Analyze clusters
        cluster_summary = {}
        for i in range(optimal_k):
            cluster_mask = cluster_labels == i
            cluster_data = df.iloc[cluster_mask]

            cluster_summary[f'cluster_{i}'] = {
                'size': int(np.sum(cluster_mask)),
                'percentage': float(np.sum(cluster_mask) / len(df) * 100),
                'centroid': {col: float(np.mean(df[col][cluster_mask])) for col in df.columns},
                'characteristics': self._describe_cluster(cluster_data, df)
            }

        silhouette = self._calculate_silhouette_score(scaled_data, cluster_labels)

        return {
            'optimal_clusters': int(optimal_k),
            'cluster_summary': cluster_summary,
            'silhouette_score': silhouette,
            'total_samples': len(df)
        }

    def _describe_cluster(self, cluster_data: pd.DataFrame, all_data: pd.DataFrame) -> Dict:
        """Describe cluster characteristics"""
        characteristics = {}

        for col in cluster_data.columns:
            cluster_mean = float(cluster_data[col].mean()) if not cluster_data[col].empty else 0.0
            overall_mean = float(all_data[col].mean()) if not all_data[col].empty else 0.0

            if overall_mean == 0:
                characteristics[col] = 'normal'
            elif cluster_mean > overall_mean * 1.2:
                characteristics[col] = 'high'
            elif cluster_mean < overall_mean * 0.8:
                characteristics[col] = 'low'
            else:
                characteristics[col] = 'normal'

        return characteristics

    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score"""
        try:
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(data, labels))
        except Exception:
            return 0.0


class AdvancedAnalyticsEngine:
    """Main analytics engine combining all analysis capabilities"""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.config = AnalyticsConfig()

        # Initialize analyzers
        self.statistical_analyzer = StatisticalAnalyzer(self.config)
        self.forecasting_engine = ForecastingEngine(self.config)
        self.cluster_analyzer = ClusterAnalyzer(self.config)

        logger.info("Advanced Analytics Engine initialized")

    def analyze_metric_comprehensive(self, metric_name: str, hours: int = 24) -> Dict:
        """Comprehensive analysis of a single metric"""
        try:
            # Get data
            query = '''
                SELECT timestamp, value
                FROM v_all_metrics
                WHERE metric_name = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp
            '''

            data = self.db_manager.get_cached_query(
                f"metric_analysis_{metric_name}_{hours}h",
                query,
                (metric_name, hours)
            )

            if not data or len(data) < 10:
                return {'error': 'Insufficient data for comprehensive analysis'}

            timestamps = [row['timestamp'] for row in data]
            values = [row['value'] for row in data]

            # Perform all types of analysis
            descriptive = self.statistical_analyzer.calculate_descriptive_stats(values)
            outlier_analysis = self.statistical_analyzer.detect_outliers(values, 'iqr')
            trend = self.statistical_analyzer.trend_analysis(timestamps, values)
            forecast_linear = self.forecasting_engine.linear_forecast(timestamps, values)
            forecast_exponential = self.forecasting_engine.exponential_smoothing_forecast(values)

            analysis_result = {
                'metric_name': metric_name,
                'analysis_period_hours': hours,
                'sample_count': len(values),
                'descriptive_stats': descriptive,
                'outlier_analysis': outlier_analysis,
                'trend_analysis': trend,
                'forecast_linear': forecast_linear,
                'forecast_exponential': forecast_exponential,
                'analyzed_at': datetime.utcnow().isoformat()
            }

            # Add seasonal forecast if enough data
            if len(values) >= 48:  # 2 days of hourly data
                analysis_result['forecast_seasonal'] = self.forecasting_engine.seasonal_naive_forecast(
                    values, seasonal_period=24, periods=12
                )

            return analysis_result

        except Exception as e:
            logger.error(f"Comprehensive metric analysis error: {e}")
            return {'error': str(e)}

    def cross_metric_analysis(self, metric_names: List[str], hours: int = 24) -> Dict:
        """Analyze relationships between multiple metrics"""
        try:
            if len(metric_names) < 2:
                return {'error': 'Need at least 2 metrics for cross-metric analysis'}

            # Get data for all metrics
            data_dict: Dict[str, List[float]] = {}

            for metric_name in metric_names:
                query = '''
                    SELECT value
                    FROM v_all_metrics
                    WHERE metric_name = ?
                    AND timestamp >= datetime('now', '-' || ? || ' hours')
                    ORDER BY timestamp
                '''

                data = self.db_manager.get_cached_query(
                    f"cross_metric_{metric_name}_{hours}h",
                    query,
                    (metric_name, hours)
                )

                if data:
                    data_dict[metric_name] = [row['value'] for row in data]

            if len(data_dict) < 2:
                return {'error': 'Insufficient data for cross-metric analysis'}

            # Ensure all metrics have the same number of data points
            min_length = min(len(values) for values in data_dict.values())
            for metric in data_dict:
                data_dict[metric] = data_dict[metric][:min_length]

            # Perform correlation analysis
            correlation_analysis = self.statistical_analyzer.correlation_analysis(data_dict)

            # Perform cluster analysis
            cluster_analysis = self.cluster_analyzer.performance_clustering(data_dict)

            return {
                'metrics_analyzed': metric_names,
                'analysis_period_hours': hours,
                'correlation_analysis': correlation_analysis,
                'cluster_analysis': cluster_analysis,
                'analyzed_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Cross-metric analysis error: {e}")
            return {'error': str(e)}

    def system_health_analysis(self) -> Dict:
        """Comprehensive system health analysis"""
        try:
            # Get recent system metrics
            query = '''
                SELECT
                    metric_type,
                    metric_name,
                    AVG(value) as avg_value,
                    MIN(value) as min_value,
                    MAX(value) as max_value,
                    COUNT(*) as sample_count
                FROM v_all_metrics
                WHERE timestamp >= datetime('now', '-1 hour')
                GROUP BY metric_type, metric_name
            '''

            data = self.db_manager.get_cached_query('system_health_analysis', query)

            if not data:
                return {'error': 'No recent system data available'}

            # Analyze each metric type
            health_scores = {}
            alerts = []

            for row in data:
                metric_type = row.get('metric_type')
                metric_name = row.get('metric_name')
                avg_value = float(row.get('avg_value', 0))
                max_value = float(row.get('max_value', 0))

                # Calculate health score based on metric type
                if metric_type in ['cpu', 'memory', 'disk']:
                    if avg_value > 90:
                        health_score = 0  # Critical
                        alerts.append(f"Critical {metric_type} usage: {avg_value:.1f}%")
                    elif avg_value > 80:
                        health_score = 30  # Poor
                        alerts.append(f"High {metric_type} usage: {avg_value:.1f}%")
                    elif avg_value > 70:
                        health_score = 60  # Fair
                    elif avg_value > 50:
                        health_score = 80  # Good
                    else:
                        health_score = 100  # Excellent
                else:
                    # Default scoring for other metrics
                    health_score = 80

                health_scores[f"{metric_type}_{metric_name}"] = {
                    'score': health_score,
                    'current_value': avg_value,
                    'status': self._get_health_status(health_score)
                }

            # Calculate overall health score
            if health_scores:
                overall_score = sum(score['score'] for score in health_scores.values()) / len(health_scores)
            else:
                overall_score = 0

            return {
                'overall_health_score': round(overall_score, 1),
                'overall_status': self._get_health_status(overall_score),
                'individual_scores': health_scores,
                'alerts': alerts,
                'recommendations': self._generate_health_recommendations(health_scores, alerts),
                'analyzed_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"System health analysis error: {e}")
            return {'error': str(e)}

    def _get_health_status(self, score: float) -> str:
        """Convert health score to status string"""
        if score >= 90:
            return 'excellent'
        elif score >= 70:
            return 'good'
        elif score >= 50:
            return 'fair'
        elif score >= 30:
            return 'poor'
        else:
            return 'critical'

    def _generate_health_recommendations(self, health_scores: Dict, alerts: List[str]) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []

        for metric, score_info in health_scores.items():
            if score_info['score'] < 50:
                if 'cpu' in metric:
                    recommendations.append("Consider optimizing CPU-intensive processes")
                elif 'memory' in metric:
                    recommendations.append("Review memory usage and consider increasing available memory or optimizing usage")
                elif 'disk' in metric:
                    recommendations.append("Clean up disk space or add storage capacity")

        if len(alerts) > 3:
            recommendations.append("Multiple performance issues detected - perform a comprehensive systems review")

        return recommendations

    def capacity_forecast(self, resource_type: str, days_ahead: int = 30) -> Dict:
        """Forecast resource capacity requirements"""
        try:
            # Get historical data for capacity planning
            query = '''
                SELECT timestamp, value
                FROM v_all_metrics
                WHERE metric_type = ?
                AND timestamp >= datetime('now', '-30 days')
                ORDER BY timestamp
            '''

            data = self.db_manager.get_cached_query(
                f"capacity_forecast_{resource_type}_{days_ahead}d",
                query,
                (resource_type,)
            )

            if not data or len(data) < 100:
                return {'error': 'Insufficient historical data for capacity forecasting'}

            timestamps = [row['timestamp'] for row in data]
            values = [row['value'] for row in data]

            # Generate forecast
            forecast_periods = days_ahead * 24  # Hourly forecast
            forecast_result = self.forecasting_engine.linear_forecast(timestamps, values, periods=forecast_periods)

            if 'error' in forecast_result:
                return forecast_result

            # Analyze capacity requirements
            current_utilization = float(np.mean(values[-24:])) if len(values) >= 24 else float(np.mean(values))
            forecast_values = forecast_result['forecast']['values']
            max_forecast_utilization = max(forecast_values) if forecast_values else current_utilization

            # Estimate time to capacity exhaustion
            capacity_threshold = 95.0  # 95% utilization threshold

            days_to_exhaustion = None
            for i, value in enumerate(forecast_values):
                if value >= capacity_threshold:
                    days_to_exhaustion = i / 24  # Convert hours to days
                    break

            # Growth rate calculation
            if len(values) >= 168:  # 1 week of hourly data
                week_ago_avg = float(np.mean(values[-168:-144])) if len(values) >= 168 else float(np.mean(values))
                current_avg = float(np.mean(values[-24:])) if len(values) >= 24 else float(np.mean(values))
                weekly_growth_rate = (current_avg - week_ago_avg) / week_ago_avg * 100 if week_ago_avg != 0 else 0
                monthly_growth_rate = weekly_growth_rate * 4.33  # Approximate weeks per month
            else:
                weekly_growth_rate = 0
                monthly_growth_rate = 0

            return {
                'resource_type': resource_type,
                'current_utilization': round(current_utilization, 2),
                'forecast_horizon_days': days_ahead,
                'max_forecast_utilization': round(max_forecast_utilization, 2),
                'growth_rate': {
                    'weekly_percent': round(weekly_growth_rate, 2),
                    'monthly_percent': round(monthly_growth_rate, 2)
                },
                'capacity_analysis': {
                    'days_to_exhaustion': round(days_to_exhaustion, 1) if days_to_exhaustion is not None else None,
                    'capacity_sufficient': (max_forecast_utilization < capacity_threshold),
                    'risk_level': self._assess_capacity_risk(max_forecast_utilization, days_to_exhaustion)
                },
                'forecast_data': forecast_result,
                'recommendations': self._generate_capacity_recommendations(
                    resource_type, max_forecast_utilization, days_to_exhaustion
                ),
                'analyzed_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Capacity forecast error: {e}")
            return {'error': str(e)}

    def _assess_capacity_risk(self, max_utilization: float, days_to_exhaustion: Optional[float]) -> str:
        """Assess capacity risk level"""
        if days_to_exhaustion is not None and days_to_exhaustion <= 7:
            return 'critical'
        elif days_to_exhaustion is not None and days_to_exhaustion <= 30:
            return 'high'
        elif max_utilization > 85:
            return 'medium'
        else:
            return 'low'

    def _generate_capacity_recommendations(self, resource_type: str, max_utilization: float,
                                           days_to_exhaustion: Optional[float]) -> List[str]:
        """Generate capacity planning recommendations"""
        recommendations: List[str] = []

        if days_to_exhaustion is not None and days_to_exhaustion <= 30:
            if resource_type == 'cpu':
                recommendations.extend([
                    "Consider CPU upgrade or optimization",
                    "Review and optimize CPU-intensive processes",
                    "Consider load balancing across multiple cores/machines"
                ])
            elif resource_type == 'memory':
                recommendations.extend([
                    "Plan memory upgrade",
                    "Identify and optimize memory-intensive applications",
                    "Consider implementing memory caching strategies"
                ])
            elif resource_type == 'disk':
                recommendations.extend([
                    "Plan disk space expansion",
                    "Implement data archival and cleanup policies",
                    "Consider data compression techniques"
                ])

        if max_utilization > 85:
            recommendations.append(f"Monitor {resource_type} usage closely for performance degradation")

        return recommendations


# Global analytics engine instance
analytics_engine: Optional[AdvancedAnalyticsEngine] = None


def get_analytics_engine(db_manager):
    """Get global analytics engine instance"""
    global analytics_engine
    if analytics_engine is None:
        analytics_engine = AdvancedAnalyticsEngine(db_manager)
    return analytics_engine

"""
Advanced Data Analytics Engine
Author: Member 3 
"""
# ... (all your existing imports and code) ...

# Add this at the end of your file:

class DataAnalyzer:
    """Interface for analytics engine used by tests"""

    def __init__(self, db_path):
        # Import here to avoid heavy imports at module load in test contexts
        from database.models import DatabaseManager
        self.db_manager = DatabaseManager(db_path=db_path)
        self.engine = get_analytics_engine(self.db_manager)

    def get_cpu_trends(self, hours=24):
        """Returns CPU stats (average, min, max, samples) for last N hours"""
        analysis = self.engine.analyze_metric_comprehensive('cpu', hours=hours)
        if 'error' in analysis:
            return analysis
        stats = analysis.get('descriptive_stats', {})
        return {
            'average': stats.get('mean'),
            'minimum': stats.get('min'),
            'maximum': stats.get('max'),
            'samples': stats.get('count')
        }

    def get_memory_trends(self, hours=24):
        """Returns memory stats (average, min, max, samples) for last N hours"""
        memory_analysis = self.engine.analyze_metric_comprehensive('memory', hours=hours)
        if 'error' in memory_analysis:
            return memory_analysis
        stats = memory_analysis.get('descriptive_stats', {})
        swap_stats = self.engine.analyze_metric_comprehensive('swap', hours=hours).get('descriptive_stats', {})
        return {
            'memory': {
                'average': stats.get('mean'),
                'minimum': stats.get('min'),
                'maximum': stats.get('max')
            },
            'swap': {
                'average': swap_stats.get('mean'),
                'minimum': swap_stats.get('min'),
                'maximum': swap_stats.get('max')
            },
            'samples': stats.get('count')
        }

    def generate_report(self, hours=24):
        """Generates report for last N hours"""
        report = {
            'report_generated': True,
            'cpu_analysis': self.get_cpu_trends(hours),
            'memory_analysis': self.get_memory_trends(hours),
            'disk_analysis': self.engine.analyze_metric_comprehensive('disk', hours=hours),
            'network_analysis': self.engine.analyze_metric_comprehensive('network', hours=hours)
        }
        return report

# (You may want to move this code to a new file called `data_analyzer.py` if it isn't already!)
