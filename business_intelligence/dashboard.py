"""
 Business Intelligence Dashboard
 Author: Member 3
 """
 import json
 import logging
 from datetime import datetime, timedelta
 from typing import Dict, List, Optional, Any
 from dataclasses import dataclass
 logger = logging.getLogger(__name__)
 @dataclass
 class KPI:
 """Key Performance Indicator definition"""
    name: str
    description: str
    category: str
    target_value: float
    warning_threshold: float
    critical_threshold: float
    unit: str
    calculation_method: str
    data_sources: List[str]
    update_frequency: str  # 'real-time', 'hourly', 'daily'
 class BusinessIntelligenceDashboard:
 """Business Intelligence Dashboard for system monitoring"""
 def __init__(self, db_manager, analytics_engine):
    self.db_manager = db_manager
    self.analytics_engine = analytics_engine
 # Initialize KPIs
    self.kpis = self._initialize_kpis()
 # Dashboard configuration
    self.dashboard_config = {
        'refresh_interval': 300,  # 5 minutes
        'data_retention_days': 90,
        'alert_thresholds': {
            'performance_degradation': 0.8,
            'availability_threshold': 0.99,
            'response_time_threshold': 2.0
        }
    }
 logger.info("Business Intelligence Dashboard initialized")
 def _initialize_kpis(self) -> Dict[str, KPI]:
 """Initialize standard KPIs"""
    return {
        'system_availability': KPI(
            name='System Availability',
            description='Percentage of time system is operational',
            category='Availability',
            target_value=99.9,
            warning_threshold=99.0,
            critical_threshold=98.0,
            unit='percent',
            calculation_method='uptime / total_time * 100',
            data_sources=['system_events', 'metrics_current'],
            update_frequency='real-time'
        ),
        'average_response_time': KPI(
            name='Average Response Time',
            description='Average system response time',
            category='Performance',
            target_value=1.0,
            warning_threshold=2.0,
            critical_threshold=5.0,
            unit='seconds',
            calculation_method='avg(response_times)',
            data_sources=['metrics_current'],
            update_frequency='real-time'
        ),
        'cpu_efficiency': KPI(
            name='CPU Efficiency',
            description='CPU utilization efficiency score',
            category='Resource Utilization',
            target_value=70.0,
            warning_threshold=85.0,
            critical_threshold=95.0,
            unit='percent',
            calculation_method='weighted_avg(cpu_usage)',
            data_sources=['metrics_current'],
            update_frequency='real-time'
        ),
        'memory_efficiency': KPI(
            name='Memory Efficiency',
            description='Memory utilization efficiency score',
            category='Resource Utilization',
            target_value=75.0,
            warning_threshold=85.0,
            critical_threshold=95.0,
            unit='percent',
            calculation_method='avg(memory_usage)',
            data_sources=['metrics_current'],
            update_frequency='real-time'
        ),
        'error_rate': KPI(
            name='System Error Rate',
            description='Rate of system errors per hour',
            category='Reliability',
            target_value=0.0,
            warning_threshold=5.0,
            critical_threshold=20.0,
            unit='errors/hour',
            calculation_method='count(errors) / hours',
            data_sources=['system_events'],
            update_frequency='hourly'
        ),
        'capacity_utilization': KPI(
            name='Overall Capacity Utilization',
            description='Combined resource capacity utilization',
            category='Capacity',
            target_value=60.0,
            warning_threshold=80.0,
            critical_threshold=90.0,
            unit='percent',
            calculation_method='weighted_avg(cpu, memory, disk)',
            data_sources=['metrics_current'],
            update_frequency='real-time'
        )
    }
 def calculate_kpi_value(self, kpi_name: str, time_period_hours: int = 24) -> Dict:
 """Calculate current value for a specific KPI"""
    if kpi_name not in self.kpis:
        return {'error': f'KPI {kpi_name} not found'}
 kpi = self.kpis[kpi_name]
    try:
        if kpi_name == 'system_availability':
            return self._calculate_availability(time_period_hours)
        elif kpi_name == 'average_response_time':
            return self._calculate_response_time(time_period_hours)
        elif kpi_name == 'cpu_efficiency':
            return self._calculate_cpu_efficiency(time_period_hours)
        elif kpi_name == 'memory_efficiency':
            return self._calculate_memory_efficiency(time_period_hours)
        elif kpi_name == 'error_rate':
            return self._calculate_error_rate(time_period_hours)
        elif kpi_name == 'capacity_utilization':
            return self._calculate_capacity_utilization(time_period_hours)
        else:
            return {'error': f'Calculation method not implemented for {kpi_name}'}
    except Exception as e:
        logger.error(f"Error calculating KPI {kpi_name}: {e}")
        return {'error': str(e)}
 def _calculate_availability(self, hours: int) -> Dict:
 """Calculate system availability KPI"""
    query = '''
        SELECT
            COUNT(CASE WHEN severity IN ('error', 'critical') THEN 1 END) as error_events,
            COUNT(*) as total_events
        FROM system_events
        WHERE timestamp >= datetime('now', '-' || ? || ' hours')
    '''
    data = self.db_manager.get_cached_query(f'availability_{hours}h', query, (hours,))
    if not data:
        return {'value': 100.0, 'status': 'excellent'}
    error_events = data[0]['error_events'] or 0
    total_events = data[0]['total_events'] or 0
 # Calculate availability (simplified)
    if total_events == 0:
        availability = 100.0
    else:
 # Assume each error event represents 5 minutes of downtime
        downtime_minutes = error_events * 5
        total_minutes = hours * 60
        availability = max(0, (total_minutes - downtime_minutes) / total_minutes * 100)
 kpi = self.kpis['system_availability']
    status = self._get_kpi_status(availability, kpi)
    return {
        'value': round(availability, 2),
        'status': status,
        'target': kpi.target_value,
        'unit': kpi.unit,
        'details': {
            'error_events': error_events,
            'total_events': total_events,
            'downtime_minutes': error_events * 5
        }
    }
 def _calculate_response_time(self, hours: int) -> Dict:
 """Calculate average response time KPI"""
 # This would typically come from application metrics
 # For now, we'll estimate based on CPU usage
    query = '''
        SELECT AVG(value) as avg_cpu
        FROM v_all_metrics
        WHERE metric_name LIKE '%cpu%'
        AND timestamp >= datetime('now', '-' || ? || ' hours')
    '''
    data = self.db_manager.get_cached_query(f'response_time_{hours}h', query, (hours,))
    if not data or data[0]['avg_cpu'] is None:
        return {'value': 1.0, 'status': 'good'}
 avg_cpu = data[0]['avg_cpu']
 # Estimate response time based on CPU usage
    if avg_cpu > 90:
        response_time = 5.0
    elif avg_cpu > 80:
        response_time = 3.0
    elif avg_cpu > 70:
        response_time = 2.0
    elif avg_cpu > 50:
        response_time = 1.5
    else:
        response_time = 1.0
 kpi = self.kpis['average_response_time']
    status = self._get_kpi_status(response_time, kpi, inverse=True)  # Lower is better
    return {
        'value': round(response_time, 2),
        'status': status,
        'target': kpi.target_value,
        'unit': kpi.unit,
        'details': {
            'avg_cpu_usage': round(avg_cpu, 2)
        }
    }
 def _calculate_cpu_efficiency(self, hours: int) -> Dict:
 """Calculate CPU efficiency KPI"""
    query = '''
        SELECT
            AVG(value) as avg_usage,
            MIN(value) as min_usage,
            MAX(value) as max_usage,
            COUNT(*) as sample_count
        FROM v_all_metrics
        WHERE metric_type = 'cpu'
        AND timestamp >= datetime('now', '-' || ? || ' hours')
    '''
    data = self.db_manager.get_cached_query(f'cpu_efficiency_{hours}h', query, (hours,))
    if not data or data[0]['avg_usage'] is None:
        return {'value': 50.0, 'status': 'fair'}
 row = data[0]
    avg_usage = row['avg_usage']
    max_usage = row['max_usage']
 # Calculate efficiency score
 # Good efficiency: consistent usage without spikes
    usage_variance = max_usage - avg_usage
    if avg_usage > 95:
        efficiency = 20  # Too high
    elif avg_usage > 85:
        efficiency = 40
    elif avg_usage > 70:
        efficiency = 80 - (usage_variance * 2)  # Penalize high variance
    elif avg_usage > 50:
        efficiency = 90 - (usage_variance * 1.5)
    else:
        efficiency = 95  # Low but stable usage
    efficiency = max(0, min(100, efficiency))
    kpi = self.kpis['cpu_efficiency']
    status = self._get_kpi_status(efficiency, kpi, inverse=True)
    return {
        'value': round(efficiency, 1),
        'status': status,
        'target': kpi.target_value,
        'unit': kpi.unit,
        'details': {
            'avg_usage': round(avg_usage, 1),
            'max_usage': round(max_usage, 1),
            'usage_variance': round(usage_variance, 1),
            'sample_count': row['sample_count']
        }
    }
 def _calculate_memory_efficiency(self, hours: int) -> Dict:
 """Calculate memory efficiency KPI"""
    query = '''
        SELECT
            AVG(value) as avg_usage,
            MAX(value) as max_usage,
            COUNT(*) as sample_count
        FROM v_all_metrics
        WHERE metric_type = 'memory'
        AND timestamp >= datetime('now', '-' || ? || ' hours')
    '''
    data = self.db_manager.get_cached_query(f'memory_efficiency_{hours}h', query, (hours,))
    if not data or data[0]['avg_usage'] is None:
        return {'value': 50.0, 'status': 'fair'}
 row = data[0]
    avg_usage = row['avg_usage']
    kpi = self.kpis['memory_efficiency']
    status = self._get_kpi_status(avg_usage, kpi, inverse=True)
    return {
        'value': round(avg_usage, 1),
        'status': status,
        'target': kpi.target_value,
        'unit': kpi.unit,
        'details': {
            'avg_usage': round(avg_usage, 1),
            'max_usage': round(row['max_usage'], 1),
            'sample_count': row['sample_count']
        }
    }
 def _calculate_error_rate(self, hours: int) -> Dict:
 """Calculate system error rate KPI"""
    query = '''
        SELECT
            COUNT(CASE WHEN severity IN ('error', 'critical') THEN 1 END) as error_count
        FROM system_events
        WHERE timestamp >= datetime('now', '-' || ? || ' hours')
    '''
    data = self.db_manager.get_cached_query(f'error_rate_{hours}h', query, (hours,))
    if not data:
        return {'value': 0.0, 'status': 'excellent'}
    error_count = data[0]['error_count'] or 0
    error_rate = error_count / hours if hours > 0 else 0
    kpi = self.kpis['error_rate']
    status = self._get_kpi_status(error_rate, kpi, inverse=True)
    return {
        'value': round(error_rate, 2),
        'status': status,
        'target': kpi.target_value,
        'unit': kpi.unit,
        'details': {
            'total_errors': error_count,
            'time_period_hours': hours
        }
    }
 def _calculate_capacity_utilization(self, hours: int) -> Dict:
 """Calculate overall capacity utilization KPI"""
    query = '''
        SELECT
            metric_type,
            AVG(value) as avg_value
        FROM v_all_metrics
        WHERE metric_type IN ('cpu', 'memory', 'disk')
        AND timestamp >= datetime('now', '-' || ? || ' hours')
        GROUP BY metric_type
    '''
    data = self.db_manager.get_cached_query(f'capacity_util_{hours}h', query, (hours,))
    if not data:
        return {'value': 50.0, 'status': 'fair'}
 # Weight different resource types
    weights = {'cpu': 0.4, 'memory': 0.3, 'disk': 0.3}
    weighted_sum = 0
    total_weight = 0
    resource_values = {}
    for row in data:
        metric_type = row['metric_type']
        avg_value = row['avg_value']
        if metric_type in weights:
            weight = weights[metric_type]
            weighted_sum += avg_value * weight
            total_weight += weight
            resource_values[metric_type] = avg_value
    if total_weight == 0:
        capacity_utilization = 50.0
    else:
        capacity_utilization = weighted_sum / total_weight
    kpi = self.kpis['capacity_utilization']
    status = self._get_kpi_status(capacity_utilization, kpi, inverse=True)
    return {
        'value': round(capacity_utilization, 1),
        'status': status,
        'target': kpi.target_value,
        'unit': kpi.unit,
        'details': {
            'resource_breakdown': resource_values,
            'weights_used': weights
        }
    }
 def _get_kpi_status(self, value: float, kpi: KPI, inverse: bool = False) -> str:
 """Get KPI status based on value and thresholds"""
    if inverse:  # For KPIs where lower values are better
        if value <= kpi.target_value:
            return 'excellent'
        elif value <= kpi.warning_threshold:
            return 'good'
        elif value <= kpi.critical_threshold:
            return 'warning'
        else:
            return 'critical'
    else:  # For KPIs where higher values are better
        if value >= kpi.target_value:
            return 'excellent'
        elif value >= kpi.warning_threshold:
            return 'good'
        elif value >= kpi.critical_threshold:
            return 'warning'
        else:
            return 'critical'
 def get_executive_dashboard(self, time_period_hours: int = 24) -> Dict:
 """Generate executive dashboard with all KPIs"""
    try:
        dashboard_data = {
            'generated_at': datetime.utcnow().isoformat(),
            'time_period_hours': time_period_hours,
            'kpis': {},
            'summary': {
                'excellent': 0,
                'good': 0,
                'warning': 0,
                'critical': 0
            },
            'overall_health_score': 0,
            'recommendations': []
        }
        total_score = 0
        kpi_count = 0
 # Calculate all KPIs
        for kpi_name in self.kpis:
            kpi_result = self.calculate_kpi_value(kpi_name, time_period_hours)
            if 'error' not in kpi_result:
                dashboard_data['kpis'][kpi_name] = kpi_result
                dashboard_data['summary'][kpi_result['status']] += 1
 # Convert status to score for overall calculation
                status_scores = {'excellent': 100, 'good': 80, 'warning': 50, 'critical': 20}
                total_score += status_scores.get(kpi_result['status'], 50)
                kpi_count += 1
 # Calculate overall health score
        if kpi_count > 0:
            dashboard_data['overall_health_score'] = round(total_score / kpi_count, 1)
 # Generate recommendations based on KPI status
        dashboard_data['recommendations'] = self._generate_executive_recommendations(dashboard_data['kpis'])
        return dashboard_data
    except Exception as e:
        logger.error(f"Executive dashboard generation error: {e}")
        return {'error': str(e)}
 def _generate_executive_recommendations(self, kpis: Dict) -> List[str]:
 """Generate executive-level recommendations"""
    recommendations = []
    for kpi_name, kpi_data in kpis.items():
        if kpi_data['status'] == 'critical':
            if kpi_name == 'system_availability':
                recommendations.append("URGENT: System availability is below acceptable levels. Investigate service interruptions.")
            elif kpi_name == 'cpu_efficiency':
                recommendations.append("CPU resources are critically overutilized - consider scaling up or optimizing workloads.")
            elif kpi_name == 'memory_efficiency':
                recommendations.append("Memory resources are critically low - immediate action required to prevent system crashes.")
            elif kpi_name == 'error_rate':
                recommendations.append("System error rate is critically high - investigate root cause and deploy fixes.")
        elif kpi_data['status'] == 'warning':
            if kpi_name == 'capacity_utilization':
                recommendations.append("Resource utilization approaching limits - plan for future capacity increases.")
            elif kpi_name == 'average_response_time':
                recommendations.append("System response times are elevated - performance optimization is recommended.")
 # Overall recommendations
    critical_count = sum(1 for kpi in kpis.values() if kpi['status'] == 'critical')
    warning_count = sum(1 for kpi in kpis.values() if kpi['status'] == 'warning')
    if critical_count >= 2:
        recommendations.insert(0, "Multiple critical issues detected - comprehensive system review is required immediately.")
    elif critical_count == 1 and warning_count >= 2:
        recommendations.append("System health degrading - proactive maintenance recommended to prevent further issues.")
    return recommendations[:5]  # Limit to top 5 recommendations
 def get_trend_analysis(self, kpi_name: str, days: int = 7) -> Dict:
 """Get trend analysis for a specific KPI over time"""
    if kpi_name not in self.kpis:
        return {'error': f'KPI {kpi_name} not found'}
    try:
 # Calculate KPI values for each day
        trend_data = []
        for day in range(days, 0, -1):
            date_point = datetime.utcnow() - timedelta(days=day)
 # For simplicity, we'll calculate KPI for each day
 # In a real implementation, this would be pre-calculated and stored
            kpi_value = self.calculate_kpi_value(kpi_name, 24)  # 24-hour window
            if 'error' not in kpi_value:
                trend_data.append({
                    'date': date_point.strftime('%Y-%m-%d'),
                    'value': kpi_value['value'],
                    'status': kpi_value['status']
                })
        if not trend_data:
            return {'error': 'No trend data available'}
 # Analyze trend
        values = [point['value'] for point in trend_data]
        if len(values) >= 2:
            trend_direction = 'improving' if values[-1] > values[0] else 'declining'
            change_percent = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
        else:
            trend_direction = 'stable'
            change_percent = 0
        return {
            'kpi_name': kpi_name,
            'trend_period_days': days,
            'trend_data': trend_data,
            'analysis': {
                'direction': trend_direction,
                'change_percent': round(change_percent, 2),
                'current_value': values[-1] if values else 0,
                'average_value': round(sum(values) / len(values), 2) if values else 0
            },
            'generated_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Trend analysis error for {kpi_name}: {e}")
        return {'error': str(e)}
 def export_dashboard_data(self, format_type: str = 'json', time_period_hours: int = 24) -> Dict:
 """Export dashboard data in various formats"""
    try:
        dashboard_data = self.get_executive_dashboard(time_period_hours)
        if format_type == 'json':
            return {
                'format': 'json',
                'data': dashboard_data,
                'exported_at': datetime.utcnow().isoformat()
            }
        elif format_type == 'csv':
 # Convert KPI data to CSV format
            csv_data = []
            for kpi_name, kpi_data in dashboard_data.get('kpis', {}).items():
                csv_data.append({
                    'kpi_name': kpi_name,
                    'value': kpi_data['value'],
                    'status': kpi_data['status'],
                    'target': kpi_data['target'],
                    'unit': kpi_data['unit']
                })
            return {
                'format': 'csv',
                'data': csv_data,
                'exported_at': datetime.utcnow().isoformat()
            }
        else:
            return {'error': f'Unsupported format: {format_type}'}
    except Exception as e:
        logger.error(f"Dashboard export error: {e}")
        return {'error': str(e)}
 # Global BI dashboard instance
bi_dashboard = None
def get_bi_dashboard(db_manager, analytics_engine):
 """Get global BI dashboard instance"""
    global bi_dashboard
    if bi_dashboard is None:
        bi_dashboard = BusinessIntelligenceDashboard(db_manager, analytics_engine)
    return bi_dashboard
