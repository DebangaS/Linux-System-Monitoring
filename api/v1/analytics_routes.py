"""
 Analytics API Routes
 Author: Member 3
 """
 from flask import Blueprint, request, jsonify
 from database.advanced_models import get_advanced_db_manager
 from analytics.data_analytics import get_analytics_engine
 from business_intelligence.dashboard import get_bi_dashboard
 import logging
 from datetime import datetime
 logger = logging.getLogger(__name__)
 analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/v1/analytics')
 # Initialize components
 db_manager = get_advanced_db_manager()
 analytics_engine = get_analytics_engine(db_manager)
 bi_dashboard = get_bi_dashboard(db_manager, analytics_engine)
 @analytics_bp.route('/metrics/<metric_name>/analysis', methods=['GET'])
 def get_metric_analysis(metric_name):
 """Get comprehensive analysis for a specific metric"""
    try:
        hours = request.args.get('hours', 24, type=int)
        if hours > 168:  # Limit to 1 week
            return jsonify({'error': 'Maximum analysis period is 168 hours (1 week)'}), 400
        analysis = analytics_engine.analyze_metric_comprehensive(metric_name, hours)
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Metric analysis error: {e}")
        return jsonify({'error': 'Analysis failed'}), 500
 @analytics_bp.route('/metrics/cross-analysis', methods=['POST'])
 def cross_metric_analysis():
 """Analyze relationships between multiple metrics"""
    try:
        data = request.get_json()
        if not data or 'metrics' not in data:
            return jsonify({'error': 'Metrics list required in request body'}), 400
        metrics = data['metrics']
        hours = data.get('hours', 24)
        if not isinstance(metrics, list) or len(metrics) < 2:
            return jsonify({'error': 'At least 2 metrics required for cross-analysis'}), 400
        if hours > 168:
            return jsonify({'error': 'Maximum analysis period is 168 hours (1 week)'}), 400
        analysis = analytics_engine.cross_metric_analysis(metrics, hours)
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Cross-metric analysis error: {e}")
        return jsonify({'error': 'Cross-analysis failed'}), 500
 @analytics_bp.route('/system/health', methods=['GET'])
 def system_health_analysis():
 """Get comprehensive system health analysis"""
    try:
        analysis = analytics_engine.system_health_analysis()
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"System health analysis error: {e}")
        return jsonify({'error': 'Health analysis failed'}), 500
 @analytics_bp.route('/capacity/forecast/<resource_type>', methods=['GET'])
 def capacity_forecast(resource_type):
 """Get capacity forecast for a specific resource"""
    try:
        days = request.args.get('days', 30, type=int)
        if days > 365:  # Limit to 1 year
            return jsonify({'error': 'Maximum forecast period is 365 days'}), 400
        forecast = analytics_engine.capacity_forecast(resource_type, days)
        return jsonify(forecast)
    except Exception as e:
        logger.error(f"Capacity forecast error: {e}")
        return jsonify({'error': 'Capacity forecast failed'}), 500
 @analytics_bp.route('/dashboard/executive', methods=['GET'])
 def executive_dashboard():
 """Get executive dashboard with KPIs"""
    try:
        hours = request.args.get('hours', 24, type=int)
        if hours > 168:
            return jsonify({'error': 'Maximum dashboard period is 168 hours (1 week)'}), 400
        dashboard_data = bi_dashboard.get_executive_dashboard(hours)
        return jsonify(dashboard_data)
    except Exception as e:
        logger.error(f"Executive dashboard error: {e}")
        return jsonify({'error': 'Dashboard generation failed'}), 500
 @analytics_bp.route('/kpi/<kpi_name>', methods=['GET'])
 def get_kpi_value(kpi_name):
 """Get current value for a specific KPI"""
    try:
        hours = request.args.get('hours', 24, type=int)
        kpi_result = bi_dashboard.calculate_kpi_value(kpi_name, hours)
        return jsonify(kpi_result)
    except Exception as e:
        logger.error(f"KPI calculation error: {e}")
        return jsonify({'error': 'KPI calculation failed'}), 500
 @analytics_bp.route('/kpi/<kpi_name>/trend', methods=['GET'])
 def get_kpi_trend(kpi_name):
 """Get trend analysis for a specific KPI"""
    try:
        days = request.args.get('days', 7, type=int)
        if days > 30:
            return jsonify({'error': 'Maximum trend period is 30 days'}), 400
        trend_analysis = bi_dashboard.get_trend_analysis(kpi_name, days)
        return jsonify(trend_analysis)
    except Exception as e:
        logger.error(f"KPI trend analysis error: {e}")
        return jsonify({'error': 'Trend analysis failed'}), 500
 @analytics_bp.route('/dashboard/export', methods=['GET'])
 def export_dashboard():
 """Export dashboard data in various formats"""
    try:
        format_type = request.args.get('format', 'json')
        hours = request.args.get('hours', 24, type=int)
        if format_type not in ['json', 'csv']:
            return jsonify({'error': 'Supported formats: json, csv'}), 400
        export_data = bi_dashboard.export_dashboard_data(format_type, hours)
        return jsonify(export_data)
    except Exception as e:
        logger.error(f"Dashboard export error: {e}")
        return jsonify({'error': 'Dashboard export failed'}), 500
 @analytics_bp.route('/database/stats', methods=['GET'])
 def database_statistics():
 """Get advanced database statistics"""
    try:
 # Get basic database stats
        basic_stats = db_manager.get_cached_query(
            'database_basic_stats',
            '''
            SELECT
                (SELECT COUNT(*) FROM metrics_current) as current_metrics,
                (SELECT COUNT(*) FROM metrics_historical) as historical_metrics,
                (SELECT COUNT(*) FROM system_events) as total_events,
                (SELECT COUNT(*) FROM process_metrics_advanced) as process_metrics
            '''
        )
 # Get table sizes
        table_stats = db_manager.get_cached_query(
            'database_table_stats',
            '''
            SELECT
                name as table_name
            FROM sqlite_master
            WHERE type='table'
            ORDER BY name
            '''
        )
        return jsonify({
            'basic_statistics': basic_stats[0] if basic_stats else {},
            'table_statistics': table_stats,
            'generated_at': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Database statistics error: {e}")
        return jsonify({'error': 'Database statistics failed'}), 500
 @analytics_bp.route('/query/performance', methods=['POST'])
 def analyze_query_performance():
 """Analyze query performance"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query required in request body'}), 400
        query = data['query']
        params = tuple(data.get('params', []))
 # Security check - only allow SELECT queries
        if not query.strip().upper().startswith('SELECT'):
            return jsonify({'error': 'Only SELECT queries are allowed'}), 400
        analysis = db_manager.analyze_query_performance(query, params)
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Query performance analysis error: {e}")
        return jsonify({'error': 'Query analysis failed'}), 500
 @analytics_bp.route('/events/create', methods=['POST'])
 def create_system_event():
 """Create a new system event"""
    try:
        data = request.get_json()
        required_fields = ['event_type', 'severity', 'source', 'title']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Required field missing: {field}'}), 400
        event_id = db_manager.create_system_event(
            event_type=data['event_type'],
            severity=data['severity'],
            source=data['source'],
            title=data['title'],
            description=data.get('description'),
            details=data.get('details'),
            tags=data.get('tags'),
            impact_score=data.get('impact_score', 0),
            related_metrics=data.get('related_metrics')
        )
        if event_id:
            return jsonify({
                'event_id': event_id,
                'message': 'System event created successfully'
            }), 201
        else:
            return jsonify({'error': 'Failed to create system event'}), 500
    except Exception as e:
        logger.error(f"System event creation error: {e}")
        return jsonify({'error': 'Event creation failed'}), 500
 @analytics_bp.route('/metrics/store', methods=['POST'])
 def store_advanced_metric():
 """Store an advanced metric"""
    try:
        data = request.get_json()
        required_fields = ['metric_type', 'metric_name', 'value']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Required field missing: {field}'}), 400
        success = db_manager.store_advanced_metric(
            metric_type=data['metric_type'],
            metric_name=data['metric_name'],
            value=float(data['value']),
            metadata=data.get('metadata'),
            source=data.get('source', 'api'),
            tags=data.get('tags')
        )
        if success:
            return jsonify({'message': 'Metric stored successfully'}), 201
        else:
            return jsonify({'error': 'Failed to store metric'}), 500
    except (ValueError, TypeError) as e:
        return jsonify({'error': 'Invalid metric value'}), 400
    except Exception as e:
        logger.error(f"Advanced metric storage error: {e}")
        return jsonify({'error': 'Metric storage failed'}), 500
 @analytics_bp.route('/dashboard/kpis', methods=['GET'])
 def list_available_kpis():
 """List all available KPIs"""
    try:
        kpi_list = []
        for kpi_name, kpi in bi_dashboard.kpis.items():
            kpi_list.append({
                'name': kpi_name,
                'display_name': kpi.name,
                'description': kpi.description,
                'category': kpi.category,
                'unit': kpi.unit,
                'target_value': kpi.target_value,
                'update_frequency': kpi.update_frequency
            })
        return jsonify({
            'kpis': kpi_list,
            'total_count': len(kpi_list)
        })
    except Exception as e:
        logger.error(f"KPI listing error: {e}")
        return jsonify({'error': 'Failed to list KPIs'}), 500
 @analytics_bp.route('/health', methods=['GET'])
 def analytics_health_check():
 """Analytics service health check"""
    try:
 # Test database connection
        test_query = "SELECT 1 as test"
        result = db_manager.get_cached_query('health_check', test_query)
        if result and result[0]['test'] == 1:
            db_status = 'healthy'
        else:
            db_status = 'unhealthy'
        return jsonify({
            'status': 'healthy' if db_status == 'healthy' else 'degraded',
            'database_status': db_status,
            'analytics_engine_status': 'healthy',
            'bi_dashboard_status': 'healthy',
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Analytics health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500
