"""
System Monitor - Main Application
Author: Member 1
Updated for Day 4 - Production Ready
"""

import os
import time
import logging
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache

from config import get_config
from auth.routes import auth_bp
from api.v1.routes import api_v1_bp
from api.documentation import docs_bp
from monitoring.health import health_monitor
from database.models import DatabaseManager
from utils.performance_monitor import PerformanceMonitor


def create_app(config_name=None):
    """Application factory"""
    app = Flask(__name__)

    # Load configuration
    config_class = get_config()
    app.config.from_object(config_class)

    # Initialize extensions
    cors = CORS(app, origins=app.config.get('CORS_ORIGINS', []))

    # Rate limiting
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        storage_uri=app.config.get('RATELIMIT_STORAGE_URL'),
        default_limits=[app.config.get('RATELIMIT_DEFAULT')]
    )

    # Caching
    cache = Cache(app)

    # Performance monitoring
    perf_monitor = PerformanceMonitor()

    # Initialize database
    db_manager = DatabaseManager()

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_v1_bp)
    app.register_blueprint(docs_bp)

    # Error handlers
    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': str(e.description)
        }), 429

    @app.errorhandler(500)
    def internal_error(e):
        app.logger.error(f"Internal server error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            'error': 'Not found',
            'message': 'The requested resource was not found'
        }), 404

    # Health check endpoint
    @app.route('/api/v1/health')
    @limiter.exempt
    def health_check():
        """Comprehensive health check endpoint"""
        try:
            health_data = health_monitor.get_comprehensive_health()
            if health_data['overall_status'] == 'unhealthy':
                return jsonify(health_data), 503
            else:
                return jsonify(health_data), 200
        except Exception as e:
            app.logger.error(f"Health check failed: {str(e)}")
            return jsonify({
                'status': 'unhealthy',
                'error': 'Health check failed',
                'timestamp': datetime.utcnow().isoformat()
            }), 503

    # Performance monitoring endpoint
    @app.route('/api/v1/performance')
    @limiter.limit("10/minute")
    def performance_stats():
        """Get application performance statistics"""
        try:
            return jsonify(perf_monitor.get_performance_summary())
        except Exception as e:
            app.logger.error(f"Performance stats error: {str(e)}")
            return jsonify({'error': 'Performance stats unavailable'}), 500

    # Request logging middleware
    @app.before_request
    def log_request_info():
        """Log request information"""
        if app.config.get('LOG_REQUESTS', False):
            app.logger.info(f"Request: {request.method} {request.url}")

    # Performance monitoring middleware
    @app.before_request
    def start_timer():
        """Start request timer"""
        request.start_time = time.time()

    @app.after_request
    def log_response_info(response):
        """Log response information and performance"""
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            if app.config.get('LOG_PERFORMANCE', False):
                app.logger.info(
                    f"Response: {response.status_code} "
                    f"Duration: {duration:.3f}s "
                    f"Path: {request.path}"
                )
        return response

    # Initialize configuration hooks if available
    if hasattr(config_class, "init_app"):
        config_class.init_app(app)

    return app


# Create application instance
app = create_app()

if __name__ == '__main__':
    # Development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config.get('DEBUG', False),
        threaded=True
    )
