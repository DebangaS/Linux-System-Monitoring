"""
Database API routes for system monitoring
Author: System
"""

from flask import Blueprint, jsonify
from datetime import datetime, timezone
from database.models import db_manager

# Create database API blueprint
db_api_bp = Blueprint('database', __name__, url_prefix='/api/database')

@db_api_bp.route('/stats')
def database_stats():
    """Get database statistics"""
    try:
        stats = db_manager.get_database_stats()
        return jsonify({
            'success': True,
            'data': stats,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

@db_api_bp.route('/health')
def database_health():
    """Check database health"""
    try:
        # Try to get a simple stat to test connection
        stats = db_manager.get_database_stats()
        is_healthy = isinstance(stats, dict) and len(stats) > 0
        
        return jsonify({
            'status': 'healthy' if is_healthy else 'unhealthy',
            'connected': is_healthy,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'connected': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500
