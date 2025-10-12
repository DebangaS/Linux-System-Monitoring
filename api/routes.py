"""
Main API routes for system monitoring
Author: System
"""

from flask import Blueprint, jsonify
from datetime import datetime, timezone

# Create main API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/health')
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'service': 'system-monitor-api'
    })

@api_bp.route('/version')
def version():
    """API version endpoint"""
    return jsonify({
        'version': '1.0.0',
        'api_version': 'v1',
        'timestamp': datetime.now(timezone.utc).isoformat()
    })
