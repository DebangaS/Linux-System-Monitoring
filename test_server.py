#!/usr/bin/env python3
"""
Minimal test server for load tests
"""
import os
import sys
import threading
import time
from flask import Flask, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

def create_minimal_app():
    """Create minimal app for load testing"""
    app = Flask(__name__)
    CORS(app, origins='*')
    
    # Minimal Socket.IO setup
    socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')
    
    @app.route('/api/v1/health')
    def health():
        return jsonify({'status': 'ok'}), 200
    
    @app.route('/api/v1/system/resources')
    def resources():
        return jsonify({
            'success': True, 
            'data': {
                'cpu': {'usage_percent': 25.0},
                'memory': {'usage_percent': 60.0},
                'disk': {'total_usage_percent': 45.0},
                'network': {'sent_rate_kbps': 10.0, 'recv_rate_kbps': 15.0}
            }
        }), 200
    
    @app.route('/api/v1/system/cpu')
    def cpu():
        return jsonify({'success': True, 'data': {'usage_percent': 25.0}}), 200
    
    @app.route('/api/v1/system/memory')
    def memory():
        return jsonify({'success': True, 'data': {'usage_percent': 60.0}}), 200
    
    @app.route('/api/v1/system/processes')
    def processes():
        return jsonify({'success': True, 'data': []}), 200
    
    @app.route('/api/v1/system/info')
    def info():
        return jsonify({'success': True, 'data': {'timestamp': '2024-01-01T00:00:00'}}), 200
    
    @socketio.on('connect')
    def handle_connect():
        print("Client connected")
    
    @socketio.on('disconnect')
    def handle_disconnect():
        print("Client disconnected")
    
    return app, socketio

if __name__ == '__main__':
    app, socketio = create_minimal_app()
    print("Starting minimal test server on 127.0.0.1:5000...")
    try:
        socketio.run(app, host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Server failed to start: {e}")
        import sys
        sys.exit(1)
