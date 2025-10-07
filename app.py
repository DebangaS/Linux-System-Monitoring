"""
Enhanced Flask Application with Database Integration
Author: Member 1 (updating previous work)
"""
import os
import logging
import threading
import time
from datetime import datetime

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Import configuration and monitors
from config import config
from monitors.system_monitor import system_monitor
from database.models import db_manager


def create_app(config_name: str | None = None):
    """Enhanced application factory with database integration"""
    if config_name is None:
        config_name = os.environ.get('FLASK_CONFIG', 'default')

    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # Ensure log directory exists before configuring logging
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/exports', exist_ok=True)
    os.makedirs('data/backups', exist_ok=True)

    # Setup logging
    setup_logging(app)

    # Initialize extensions
    CORS(app, origins=app.config.get('CORS_ORIGINS', '*'))
    socketio = SocketIO(app, cors_allowed_origins=app.config.get('CORS_ORIGINS', '*'))

    # Initialize database (within app context)
    try:
        with app.app_context():
            db_manager.init_database()
        app.logger.info("Database initialized successfully")
    except Exception as e:
        app.logger.error(f"Database initialization failed: {e}")

    # Register blueprints (if present)
    try:
        from api.routes import api_bp
        from api.database_routes import db_api_bp

        app.register_blueprint(api_bp)
        app.register_blueprint(db_api_bp)
    except Exception:
        app.logger.debug("API blueprints not found or failed to import; continuing without them.")

    # Register routes and socket events
    register_routes(app)
    register_socketio_events(socketio, app)

    # Start background monitoring with database storage
    start_background_monitoring(app, socketio)

    # Schedule database cleanup task
    schedule_database_cleanup(app)

    app.logger.info(f"{app.config.get('APP_NAME', 'Resource Monitor')} v{app.config.get('APP_VERSION', '0.0.0')} with database startup")

    return app, socketio


def start_background_monitoring(app, socketio):
    """Enhanced background monitoring with database storage"""

    def monitor_loop():
        with app.app_context():
            # simple counter stored as attribute on function
            monitor_loop.counter = getattr(monitor_loop, 'counter', 0)
            while True:
                try:
                    # Get current system resources
                    resources = system_monitor.get_all_resources()

                    # Store in database
                    store_success = db_manager.store_system_metrics(
                        cpu=resources.get('cpu'),
                        memory=resources.get('memory'),
                        disk=resources.get('disk'),
                        network=resources.get('network'),
                    )
                    if store_success:
                        app.logger.debug("System metrics stored in database")

                    # Store process snapshot periodically (every 5th iteration)
                    monitor_loop.counter += 1
                    if monitor_loop.counter % 5 == 0:
                        try:
                            from monitors.process_monitor import process_monitor

                            processes = process_monitor.get_all_processes(limit=50)
                            db_manager.store_process_snapshot(processes)
                        except Exception as e:
                            app.logger.debug(f"Process snapshot failed: {e}")

                    # Emit to all connected clients
                    try:
                        socketio.emit('system_update', resources)
                    except Exception as e:
                        app.logger.debug(f"Socket emit failed: {e}")

                    # Check and store alerts
                    alerts = system_monitor.get_alerts()
                    if alerts:
                        for alert in alerts:
                            db_manager.store_system_alert(
                                alert.get('type'),
                                alert.get('level'),
                                alert.get('message'),
                                alert.get('value'),
                            )
                        socketio.emit('system_alerts', {'alerts': alerts})

                    # Wait for next update
                    time.sleep(app.config.get('MONITORING_INTERVAL', 2))
                except Exception as e:
                    app.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(5)

    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()


def schedule_database_cleanup(app):
    """Schedule periodic database cleanup"""

    def cleanup_task():
        with app.app_context():
            while True:
                try:
                    # Run cleanup every hour
                    time.sleep(3600)
                    # Clean up old data (keep last 7 days by default)
                    cleanup_success = db_manager.cleanup_old_data(days_to_keep=7)
                    if cleanup_success:
                        app.logger.info("Database cleanup completed")
                except Exception as e:
                    app.logger.error(f"Error in database cleanup: {e}")
                    # Wait an hour before retrying
                    time.sleep(3600)

    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()


def setup_logging(app):
    """Enhanced logging setup"""
    log_file = app.config.get('LOG_FILE', 'data/logs/resource_monitor.log')

    # Make sure the parent directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if not app.debug:
        # Production logging setup
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        )
        file_handler.setFormatter(formatter)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == file_handler.baseFilename
                   for h in app.logger.handlers):
            app.logger.addHandler(file_handler)

    # Always set logger level to INFO (can be overridden by config)
    app.logger.setLevel(app.config.get('LOG_LEVEL', logging.INFO))
    app.logger.info('Resource Monitor Dashboard with database startup')


def register_routes(app):
    """Register all application routes"""

    @app.route('/')
    def index():
        """Main dashboard page"""
        return render_template('index.html',
                               app_name=app.config.get('APP_NAME', 'Resource Monitor'),
                               app_version=app.config.get('APP_VERSION', '0.0.0'))

    @app.route('/dashboard')
    def dashboard():
        """Dashboard page"""
        return render_template('dashboard.html',
                               app_name=app.config.get('APP_NAME', 'Resource Monitor'))

    @app.route('/history')
    def history():
        """Historical data page"""
        return render_template('history.html',
                               app_name=app.config.get('APP_NAME', 'Resource Monitor'))

    @app.route('/analytics')
    def analytics():
        """Data analytics page"""
        return render_template('analytics.html',
                               app_name=app.config.get('APP_NAME', 'Resource Monitor'))

    @app.route('/api/status')
    def api_status():
        """Enhanced API status with database info"""
        try:
            db_stats = db_manager.get_database_stats()
            return jsonify({
                'status': 'running',
                'app_name': app.config.get('APP_NAME', 'Resource Monitor'),
                'version': app.config.get('APP_VERSION', '0.0.0'),
                'timestamp': datetime.utcnow().isoformat(),
                'monitoring_interval': app.config.get('MONITORING_INTERVAL', None),
                'monitoring_active': True,
                'database': {
                    'connected': True,
                    'metrics_count': db_stats.get('metrics_count', 0),
                    'process_snapshots_count': db_stats.get('process_snapshots_count', 0),
                    'alerts_count': db_stats.get('alerts_count', 0),
                    'db_size_mb': round(db_stats.get('db_size_mb', 0), 2)
                }
            })
        except Exception as e:
            app.logger.error(f"Error getting database stats: {e}")
            return jsonify({
                'status': 'running',
                'app_name': app.config.get('APP_NAME', 'Resource Monitor'),
                'version': app.config.get('APP_VERSION', '0.0.0'),
                'timestamp': datetime.utcnow().isoformat(),
                'database': {
                    'connected': False,
                    'error': str(e)
                }
            }), 500

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500


def register_socketio_events(socketio, app):
    """Enhanced SocketIO events with database integration"""

    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        emit('status', {'msg': 'Connected to Resource Monitor Dashboard with Database'})
        app.logger.info(f'Client connected at {datetime.now()}')
        # Send initial system data and recent history
        try:
            resources = system_monitor.get_all_resources()
            emit('system_update', resources)
            # Send recent alerts
            recent_alerts = db_manager.get_system_alerts(hours=1, acknowledged=False)
            if recent_alerts:
                emit('recent_alerts', {'alerts': recent_alerts})
        except Exception as e:
            app.logger.error(f"Error sending initial data: {e}")

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        app.logger.info(f'Client disconnected at {datetime.now()}')

    @socketio.on('request_historical_data')
    def handle_historical_request(data=None):
        """Handle request for historical data"""
        try:
            metric_type = data.get('type', 'cpu') if data else 'cpu'
            hours = int(data.get('hours', 24)) if data else 24
            historical_data = db_manager.get_historical_metrics(metric_type, hours)
            emit('historical_data', {
                'type': metric_type,
                'data': historical_data,
                'hours': hours
            })
        except Exception as e:
            app.logger.error(f"Error handling historical data request: {e}")
            emit('error', {'message': 'Failed to get historical data'})

    @socketio.on('acknowledge_alert')
    def handle_acknowledge_alert(data):
        """Handle alert acknowledgment"""
        try:
            alert_id = data.get('alert_id') if data else None
            if alert_id:
                success = db_manager.acknowledge_alert(alert_id)
                emit('alert_acknowledged', {
                    'alert_id': alert_id,
                    'success': success
                })
            else:
                emit('error', {'message': 'Missing alert_id'})
        except Exception as e:
            app.logger.error(f"Error acknowledging alert: {e}")
            emit('error', {'message': 'Failed to acknowledge alert'})


# Create the app instance
app, socketio = create_app()

if __name__ == '__main__':
    # Use config debug flag if present
    debug_mode = app.config.get('DEBUG', True)
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=5000)

from flask import Flask, render_template
from api.data_routes import data_bp

app = Flask(__name__)
app.config['APP_NAME'] = "System Analytics Dashboard"

@app.route('/analytics')
def analytics():
    """Analytics and historical data page"""
    return render_template('analytics.html', app_name=app.config['APP_NAME'])

# Register data blueprint
app.register_blueprint(data_bp)

if __name__ == '__main__':
    app.run(debug=True)

