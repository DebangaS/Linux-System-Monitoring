"""
Data export and analysis API routes
Author: Member 4
"""

from flask import Blueprint, jsonify, request, Response
from datetime import datetime
import csv
import json
import io
import logging
from database.models import db_manager
from database.data_analyzer import data_analyzer

# Create data API blueprint
data_bp = Blueprint('data', __name__, url_prefix='/api/v1/data')

# Setup logging
logger = logging.getLogger(__name__)


@data_bp.route('/historical', methods=['POST'])
def get_historical_data():
    """Get historical data with date range and granularity"""
    try:
        data = request.get_json()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        granularity = data.get('granularity', '5min')

        if not start_date or not end_date:
            return jsonify({
                'success': False,
                'error': 'Start date and end date are required'
            }), 400

        # Parse ISO format dates
        start_dt = datetime.fromisoformat(start_date.replace('T', ' '))
        end_dt = datetime.fromisoformat(end_date.replace('T', ' '))
        hours = (end_dt - start_dt).total_seconds() / 3600

        # Fetch data for all metric types
        cpu_data = db_manager.get_historical_metrics('cpu', hours=int(hours))
        memory_data = db_manager.get_historical_metrics('memory', hours=int(hours))
        disk_data = db_manager.get_historical_metrics('disk', hours=int(hours))
        network_data = db_manager.get_historical_metrics('network', hours=int(hours))

        def filter_by_date(data_list):
            return [
                item for item in data_list
                if start_dt <= datetime.fromisoformat(item['timestamp']) <= end_dt
            ]

        filtered_data = {
            'cpu': filter_by_date(cpu_data),
            'memory': filter_by_date(memory_data),
            'disk': filter_by_date(disk_data),
            'network': filter_by_date(network_data)
        }

        def apply_granularity(data_list, granularity):
            if granularity == '1min' or len(data_list) <= 100:
                return data_list
            sample_rate = {
                '5min': 5,
                '15min': 15,
                '1hour': 60
            }.get(granularity, 5)
            return data_list[::sample_rate]

        sampled_data = {
            key: apply_granularity(value, granularity)
            for key, value in filtered_data.items()
        }

        # Summary statistics
        summary = {
            'data_points': sum(len(v) for v in sampled_data.values()),
            'time_range': f"{hours:.1f} hours",
            'avg_cpu': calculate_average(
                [item.get('usage_percent', 0) for item in sampled_data['cpu']]
            ),
            'peak_memory': max(
                [item.get('usage_percent', 0) for item in sampled_data['memory']],
                default=0
            ),
            'start_date': start_date,
            'end_date': end_date
        }

        return jsonify({
            'success': True,
            'data': sampled_data,
            'summary': summary,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve historical data',
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@data_bp.route('/export/csv', methods=['POST'])
def export_csv():
    """Export data to CSV format"""
    try:
        data = request.get_json()
        data_type = data.get('data_type', 'all')
        time_range = data.get('time_range', {})

        if time_range.get('start') and time_range.get('end'):
            start_dt = datetime.fromisoformat(time_range['start'].replace('T', ' '))
            end_dt = datetime.fromisoformat(time_range['end'].replace('T', ' '))
            hours = (end_dt - start_dt).total_seconds() / 3600
        else:
            hours = 24

        output = io.StringIO()
        writer = csv.writer(output)

        if data_type == 'all':
            cpu_data = db_manager.get_historical_metrics('cpu', hours=int(hours))
            memory_data = db_manager.get_historical_metrics('memory', hours=int(hours))
            network_data = db_manager.get_historical_metrics('network', hours=int(hours))

            writer.writerow(['timestamp', 'cpu_percent', 'memory_percent', 'memory_total_gb',
                             'network_sent_kbps', 'network_recv_kbps'])

            all_timestamps = sorted({
                *[d['timestamp'] for d in cpu_data],
                *[d['timestamp'] for d in memory_data],
                *[d['timestamp'] for d in network_data]
            })

            for ts in all_timestamps:
                cpu_item = next((i for i in cpu_data if i['timestamp'] == ts), {})
                mem_item = next((i for i in memory_data if i['timestamp'] == ts), {})
                net_item = next((i for i in network_data if i['timestamp'] == ts), {})

                writer.writerow([
                    ts,
                    cpu_item.get('usage_percent', ''),
                    mem_item.get('usage_percent', ''),
                    round(mem_item.get('total', 0) / (1024 ** 3), 2) if mem_item.get('total') else '',
                    net_item.get('sent_rate_kbps', ''),
                    net_item.get('recv_rate_kbps', '')
                ])
        else:
            data_list = db_manager.get_historical_metrics(data_type, hours=int(hours))
            if not data_list:
                return jsonify({'success': False, 'error': 'No data available'}), 404

            headers = list(data_list[0].keys())
            writer.writerow(headers)
            for item in data_list:
                writer.writerow([item.get(k, '') for k in headers])

        output.seek(0)
        filename = f"{data_type}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )

    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        return jsonify({'success': False, 'error': 'Failed to export CSV data'}), 500


@data_bp.route('/export/json', methods=['POST'])
def export_json():
    """Export data to JSON format"""
    try:
        data = request.get_json()
        data_type = data.get('data_type', 'all')
        time_range = data.get('time_range', {})

        if time_range.get('start') and time_range.get('end'):
            start_dt = datetime.fromisoformat(time_range['start'].replace('T', ' '))
            end_dt = datetime.fromisoformat(time_range['end'].replace('T', ' '))
            hours = (end_dt - start_dt).total_seconds() / 3600
        else:
            hours = 24

        if data_type == 'all':
            export_data = {
                'cpu': db_manager.get_historical_metrics('cpu', hours=int(hours)),
                'memory': db_manager.get_historical_metrics('memory', hours=int(hours)),
                'disk': db_manager.get_historical_metrics('disk', hours=int(hours)),
                'network': db_manager.get_historical_metrics('network', hours=int(hours))
            }
        else:
            export_data = {
                'data': db_manager.get_historical_metrics(data_type, hours=int(hours))
            }

        export_data['export_info'] = {
            'exported_at': datetime.utcnow().isoformat(),
            'data_type': data_type,
            'time_range_hours': hours
        }

        json_str = json.dumps(export_data, indent=2, default=str)
        filename = f"{data_type}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        return Response(
            json_str,
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )

    except Exception as e:
        logger.error(f"Error exporting JSON: {e}")
        return jsonify({'success': False, 'error': 'Failed to export JSON data'}), 500


@data_bp.route('/analysis/summary', methods=['GET'])
def get_analysis_summary():
    """Get analysis summary for specified time period"""
    try:
        hours = request.args.get('hours', 24, type=int)
        analysis_report = data_analyzer.generate_system_report(hours=hours)

        return jsonify({
            'success': True,
            'data': analysis_report,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting analysis summary: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve analysis summary'
        }), 500


def calculate_average(values):
    """Calculate average of numeric values"""
    numeric_values = [v for v in values if isinstance(v, (int, float)) and v != 0]
    return sum(numeric_values) / len(numeric_values) if numeric_values else 0


# Error handlers
@data_bp.errorhandler(400)
def bad_request(error):
    return jsonify({'success': False, 'error': 'Bad request'}), 400


@data_bp.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Data endpoint not found'}), 404


@data_bp.errorhandler(500)
def internal_error(error):
    logger.error(f"Data API internal error: {error}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500
