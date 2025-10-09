"""
Advanced Sensor Integration and IoT Monitoring
Author: Member 2
"""
import time
import threading
import asyncio
import logging
import json
import requests
import serial
import glob
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    timestamp: str
    metadata: Dict[str, Any]
    quality_score: float

    def to_dict(self):
        return asdict(self)

@dataclass
class DeviceStatus:
    device_id: str
    device_type: str
    status: str
    last_seen: str
    battery_level: Optional[float]
    signal_strength: Optional[float]
    firmware_version: Optional[str]

    def to_dict(self):
        return asdict(self)

class TemperatureSensorManager:
    def __init__(self):
        self.sensors = {}
        self.readings = defaultdict(lambda: deque(maxlen=1000))

    def discover_temperature_sensors(self) -> List[str]:
        sensors = []
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            for sensor_name, sensor_list in temps.items():
                for i, sensor in enumerate(sensor_list):
                    sensor_id = f"temp_{sensor_name}_{i}"
                    sensors.append(sensor_id)
                    self.sensors[sensor_id] = {
                        'type': 'temperature',
                        'source': 'psutil',
                        'sensor_name': sensor_name,
                        'sensor_index': i,
                        'label': sensor.label or f"Sensor {i}"
                    }
        except Exception as e:
            logger.debug(f"Could not discover psutil temperature sensors: {e}")

        try:
            thermal_zones = glob.glob('/sys/class/thermal/thermal_zone*/temp')
            for i, zone_path in enumerate(thermal_zones):
                sensor_id = f"thermal_zone_{i}"
                sensors.append(sensor_id)
                self.sensors[sensor_id] = {
                    'type': 'temperature',
                    'source': 'thermal_zone',
                    'path': zone_path,
                    'zone_index': i
                }
        except Exception as e:
            logger.debug(f"Could not discover thermal zone sensors: {e}")

        try:
            w1_devices = glob.glob('/sys/bus/w1/devices/28-*/w1_slave')
            for device_path in w1_devices:
                device_id = device_path.split('/')[-2]
                sensor_id = f"ds18b20_{device_id}"
                sensors.append(sensor_id)
                self.sensors[sensor_id] = {
                    'type': 'temperature',
                    'source': 'ds18b20',
                    'path': device_path,
                    'device_id': device_id
                }
        except Exception as e:
            logger.debug(f"Could not discover DS18B20 sensors: {e}")

        return sensors

    def read_temperature_sensor(self, sensor_id: str) -> Optional[SensorReading]:
        try:
            if sensor_id not in self.sensors:
                return None

            sensor_config = self.sensors[sensor_id]
            temperature = None

            if sensor_config['source'] == 'psutil':
                import psutil
                temps = psutil.sensors_temperatures()
                sensor_name = sensor_config['sensor_name']
                sensor_index = sensor_config['sensor_index']
                if sensor_name in temps and len(temps[sensor_name]) > sensor_index:
                    temperature = temps[sensor_name][sensor_index].current

            elif sensor_config['source'] == 'thermal_zone':
                with open(sensor_config['path'], 'r') as f:
                    temp_raw = int(f.read().strip())
                    temperature = temp_raw / 1000.0

            elif sensor_config['source'] == 'ds18b20':
                with open(sensor_config['path'], 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2 and 'YES' in lines[0]:
                        temp_output = lines[1].find('t=')
                        if temp_output != -1:
                            temp_string = lines[1][temp_output+2:]
                            temperature = float(temp_string) / 1000.0

            if temperature is not None:
                reading = SensorReading(
                    sensor_id=sensor_id,
                    sensor_type='temperature',
                    value=temperature,
                    unit='°C',
                    timestamp=datetime.utcnow().isoformat(),
                    metadata=sensor_config.copy(),
                    quality_score=1.0
                )
                self.readings[sensor_id].append(reading)
                return reading

        except Exception as e:
            logger.error(f"Error reading temperature sensor {sensor_id}: {e}")
        return None

    def get_all_temperature_readings(self) -> List[SensorReading]:
        readings = []
        for sensor_id in self.sensors:
            reading = self.read_temperature_sensor(sensor_id)
            if reading:
                readings.append(reading)
        return readings

class NetworkSensorManager:
    def __init__(self):
        self.devices = {}
        self.device_status = {}

    def register_http_device(self, device_id: str, endpoint_url: str, device_type: str, auth_header: Optional[str] = None):
        try:
            self.devices[device_id] = {
                'type': 'http',
                'device_type': device_type,
                'endpoint': endpoint_url,
                'auth_header': auth_header,
                'registered_at': datetime.utcnow()
            }
            logger.info(f"Registered HTTP device {device_id} at {endpoint_url}")
        except Exception as e:
            logger.error(f"Error registering HTTP device: {e}")

    def read_http_device(self, device_id: str) -> Optional[List[SensorReading]]:
        try:
            if device_id not in self.devices:
                return None

            device_config = self.devices[device_id]
            headers = {}
            if device_config.get('auth_header'):
                headers['Authorization'] = device_config['auth_header']

            response = requests.get(device_config['endpoint'], headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                readings = self._parse_device_data(device_id, data)
                self.device_status[device_id] = DeviceStatus(
                    device_id=device_id,
                    device_type=device_config['device_type'],
                    status='online',
                    last_seen=datetime.utcnow().isoformat(),
                    battery_level=data.get('battery_level'),
                    signal_strength=data.get('signal_strength'),
                    firmware_version=data.get('firmware_version')
                )
                return readings
            else:
                logger.warning(f"HTTP device {device_id} returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Error reading HTTP device {device_id}: {e}")
            self.device_status[device_id] = DeviceStatus(
                device_id=device_id,
                device_type=self.devices.get(device_id, {}).get('device_type', 'unknown'),
                status='offline',
                last_seen=datetime.utcnow().isoformat(),
                battery_level=None,
                signal_strength=None,
                firmware_version=None
            )
        return None

    def _parse_device_data(self, device_id: str, data: Dict) -> List[SensorReading]:
        readings = []
        try:
            if 'sensors' in data:
                for sensor_data in data['sensors']:
                    reading = SensorReading(
                        sensor_id=f"{device_id}_{sensor_data.get('type', 'unknown')}",
                        sensor_type=sensor_data.get('type', 'unknown'),
                        value=sensor_data.get('value', 0),
                        unit=sensor_data.get('unit', ''),
                        timestamp=sensor_data.get('timestamp', datetime.utcnow().isoformat()),
                        metadata={'device_id': device_id},
                        quality_score=sensor_data.get('quality', 1.0)
                    )
                    readings.append(reading)
            else:
                for key, value in data.items():
                    if key in ['temperature', 'humidity', 'pressure', 'light', 'motion']:
                        unit_map = {
                            'temperature': '°C',
                            'humidity': '%',
                            'pressure': 'hPa',
                            'light': 'lux',
                            'motion': 'bool'
                        }
                        reading = SensorReading(
                            sensor_id=f"{device_id}_{key}",
                            sensor_type=key,
                            value=float(value),
                            unit=unit_map.get(key, ''),
                            timestamp=datetime.utcnow().isoformat(),
                            metadata={'device_id': device_id},
                            quality_score=1.0
                        )
                        readings.append(reading)
        except Exception as e:
            logger.error(f"Error parsing device data: {e}")
        return readings

    def get_all_device_status(self) -> List[DeviceStatus]:
        return list(self.device_status.values())

class AdvancedSensorIntegration:
    def __init__(self):
        self.temperature_manager = TemperatureSensorManager()
        self.network_manager = NetworkSensorManager()
        self.monitoring_active = False
        self.monitoring_interval = 30
        self.monitoring_thread = None
        self.all_readings = deque(maxlen=10000)
        self.reading_callbacks = []
        logger.info("Advanced Sensor Integration initialized")

    def start_monitoring(self, interval: int = 30):
        try:
            self.monitoring_interval = interval
            self.monitoring_active = True
            self._discover_all_sensors()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info(f"Started sensor monitoring with {interval}s interval")
        except Exception as e:
            logger.error(f"Error starting sensor monitoring: {e}")

    def stop_monitoring(self):
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped sensor monitoring")

    def _discover_all_sensors(self):
        try:
            temp_sensors = self.temperature_manager.discover_temperature_sensors()
            logger.info(f"Discovered {len(temp_sensors)} temperature sensors")
        except Exception as e:
            logger.error(f"Error discovering sensors: {e}")

    def _monitoring_loop(self):
        while self.monitoring_active:
            try:
                all_readings = []
                temp_readings = self.temperature_manager.get_all_temperature_readings()
                all_readings.extend(temp_readings)
                for device_id in self.network_manager.devices:
                    device_readings = self.network_manager.read_http_device(device_id)
                    if device_readings:
                        all_readings.extend(device_readings)
                for reading in all_readings:
                    self.all_readings.append(reading)
                    for callback in self.reading_callbacks:
                        try:
                            callback(reading)
                        except Exception as e:
                            logger.error(f"Error in reading callback: {e}")
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in sensor monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def add_reading_callback(self, callback: Callable[[SensorReading], None]):
        self.reading_callbacks.append(callback)

    def register_iot_device(self, device_id: str, device_config: Dict[str, Any]):
        try:
            device_type = device_config.get('type', 'http')
            if device_type == 'http':
                self.network_manager.register_http_device(
                    device_id=device_id,
                    endpoint_url=device_config['endpoint'],
                    device_type=device_config.get('device_type', 'generic'),
                    auth_header=device_config.get('auth_header')
                )
            logger.info(f"Registered IoT device {device_id} of type {device_type}")
        except Exception as e:
            logger.error(f"Error registering IoT device {device_id}: {e}")

    def get_latest_readings(self, sensor_type: Optional[str] = None, count: int = 100) -> List[SensorReading]:
        try:
            if sensor_type:
                filtered = [r for r in self.all_readings if r.sensor_type == sensor_type]
                return filtered[-count:]
            return list(self.all_readings)[-count:]
        except Exception as e:
            logger.error(f"Error getting latest readings: {e}")
            return []

    def get_device_summary(self) -> Dict[str, Any]:
        try:
            summary = {
                'temperature_sensors': len(self.temperature_manager.sensors),
                'network_devices': len(self.network_manager.devices),
                'total_readings': len(self.all_readings),
                'monitoring_active': self.monitoring_active,
                'monitoring_interval': self.monitoring_interval,
                'timestamp': datetime.utcnow().isoformat(),
                'device_status': [status.to_dict() for status in self.network_manager.get_all_device_status()]
            }
            return summary
        except Exception as e:
            logger.error(f"Error getting device summary: {e}")
            return {'error': str(e)}

    def export_sensor_data(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        try:
            return [r.to_dict() for r in self.all_readings if start_time <= datetime.fromisoformat(r.timestamp) <= end_time]
        except Exception as e:
            logger.error(f"Error exporting sensor data: {e}")
            return []

sensor_integration = None

def get_sensor_integration():
    global sensor_integration
    if sensor_integration is None:
        sensor_integration = AdvancedSensorIntegration()
    return sensor_integration
