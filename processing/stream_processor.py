"""
Real-Time Data Processing & Stream Analytics
Author: Member 2
"""
import asyncio
import time
import threading
import logging
import json
import statistics
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue

logger = logging.getLogger(__name__)

@dataclass
class StreamEvent:
    """Real-time stream event"""
    event_id: str
    event_type: str
    source: str
    data: Dict[str, Any]
    timestamp: str
    processing_latency_ms: float = 0.0

    def to_dict(self):
        return asdict(self)

@dataclass
class ProcessingResult:
    """Stream processing result"""
    result_id: str
    input_events: List[str]
    processor_name: str
    result_data: Dict[str, Any]
    processing_time_ms: float
    timestamp: str

    def to_dict(self):
        return asdict(self)

@dataclass
class StreamMetrics:
    """Stream processing metrics"""
    events_processed: int
    events_per_second: float
    average_latency_ms: float
    error_count: int
    error_rate: float
    timestamp: str

    def to_dict(self):
        return asdict(self)

class StreamProcessor:
    """Individual stream processor"""
    def __init__(self, name: str, process_func: Callable[[List[StreamEvent]], ProcessingResult]):
        self.name = name
        self.process_func = process_func
        self.input_queue = queue.Queue(maxsize=1000)
        self.output_callbacks = []
        self.metrics = {
            'processed': 0,
            'errors': 0,
            'total_processing_time': 0.0,
            'last_processed': None
        }
        self.active = False
        self.processing_thread = None

    def start(self):
        if self.active:
            return
        self.active = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info(f"Started stream processor: {self.name}")

    def stop(self):
        self.active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        logger.info(f"Stopped stream processor: {self.name}")

    def add_event(self, event: StreamEvent):
        try:
            self.input_queue.put_nowait(event)
        except queue.Full:
            logger.warning(f"Input queue full for processor {self.name}")

    def add_output_callback(self, callback: Callable[[ProcessingResult], None]):
        self.output_callbacks.append(callback)

    def _processing_loop(self):
        while self.active:
            try:
                events = []
                timeout = 0.1
                try:
                    first_event = self.input_queue.get(timeout=timeout)
                    events.append(first_event)
                except queue.Empty:
                    continue

                while len(events) < 100 and not self.input_queue.empty():
                    try:
                        event = self.input_queue.get_nowait()
                        events.append(event)
                    except queue.Empty:
                        break

                if events:
                    start_time = time.time()
                    try:
                        result = self.process_func(events)
                        processing_time = (time.time() - start_time) * 1000
                        result.processing_time_ms = processing_time

                        self.metrics['processed'] += len(events)
                        self.metrics['total_processing_time'] += processing_time
                        self.metrics['last_processed'] = datetime.utcnow()

                        for callback in self.output_callbacks:
                            try:
                                callback(result)
                            except Exception as e:
                                logger.error(f"Error in output callback: {e}")

                    except Exception as e:
                        logger.error(f"Error in processor {self.name}: {e}")
                        self.metrics['errors'] += 1
            except Exception as e:
                logger.error(f"Error in processing loop for {self.name}: {e}")
                time.sleep(1)

    def get_metrics(self) -> Dict[str, Any]:
        processed = self.metrics['processed']
        total_time = self.metrics['total_processing_time']
        return {
            'name': self.name,
            'events_processed': processed,
            'errors': self.metrics['errors'],
            'average_processing_time_ms': total_time / processed if processed > 0 else 0,
            'last_processed': self.metrics['last_processed'].isoformat() if self.metrics['last_processed'] else None,
            'queue_size': self.input_queue.qsize(),
            'active': self.active
        }

class RealTimeStreamProcessor:
    def __init__(self):
        self.processors = {}
        self.event_history = deque(maxlen=10000)
        self.result_history = deque(maxlen=5000)
        self.stream_metrics = {
            'events_received': 0,
            'events_processed': 0,
            'processing_errors': 0,
            'start_time': None
        }
        self.event_routing = defaultdict(list)
        logger.info("Real-time stream processor initialized")

    def start_processing(self):
        try:
            self.stream_metrics['start_time'] = datetime.utcnow()
            self._initialize_builtin_processors()
            for processor in self.processors.values():
                processor.start()
            logger.info("Stream processing started")
        except Exception as e:
            logger.error(f"Error starting stream processing: {e}")

    def stop_processing(self):
        for processor in self.processors.values():
            processor.stop()
        logger.info("Stream processing stopped")

    def _initialize_builtin_processors(self):
        try:
            self.add_processor('system_aggregator', self._create_system_aggregator_processor())
            self.add_processor('anomaly_detector', self._create_anomaly_detector_processor())
            self.add_processor('performance_calculator', self._create_performance_calculator_processor())
            self.add_processor('alert_generator', self._create_alert_generator_processor())

            self.route_events('system_metric', ['system_aggregator', 'anomaly_detector'])
            self.route_events('sensor_reading', ['performance_calculator'])
            self.route_events('anomaly_detected', ['alert_generator'])
        except Exception as e:
            logger.error(f"Error initializing processors: {e}")

    def _create_system_aggregator_processor(self):
        def process_system_metrics(events: List[StreamEvent]) -> ProcessingResult:
            try:
                metric_groups = defaultdict(list)
                for event in events:
                    if event.event_type == 'system_metric':
                        metric_type = event.data.get('metric_type')
                        value = event.data.get('value')
                        if metric_type and value is not None:
                            metric_groups[metric_type].append(value)

                aggregations = {}
                for metric_type, values in metric_groups.items():
                    if values:
                        aggregations[metric_type] = {
                            'count': len(values),
                            'mean': statistics.mean(values),
                            'min': min(values),
                            'max': max(values),
                            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                            'median': statistics.median(values)
                        }

                return ProcessingResult(
                    result_id=f"agg_{int(time.time() * 1000)}",
                    input_events=[e.event_id for e in events],
                    processor_name='system_aggregator',
                    result_data={'aggregations': aggregations},
                    processing_time_ms=0,
                    timestamp=datetime.utcnow().isoformat()
                )
            except Exception as e:
                logger.error(f"Error in system aggregator: {e}")
                raise
        return process_system_metrics

    def _create_anomaly_detector_processor(self):
        thresholds = {
            'cpu': {'warning': 80, 'critical': 95},
            'memory': {'warning': 85, 'critical': 95},
            'disk': {'warning': 90, 'critical': 98},
            'temperature': {'warning': 70, 'critical': 85}
        }
        def detect(events: List[StreamEvent]) -> ProcessingResult:
            anomalies = []
            for e in events:
                if e.event_type == 'system_metric':
                    metric = e.data.get('metric_type')
                    value = e.data.get('value', 0)
                    if metric in thresholds:
                        t = thresholds[metric]
                        severity = 'critical' if value >= t['critical'] else 'warning' if value >= t['warning'] else None
                        if severity:
                            anomalies.append({
                                'event_id': e.event_id,
                                'metric_type': metric,
                                'value': value,
                                'severity': severity,
                                'threshold': t[severity],
                                'timestamp': e.timestamp
                            })
            return ProcessingResult(
                result_id=f"anom_{int(time.time() * 1000)}",
                input_events=[e.event_id for e in events],
                processor_name='anomaly_detector',
                result_data={'anomalies': anomalies},
                processing_time_ms=0,
                timestamp=datetime.utcnow().isoformat()
            )
        return detect

    def _create_performance_calculator_processor(self):
        def calc(events: List[StreamEvent]) -> ProcessingResult:
            latencies = [e.processing_latency_ms for e in events if e.processing_latency_ms > 0]
            performance = {}
            if latencies:
                performance['latency'] = {
                    'average_ms': statistics.mean(latencies),
                    'min_ms': min(latencies),
                    'max_ms': max(latencies),
                    'percentile_95': float(np.percentile(latencies, 95))
                }
            if len(events) > 1:
                t0 = datetime.fromisoformat(events[0].timestamp)
                t1 = datetime.fromisoformat(events[-1].timestamp)
                span = (t1 - t0).total_seconds() or 1
                performance['throughput'] = {
                    'events_per_second': len(events) / span,
                    'total_events': len(events),
                    'time_span_seconds': span
                }
            return ProcessingResult(
                result_id=f"perf_{int(time.time() * 1000)}",
                input_events=[e.event_id for e in events],
                processor_name='performance_calculator',
                result_data=performance,
                processing_time_ms=0,
                timestamp=datetime.utcnow().isoformat()
            )
        return calc

    def _create_alert_generator_processor(self):
        def gen(events: List[StreamEvent]) -> ProcessingResult:
            alerts = []
            for e in events:
                if e.event_type == 'anomaly_detected':
                    severity = e.data.get('severity', 'info')
                    metric = e.data.get('metric_type', 'unknown')
                    value = e.data.get('value', 0)
                    alerts.append({
                        'alert_id': f"alert_{int(time.time() * 1000)}_{metric}",
                        'type': 'anomaly',
                        'severity': severity,
                        'message': f"{metric.upper()} anomaly detected: {value}",
                        'metric_type': metric,
                        'value': value,
                        'timestamp': e.timestamp,
                        'source_event': e.event_id
                    })
            return ProcessingResult(
                result_id=f"alert_{int(time.time() * 1000)}",
                input_events=[e.event_id for e in events],
                processor_name='alert_generator',
                result_data={'alerts': alerts},
                processing_time_ms=0,
                timestamp=datetime.utcnow().isoformat()
            )
        return gen

    def add_processor(self, name: str, func: Callable[[List[StreamEvent]], ProcessingResult]):
        p = StreamProcessor(name, func)
        p.add_output_callback(self._handle_processor_result)
        self.processors[name] = p

    def route_events(self, event_type: str, processors: List[str]):
        self.event_routing[event_type] = processors

    def add_event(self, event: StreamEvent):
        self.event_history.append(event)
        for p_name in self.event_routing.get(event.event_type, []):
            if p_name in self.processors:
                self.processors[p_name].add_event(event)
        self.stream_metrics['events_received'] += 1

    def _handle_processor_result(self, result: ProcessingResult):
        self.result_history.append(result)
        self.stream_metrics['events_processed'] += len(result.input_events)

    def get_stream_metrics(self) -> StreamMetrics:
        try:
            uptime = (datetime.utcnow() - self.stream_metrics['start_time']).total_seconds() if self.stream_metrics['start_time'] else 1
            eps = self.stream_metrics['events_processed'] / uptime
            error_rate = self.stream_metrics['processing_errors'] / max(1, self.stream_metrics['events_processed'])
            latencies = [e.processing_latency_ms for e in list(self.event_history)[-1000:] if e.processing_latency_ms > 0]
            avg_lat = statistics.mean(latencies) if latencies else 0
            return StreamMetrics(
                events_processed=self.stream_metrics['events_processed'],
                events_per_second=eps,
                average_latency_ms=avg_lat,
                error_count=self.stream_metrics['processing_errors'],
                error_rate=error_rate,
                timestamp=datetime.utcnow().isoformat()
            )
        except Exception as e:
            logger.error(f"Error getting stream metrics: {e}")
            return StreamMetrics(0, 0, 0, 0, 0, datetime.utcnow().isoformat())

    def get_processor_metrics(self) -> Dict[str, Dict[str, Any]]:
        return {n: p.get_metrics() for n, p in self.processors.items()}

    def get_recent_results(self, count: int = 100) -> List[ProcessingResult]:
        return list(self.result_history)[-count:]

    def get_event_summary(self) -> Dict[str, Any]:
        try:
            types = defaultdict(int)
            sources = defaultdict(int)
            for e in self.event_history:
                types[e.event_type] += 1
                sources[e.source] += 1
            return {
                'total_events': len(self.event_history),
                'event_types': dict(types),
                'sources': dict(sources),
                'processors': len(self.processors),
                'active_processors': sum(1 for p in self.processors.values() if p.active),
                'routing_rules': len(self.event_routing),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting event summary: {e}")
            return {'error': str(e)}

global_stream_processor = None

def get_stream_processor():
    global global_stream_processor
    if global_stream_processor is None:
        global_stream_processor = RealTimeStreamProcessor()
    return global_stream_processor
