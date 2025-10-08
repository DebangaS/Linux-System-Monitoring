"""
Real-time Data Stream Processing
Author: Member 2 
Provides:
 - StreamMessage: dataclass for stream messages
 - DataStream: per-stream buffering, subscriber management, background batching/dispatch
 - StreamManager: manages multiple DataStream instances + websocket (python-socketio) integration
 - DeltaCompressor: simple delta-compression helper with stats
"""
import asyncio
import json
import time
import gzip
import base64
import threading
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import logging
from dataclasses import dataclass, asdict
from queue import Queue, Empty
import inspect

import socketio

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class StreamMessage:
    """Stream message data structure"""
    message_type: str
    data: Dict[str, Any]
    timestamp: str
    source: str
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical


class DataStream:
    """Real-time data stream handler"""

    def __init__(self, stream_name: str, buffer_size: int = 1000):
        self.stream_name = stream_name
        self.buffer_size = buffer_size

        # Stream data
        self.message_buffer: deque = deque(maxlen=buffer_size)
        self._buffer_lock = threading.Lock()

        self.subscribers: List[Dict] = []
        self.filters: List[Callable[[StreamMessage], bool]] = []

        # Statistics
        self.messages_processed = 0
        self.messages_dropped = 0
        self.last_message_time: Optional[datetime] = None

        # Configuration
        self.compression_enabled = True
        self.batch_size = 10
        self.flush_interval = 1.0

        # State
        self.is_active = False
        self.processing_thread: Optional[threading.Thread] = None

        logger.info(f"Data stream '{stream_name}' initialized")

    def add_subscriber(self,
                       callback: Callable[[StreamMessage], None],
                       filter_func: Optional[Callable[[StreamMessage], bool]] = None) -> None:
        """Add a subscriber to the stream"""
        subscriber = {
            'callback': callback,
            'filter': filter_func,
            'added_at': datetime.utcnow(),
            'messages_received': 0
        }
        self.subscribers.append(subscriber)
        logger.info(f"Subscriber added to stream '{self.stream_name}'")

    def remove_subscriber(self, callback: Callable) -> None:
        """Remove a subscriber from the stream"""
        self.subscribers = [s for s in self.subscribers if s['callback'] != callback]

    def add_message(self, message: StreamMessage) -> None:
        """Add a message to the stream"""
        try:
            with self._buffer_lock:
                self.message_buffer.append(message)
            self.messages_processed += 1
            self.last_message_time = datetime.utcnow()

            # Notify subscribers immediately for high priority messages
            if message.priority >= 3:
                self._notify_subscribers(message)
        except Exception as e:
            logger.error(f"Error adding message to stream '{self.stream_name}': {e}")
            self.messages_dropped += 1

    def _notify_subscribers(self, message: StreamMessage) -> None:
        """Notify all subscribers of a new message. Accepts sync or async callbacks."""
        for subscriber in list(self.subscribers):
            try:
                # Apply filter if present
                if subscriber['filter'] and not subscriber['filter'](message):
                    continue

                cb = subscriber['callback']
                if inspect.iscoroutinefunction(cb):
                    # schedule coroutine
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(cb(message))
                    except RuntimeError:
                        # no running loop; run in new background loop
                        def _run_coroutine():
                            asyncio.run(cb(message))
                        threading.Thread(target=_run_coroutine, daemon=True).start()
                else:
                    # sync callback - call in background thread to avoid blocking
                    threading.Thread(target=self._safe_sync_callback, args=(cb, message), daemon=True).start()

                subscriber['messages_received'] += 1
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")

    @staticmethod
    def _safe_sync_callback(cb: Callable[[StreamMessage], None], message: StreamMessage) -> None:
        try:
            cb(message)
        except Exception as exc:
            logger.error(f"Subscriber callback error: {exc}")

    def start_processing(self) -> None:
        """Start background processing thread which batches and dispatches messages"""
        if self.is_active:
            return
        self.is_active = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info(f"Data stream '{self.stream_name}' processing started")

    def stop_processing(self, join_timeout: float = 2.0) -> None:
        """Stop background processing thread"""
        self.is_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=join_timeout)
        logger.info(f"Data stream '{self.stream_name}' processing stopped")

    def _processing_loop(self) -> None:
        """Background loop that flushes messages in batches and notifies subscribers"""
        while self.is_active:
            try:
                batch: List[StreamMessage] = []
                with self._buffer_lock:
                    while len(batch) < self.batch_size and self.message_buffer:
                        try:
                            batch.append(self.message_buffer.popleft())
                        except IndexError:
                            break

                # Notify subscribers for non-high-priority messages (high already immediately notified)
                for msg in batch:
                    if msg.priority < 3:
                        self._notify_subscribers(msg)

                time.sleep(self.flush_interval)
            except Exception as e:
                logger.error(f"DataStream processing error for '{self.stream_name}': {e}")
                time.sleep(self.flush_interval)

    def get_recent_messages(self, count: int = 10, since: Optional[datetime] = None) -> List[StreamMessage]:
        """Get recent messages from the buffer"""
        with self._buffer_lock:
            messages = list(self.message_buffer)
        if since:
            filtered = []
            for m in messages:
                try:
                    m_time = datetime.fromisoformat(m.timestamp.replace('Z', '+00:00'))
                except Exception:
                    try:
                        m_time = datetime.fromisoformat(m.timestamp)
                    except Exception:
                        continue
                if m_time > since:
                    filtered.append(m)
            messages = filtered
        return messages[-count:] if count else messages

    def get_statistics(self) -> Dict:
        """Get stream statistics"""
        return {
            'stream_name': self.stream_name,
            'messages_processed': self.messages_processed,
            'messages_dropped': self.messages_dropped,
            'current_buffer_size': len(self.message_buffer),
            'max_buffer_size': self.buffer_size,
            'subscriber_count': len(self.subscribers),
            'last_message_time': self.last_message_time.isoformat() + "Z" if self.last_message_time else None,
            'is_active': self.is_active,
            'subscriber_stats': [
                {
                    'added_at': s['added_at'].isoformat() + "Z",
                    'messages_received': s['messages_received']
                } for s in self.subscribers
            ]
        }


class StreamManager:
    """Manager for multiple data streams and websocket integration"""

    def __init__(self):
        self.streams: Dict[str, DataStream] = {}
        self.global_stats = {
            'total_messages': 0,
            'total_subscribers': 0,
            'streams_active': 0
        }

        # WebSocket integration (python-socketio)
        # The ASGI/Wsgi app integration is left to the user; we just expose the AsyncServer instance here.
        self.sio = socketio.AsyncServer(cors_allowed_origins="*")
        self.websocket_clients: Dict[str, Dict] = {}
        self._setup_websocket_handlers()
        logger.info("Stream manager initialized")

    def create_stream(self, stream_name: str, buffer_size: int = 1000) -> DataStream:
        """Create a new data stream"""
        if stream_name in self.streams:
            return self.streams[stream_name]
        stream = DataStream(stream_name, buffer_size)
        # start background processing for stream by default
        stream.start_processing()
        self.streams[stream_name] = stream
        logger.info(f"Stream '{stream_name}' created")
        return stream

    def get_stream(self, stream_name: str) -> Optional[DataStream]:
        """Get an existing stream"""
        return self.streams.get(stream_name)

    def publish_message(self, stream_name: str, message_type: str,
                        data: Dict, source: str = 'system', priority: int = 1) -> None:
        """Publish a message to a stream"""
        stream = self.get_stream(stream_name)
        if not stream:
            logger.warning(f"Stream '{stream_name}' not found")
            return

        message = StreamMessage(
            message_type=message_type,
            data=data,
            timestamp=datetime.utcnow().isoformat() + "Z",
            source=source,
            priority=priority
        )
        stream.add_message(message)
        self.global_stats['total_messages'] += 1

        # Send to WebSocket clients if connected (schedule safely)
        coro = self._send_to_websocket_clients(stream_name, message)
        self._schedule_coroutine(coro)

    def _schedule_coroutine(self, coro: asyncio.coroutine) -> None:
        """Schedule a coroutine for execution, whether or not an event loop is currently running"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            # no running loop — run in a dedicated background loop thread
            def _runner():
                new_loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(new_loop)
                    new_loop.run_until_complete(coro)
                finally:
                    try:
                        new_loop.close()
                    except Exception:
                        pass

            threading.Thread(target=_runner, daemon=True).start()

    def _setup_websocket_handlers(self) -> None:
        """Setup WebSocket event handlers"""
        @self.sio.event
        async def connect(sid, environ):
            logger.info(f"WebSocket client connected: {sid}")
            self.websocket_clients[sid] = {
                'connected_at': datetime.utcnow(),
                'subscriptions': [],
                'messages_sent': 0
            }
            await self.sio.emit('connection_established', {
                'client_id': sid,
                'server_time': datetime.utcnow().isoformat() + "Z",
                'available_streams': list(self.streams.keys())
            }, room=sid)

        @self.sio.event
        async def disconnect(sid):
            logger.info(f"WebSocket client disconnected: {sid}")
            if sid in self.websocket_clients:
                del self.websocket_clients[sid]

        @self.sio.event
        async def subscribe_stream(sid, data):
            """Handle stream subscription from client"""
            stream_name = data.get('stream_name')
            filters = data.get('filters', {})
            if sid in self.websocket_clients:
                client_info = self.websocket_clients[sid]
                if stream_name not in client_info['subscriptions']:
                    client_info['subscriptions'].append(stream_name)
                await self.sio.emit('subscription_confirmed', {
                    'stream_name': stream_name,
                    'filters': filters
                }, room=sid)
                logger.info(f"Client {sid} subscribed to stream '{stream_name}'")

        @self.sio.event
        async def unsubscribe_stream(sid, data):
            """Handle stream unsubscription from client"""
            stream_name = data.get('stream_name')
            if sid in self.websocket_clients:
                client_info = self.websocket_clients[sid]
                if stream_name in client_info['subscriptions']:
                    client_info['subscriptions'].remove(stream_name)
                await self.sio.emit('unsubscription_confirmed', {
                    'stream_name': stream_name
                }, room=sid)
                logger.info(f"Client {sid} unsubscribed from stream '{stream_name}'")

        @self.sio.event
        async def get_stream_stats(sid, data):
            """Send stream statistics to client"""
            stream_name = data.get('stream_name')
            if stream_name in self.streams:
                stats = self.streams[stream_name].get_statistics()
                await self.sio.emit('stream_stats', stats, room=sid)
            else:
                await self.sio.emit('error', {
                    'message': f"Stream '{stream_name}' not found"
                }, room=sid)

    async def _send_to_websocket_clients(self, stream_name: str, message: StreamMessage) -> None:
        """Send message to subscribed WebSocket clients"""
        if not self.websocket_clients:
            return

        message_dict = asdict(message)
        message_data = {
            'stream_name': stream_name,
            'message': message_dict
        }

        # Compress large messages (compress payload only)
        payload_json = json.dumps(message_dict['data'])
        if len(payload_json) > 1024:
            compressed = gzip.compress(payload_json.encode())
            message_data['compressed'] = True
            message_data['data_base64'] = base64.b64encode(compressed).decode('ascii')
            # Keep a small metadata version in 'message' for quick display
            message_data['message']['data'] = {'_compressed': True, 'original_size': len(payload_json)}
        else:
            message_data['compressed'] = False

        # emit to subscribed clients
        for client_id, client_info in list(self.websocket_clients.items()):
            if stream_name in client_info.get('subscriptions', []):
                try:
                    await self.sio.emit('stream_message', message_data, room=client_id)
                    client_info['messages_sent'] += 1
                except Exception as e:
                    logger.error(f"Error sending to WebSocket client {client_id}: {e}")

    def get_global_statistics(self) -> Dict:
        """Get global stream statistics"""
        self.global_stats.update({
            'total_subscribers': sum(len(s.subscribers) for s in self.streams.values()),
            'streams_active': len([s for s in self.streams.values() if s.is_active]),
            'websocket_clients': len(self.websocket_clients),
            'streams': {name: stream.get_statistics() for name, stream in self.streams.items()}
        })
        return dict(self.global_stats)


class DeltaCompressor:
    """Delta compression for efficient data transmission"""

    def __init__(self):
        self.last_values: Dict[str, Dict] = {}
        self.compression_stats = defaultdict(lambda: {'original_size': 0, 'compressed_size': 0})

    def compress_data(self, stream_name: str, data: Dict) -> Dict:
        """Compress data using delta compression"""
        if stream_name not in self.last_values:
            self.last_values[stream_name] = dict(data)
            return {'type': 'full', 'data': data}

        delta: Dict = {}
        last = self.last_values[stream_name]
        for key, value in data.items():
            if key not in last or last[key] != value:
                delta[key] = value

        # Update last values (shallow copy)
        self.last_values[stream_name] = dict(data)

        original_size = len(json.dumps(data))
        compressed_size = len(json.dumps(delta))
        stats = self.compression_stats[stream_name]
        stats['original_size'] += original_size
        stats['compressed_size'] += compressed_size

        return {
            'type': 'delta',
            'data': delta,
            'compression_ratio': (compressed_size / original_size) if original_size > 0 else 1.0
        }

    def get_compression_stats(self) -> Dict:
        """Get compression statistics"""
        return dict(self.compression_stats)


# Global instances
stream_manager = StreamManager()
delta_compressor = DeltaCompressor()


if __name__ == "__main__":
    # Minimal smoke-test / demo
    logging.basicConfig(level=logging.INFO)
    sm = stream_manager
    stream = sm.create_stream("telemetry", buffer_size=200)

    # sample subscriber
    def print_subscriber(msg: StreamMessage):
        print(f"[SUB] {msg.timestamp} {msg.message_type} from {msg.source} priority={msg.priority}")

    stream.add_subscriber(print_subscriber)

    # publish some messages
    for i in range(5):
        sm.publish_message("telemetry", "metric.update", {"value": i, "i": i}, source="demo", priority=1)
        time.sleep(0.2)

    # publish a high-priority message (immediate notify)
    sm.publish_message("telemetry", "alert", {"msg": "high cpu"}, source="demo", priority=4)

    # wait a bit to allow background processing to run
    time.sleep(2)
    print("Global stats:", json.dumps(sm.get_global_statistics(), indent=2, default=str))
