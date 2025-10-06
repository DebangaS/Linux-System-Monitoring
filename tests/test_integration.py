"""
Integration tests for system monitoring components
Author: Member 5
"""
import unittest
import time
from app import create_app

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up test environment"""
        self.app, self.socketio = create_app('development')
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        self.socketio_client = self.socketio.test_client(self.app)
    
    def test_app_startup_with_monitoring(self):
        """Test app starts successfully with monitoring enabled"""
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertEqual(data['status'], 'running')
        self.assertIn('monitoring_active', data)
    
    def test_system_resources_endpoint(self):
        """Test system resources API endpoint"""
        response = self.client.get('/api/v1/system/resources')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn('data', data)
        
        # Check all resource types are present
        resources = data['data']
        self.assertIn('cpu', resources)
        self.assertIn('memory', resources)
        self.assertIn('disk', resources)
        self.assertIn('network', resources)
    
    def test_websocket_connection(self):
        """Test WebSocket connectivity"""
        received = self.socketio_client.get_received()
        self.assertGreater(len(received), 0)
    
    def test_real_time_data_flow(self):
        """Test real-time data updates via WebSocket"""
        # Emit request for system data
        self.socketio_client.emit('request_system_data')
        
        # Wait for response
        time.sleep(1)
        received = self.socketio_client.get_received()
        
        # Should receive system update
        system_updates = [msg for msg in received if msg['name'] == 'system_update']
        self.assertGreater(len(system_updates), 0)

if __name__ == '__main__':
    unittest.main()
