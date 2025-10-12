#!/usr/bin/env python3
"""
Startup script for Linux System Monitoring with Frontend
Author: System
"""

import os
import sys

def main():
    """Start the system monitoring application with frontend"""
    
    print("🚀 Starting Linux System Monitoring with Frontend...")
    print("=" * 50)
    
    # Set environment variables for production mode
    os.environ['APP_ENABLE_BACKGROUND'] = '1'
    os.environ['FLASK_ENV'] = 'production'
    
    # Import and run the app
    try:
        from app import create_app
        
        print("✅ Creating Flask application...")
        app, socketio = create_app('production')
        
        print("✅ Initializing database...")
        print("✅ Starting background monitoring...")
        print("✅ Setting up real-time data streaming...")
        
        print("\n🌐 Frontend URLs:")
        print("   Home Page:     http://127.0.0.1:5000/")
        print("   Dashboard:     http://127.0.0.1:5000/dashboard")
        print("   History:       http://127.0.0.1:5000/history")
        print("   Analytics:     http://127.0.0.1:5000/analytics")
        
        print("\n🔗 API Endpoints:")
        print("   Status:        http://127.0.0.1:5000/api/status")
        print("   Resources:     http://127.0.0.1:5000/api/v1/system/resources")
        print("   Processes:     http://127.0.0.1:5000/api/v1/system/processes")
        print("   Database:      http://127.0.0.1:5000/api/database/stats")
        
        print("\n" + "=" * 50)
        print("🎯 System Monitor is running!")
        print("   Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Start the server
        socketio.run(
            app, 
            host='127.0.0.1', 
            port=5000, 
            debug=False,
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down System Monitor...")
        print("✅ Server stopped successfully")
        
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
