# API Documentation - Day 2 Updates

## New Endpoints Added

### System Resources
- **GET** `/api/v1/system/resources` - Get all system resource data
- **GET** `/api/v1/system/cpu` - Get CPU usage details
- **GET** `/api/v1/system/memory` - Get memory usage details
- **GET** `/api/v1/system/disk` - Get disk usage details
- **GET** `/api/v1/system/network` - Get network I/O details

### Process Management
- **GET** `/api/v1/system/processes` - Get running processes
- **GET** `/api/v1/system/processes/{pid}` - Get specific process details
- **POST** `/api/v1/system/processes/{pid}/kill` - Terminate process

### System Information
- **GET** `/api/v1/system/info` - Get system information
- **GET** `/api/v1/system/alerts` - Get system alerts
- **GET** `/api/v1/system/history` - Get historical data

## Response Format
All endpoints return JSON with success/error status:
{
"success": true,
"data": {...},
"timestamp": "2025-10-06T14:30:00.000Z"
}

## WebSocket Events
- `system_update` - Real-time system data updates
- `system_alerts` - Alert notifications
- `processes_update` - Process data updates
