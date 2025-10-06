# Linux-System-Monitoring
# Real-Time System Resource Monitoring Dashboard

A simple, real-time web dashboard to monitor system resources (CPU, RAM, Disk, Network).

## Features
- Live CPU, Memory, Disk, Network charts
- System info display
- Alerts for thresholds
- Export data (CSV/JSON)
- WebSocket for real-time updates
- Mobile-friendly design

## Setup Instructions

### Clone the Repository
git clone https://github.com/your-username/resource-monitoring.git
cd resource-monitoring
### Create a Virtual Environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

### Install Dependencies
pip install -r requirements.txt

### Run the Application
python run.py

### Access the Dashboard
Open your browser and go to:
- [http://localhost:5000](http://localhost:5000)

## Usage
- View real-time system metrics on the dashboard.
- Access system info, alerts, and historical data.
- Filter and export data.

## Testing
Run the initial tests:
python -m unittest discover tests/

## Project Structure
resource-monitoring/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── run.py
├── config.py
├── monitors/
│ ├── init.py
│ └── system_monitor.py
├── templates/
│ ├── base.html
│ └── index.html
├── static/
│ └── ...
├── api/
│ ├── init.py
│ └── routes.py
└── tests/
└── test_basic.py

## Contribution
- Create feature branches
- Push code regularly
- Open pull requests for review
- Follow coding standards
  
## License
This project is licensed under MIT License.

## Support
- Submit issues for bugs or enhancements
- Contact the team via GitHub discussions

---
**End of setup instructions.**

## Day 2 Updates - Enhanced Monitoring

### New Features Added
- ✅ **Real-time system monitoring** with psutil integration
- ✅ **CPU monitoring** - Usage, per-core, frequency, detailed metrics
- ✅ **Memory monitoring** - RAM, swap, detailed breakdown
- ✅ **Disk monitoring** - All partitions, usage statistics
- ✅ **Network monitoring** - I/O rates, data transfer
- ✅ **Process management** - List, filter, search, terminate
- ✅ **System alerts** - Threshold-based notifications
- ✅ **Enhanced logging** - Multi-level, rotation, archiving
- ✅ **WebSocket updates** - Real-time data streaming

### Testing
Run all Day 2 tests:
python -m unittest discover tests/ -v
python -m unittest tests.test_integration
python -m unittest tests.test_performance

### Performance Metrics
- CPU monitoring: < 1 second response time
- Memory monitoring: < 0.5 second response time  
- Process listing: < 2 seconds for 50 processes
- WebSocket updates: Real-time (2-second intervals)
Commit Message: "Day 2: Add comprehensive testing suite for system monitoring features

Integration tests for API endpoints and WebSocket functionality

Performance tests for monitoring functions and concurrent access

System monitoring tests with validation and edge cases

Updated documentation with new API endpoints and features

Load testing for real-time data updates and concurrent requests

Test Coverage:
✅ System resource monitoring (CPU, Memory, Disk, Network)
✅ Process management and filtering
✅ API endpoint integration
✅ WebSocket real-time communication
✅ Performance and concurrency validation
✅ Error handling and edge cases

Ready for Day 3: Database integration testing"
