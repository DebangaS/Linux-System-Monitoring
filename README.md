# Linux System Monitoring

A comprehensive real-time system monitoring dashboard with web frontend.

## ✨ Features

- 📊 **Real-time Dashboard** - Live CPU, Memory, Disk, Network monitoring
- 📈 **Historical Data** - View performance trends over time
- 🔍 **Analytics** - Advanced system insights and health indicators
- 💾 **Database Storage** - SQLite database for metrics persistence
- 🌐 **Web Frontend** - Beautiful, responsive web interface
- ⚡ **WebSocket Updates** - Real-time data streaming
- 🧪 **Comprehensive Tests** - Full test suite included

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/your-username/Linux-System-Monitoring.git
cd Linux-System-Monitoring
pip install -r requirements.txt
```

### 2. Run with Frontend
```bash
python start_frontend.py
```

### 3. Access Dashboard
- **Home:** http://127.0.0.1:5000/
- **Dashboard:** http://127.0.0.1:5000/dashboard
- **History:** http://127.0.0.1:5000/history
- **Analytics:** http://127.0.0.1:5000/analytics

## 🧪 Testing

Run the comprehensive test suite:
```bash
python -m pytest tests/test_comprehensive.py -v
```

## 🏗️ GitHub Actions

Minimal CI/CD workflow that:
- ✅ Runs tests on push/PR to main branch
- ✅ Uses Python 3.12
- ✅ Simple and fast execution

## 📁 Project Structure

```
Linux-System-Monitoring/
├── app.py                 # Main Flask application
├── start_frontend.py      # Easy startup script
├── requirements.txt       # Dependencies
├── templates/            # HTML templates
│   ├── index.html
│   ├── dashboard.html
│   ├── history.html
│   └── analytics.html
├── api/                  # API routes
├── database/             # Database models
├── monitors/             # System monitoring
├── tests/               # Test suite
│   └── test_comprehensive.py
└── .github/workflows/   # GitHub Actions
    └── ci-cd.yml
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/test_comprehensive.py`
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**Happy Monitoring! 🚀**
