@echo off
echo.
echo ========================================
echo   Linux System Monitoring Frontend
echo ========================================
echo.
echo Starting the system monitor with frontend...
echo.

set APP_ENABLE_BACKGROUND=1
set FLASK_ENV=production

python start_frontend.py

pause
