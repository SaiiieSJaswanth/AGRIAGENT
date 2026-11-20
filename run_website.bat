@echo off
title AGRIAGENT 3D Agricultural Intelligence System
echo =====================================================================
echo           AGRIAGENT 3D AGRICULTURAL INTELLIGENCE SYSTEM
echo =====================================================================
echo.
echo [1/2] Starting AGRIAGENT Local 3D Web Server...
echo [2/2] Opening browser at: http://127.0.0.1:5000
echo.
echo Press Ctrl+C in this window anytime to stop the server.
echo =====================================================================
echo.

start "" http://127.0.0.1:5000
python app.py
if errorlevel 1 (
    echo.
    echo Server stopped or error occurred.
    pause
)
