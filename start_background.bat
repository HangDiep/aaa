@echo off
echo ========================================
echo Starting Chatbot Services (Background)
echo ========================================

REM Start ngrok in background (minimized)
echo [1/2] Starting ngrok (minimized)...
start /min "Ngrok" cmd /c "ngrok http 8000"

REM Wait 3 seconds
timeout /t 3 /nobreak >nul

REM Start server in background (minimized)
echo [2/2] Starting server (minimized)...
start /min "Server" cmd /c "cd /d %~dp0 && uv run uvicorn chat_fixed:app --host 0.0.0.0 --port 8000 --workers 1"

echo.
echo ========================================
echo Services started in background!
echo ========================================
echo.
echo To view ngrok URL:
echo   1. Open http://localhost:4040 in browser
echo   2. Or check Task Manager for "ngrok" process
echo.
echo To stop services:
echo   Run: stop_all.bat
echo.
pause
