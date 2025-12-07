@echo off
echo ========================================
echo Starting Chatbot Services
echo ========================================

REM Start ngrok in a new window
echo [1/2] Starting ngrok on port 8000...
start "Ngrok Tunnel" cmd /k "ngrok http 8000"

REM Wait 3 seconds for ngrok to initialize
timeout /t 3 /nobreak >nul

REM Start uvicorn server in a new window
echo [2/2] Starting FastAPI server...
start "Chatbot Server" cmd /k "cd /d %~dp0 && uv run uvicorn chat_fixed:app --host 0.0.0.0 --port 8000 --workers 1"

echo.
echo ========================================
echo All services started!
echo ========================================
echo.
echo Ngrok: Check the "Ngrok Tunnel" window for URL
echo Server: Check the "Chatbot Server" window for logs
echo.
echo Press any key to exit this launcher...
pause >nul
