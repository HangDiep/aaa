@echo off
echo ========================================
echo Stopping Chatbot Services
echo ========================================

REM Kill ngrok process
echo [1/2] Stopping ngrok...
taskkill /F /IM ngrok.exe >nul 2>&1
if %errorlevel% == 0 (
    echo   ✓ Ngrok stopped
) else (
    echo   ✗ Ngrok not running
)

REM Kill Python/uvicorn process (be careful - this kills ALL python processes)
echo [2/2] Stopping server...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Chatbot Server*" >nul 2>&1
if %errorlevel% == 0 (
    echo   ✓ Server stopped
) else (
    echo   ✗ Server not running
)

echo.
echo ========================================
echo All services stopped!
echo ========================================
pause
