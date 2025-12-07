@echo off
echo ========================================
echo Checking Ngrok URL
echo ========================================
echo.
echo Opening ngrok web interface...
echo.
start http://localhost:4040
echo.
echo The ngrok dashboard will open in your browser.
echo Look for the "Forwarding" URL (https://...ngrok-free.app)
echo.
echo Copy that URL and use it in your n8n workflow!
echo.
pause
