#!/bin/bash

echo "========================================"
echo "Checking Ngrok URL"
echo "========================================"
echo ""
echo "Opening ngrok web interface..."
echo ""

# Try different browsers
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:4040
elif command -v open &> /dev/null; then
    open http://localhost:4040
elif command -v start &> /dev/null; then
    start http://localhost:4040
else
    echo "Please open this URL in your browser:"
    echo "http://localhost:4040"
fi

echo ""
echo "Look for the 'Forwarding' URL (https://...ngrok-free.app)"
echo "Copy that URL and use it in your n8n workflow!"
echo ""
