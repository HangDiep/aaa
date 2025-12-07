#!/bin/bash

echo "========================================"
echo "Starting Chatbot Services"
echo "========================================"

# Start ngrok in background
echo "[1/2] Starting ngrok on port 8000..."
ngrok http 8000 &
NGROK_PID=$!

# Wait 3 seconds for ngrok to initialize
sleep 3

# Start uvicorn server
echo "[2/2] Starting FastAPI server..."
cd "$(dirname "$0")"
uv run uvicorn chat_fixed:app --host 0.0.0.0 --port 8000 --workers 1 &
SERVER_PID=$!

echo ""
echo "========================================"
echo "All services started!"
echo "========================================"
echo ""
echo "Ngrok PID: $NGROK_PID"
echo "Server PID: $SERVER_PID"
echo ""
echo "To view ngrok URL: http://localhost:4040"
echo "To stop services: ./stop_all.sh"
echo ""
echo "Press Ctrl+C to stop this script (services will keep running)"
echo ""

# Wait for user interrupt
wait
