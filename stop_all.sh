#!/bin/bash

echo "========================================"
echo "Stopping Chatbot Services"
echo "========================================"

# Kill ngrok process
echo "[1/2] Stopping ngrok..."
pkill -f "ngrok http" && echo "  ✓ Ngrok stopped" || echo "  ✗ Ngrok not running"

# Kill uvicorn process
echo "[2/2] Stopping server..."
pkill -f "uvicorn chat_fixed:app" && echo "  ✓ Server stopped" || echo "  ✗ Server not running"

echo ""
echo "========================================"
echo "All services stopped!"
echo "========================================"
