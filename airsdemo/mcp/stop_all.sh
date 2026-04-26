#!/bin/bash
echo "🛑 Stopping all MCP servers and Multi-Agent Demo..."
if [ -f .pids ]; then
    PIDS=$(cat .pids)
    echo "   Killing processes: $PIDS"
    kill $PIDS 2>/dev/null
    rm .pids
    echo "✅ All servers stopped"
else
    echo "⚠️  No .pids file found. Killing by port..."
    # Kill processes on known ports
    for port in 8079 9001 9002 9003 9004 9005 9006; do
        PID=$(lsof -ti:$port 2>/dev/null)
        if [ ! -z "$PID" ]; then
            echo "   Killing process on port $port (PID: $PID)"
            kill $PID 2>/dev/null
        fi
    done
    echo "✅ Done"
fi