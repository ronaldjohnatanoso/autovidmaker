#!/bin/bash


PORT=9222

echo "🛑 Looking for Chrome processes with remote debugging on port $PORT..."
PIDS=$(ps aux | grep "chrome.*--remote-debugging-port=$PORT" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
  echo "✅ No Chrome instance using remote debugging on port $PORT."
else
  echo "🔍 Matching processes:"
  ps aux | grep "chrome.*--remote-debugging-port=$PORT" | grep -v grep

  echo "🔪 Killing all matching Chrome processes..."
  for pid in $PIDS; do
    kill -9 "$pid" && echo "🔪 Killed PID $pid"
  done
fi
