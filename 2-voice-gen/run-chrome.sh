#!/bin/bash

# Path to the cloned profile
PROFILE_DIR="./puppeteer-profile"

# Check if profile exists
if [ ! -d "$PROFILE_DIR" ]; then
  echo "❌ Profile directory '$PROFILE_DIR' not found. Run the copy script first."
  exit 1
fi

# Kill any existing Chrome instances using remote debugging port 9222
EXISTING_PIDS=$(ps aux | grep "chrome.*--remote-debugging-port=9222" | grep -v grep | awk '{print $2}')

if [ -n "$EXISTING_PIDS" ]; then
  echo "🛑 Found Chrome using remote debugging port 9222. Terminating..."
  echo "$EXISTING_PIDS" | xargs kill -9
  sleep 1
fi

# Default Chrome flags
CHROME_FLAGS="--remote-debugging-port=9222 --user-data-dir=\"$PWD/puppeteer-profile\""

# Parse arguments
for arg in "$@"; do
  case $arg in
    --headless)
      CHROME_FLAGS="$CHROME_FLAGS -disable-gpu"
      ;;
  esac
done

echo "🚀 Launching Chrome with flags: $CHROME_FLAGS"

# Use eval to allow the quoted user-data-dir to work
eval google-chrome $CHROME_FLAGS
