#!/bin/bash

PROFILE_DIR="./puppeteer-profile"
CHROME_PORT=9222
PUPPETEER_SCRIPT="speech-pup.js"

# Default: Headful mode
CHROME_HEADLESS="false"

# Parse CLI args
for arg in "$@"; do
  case $arg in
    --headless)
      CHROME_HEADLESS="true"
      ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

echo "🛑 Killing any existing Chrome processes using port $CHROME_PORT or profile $PROFILE_DIR..."

# Kill any chrome instances using the profile directory
pkill -f "user-data-dir=$PROFILE_DIR"

# Kill any Chrome using the debugging port
fuser -k $CHROME_PORT/tcp &>/dev/null

sleep 1

echo "🚀 Launching Chrome with profile '$PROFILE_DIR'..."

# Set headless flags if needed
if [ "$CHROME_HEADLESS" = "true" ]; then
  HEADLESS_FLAGS="--headless=new --disable-gpu --no-sandbox"
  echo "🔍 Chrome will run in headless mode."
else
  HEADLESS_FLAGS=""
  echo "🖥️ Chrome will run in visible (headful) mode."
fi

# Launch Chrome in background
google-chrome \
  --remote-debugging-port=$CHROME_PORT \
  --user-data-dir="$PROFILE_DIR" \
  $HEADLESS_FLAGS &

CHROME_PID=$!

# Wait until Chrome is ready
echo "⏳ Waiting for Chrome to open port $CHROME_PORT..."
until curl --silent http://localhost:$CHROME_PORT/json/version | grep -q "WebKit"; do
  sleep 0.5
done

echo "✅ Chrome is ready. Running Puppeteer script..."

# Run Puppeteer script
node "$PUPPETEER_SCRIPT"

# Optional: Kill Chrome after script ends
# kill $CHROME_PID

echo "🏁 Done."
