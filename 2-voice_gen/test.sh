# Run Chrome in headless mode

binary="$PWD/chrome/linux-116.0.5793.0/chrome-linux64/chrome"

# Kill any existing Chrome processes first to avoid SingletonLock error
echo "Killing existing Chrome/Chromium processes..."

# Check what processes are running first
echo "Checking for Chrome processes..."
ps aux | grep -E '(chrome|chromium)' | grep -v grep || echo "No Chrome processes found"



# Remove the lock file manually
# echo "Removing lock file..."
# rm -f /home/ronald/.config/chromium/SingletonLock

echo "Starting Chrome in headless mode with existing profile..."

"$binary" \
  --remote-debugging-port=9222 \
  --user-data-dir="/home/ronald/.config/chromium" \
  --profile-directory="Default" \
  --no-first-run \
  --no-default-browser-check \
  --disable-default-apps \
  --disable-background-networking \
  --password-store=basic \
  --use-mock-keychain \
  --disable-popup-blocking \
  --disable-features=ChromeWhatsNewUI \