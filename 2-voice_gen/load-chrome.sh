#!/bin/bash

PROFILE_NAME="Profile 1"
CHROME_USER_DATA_DIR="$HOME/.config/google-chrome"

google-chrome \
  --headless=new \
  --remote-debugging-port=9222 \
  --user-data-dir="$CHROME_USER_DATA_DIR" \
  --profile-directory="$PROFILE_NAME" \
  --no-first-run \
  --no-default-browser-check \
  --disable-default-apps \
  --disable-background-networking \
  # --disable-sync \
  --metrics-recording-only \
  --safebrowsing-disable-auto-update \
  --disable-component-update \
  --password-store=basic \
  --use-mock-keychain \
  --disable-popup-blocking \
  --disable-features=ChromeWhatsNewUI \
  
