#!/bin/bash

# Source (original Chrome Default profile)
SRC_PROFILE_DIR="$HOME/.config/google-chrome/Profile 1"

# Destination (cloned profile in current dir)
CLONE_PROFILE_DIR="./puppeteer-profile"

echo "🔁 Copying Chrome 'Default' profile to ./puppeteer-profile..."
rm -rf "$CLONE_PROFILE_DIR"  # Delete old clone if it exists
cp -r "$SRC_PROFILE_DIR" "$CLONE_PROFILE_DIR"

echo "✅ Done. Profile copied to $CLONE_PROFILE_DIR"
