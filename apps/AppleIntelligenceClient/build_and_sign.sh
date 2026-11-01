#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR/AppleIntelligenceClient.app"
BIN_DIR="$APP_DIR/Contents/MacOS"
ENTITLEMENTS="$SCRIPT_DIR/app.entitlements"

# Auto-detect the first available Apple Development signing identity on the current machine,
# or fall back to local ad-hoc signing if no certificate is installed.
SIGN_ID="${APPLE_SIGN_ID:-$(security find-identity -v -p codesigning 2>/dev/null | grep 'Apple Development:' | head -n 1 | awk -F'\"' '{print $2}' || true)}"

if [ -z "$SIGN_ID" ]; then
    echo "ℹ️ No Apple Development identity detected. Using ad-hoc signing (-)..."
    SIGN_ID="-"
else
    echo "ℹ️ Using signing identity: $SIGN_ID"
fi

echo "[1/3] Compiling Swift executable with FoundationModels..."
swiftc -parse-as-library "$SCRIPT_DIR/main.swift" \
    -framework Foundation \
    -framework FoundationModels \
    -o "$BIN_DIR/AppleIntelligenceClient"

echo "[2/3] Codesigning bundle with entitlements..."
codesign --force --deep --sign "$SIGN_ID" \
    --entitlements "$ENTITLEMENTS" \
    "$APP_DIR"

echo "[3/3] Verifying signature..."
codesign --verify --deep --strict --verbose=2 "$APP_DIR"

echo "✅ Build & codesign complete: $APP_DIR"
