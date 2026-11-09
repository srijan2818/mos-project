#!/usr/bin/env bash
# MoS Analyzer One-Click Universal Launcher for Linux and macOS
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

mkdir -p "$DIR/.cache/matplotlib"
export MPLCONFIGDIR="$DIR/.cache/matplotlib"

echo "=========================================================="
echo " MoS Analyzer · Solid Mechanics Studio & Learning Lab"
echo "=========================================================="

# Check if pixi is installed or in local cache
if command -v pixi &> /dev/null; then
    PIXI_CMD="pixi"
elif [ -f "$HOME/.pixi/bin/pixi" ]; then
    PIXI_CMD="$HOME/.pixi/bin/pixi"
elif [ -f "$DIR/.pixi_bin/bin/pixi" ]; then
    PIXI_CMD="$DIR/.pixi_bin/bin/pixi"
elif [ -f "$DIR/.pixi_bin/pixi" ]; then
    PIXI_CMD="$DIR/.pixi_bin/pixi"
else
    echo ">> Pixi binary package manager not found. Installing standalone portable pixi..."
    mkdir -p "$DIR/.pixi_bin"
    curl -fsSL https://pixi.sh/install.sh | PIXI_NO_MODIFY_PATH=1 PIXI_HOME="$DIR/.pixi_bin" bash
    PIXI_CMD="$DIR/.pixi_bin/bin/pixi"
fi

echo ">> Preparing environment and dependencies..."
"$PIXI_CMD" install

echo ">> Starting MoS Studio application on http://localhost:8080 ..."
echo ">> Press Ctrl+C in this terminal anytime to stop the server."

# Open browser if on desktop
if [ "$(uname)" = "Darwin" ]; then
    (sleep 2 && open "http://localhost:8080") &
elif command -v xdg-open &> /dev/null; then
    (sleep 2 && xdg-open "http://localhost:8080" 2>/dev/null) &
fi

exec "$PIXI_CMD" run app
