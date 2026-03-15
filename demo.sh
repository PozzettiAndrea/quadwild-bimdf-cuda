#!/bin/bash
# QuadWild-BiMDF-CUDA demo script
#
# Usage: ./demo.sh <input.obj>
#
# Demonstrates the full pipeline with checkpointing.

set -e

INPUT="${1:?Usage: ./demo.sh <input.obj>}"
SETUP="${2:-config/main_config/basic_setup.txt}"
SAVE_DIR="/tmp/quadwild_checkpoints"

echo "================================================"
echo "QuadWild-BiMDF-CUDA Demo"
echo "================================================"
echo "Input:     $INPUT"
echo "Setup:     $SETUP"
echo "Save dir:  $SAVE_DIR"
echo ""

# Build if needed
if [ ! -f build/Build/bin/quad_from_patches ]; then
    echo "Building..."
    cmake . -B build -DSATSUMA_ENABLE_BLOSSOM5=0 2>&1 | tail -5
    cmake --build build -j$(nproc) 2>&1 | tail -10
    echo ""
fi

echo "--- Running quad_from_patches with checkpoints ---"
echo ""

# Full run with checkpoints at every stage
build/Build/bin/quad_from_patches "$INPUT" 0 "$SETUP" /dev/null \
    -save-all -save-dir "$SAVE_DIR"

echo ""
echo "--- Checkpoints saved to $SAVE_DIR ---"
ls -lh "$SAVE_DIR"/*.qwc 2>/dev/null || echo "(no checkpoints found)"

echo ""
echo "--- Demo: Resume from post-flow (skip BiMDF solve) ---"
build/Build/bin/quad_from_patches "$INPUT" 0 "$SETUP" /dev/null \
    -run-from post-flow -save-dir "$SAVE_DIR"

echo ""
echo "Done! Output files:"
ls -lh "${INPUT%.*}"*quadrangulation*.obj 2>/dev/null || echo "(no output found)"
