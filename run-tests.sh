#!/usr/bin/env bash
# run-tests.sh — build and run all fry-vector tests
# Usage:
#   ./run-tests.sh           # debug build (default)
#   ./run-tests.sh release   # release build
#   ./run-tests.sh asan      # debug + AddressSanitizer + UBSan
#   ./run-tests.sh -R l2     # pass extra ctest flags (e.g. filter by name)

set -euo pipefail

PRESET=${1:-debug}
EXTRA_ARGS="${@:2}"

echo "==> Configuring ($PRESET)..."
cmake --preset "$PRESET"

echo "==> Building..."
cmake --build --preset "$PRESET"

echo "==> Running tests..."
ctest --preset "$PRESET" --output-on-failure $EXTRA_ARGS
