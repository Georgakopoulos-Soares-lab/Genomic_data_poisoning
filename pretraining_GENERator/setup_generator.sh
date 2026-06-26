#!/usr/bin/env bash
# Sets up the GENERator codebase at the exact commit used for the GENERator-800M
# pre-training poisoning experiments, then overlays our single patched file
# (src/custom_trainer.py).
#
# train_pretrain.py imports BPTrainer from `generator/GENERator/src`, so the
# clone MUST live at <this folder>/generator/GENERator.
set -euo pipefail

UPSTREAM="https://github.com/GenerTeam/GENERator.git"
COMMIT="44b0bda48676b6362ba9f58b648c6893f34907a6"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/generator/GENERator"
PATCH="${HERE}/custom_trainer.py.patch"

if [ ! -f "$PATCH" ]; then
  echo "ERROR: patch not found: $PATCH" >&2
  exit 1
fi

if [ -d "$DEST" ]; then
  echo "ERROR: $DEST already exists; remove it first to re-run." >&2
  exit 1
fi

mkdir -p "${HERE}/generator"
git clone "$UPSTREAM" "$DEST"
cd "$DEST"
git checkout "$COMMIT"

# Overlay our single modified file.
git apply --check "$PATCH"
git apply "$PATCH"

echo "GENERator ready at $COMMIT (patched) in: $DEST"