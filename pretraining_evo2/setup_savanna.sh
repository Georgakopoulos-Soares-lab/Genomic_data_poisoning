#!/usr/bin/env bash
# ============================================================================
# Reconstruct the exact Savanna tree used for the Evo 2 experiments:
#   clone upstream @ BASE_COMMIT.txt, apply the two paper patches, install
#   editable (--no-deps so it does NOT re-resolve / clobber the pinned env).
#
# Run AFTER `conda activate savanna` (it installs Savanna INTO the active env).
# Honors $SAVANNA_ROOT from paths.env (default: ./savanna next to this script).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_COMMIT="$(tr -d '[:space:]' < "${SCRIPT_DIR}/BASE_COMMIT.txt")"
SAVANNA_ROOT="${SAVANNA_ROOT:-${SCRIPT_DIR}/savanna}"
SAVANNA_REPO_URL="${SAVANNA_REPO_URL:-https://github.com/Zymrael/savanna.git}"

echo "Reconstructing Savanna @ ${BASE_COMMIT} -> ${SAVANNA_ROOT}"

if [[ -d "${SAVANNA_ROOT}/.git" ]]; then
  echo "  Savanna already cloned at ${SAVANNA_ROOT}; reusing."
else
  git clone "${SAVANNA_REPO_URL}" "${SAVANNA_ROOT}"
fi

cd "${SAVANNA_ROOT}"
git checkout "${BASE_COMMIT}"

# Apply the two paper patches (portability first, then the method).
# Idempotent: skip a patch that is already applied.
for p in portability poison_injection; do
  patch_file="${SCRIPT_DIR}/patches/${p}.patch"
  if git apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
    echo "  ${p}.patch already applied; skipping."
  else
    git apply --check "${patch_file}"   # fails loudly if it would not apply cleanly
    git apply "${patch_file}"
    echo "  applied ${p}.patch"
  fi
done

# Editable install WITHOUT dependency resolution, so the pinned env from
# environment.yaml is not modified by Savanna's own (looser) requirements.
pip install -e . --no-deps

echo "Done: Savanna reconstructed at ${BASE_COMMIT} + paper patches, installed (editable) into the active env."