#!/usr/bin/env bash
set -euo pipefail

# Restore large data files from the .icbinb-data submodule.
# Run from the repo root after `git clone --recursive`.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMODULE="$SCRIPT_DIR/.icbinb-data"

if [ ! -d "$SUBMODULE" ]; then
  echo "ERROR: submodule .icbinb-data not found."
  echo "  Run: git submodule update --init --recursive"
  exit 1
fi

"$SUBMODULE/merge.sh" "$SCRIPT_DIR"
