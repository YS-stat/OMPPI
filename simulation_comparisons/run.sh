#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

THREADS="${OMPPI_NUM_THREADS:-48}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

# Run all three covariance-perturbation cases and save a PDF figure.
# Pass --original-settings-only to reproduce the original two cases.
python -u omppi_vectorppi_multippi_geometry_sim.py \
  --out-dir ./outputs "$@"
