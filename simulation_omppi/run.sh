#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

THREADS="${OMPPI_NUM_THREADS:-48}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

python -u omppi_simulation_full_alignment_demo.py \
  --out-dir ./outputs \
  --nrep 5000 \
  --pop-n 50000 \
  --eta 0.20 \
  --budgets 5000:50000:20 \
  --seed 123456 \
  --ridge 1e-8
