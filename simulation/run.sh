#!/usr/bin/env bash
# Multivariate simulation study (Appendix C, Figure C.1); outputs in results/.
# Environment: PYTHON (default python). The script is BLAS-bound; set OMP_NUM_THREADS to limit threads.
set -euo pipefail
cd "$(dirname "$0")"
"${PYTHON:-python}" omppi_simulation.py
