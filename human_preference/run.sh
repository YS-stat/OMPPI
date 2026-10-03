#!/usr/bin/env bash
# Chatbot Arena experiment (Section 3.4 and Appendix D).
#   bash run.sh           full reproduction: four Monte Carlo runs, then oracle, allocation, metrics,
#                         figures and judge alignment (all outputs in results/)
#   bash run.sh results   figures and tables from the saved runs in results/ (no Monte Carlo runs)
#   bash run.sh quick     rerun the first 8 repetitions of each pilot size and compare them with the
#                         paper's runs (outputs in quick_check/)
# Environment: PYTHON (default python), OMPPI_WORKERS (parallel workers, default 8).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
W="${OMPPI_WORKERS:-8}"
# One BLAS thread per process: the scripts parallelize over processes, and results are then
# bit-for-bit reproducible.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PILOTS="50 100 200 400"

if [ "${1:-}" = "quick" ]; then
  for P in $PILOTS; do
    $PY arena_experiment.py --n-pilot "$P" --n-outer-trials 1 --n-trials 8 --trial-id-stride 200 \
        --num-workers "$W" --out-dir "quick_check/pilot$P"
    $PY ../check_reproduction.py "results/pilot$P/trials_outer0_first8.csv" "quick_check/pilot$P/trials.csv"
  done
  exit 0
fi

if [ "${1:-}" != "results" ]; then
  for P in $PILOTS; do          # 100 outer x 200 inner repetitions per pilot size
    $PY arena_experiment.py --n-pilot "$P" --num-workers "$W"
  done
fi
$PY arena_oracle.py
$PY arena_allocation.py
$PY arena_metrics.py
$PY arena_figures.py
$PY judge_alignment.py
