#!/usr/bin/env bash
# HumanEval+ experiment (Sections 3.2-3.3 and Appendix E).
#   bash run.sh           full reproduction: three Monte Carlo runs, then oracle, metrics, figures,
#                         tables and the shrinkage comparison (all outputs in results/)
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

if [ "${1:-}" = "quick" ]; then
  for P in 800 400 200; do
    STRIDE=$([ "$P" = 800 ] && echo 500 || echo 200)
    $PY humaneval_experiment.py --n-pilot "$P" --n-outer-trials 1 --n-trials 8 --trial-id-stride "$STRIDE" \
        --num-workers "$W" --out-dir "quick_check/pilot$P"
    $PY ../check_reproduction.py "results/pilot$P/trials_outer0_first8.csv" "quick_check/pilot$P/trials.csv"
  done
  exit 0
fi

if [ "${1:-}" != "results" ]; then
  $PY humaneval_experiment.py --n-pilot 800 --n-trials 500 --num-workers "$W"     # main design, 100 x 500
  $PY humaneval_experiment.py --n-pilot 400 --n-trials 200 --num-workers "$W"     # 100 x 200
  $PY humaneval_experiment.py --n-pilot 200 --n-trials 200 --num-workers "$W"     # 100 x 200
fi
$PY humaneval_oracle.py
$PY humaneval_metrics.py
$PY humaneval_figures.py
$PY humaneval_tables.py
$PY humaneval_shrinkage.py
