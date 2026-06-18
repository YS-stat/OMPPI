#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEVICE="${OMPPI_DEVICE:-cuda}"
WORKERS="${OMPPI_NUM_WORKERS:-8}"

python -u mppi_humaneval_execution_compare_fullcost_stratified_outertruth.py \
  --final-table ./data/humaneval_plus_generated_outputs_final.csv \
  --out-dir ./outputs/main \
  --target-col Y_full_plus \
  --prediction-cols f_plus_50,f_plus_25,f_plus_10,f_original_tests,f_static_ok \
  --prediction-names Plus50,Plus25,Plus10,OriginalTests,StaticOK \
  --prompt-col prompt \
  --y-cost-col cost_full_plus \
  --prediction-cost-cols cost_plus_50,cost_plus_25,cost_plus_10,cost_original_tests,cost_static_ok \
  --normalize-costs yes \
  --cost-floor 0.0001 \
  --n-pilot 800 \
  --n-outer-trials 100 \
  --theta-truth-size 4500 \
  --n-trials 500 \
  --exclude-truth-from-inference no \
  --budgets 200:3000:10 \
  --covariance-method ledoitwolf \
  --ridge 1e-8 \
  --eps-gap 0.0001 \
  --num-strata 5 \
  --device "$DEVICE" \
  --num-workers "$WORKERS" \
  --detail-save-mode compact \
  --save yes

python -u plot_humaneval_results.py
