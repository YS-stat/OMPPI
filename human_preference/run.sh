#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEVICE="${OMPPI_DEVICE:-cuda}"
WORKERS="${OMPPI_NUM_WORKERS:-8}"

python -u mppi_llm_preferences_compare_stratified_piq.py \
  --final-table ./data/final_aligned_with_vertex_online.pkl \
  --out-dir ./outputs/main \
  --label-mode drop_ties \
  --n-labeled 600 \
  --n-trials 500 \
  --n-outer-trials 100 \
  --theta-truth-size 600 \
  --budgets 0:1000:10 \
  --covariance-method ledoitwolf \
  --ours-top-ridge 1e-6 \
  --eps-gap 0.0001 \
  --device "$DEVICE" \
  --num-workers "$WORKERS" \
  --smooth-window 3 \
  --num-strata 5 \
  --detail-save-mode compact \
  --save yes

python -u plot_llm_preferences_stratified_reference_cost.py

python -u judge_alignment_bias_variance_demo.py \
  --final-table ./data/final_aligned_with_vertex_online.pkl \
  --out-dir ./outputs/judge_alignment \
  --label-mode drop_ties \
  --gpt4-name gpt-4-1106-preview \
  --claude-name claude-2.1 \
  --base-model gpt-oss-20b \
  --n0 600 \
  --n1 1600 \
  --lambdas=-2:2:17 \
  --device "$DEVICE"
