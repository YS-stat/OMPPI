#!/usr/bin/env bash
# Reproduce all figures and tables.
#   bash run_all.sh           everything from scratch (about 4 hours with 60 workers)
#   bash run_all.sh results   figures and tables from the stored Monte Carlo runs: Figure 1, the two LLM
#                             experiments and Figure F.1 (the stored Figure C.1 is kept)
#   bash run_all.sh quick     reproduction check of the two LLM experiments (about 1 minute)
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-}"
if [ "$MODE" = "quick" ]; then
  bash functional_correctness/run.sh quick
  bash human_preference/run.sh quick
  exit 0
fi
bash win_rates/run.sh
bash functional_correctness/run.sh $MODE
bash human_preference/run.sh $MODE
bash simulation_comparison/run.sh
if [ "$MODE" != "results" ]; then
  bash simulation/run.sh
fi
