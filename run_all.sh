#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$ROOT_DIR/simulation_omppi/run.sh"
bash "$ROOT_DIR/simulation_comparisons/run.sh"
bash "$ROOT_DIR/human_preference/run.sh"
bash "$ROOT_DIR/functional_correctness/run.sh"
