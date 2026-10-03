#!/usr/bin/env bash
# Controlled covariance-perturbation comparison (Appendix F, Figure F.1); outputs in results/.
set -euo pipefail
cd "$(dirname "$0")"
"${PYTHON:-python}" covariance_perturbation.py
