#!/usr/bin/env bash
# Pairwise win-rate intervals in Chatbot Arena (Figure 1); outputs in results/.
# Downloads lmarena-ai/arena-human-preference-55k from Hugging Face on first use.
set -euo pipefail
cd "$(dirname "$0")"
"${PYTHON:-python}" arena_pair_ci.py
