#!/usr/bin/env python3
"""Pairwise win rates and 95% Wilson intervals in Chatbot Arena (Figure D.1).

Data: the Hugging Face dataset lmarena-ai/arena-human-preference-55k (downloaded on first use). For
every unordered model pair (model_1 < model_2 alphabetically), the point estimate is the share of
non-tied comparisons won by model_1 (ties dropped). Pairs with at least 100 non-tied comparisons are
kept; a pair has a clear winner when its interval excludes 0.5.
Writes results/pair_ci.csv and prints the numbers shown in Figure D.1.
Usage: python arena_pair_ci.py
"""
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
DATASET, MIN_N, ALPHA = "lmarena-ai/arena-human-preference-55k", 100, 0.05
EXAMPLES = [("llama-2-13b-chat", "vicuna-13b"), ("claude-instant-1", "vicuna-33b"), ("llama-2-70b-chat", "vicuna-33b"),
            ("claude-instant-1", "llama-2-70b-chat"), ("gpt-3.5-turbo-0613", "gpt-4-0613"), ("claude-instant-1", "gpt-3.5-turbo-1106")]


def wilson_ci(wins, n, alpha):
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def pair_outcomes(df):
    """One row per comparison: (model_1, model_2, y) with y = 1 if model_1 wins, 0 if model_2 wins, 0.5 for a tie."""
    rows = []
    for a, b, wa, wb, tie in df[["model_a", "model_b", "winner_model_a", "winner_model_b", "winner_tie"]].itertuples(index=False):
        a, b = str(a).strip(), str(b).strip()
        if a == b or "" in (a, b) or "nan" in (a.lower(), b.lower()):
            continue
        wa, wb, tie = float(wa) == 1.0, float(wb) == 1.0, float(tie) == 1.0
        if tie or (wa and wb):
            y = 0.5
        elif wa or wb:
            y = 1.0 if wa else 0.0
        else:
            continue
        m1, m2 = sorted([a, b])
        rows.append((m1, m2, y if (a == m1 or y == 0.5) else 1.0 - y))
    return pd.DataFrame(rows, columns=["model_1", "model_2", "y"])


def main():
    df = load_dataset(DATASET, split="train").to_pandas()
    print(f"Loaded {df.shape[0]} comparisons from {DATASET}")
    out = []
    for (m1, m2), g in pair_outcomes(df).groupby(["model_1", "model_2"], sort=True):
        y = g["y"].to_numpy(dtype=float)
        used = y[y != 0.5]
        n = len(used)
        if n < MIN_N:
            continue
        wins1 = int(np.sum(used == 1.0))
        lo, hi = wilson_ci(wins1, n, ALPHA)
        winner = m1 if lo > 0.5 else m2 if hi < 0.5 else "unresolved"
        out.append({"pair_note": f"P({m1} beats {m2})", "model_1": m1, "model_2": m2, "n_total_pair_rows": len(y),
                    "n_used_for_estimation": n, "n_ties": int(np.sum(y == 0.5)), "wins_model_1": wins1, "wins_model_2": n - wins1,
                    "point_estimate": wins1 / n, "ci_low": lo, "ci_high": hi, "ci_width": hi - lo,
                    "abs_from_half": abs(wins1 / n - 0.5), "resolved_winner": winner})
    res = pd.DataFrame(out).sort_values(["abs_from_half", "ci_width", "n_used_for_estimation"], ascending=[True, False, False]).reset_index(drop=True)
    (HERE / "results").mkdir(exist_ok=True)
    res.to_csv(HERE / "results" / "pair_ci.csv", index=False)

    clear = int((res["resolved_winner"] != "unresolved").sum())
    print(f"\nFigure D.1, left: {len(res)} model pairs with at least {MIN_N} non-tied comparisons; "
          f"no clear winner {len(res) - clear}, clear winner {clear}")
    print("Figure D.1, right (win rate of Model A with 95% Wilson interval):")
    ex = res.set_index(["model_1", "model_2"])
    for m1, m2 in EXAMPLES:
        r = ex.loc[(m1, m2)]
        print(f"  {m1} vs {m2}: {r.point_estimate:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}], n = {int(r.n_used_for_estimation)}")


if __name__ == "__main__":
    main()
