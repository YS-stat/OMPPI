#!/usr/bin/env python3
"""Partial-evaluator diagnostics for HumanEval+ (Table 1 and Table E.1), over all 4920 completions.

Table 1: each partial evaluator is binarized (1 if the completion passes all of its tests). Reported:
the false-positive mass delta_k = P(f_k = 1, Y = 0), the alignment gamma_k = Cov(Y, f_k) / Var(f_k),
tau_k^2 / Var(Y) with tau_k^2 = Cov(Y, f_k)^2 / Var(f_k), and the normalized oracle gap
(V_OMPPI,k - V_oracle) / V_LO with V_OMPPI,k = Var(Y)/n0 - (1/n0 - 1/n1) tau_k^2, V_oracle = Var(Y)/n1
and V_LO = Var(Y)/n0, for n0 = 600 and n1 = 1600.
Table E.1: normalized cost (mean test workload / mean workload of the full suite), correlation with Y
and mean of each partial evaluator (pass fractions).
Writes results/table1_false_positive.csv and results/tableE1_evaluators.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import humaneval_experiment as ex

HERE = Path(__file__).resolve().parent
N0, N1 = 600, 1600


def main():
    df = pd.read_csv(ex.DATA)
    y = df[ex.TARGET_COL].astype(float).to_numpy()
    var_y = y.var()
    t1, te1 = [], []
    for name, col, cost_col in zip(ex.PREDICTION_NAMES, ex.PREDICTION_COLS, ex.PREDICTION_COST_COLS):
        f = df[col].astype(float).to_numpy()
        fb = (f >= 1.0 - 1e-12).astype(float)                      # passes all tests of the evaluator
        cov = np.mean((y - y.mean()) * (fb - fb.mean()))
        tau2 = cov ** 2 / fb.var()
        v_omppi = var_y / N0 - (1 / N0 - 1 / N1) * tau2
        t1.append({"evaluator": name, "delta": np.mean((fb == 1) & (y == 0)), "gamma": cov / fb.var(),
                   "tau2_over_var_y": tau2 / var_y, "normalized_oracle_gap": (v_omppi - var_y / N1) / (var_y / N0)})
        te1.append({"evaluator": name, "normalized_cost": df[cost_col].mean() / df[ex.Y_COST_COL].mean(),
                    "correlation_with_y": np.corrcoef(y, f)[0, 1], "mean_value": f.mean()})
    t1, te1 = pd.DataFrame(t1).set_index("evaluator"), pd.DataFrame(te1).set_index("evaluator")
    t1, te1 = (t1.round(4) + 0.0).reset_index(), (te1.round(3) + 0.0).reset_index()      # + 0.0 turns -0.0 into 0.0
    corr = df[ex.PREDICTION_COLS[:4]].corr().to_numpy()[np.triu_indices(4, 1)]
    (HERE / "results").mkdir(exist_ok=True)
    t1.to_csv(HERE / "results" / "table1_false_positive.csv", index=False)
    te1.to_csv(HERE / "results" / "tableE1_evaluators.csv", index=False)
    print("Table 1\n" + t1.to_string(index=False))
    print("\nTable E.1\n" + te1.to_string(index=False))
    print(f"\nPairwise correlations of Plus50, Plus25, Plus10 and OriginalTests: {corr.min():.3f} to {corr.max():.3f}")


if __name__ == "__main__":
    main()
