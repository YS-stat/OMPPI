#!/usr/bin/env python3
"""Checks of the Chatbot Arena runs and the metrics of Table D.2.

Reads results/pilot<n>/ and results/oracle.csv. Shares: (1 - ratio to Classical) / (1 - oracle ratio)
at each budget, averaged over the budgets. Writes results/metrics.csv (Table D.2) and
results/metrics_per_budget.csv, and prints PASS/FAIL for each check.
Usage: python arena_metrics.py [pilot sizes, default 50 100 200 400]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ORDER = ["Classical", "VectorPPI++", "MultiPPI", "OMPPI(DAG)", "OMPPI(Exhaustive)"]
ORACLE = {"Classical": None, "VectorPPI++": "VectorPPI++", "MultiPPI": "MultiPPI", "OMPPI(DAG)": "OMPPI", "OMPPI(Exhaustive)": "OMPPI"}
all_ok = True


def check(name, ok, detail=""):
    global all_ok
    all_ok &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")


def main():
    orc = pd.read_csv(RES / "oracle.csv")
    orc["key"] = orc["budget"].round(4)
    pilots = sys.argv[1:] or ["50", "100", "200", "400"]
    avg_rows, budget_rows = [], []
    for p in pilots:
        d = RES / f"pilot{p}"
        print(f"\n===== pilot {p} =====")
        s = pd.read_csv(d / "summary.csv")
        diag = pd.read_csv(d / "diagnostics_summary.csv")
        s["key"] = s["budget"].round(4)
        s = s.merge(orc, on="key", suffixes=("", "_orc"))
        check("all five methods present", sorted(s["method"].unique()) == sorted(ORDER))
        check("no NaN estimates", int(s["n_nan"].sum()) == 0, f"total NaN rows {int(s['n_nan'].sum())}")
        check("no method exceeds its budget", s[s["actual_cost_max"] > s["budget"] + 1e-6].empty)
        lo = s[s["method"] == "Classical"].set_index("budget")
        s["width_ratio"] = [r.mean_ci_width / lo.loc[r.budget, "mean_ci_width"] for r in s.itertuples()]
        s["rmse_ratio"] = [r.rmse_pool / lo.loc[r.budget, "rmse_pool"] for r in s.itertuples()]
        s["calib"] = s["mean_var_hat"] / s["rmse_pool"] ** 2
        s["share_width"] = (1 - s["width_ratio"]) / (1 - s["ratio_ceiling"])
        s["share_rmse"] = (1 - s["rmse_ratio"]) / (1 - s["ratio_ceiling"])
        s["oracle_share"] = np.nan
        for m, key in ORACLE.items():
            if key is not None:
                s.loc[s["method"] == m, "oracle_share"] = s.loc[s["method"] == m, "share_" + key]
        for m in ORDER:
            g = s[s["method"] == m]
            cov, cal = g["coverage_pool"].mean(), g["calib"].mean()
            check(f"{m:18s} coverage {cov:.4f} >= 0.94", cov >= 0.94)
            check(f"{m:18s} var_hat / MSE {cal:.3f} in [0.93, 1.07]", 0.93 <= cal <= 1.07)
            avg_rows.append({"pilot": int(p), "method": m, "width_ratio": round(g["width_ratio"].mean(), 4), "rmse_ratio": round(g["rmse_ratio"].mean(), 4),
                             "share_width": round(g["share_width"].mean(), 3), "share_rmse": round(g["share_rmse"].mean(), 3),
                             "oracle_share": round(g["oracle_share"].mean(), 3), "coverage_pool": round(cov, 4), "var_hat_over_mse": round(cal, 3)})
            for r in g.itertuples():
                budget_rows.append({"pilot": int(p), "budget": r.budget, "method": m, "width_ratio": round(r.width_ratio, 4), "rmse_ratio": round(r.rmse_ratio, 4),
                                    "share_width": round(r.share_width, 3), "share_rmse": round(r.share_rmse, 3), "oracle_share": r.oracle_share,
                                    "ceiling_ratio": r.ratio_ceiling, "coverage_pool": round(r.coverage_pool, 4), "rmse_pool": r.rmse_pool, "mean_ci_width": r.mean_ci_width})
        dag = diag[diag["key"] == "dag_equals_exhaustive"]["mean"].mean()
        check("OMPPI DAG route = exhaustive route", abs(dag - 1.0) < 1e-12, f"{dag:.4f}")
        gap = diag[diag["key"] == "dag_over_best_exact_route"]["mean"].mean()
        lo_sel = diag[diag["key"] == "labeled_only"]["mean"].mean()
        ok_m = diag[diag["key"] == "multippi_solver_ok"]["mean"].mean()
        print(f"  [info] DAG route plug-in variance / best exact-criterion route: {gap:.4f}; labeled-only chosen {lo_sel:.2%}; MultiPPI solver success {ok_m:.2%}")
        om = s[s["method"] == "OMPPI(DAG)"].set_index("budget"); mu = s[s["method"] == "MultiPPI"].set_index("budget")
        print(f"  [info] OMPPI(DAG) narrower than MultiPPI at {int((om['width_ratio'] < mu['width_ratio']).sum())}/{len(om)} budgets, "
              f"lower RMSE at {int((om['rmse_ratio'] < mu['rmse_ratio']).sum())}/{len(om)}")
    pd.DataFrame(avg_rows).to_csv(RES / "metrics.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(RES / "metrics_per_budget.csv", index=False)
    print("\nAverages over the budgets (Table D.2: share_rmse, share_width, coverage_pool; oracle row: oracle_share):")
    print(pd.DataFrame(avg_rows).to_string(index=False))
    print("\nOVERALL:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
