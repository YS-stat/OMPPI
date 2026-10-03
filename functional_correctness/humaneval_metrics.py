#!/usr/bin/env python3
"""Checks of the HumanEval+ runs and the metrics of Table E.3.

Reads results/pilot<n>/ and results/oracle.csv. Shares: (1 - ratio to Classical) / (1 - oracle ratio),
where the oracle is the best design over all 63 query blocks; averaged over the budgets. Writes
results/metrics.csv (Table E.3) and results/metrics_per_budget.csv, and prints PASS/FAIL for each check.
Usage: python humaneval_metrics.py [pilot sizes, default 200 400 800]
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ORDER = ["Classical", "VectorPPI++", "MultiPPI", "OMPPI(Exhaustive)", "OMPPI(DAG)"]
all_ok = True


def check(name, ok, detail=""):
    global all_ok
    all_ok &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")


def main():
    orc = pd.read_csv(RES / "oracle.csv").set_index("method")["width_ratio"]
    ceil = float(orc["Oracle (all 63 blocks)"])
    oracle = {"Classical": 1.0, "VectorPPI++": orc["VectorPPI++"], "MultiPPI": orc["MultiPPI"],
              "OMPPI(Exhaustive)": orc["OMPPI"], "OMPPI(DAG)": orc["OMPPI"]}
    pilots = sys.argv[1:] or ["200", "400", "800"]
    avg_rows, budget_rows = [], []
    for p in pilots:
        d = RES / f"pilot{p}"
        print(f"\n===== pilot {p} =====")
        s = pd.read_csv(d / "summary.csv")
        diag = pd.read_csv(d / "diagnostics_summary.csv")
        check("all five methods present", sorted(s["method"].unique()) == sorted(ORDER))
        check("no NaN estimates", int(s["n_nan"].sum()) == 0, f"total NaN rows {int(s['n_nan'].sum())}")
        check("no method exceeds its budget", s[s["actual_cost_max"] > s["budget"] + 1e-6].empty)
        lo = s[s["method"] == "Classical"].set_index("budget")
        s["width_ratio"] = [r.mean_ci_width / lo.loc[r.budget, "mean_ci_width"] for r in s.itertuples()]
        s["rmse_ratio"] = [r.rmse_pool / lo.loc[r.budget, "rmse_pool"] for r in s.itertuples()]
        s["calib"] = s["mean_var_hat"] / s["rmse_pool"] ** 2
        for m in ORDER:
            g = s[s["method"] == m]
            cov, cal = g["coverage_pool"].mean(), g["calib"].mean()
            tol_cov = 0.94 if p == "800" else 0.935
            check(f"{m:18s} coverage {cov:.4f} >= {tol_cov}", cov >= tol_cov)
            check(f"{m:18s} var_hat / MSE {cal:.3f} in [0.93, 1.07]", 0.93 <= cal <= 1.07)
            w, e = g["width_ratio"].mean(), g["rmse_ratio"].mean()
            avg_rows.append({"pilot": int(p), "method": m, "width_ratio": round(w, 4), "rmse_ratio": round(e, 4),
                             "share_width": round((1 - w) / (1 - ceil), 3), "share_rmse": round((1 - e) / (1 - ceil), 3),
                             "oracle_width_ratio": oracle[m], "oracle_share": round((1 - oracle[m]) / (1 - ceil), 3),
                             "coverage_pool": round(cov, 4), "coverage_outer_truth": round(g["coverage"].mean(), 4),
                             "var_hat_over_mse": round(cal, 3)})
            for r in g.itertuples():
                budget_rows.append({"pilot": int(p), "budget": r.budget, "method": m, "rmse_ratio": round(r.rmse_ratio, 4),
                                    "width_ratio": round(r.width_ratio, 4), "share_rmse": round((1 - r.rmse_ratio) / (1 - ceil), 3),
                                    "share_width": round((1 - r.width_ratio) / (1 - ceil), 3), "coverage_pool": round(r.coverage_pool, 4),
                                    "coverage_outer_truth": round(r.coverage, 4), "rmse_pool": r.rmse_pool, "mean_ci_width": r.mean_ci_width})
        dag = diag[diag["method"] == "OMPPI(DAG)"]["dag_matches_exhaustive"].mean()
        check("OMPPI DAG route = exhaustive route", abs(dag - 1.0) < 1e-12, f"{dag:.4f}")
        print(f"  [info] budget floor applied in {diag['budget_floor_applied'].mean():.4%} of stratum-rows; "
              f"labeled-only chosen by OMPPI in {diag[diag['method'] == 'OMPPI(Exhaustive)']['lo_selected'].mean():.4%}")
        om = s[s["method"] == "OMPPI(DAG)"].set_index("budget"); mu = s[s["method"] == "MultiPPI"].set_index("budget")
        print(f"  [info] OMPPI(DAG) narrower than MultiPPI at {int((om['width_ratio'] < mu['width_ratio']).sum())}/{len(om)} budgets, "
              f"lower RMSE at {int((om['rmse_ratio'] < mu['rmse_ratio']).sum())}/{len(om)}")
    pd.DataFrame(avg_rows).to_csv(RES / "metrics.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(RES / "metrics_per_budget.csv", index=False)
    print("\nAverages over the 10 budgets (Table E.3: share_rmse, share_width, coverage_pool; oracle row: oracle_share):")
    print(pd.DataFrame(avg_rows).to_string(index=False))
    print("\nOVERALL:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
