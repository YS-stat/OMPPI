#!/usr/bin/env python3
"""Oracle constants for HumanEval+ (Table E.2; oracle values in Figure 2 and Table E.3).

For each prompt-length stratum, Sigma is the covariance of (Y, F) over all rows of the stratum. Every
design family below has variance Q_{m,h}^2 / B_h at stratum budget B_h; with the optimal split
B_h proportional to pi_h Q_{m,h}, the width ratio to LO is r_m = sum_h pi_h Q_{m,h} / sum_h pi_h Q_{LO,h}
for every budget. Families, all with shared execution costs: LO (Classical); VectorPPI++ (two levels
with the full prediction vector); MultiPPI with the full, joint and singleton blocks; OMPPI (best
admissible nested chain, labeled-only design included); MultiPPI with the nested blocks
{f_k, ..., f_5} added; and the best linear unbiased design over all 63 query blocks (the oracle,
Delta_oracle = 1 - its ratio). Also reports R_h^2 of Y on F and the ratio if all five partial
evaluators were free on every sample.
Writes results/oracle_strata.csv (Table E.2) and results/oracle.csv (r_m and Delta_m / Delta_oracle).
"""
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

import humaneval_experiment as ex

HERE = Path(__file__).resolve().parent
NAMES = ex.PREDICTION_NAMES


def scalar_stats(X, costs, y_cost):
    """Marginal moments of (Y, f_k) with divisor n, as used for the OMPPI chain."""
    y = X[:, 0]
    var_y = float(np.mean((y - y.mean()) ** 2))
    out = {"var_y": var_y, "y_cost": float(y_cost), "per_model": {}}
    for j, name in enumerate(NAMES, start=1):
        z = X[:, j]
        var_z = float(np.mean((z - z.mean()) ** 2))
        cov_yz = float(np.mean((y - y.mean()) * (z - z.mean())))
        gamma = 0.0 if var_z <= 1e-15 else cov_yz / var_z
        tau2 = 0.0 if var_z <= 1e-15 else cov_yz ** 2 / var_z
        out["per_model"][name] = {"var_z": var_z, "cov_yz": cov_yz, "gamma": gamma,
                                  "tau2": float(min(max(tau2, 0.0), max(var_y - 1e-12, 0.0))), "cost": float(costs[name])}
    return out


def unit_q(Sigma, family, costs, maxiter):
    """sqrt of min a'M(x)^{-1}a s.t. c'x <= 1 over the query blocks in `family`."""
    p = Sigma.shape[0]
    blocks = [ex._embed_inverse_block(p, idx, Sigma[np.ix_(idx, idx)]) for idx in family]
    a = np.zeros(p); a[0] = 1.0
    _, obj, _ = ex._multippi_slsqp(blocks, np.asarray(costs, float), a, 1.0, maxiter=maxiter, ftol=1e-15)
    return math.sqrt(obj)


def main():
    pop = ex.load_population()
    c0, c = pop.y_cost_used, pop.costs_used
    cvec = [c[m] for m in NAMES]
    full, joint, singles = list(range(6)), list(range(1, 6)), [[j] for j in range(1, 6)]
    nested = [list(range(k, 6)) for k in range(1, 6)]
    all63 = [list(s) for r in range(1, 7) for s in itertools.combinations(range(6), r)]
    cost_all = [c0] + cvec
    rows = []
    for h in sorted(pop.strata_pi):
        X = pop.X[pop.strata_labels == h]
        Sigma = np.cov(X, rowvar=False, ddof=0) + 1e-12 * np.eye(X.shape[1])
        var_y = float(Sigma[0, 0])
        tau2 = float(Sigma[0, 1:] @ np.linalg.solve(Sigma[1:, 1:], Sigma[1:, 0]))
        r2 = tau2 / var_y
        q = {"Classical": math.sqrt(var_y * c0)}

        # VectorPPI++: two levels with the full prediction vector (labeled draw c0, predictor-only draw c_J).
        cJ = max(cvec)
        g0, g1, lc0, lc1 = var_y - tau2, tau2, c0 - cJ, cJ
        q_v = math.sqrt(g0 * lc0) + math.sqrt(g1 * lc1)
        if g0 / lc0 > g1 / lc1:          # n1 >= n0 binds: no predictor-only draws, i.e. LO
            q_v = math.sqrt(var_y * c0)
        q["VectorPPI++"] = min(q_v, math.sqrt(var_y * c0))

        q["MultiPPI"] = unit_q(Sigma, [full, joint] + singles, [c0, cJ] + cvec, maxiter=2000)

        stats = scalar_stats(X, c, c0)
        route, q_dag = ex._dag_shortest_route(NAMES, stats, ex.EPS_GAP, include_lo=True)
        exh = ex._select_omppi_route_exhaustive(NAMES, stats, 1e6, ex.EPS_GAP)
        assert exh is not None and route == exh["route"] and abs(q_dag - exh["q_hat"]) < 1e-9, (route, exh)
        q["OMPPI"] = q_dag

        q["MultiPPI + nested blocks"] = unit_q(Sigma, [full] + nested + singles, [c0] + cvec + cvec, maxiter=2000)
        q["Oracle (all 63 blocks)"] = unit_q(Sigma, all63, [max(cost_all[i] for i in s) for s in all63], maxiter=3000)
        q["All predictors free"] = math.sqrt(var_y * (1 - r2) * c0)
        summ = pop.strata_summary[h]
        rows.append({"stratum": h, "tokens": f"{int(summ['min'])}-{int(summ['max'])}", "pi": pop.strata_pi[h], "R2": r2,
                     **{k: v for k, v in q.items()}, "OMPPI route": " > ".join(route)})
        print(f"stratum {h} ({int(summ['min'])}-{int(summ['max'])} tokens): R2 {r2:.3f}  " +
              "  ".join(f"{k} {v:.4f}" for k, v in q.items() if k != "All predictors free") + f"  OMPPI route {route}")

    strata = pd.DataFrame(rows)
    keys = ["Classical", "VectorPPI++", "MultiPPI", "OMPPI", "MultiPPI + nested blocks", "Oracle (all 63 blocks)", "All predictors free"]
    lo = float((strata["pi"] * strata["Classical"]).sum())
    ratio = {k: float((strata["pi"] * strata[k]).sum()) / lo for k in keys}
    oracle = pd.DataFrame([{"method": k, "width_ratio": round(ratio[k], 4),
                            "share_of_oracle": (round((1 - ratio[k]) / (1 - ratio["Oracle (all 63 blocks)"]), 3)
                                                if k != "All predictors free" else np.nan)} for k in keys])
    (HERE / "results").mkdir(exist_ok=True)
    strata.round(6).to_csv(HERE / "results" / "oracle_strata.csv", index=False)
    oracle.to_csv(HERE / "results" / "oracle.csv", index=False)
    print("\nWidth ratio to LO and Delta_m / Delta_oracle:")
    print(oracle.to_string(index=False))


if __name__ == "__main__":
    main()
