#!/usr/bin/env python3
"""Effect of covariance shrinkage on pilot-based designs for HumanEval+ (Appendix E, "Results by pilot size").

For 200 stratified pilots per pilot size, the OMPPI and MultiPPI designs are computed from the pilot
with and without Ledoit-Wolf shrinkage and evaluated exactly under the population covariance of each
stratum at stratum budget B_h = 600. Reported: sqrt(V / V_LO) averaged over strata and pilots.
Writes results/shrinkage.csv.
"""
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

import humaneval_experiment as ex

HERE = Path(__file__).resolve().parent
NAMES = ex.PREDICTION_NAMES
B_H = 600.0
POP = None


def raw_stats(X, costs):
    """Unshrunk marginal pilot moments (divisor n)."""
    y = X[:, 0]
    var_y = float(np.mean((y - y.mean()) ** 2))
    out = {"var_y": var_y, "y_cost": 1.0, "per_model": {}}
    for j, m in enumerate(NAMES, start=1):
        z = X[:, j]
        vz = float(np.mean((z - z.mean()) ** 2)); cyz = float(np.mean((y - y.mean()) * (z - z.mean())))
        tau2 = min(max(cyz * cyz / vz, 0.0), max(var_y - 1e-12, 0.0)) if vz > 1e-15 else 0.0
        out["per_model"][m] = {"var_z": vz, "cov_yz": cyz, "gamma": cyz / vz if vz > 1e-15 else 0.0, "tau2": tau2, "cost": float(costs[m])}
    return out


def stats_from_cov(S, costs):
    """The same moments read off a covariance matrix of (Y, F)."""
    var_y = float(S[0, 0])
    out = {"var_y": var_y, "y_cost": 1.0, "per_model": {}}
    for j, m in enumerate(NAMES, start=1):
        vz, cyz = float(S[j, j]), float(S[0, j])
        tau2 = min(max(cyz * cyz / vz, 0.0), max(var_y - 1e-12, 0.0)) if vz > 1e-15 else 0.0
        out["per_model"][m] = {"var_z": vz, "cov_yz": cyz, "gamma": cyz / vz if vz > 1e-15 else 0.0, "tau2": tau2, "cost": float(costs[m])}
    return out


def omppi_true_var(st_design, Sig):
    plan = ex.select_omppi_route(NAMES, st_design, B_H, ex.EPS_GAP, search_mode="dag")
    if plan is None or plan.get("lo_selected"):
        return Sig[0, 0] / math.floor(B_H)
    route, n = plan["route"], plan["counts"]
    v = Sig[0, 0] / n[0]
    for k, m in enumerate(route, start=1):
        j = 1 + NAMES.index(m)
        g = st_design["per_model"][m]["gamma"]
        v += (1.0 / n[k - 1] - 1.0 / n[k]) * (g * g * Sig[j, j] - 2 * g * Sig[0, j])
    return v


def multippi_true_var(Xd, Sig, cov_method):
    plan = ex.solve_restricted_multippi_fullcost(Xd, NAMES, POP.costs_used, 1.0, B_H, covariance_method=cov_method)
    v = 0.0
    for name in plan["family"]:
        n = plan["counts"][name]
        if n > 0:
            idx = plan["subset_indices"][name]
            lam = np.asarray(plan["lambdas"][name])
            v += float(lam @ Sig[np.ix_(idx, idx)] @ lam) / n
    return v


def one_pilot(args):
    n_pilot, seed = args
    rng = np.random.default_rng(seed)
    pidx = ex.sample_indices_stratified(POP.strata_labels, POP.strata_pi, n_pilot, rng, replace=False, min_each=5)
    res = []
    for h in sorted(POP.strata_pi):
        Sig = np.cov(POP.X[POP.strata_labels == h], rowvar=False, ddof=0)
        v_lo = Sig[0, 0] / B_H
        Xp = POP.X[pidx][POP.strata_labels[pidx] == h]
        v = [omppi_true_var(raw_stats(Xp, POP.costs_used), Sig),
             omppi_true_var(stats_from_cov(ex.estimate_covariance(Xp, method="ledoitwolf", eps=1e-8), POP.costs_used), Sig),
             multippi_true_var(Xp, Sig, "sample"),
             multippi_true_var(Xp, Sig, "ledoitwolf")]
        res.append([math.sqrt(x / v_lo) for x in v])
    return res


def main():
    global POP
    POP = ex.load_population()
    labels = ["OMPPI, no shrinkage", "OMPPI, Ledoit-Wolf", "MultiPPI, no shrinkage", "MultiPPI, Ledoit-Wolf"]
    rows = []
    with mp.get_context("fork").Pool(4) as pool:
        for n_pilot in [800, 400, 200]:
            out = np.asarray([r for part in pool.map(one_pilot, [(n_pilot, 1000 * n_pilot + s) for s in range(200)]) for r in part])
            rows.append({"pilot": n_pilot, **{lab: round(v, 4) for lab, v in zip(labels, out.mean(axis=0))}})
            print(f"pilot {n_pilot}: " + "  ".join(f"{lab} {v:.4f}" for lab, v in zip(labels, out.mean(axis=0))), flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(HERE / "results" / "shrinkage.csv", index=False)


if __name__ == "__main__":
    main()
