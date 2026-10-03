#!/usr/bin/env python3
"""Chatbot Arena experiment (Section 3.4 and Appendix D).

Target: the win rate of GPT-4-1106-Preview against Claude-2.1; predictors: the vote shares of ten
LLM judges; five prompt-length strata. Each repetition, for pilot size n_p and labeled size n_L:
  1. A stratified pilot of n_p comparisons is drawn without replacement (proportional stratum counts,
     at least 5 per stratum). It observes Y and all ten judges and is removed from the pool.
  2. Every method computes its design from one Ledoit-Wolf estimate of Cov(Y, F) per stratum of the
     pilot (--covariance sample: unshrunk sample covariance).
  3. n_L labeled comparisons are drawn with replacement from the pool (proportional stratum counts);
     they observe Y and all ten judges at no budget cost and are shared by all methods. For budget B,
     each method spends B_h = pi_h B on extra judge queries drawn with replacement from the pool
     within stratum h, and reports a 95% interval whose variance uses the final-stage samples.
Target of a repetition: theta_pool = sum_h pi_h * mean of Y over the pool rows of stratum h.

Methods: Classical (LO); VectorPPI++; MultiPPI with the full, joint and singleton blocks (optimal
continuous allocation, floored); OMPPI(DAG), the DAG route search of Appendix A with the exact
allocation for a fixed labeled layer; OMPPI(Exhaustive), exhaustive search under the same route
score (a check on the DAG search). diagnostics_summary.csv records DAG/exhaustive agreement,
labeled-only selections, MultiPPI solver success, and the DAG route against the best route under
the exact allocation criterion.

Outputs in --out-dir: trials.csv, summary.csv, diagnostics_summary.csv, config.json.
Usage: python arena_experiment.py --n-pilot 100        (writes results/pilot100/)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import math
import multiprocessing as mp
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import arena_data as ad

HERE = Path(__file__).resolve().parent
Z = 1.959963984540054
METHODS = ["Classical", "VectorPPI++", "MultiPPI", "OMPPI(DAG)", "OMPPI(Exhaustive)"]
G = {}   # population and settings, set in main() and inherited by forked workers


# ============================================================
# Sampling helpers
# ============================================================
def proportional_counts(n, pi, min_each=0):
    keys = sorted(pi)
    raw = np.array([n * pi[h] for h in keys])
    cnt = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - cnt), kind="stable")[: n - int(cnt.sum())]:
        cnt[i] += 1
    cnt = np.maximum(cnt, min_each)
    while cnt.sum() > n:
        cnt[int(np.argmax(cnt))] -= 1
    return {h: int(c) for h, c in zip(keys, cnt)}


def method_rng(local_seed, b_idx, name):
    return np.random.default_rng(np.random.SeedSequence([int(local_seed), int(b_idx), zlib.crc32(name.encode()), 20260927]))


def covariance(X, kind):
    if kind == "ledoitwolf":
        return ad.estimate_covariance(X, method="ledoitwolf")
    return np.cov(X, rowvar=False, ddof=1) + 1e-8 * np.eye(X.shape[1])


# ============================================================
# OMPPI
# ============================================================
def exact_counts(g, c, n_l, B):
    """Minimize sum_j g_j / N_j s.t. sum_j c_j (N_j - n_l) <= B and n_l <= N_0 <= N_1 <= ...
    (pool adjacent violators on sqrt(g_j / c_j), then water-filling)."""
    g = np.asarray(g, float); c = np.asarray(c, float); L = len(g)
    blocks = []
    for j in range(L):
        blocks.append([g[j], c[j], j, j + 1])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop(); blocks[-1][0] += b[0]; blocks[-1][1] += b[1]; blocks[-1][3] = b[3]
    r = np.empty(L)
    for Gs, Cs, s, e in blocks:
        r[s:e] = math.sqrt(Gs / Cs)
    if B <= 0:
        return np.full(L, float(n_l))
    bp = n_l / r; order = np.argsort(bp, kind="stable"); s_cr = s_c = 0.0; t = np.inf
    for k, j in enumerate(order):
        s_cr += c[j] * r[j]; s_c += c[j]; t = (B + n_l * s_c) / s_cr
        if k + 1 == L or t <= bp[order[k + 1]]:
            break
    return np.maximum(float(n_l), r * t)


def omppi_stats(S, names, costs):
    d = np.diag(S)[1:]
    per = {m: {"tau2": float(S[0, j + 1] ** 2 / d[j]), "gamma": float(S[0, j + 1] / d[j]), "cost": float(costs[m]),
               "var_z": float(d[j]), "cov_yz": float(S[0, j + 1])} for j, m in enumerate(names)}
    return {"var_y": float(S[0, 0]), "per_model": per}


def route_gaps(route, st):
    tau = [st["per_model"][m]["tau2"] for m in route] + [0.0]
    return np.array([tau[j] - tau[j + 1] for j in range(len(route))])


def omppi_plugin_variance(route, st, n_l, B):
    if not route:
        return st["var_y"] / n_l, np.empty(0)
    g = route_gaps(route, st)
    c = np.array([st["per_model"][m]["cost"] for m in route])
    N = exact_counts(g, c, n_l, B)
    return (st["var_y"] - st["per_model"][route[0]]["tau2"]) / n_l + float(np.sum(g / N)), N


def omppi_best_exact_route(names, st, n_l, B, eps):
    ordered = sorted(names, key=lambda m: (-st["per_model"][m]["tau2"], m))
    best = (st["var_y"] / n_l, [])
    for r in range(1, len(ordered) + 1):
        for sub in itertools.combinations(ordered, r):
            g = route_gaps(sub, st)
            if st["var_y"] - st["per_model"][sub[0]]["tau2"] <= eps or np.any(g <= eps):
                continue
            v, _ = omppi_plugin_variance(list(sub), st, n_l, B)
            if v < best[0]:
                best = (v, list(sub))
    return best


def omppi_estimate(route, st, n_l, B, lab, pool, rng, col):
    """Nested estimator with pilot alignments; variance from final-stage moments."""
    y = lab[:, 0]; n_l = lab.shape[0]
    if not route:
        return float(y.mean()), float(y.var(ddof=1) / n_l), 0.0
    _, N_cont = omppi_plugin_variance(route, st, n_l, B)
    N = np.floor(N_cont + 1e-9).astype(int)
    n_extra = int(N[-1] - n_l)
    extra = pool[rng.integers(0, pool.shape[0], size=n_extra)] if n_extra > 0 else np.empty((0, lab.shape[1]))
    theta, var = float(y.mean()), float(y.var(ddof=1) / n_l)
    prev = n_l
    for j, m in enumerate(route):
        k = col[m]; gam = st["per_model"][m]["gamma"]
        z = np.concatenate([lab[:, k], extra[: N[j] - n_l, k]])
        theta += gam * (z[: N[j]].mean() - z[:prev].mean())
        var += (1.0 / prev - 1.0 / N[j]) * (gam ** 2 * z[: N[j]].var(ddof=1) - 2 * gam * np.cov(y, lab[:, k], ddof=1)[0, 1])
        prev = int(N[j])
    cost = float(sum(st["per_model"][m]["cost"] * (N[j] - n_l) for j, m in enumerate(route)))
    return theta, max(var, 1e-12), cost


# ============================================================
# MultiPPI
# ============================================================
def multippi_allocation(S, n_l, B, family, fam_cost):
    p = S.shape[0]; e0 = np.zeros(p); e0[0] = 1.0
    A_full = np.linalg.inv(S)
    A = []
    for idx in family:
        M = np.zeros((p, p)); M[np.ix_(idx, idx)] = np.linalg.inv(S[np.ix_(idx, idx)]); A.append(M)
    m = len(family)
    def u_of(w):
        n = B * np.maximum(w, 0.0) / fam_cost
        return np.linalg.solve(n_l * A_full + sum(n[i] * A[i] for i in range(m)), e0)
    f = lambda w: float(e0 @ u_of(w))
    def grad(w):
        u = u_of(w)
        return np.array([-(B / fam_cost[i]) * float(u @ A[i] @ u) for i in range(m)])
    cons = [{"type": "ineq", "fun": lambda w: 1.0 - np.sum(w), "jac": lambda w: -np.ones(m)}]
    best = (np.inf, np.zeros(m), False)
    for w0 in (np.full(m, 1.0 / m), np.eye(m)[int(np.argmin(fam_cost))]):
        res = minimize(f, w0, jac=grad, method="SLSQP", bounds=[(0, 1)] * m, constraints=cons, options={"maxiter": 500, "ftol": 1e-14})
        if f(res.x) < best[0]:
            best = (f(res.x), np.maximum(res.x, 0.0), bool(res.success))
    return np.floor(B * best[1] / fam_cost + 1e-9).astype(int), A_full, A, best[2]


def multippi_estimate(S, n_l, B, lab, pool, rng, family, fam_cost):
    n, A_full, A, ok = multippi_allocation(S, n_l, B, family, fam_cost)
    p = S.shape[0]; e0 = np.zeros(p); e0[0] = 1.0
    u = np.linalg.solve(n_l * A_full + sum(n[i] * A[i] for i in range(len(family))), e0)
    lam_full = n_l * (A_full @ u)
    vals = lab @ lam_full
    theta, var = float(vals.mean()), float(vals.var(ddof=1) / n_l)
    for i, idx in enumerate(family):
        if n[i] <= 0:
            continue
        lam = n[i] * (np.linalg.inv(S[np.ix_(idx, idx)]) @ u[idx])
        rows = pool[rng.integers(0, pool.shape[0], size=int(n[i]))][:, idx]
        v = rows @ lam
        theta += float(v.mean()); var += float(v.var(ddof=1) / n[i]) if n[i] > 1 else 0.0
    return theta, max(var, 1e-12), float(np.sum(n * fam_cost)), ok


# ============================================================
# VectorPPI++
# ============================================================
def vectorppi_estimate(S, n_l, B, lab, pool, rng, group_cost):
    y = lab[:, 0]; F = lab[:, 1:]; N = int(math.floor(B / group_cost + 1e-9))
    if N <= 0:
        return float(y.mean()), float(y.var(ddof=1) / n_l), 0.0
    lam = (N / (n_l + N)) * np.linalg.solve(S[1:, 1:], S[1:, 0])
    t_lab = y - F @ lam
    t_ext = pool[rng.integers(0, pool.shape[0], size=N)][:, 1:] @ lam
    return float(t_lab.mean() + t_ext.mean()), float(t_lab.var(ddof=1) / n_l + t_ext.var(ddof=1) / N), float(N * group_cost)


# ============================================================
# One repetition
# ============================================================
def run_trial(job):
    outer, inner = job
    X, lab_rows, pi, names = G["X"], G["rows"], G["pi"], G["names"]
    outer_seed = ad.seed_for_trial(G["seed"], 10_000_000 + outer)
    local_seed = ad.seed_for_trial(outer_seed, outer * G["stride"] + inner)
    rng = np.random.default_rng(local_seed)
    pool, lab, S = {}, {}, {}
    theta_pool = 0.0
    for h in G["strata"]:
        rows = lab_rows[h]
        pilot = rng.choice(rows, size=G["cnt_pilot"][h], replace=False)
        pool_rows = np.setdiff1d(rows, pilot)
        pool[h] = X[pool_rows]
        theta_pool += pi[h] * float(X[pool_rows, 0].mean())
        S[h] = covariance(X[pilot], G["cov"])
        lab[h] = X[rng.choice(pool_rows, size=G["cnt_lab"][h], replace=True)]
    st = {h: omppi_stats(S[h], names, G["costs"]) for h in G["strata"]}
    out, diag = [], []
    for b_idx, B in enumerate(G["budgets"]):
        res = {m: [0.0, 0.0, 0.0] for m in METHODS}
        for m in METHODS:
            mrng = method_rng(local_seed, b_idx, m)
            for h in G["strata"]:
                Bh, n_l = pi[h] * B, lab[h].shape[0]
                if m == "Classical":
                    y = lab[h][:, 0]; th, v, cst = float(y.mean()), float(y.var(ddof=1) / n_l), 0.0
                elif m == "VectorPPI++":
                    th, v, cst = vectorppi_estimate(S[h], n_l, Bh, lab[h], pool[h], mrng, G["group_cost"])
                elif m == "MultiPPI":
                    th, v, cst, ok = multippi_estimate(S[h], n_l, Bh, lab[h], pool[h], mrng, G["family"], G["fam_cost"])
                    diag.append({"budget": B, "stratum": h, "key": "multippi_solver_ok", "value": float(ok)})
                else:
                    if m == "OMPPI(DAG)":
                        route, _ = ad.select_route_dag(names, st[h], Bh, n_l, G["eps"])
                        v_dag, _ = omppi_plugin_variance(route, st[h], n_l, Bh)
                        route_ex, _ = ad.select_route_exhaustive(names, st[h], Bh, n_l, G["eps"])
                        v_best, _ = omppi_best_exact_route(names, st[h], n_l, Bh, G["eps"])
                        diag += [{"budget": B, "stratum": h, "key": "dag_equals_exhaustive", "value": float(route == route_ex)},
                                 {"budget": B, "stratum": h, "key": "dag_over_best_exact_route", "value": v_dag / v_best},
                                 {"budget": B, "stratum": h, "key": "labeled_only", "value": float(len(route) == 0)}]
                    else:
                        route, _ = ad.select_route_exhaustive(names, st[h], Bh, n_l, G["eps"])
                    th, v, cst = omppi_estimate(route, st[h], n_l, Bh, lab[h], pool[h], mrng, G["col"])
                res[m][0] += pi[h] * th; res[m][1] += pi[h] ** 2 * v; res[m][2] += cst
        for m in METHODS:
            th, v, cst = res[m]; half = Z * math.sqrt(v)
            out.append({"outer_trial": outer, "inner_trial": inner, "method": m, "budget": B, "theta_pool": theta_pool,
                        "theta_hat": th, "var_hat": v, "ci_low": th - half, "ci_high": th + half, "ci_width": 2 * half,
                        "covered_pool": float(th - half <= theta_pool <= th + half), "sq_error_pool": (th - theta_pool) ** 2,
                        "actual_cost": cst})
    return out, diag


def main():
    ap = argparse.ArgumentParser(description="Chatbot Arena experiment for one pilot size.")
    ap.add_argument("--n-pilot", type=int, required=True)
    ap.add_argument("--n-labeled", type=int, default=600)
    ap.add_argument("--budgets", default="100:1500:10", help="start:stop:count (linspace)")
    ap.add_argument("--n-outer-trials", type=int, default=100)
    ap.add_argument("--n-trials", type=int, default=200, help="inner repetitions per outer repetition")
    ap.add_argument("--trial-id-stride", type=int, default=None,
                    help="seed stride per outer repetition (default --n-trials); set 200 to rerun part of the paper's runs")
    ap.add_argument("--covariance", choices=["ledoitwolf", "sample"], default="ledoitwolf")
    ap.add_argument("--eps-gap", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--data", default=str(ad.DATA))
    ap.add_argument("--out-dir", default=None, help="default: results/pilot<n-pilot>")
    a = ap.parse_args()
    t0 = time.time()
    pop, names = ad.load_population(Path(a.data))
    pi = {int(h): float(v) for h, v in pop.strata_pi.items()}
    lo, hi, k = a.budgets.split(":")
    K = len(names)
    G.update(X=np.asarray(pop.X, float), names=names, pi=pi, strata=sorted(pi),
             rows={h: np.where(pop.strata_labels == h)[0] for h in pi},
             costs={m: float(pop.costs_used[m]) for m in names}, col={m: j + 1 for j, m in enumerate(names)},
             cnt_pilot=proportional_counts(a.n_pilot, pi, min_each=5), cnt_lab=proportional_counts(a.n_labeled, pi, min_each=2),
             budgets=[float(x) for x in np.linspace(float(lo), float(hi), int(k))], cov=a.covariance, eps=a.eps_gap, seed=a.seed,
             stride=a.trial_id_stride or a.n_trials,
             family=[list(range(1, K + 1))] + [[j] for j in range(1, K + 1)])
    G["fam_cost"] = np.array([sum(G["costs"].values())] + [G["costs"][m] for m in names])
    G["group_cost"] = float(sum(G["costs"].values()))
    out = Path(a.out_dir) if a.out_dir else HERE / "results" / f"pilot{a.n_pilot}"
    out.mkdir(parents=True, exist_ok=True)
    config = {k_: v for k_, v in vars(a).items() if k_ not in ("data", "out_dir")}
    config.update(budgets_used=G["budgets"], pilot_counts=G["cnt_pilot"], labeled_counts=G["cnt_lab"], strata_pi=pi,
                  costs_used=G["costs"], rows=int(G["X"].shape[0]), judges=names)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"rows {G['X'].shape[0]}, pilot counts {G['cnt_pilot']}, labeled counts {G['cnt_lab']}, budgets {np.round(G['budgets'], 1).tolist()}", flush=True)
    jobs = [(o, i) for o in range(a.n_outer_trials) for i in range(a.n_trials)]
    rows, diags = [], []
    with mp.get_context("fork").Pool(a.num_workers) as pool:
        for done, (r, d) in enumerate(pool.imap(run_trial, jobs, chunksize=4), start=1):
            rows += r; diags += d
            if done % 2000 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} repetitions, {time.time() - t0:.0f} s", flush=True)
    trials = pd.DataFrame(rows)
    trials.to_csv(out / "trials.csv", index=False)
    g = trials.groupby(["method", "budget"])
    summary = pd.DataFrame({
        "n_trials": g.size(), "n_nan": g["theta_hat"].apply(lambda x: int(x.isna().sum())),
        "coverage_pool": g["covered_pool"].mean(), "rmse_pool": np.sqrt(g["sq_error_pool"].mean()),
        "mean_bias_pool": g.apply(lambda x: float((x.theta_hat - x.theta_pool).mean())),
        "mean_ci_width": g["ci_width"].mean(), "mean_var_hat": g["var_hat"].mean(),
        "actual_cost_mean": g["actual_cost"].mean(), "actual_cost_max": g["actual_cost"].max()}).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(diags).groupby(["key", "budget"])["value"].agg(["mean", "min", "max", "count"]).reset_index().to_csv(out / "diagnostics_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"done in {time.time() - t0:.0f} s; outputs in {out}")


if __name__ == "__main__":
    main()
