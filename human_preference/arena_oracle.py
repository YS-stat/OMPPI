#!/usr/bin/env python3
"""Oracle reductions for the Chatbot Arena experiment (Figure D.4 and the oracle row of Table D.2).

Designs use the population covariance Sigma of each stratum of the 823 comparisons, with the labeled
layer n_{L,h} = pi_h n_L (n_L = 600) and B_h = pi_h B as in arena_experiment.py. For each budget:
width ratio to LO of OMPPI (best admissible chain, exact continuous allocation), MultiPPI (full,
joint and singleton blocks), VectorPPI++, and the best linear unbiased design over all 1023 judge
subsets (the oracle, Delta_oracle(B) = 1 - its ratio); share = (1 - r) / (1 - r_oracle).
Also reports the limit of the oracle ratio as B -> infinity (all ten judges on every comparison).
Writes results/oracle.csv.
"""
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import arena_data as ad
from arena_experiment import exact_counts

HERE = Path(__file__).resolve().parent
BUDGETS = [float(x) for x in np.linspace(100, 1500, 10)]
N_L, EPS = 600, 1e-4


def omppi(S, Sig, n_l, B, c):
    """Best admissible chain from S; returns (true variance under Sig, plug-in variance under S)."""
    var_y = S[0, 0]; d = np.diag(S)[1:]
    tau = S[0, 1:] ** 2 / d; gam = S[0, 1:] / d
    order = sorted(range(len(tau)), key=lambda k: (-tau[k], k))
    best = (var_y / n_l, (), None)
    for r in range(1, len(order) + 1):
        for sub in itertools.combinations(order, r):
            t = [tau[k] for k in sub] + [0.0]
            g = np.array([t[j] - t[j + 1] for j in range(r)])
            if var_y - t[0] <= EPS or np.any(g <= EPS):
                continue
            cc = c[list(sub)]
            if np.any(np.diff(g / cc) < -1e-12):
                continue
            N = exact_counts(g, cc, n_l, B)
            v = (var_y - t[0]) / n_l + float(np.sum(g / N))
            if v < best[0]:
                best = (v, sub, N)
    v_plug, route, N = best
    v = Sig[0, 0] / n_l
    prev = n_l
    for j, k in enumerate(route):
        v += (1.0 / prev - 1.0 / N[j]) * (gam[k] ** 2 * Sig[k + 1, k + 1] - 2 * gam[k] * Sig[0, k + 1]); prev = N[j]
    return v, v_plug


def multippi(S, Sig, n_l, B, blocks, costs):
    """Optimal continuous allocation over `blocks` (plus the fixed labeled full block) from S."""
    p = S.shape[0]; e0 = np.zeros(p); e0[0] = 1.0
    A_full = np.linalg.inv(S); m = len(blocks)
    A = []
    for idx in blocks:
        M = np.zeros((p, p)); M[np.ix_(idx, idx)] = np.linalg.inv(S[np.ix_(idx, idx)]); A.append(M)
    def parts(w):
        n = B * np.maximum(w, 0.0) / costs
        M = n_l * A_full + sum(n[i] * A[i] for i in range(m))
        return n, M, np.linalg.solve(M, e0)
    def f(w):
        return float(e0 @ parts(w)[2])
    def grad(w):
        n, M, u = parts(w)
        return np.array([-(B / costs[i]) * float(u @ A[i] @ u) for i in range(m)])
    cons = [{"type": "ineq", "fun": lambda w: 1.0 - np.sum(w), "jac": lambda w: -np.ones(m)}]
    best = (np.inf, None)
    for w0 in [np.full(m, 1.0 / m), np.eye(m)[int(np.argmin(costs))]]:
        res = minimize(f, w0, jac=grad, method="SLSQP", bounds=[(0, 1)] * m, constraints=cons, options={"maxiter": 500, "ftol": 1e-14})
        if f(res.x) < best[0]:
            best = (f(res.x), np.maximum(res.x, 0.0))
    n, M, u = parts(best[1])
    Sig_inv_parts = n_l * A_full @ Sig @ A_full + sum(n[i] * A[i] @ Sig @ A[i] for i in range(m) if n[i] > 0)
    return float(u @ Sig_inv_parts @ u), best[0]


def vectorppi(S, Sig, n_l, B, c):
    Nx = B / float(np.sum(c))
    lam = (Nx / (n_l + Nx)) * np.linalg.solve(S[1:, 1:], S[1:, 0])
    q = float(lam @ Sig[1:, 1:] @ lam)
    return (Sig[0, 0] - 2 * float(lam @ Sig[1:, 0]) + q) / n_l + (q / Nx if Nx > 0 else 0.0)


def ceiling(Sig, n_l, B, c, iters=1500):
    """Frank-Wolfe over all 1023 judge subsets (additive prices) with the fixed labeled full block.
    M(w) is affine in the budget shares w, so M((1-a)w + a e_i) = (1-a) M(w) + a M(e_i)."""
    K = Sig.shape[0] - 1; p = Sig.shape[0]; e0 = np.zeros(p); e0[0] = 1.0
    subsets = [list(s) for r in range(1, K + 1) for s in itertools.combinations(range(1, K + 1), r)]
    cost = np.array([sum(c[j - 1] for j in s) for s in subsets])
    E = np.zeros((len(subsets), p, p))
    for i, s in enumerate(subsets):
        E[i][np.ix_(s, s)] = (B / cost[i]) * np.linalg.inv(Sig[np.ix_(s, s)])
    M_lab = n_l * np.linalg.inv(Sig)
    obj = lambda M: float(e0 @ np.linalg.solve(M, e0))
    Mw = M_lab + E[int(np.argmin(cost))]
    for it in range(iters):
        u = np.linalg.solve(Mw, e0)
        gain = np.einsum("i,kij,j->k", u, E, u)
        Mv = M_lab + E[int(np.argmax(gain))]
        lo, hi = 0.0, 1.0
        for _ in range(40):
            a1, a2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if obj((1 - a1) * Mw + a1 * Mv) < obj((1 - a2) * Mw + a2 * Mv):
                hi = a2
            else:
                lo = a1
        a = 0.5 * (lo + hi)
        Mw = (1 - a) * Mw + a * Mv
    return obj(Mw)


def main():
    pop, names = ad.load_population()
    X, lab = pop.X.astype(float), pop.strata_labels
    pi = {int(h): float(v) for h, v in pop.strata_pi.items()}
    C = np.array([pop.costs_used[m] for m in names]); K = len(names)
    family = [list(range(1, K + 1))] + [[k] for k in range(1, K + 1)]
    fam_cost = np.array([C.sum()] + list(C))
    sig = {h: np.cov(X[lab == h], rowvar=False, ddof=0) for h in pi}
    num = sum(pi[h] ** 2 * (S[0, 0] - S[0, 1:] @ np.linalg.solve(S[1:, 1:], S[1:, 0])) / (pi[h] * N_L) for h, S in sig.items())
    den = sum(pi[h] ** 2 * S[0, 0] / (pi[h] * N_L) for h, S in sig.items())
    r_inf = math.sqrt(num / den)
    rows = []
    for B in BUDGETS:
        tot = dict.fromkeys(("OMPPI", "MultiPPI", "VectorPPI++", "ceiling", "LO"), 0.0)
        for h, S in sig.items():
            n_l, Bh, w2 = pi[h] * N_L, pi[h] * B, pi[h] ** 2
            tot["OMPPI"] += w2 * omppi(S, S, n_l, Bh, C)[0]
            tot["MultiPPI"] += w2 * multippi(S, S, n_l, Bh, family, fam_cost)[0]
            tot["VectorPPI++"] += w2 * vectorppi(S, S, n_l, Bh, C)
            tot["ceiling"] += w2 * ceiling(S, n_l, Bh, C)
            tot["LO"] += w2 * S[0, 0] / n_l
        r = {k: math.sqrt(v / tot["LO"]) for k, v in tot.items() if k != "LO"}
        row = {"budget": B, **{f"ratio_{k}": round(v, 5) for k, v in r.items()}}
        row.update({f"share_{k}": round((1 - r[k]) / (1 - r["ceiling"]), 4) for k in ("OMPPI", "MultiPPI", "VectorPPI++")})
        row["ratio_ceiling_inf"] = round(r_inf, 5)
        rows.append(row)
        print(f"B={B:7.1f}: " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()) + "  | share " +
              "  ".join(f"{k} {row['share_' + k]:.0%}" for k in ("OMPPI", "MultiPPI", "VectorPPI++")), flush=True)
    print(f"B=infinity, all ten judges: {r_inf:.4f}")
    (HERE / "results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(HERE / "results" / "oracle.csv", index=False)


if __name__ == "__main__":
    main()
