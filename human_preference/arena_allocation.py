#!/usr/bin/env python3
"""Allocations of the Chatbot Arena runs at the largest budget (Figure D.5).

Replays the pilots of the first 200 repetitions of each run (same seeds and draw order as
arena_experiment.py) and recomputes the OMPPI(DAG) and MultiPPI designs at B = 1500 (B_h = pi_h B,
labeled rows per stratum as in the experiment). Records, per pilot size, stratum and source, the mean
number of extra queries and their mean cost: for OMPPI a source is a judge queried on extra rows, for
MultiPPI the joint block of all ten judges or a singleton. Writes results/allocation.csv.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import arena_data as ad
import arena_experiment as ex

HERE = Path(__file__).resolve().parent
PILOTS, N_REP, B, N_L, SEED, STRIDE, EPS = [50, 100, 200, 400], 200, 1500.0, 600, 123, 200, 1e-4


def main():
    pop, names = ad.load_population()
    X = np.asarray(pop.X, float)
    pi = {int(h): float(v) for h, v in pop.strata_pi.items()}
    strata = sorted(pi)
    rows = {h: np.where(pop.strata_labels == h)[0] for h in strata}
    costs = {m: float(pop.costs_used[m]) for m in names}
    K = len(names)
    family = [list(range(1, K + 1))] + [[j] for j in range(1, K + 1)]
    fam_cost = np.array([sum(costs.values())] + [costs[m] for m in names])
    cnt_lab = ex.proportional_counts(N_L, pi, min_each=2)
    out = []
    for n_p in PILOTS:
        cnt_p = ex.proportional_counts(n_p, pi, min_each=5)
        acc = defaultdict(lambda: [0.0, 0.0])
        outer_seed = ad.seed_for_trial(SEED, 10_000_000 + 0)
        for inner in range(N_REP):
            rng = np.random.default_rng(ad.seed_for_trial(outer_seed, 0 * STRIDE + inner))
            for h in strata:
                pilot = rng.choice(rows[h], size=cnt_p[h], replace=False)
                pool_rows = np.setdiff1d(rows[h], pilot)
                S = ex.covariance(X[pilot], "ledoitwolf")
                rng.choice(pool_rows, size=cnt_lab[h], replace=True)          # labeled draw (keeps the draw order)
                n_l, Bh = cnt_lab[h], pi[h] * B
                st = ex.omppi_stats(S, names, costs)
                route, _ = ad.select_route_dag(names, st, Bh, n_l, EPS)
                if route:
                    _, N = ex.omppi_plugin_variance(route, st, n_l, Bh)
                    for j, m in enumerate(route):
                        extra = int(np.floor(N[j] + 1e-9)) - n_l
                        acc[("OMPPI", h, m)][0] += extra; acc[("OMPPI", h, m)][1] += extra * costs[m]
                n, *_ = ex.multippi_allocation(S, n_l, Bh, family, fam_cost)
                for i, src in enumerate(["Joint"] + names):
                    acc[("MultiPPI", h, src)][0] += n[i]; acc[("MultiPPI", h, src)][1] += n[i] * fam_cost[i]
        for (meth, h, src), (cnt, cst) in sorted(acc.items()):
            out.append({"pilot": n_p, "method": meth, "stratum": h, "source": src,
                        "mean_extra_queries": cnt / N_REP, "mean_cost": cst / N_REP})
        print(f"pilot {n_p}: done", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    pd.DataFrame(out).to_csv(HERE / "results" / "allocation.csv", index=False)
    summ = pop.strata_token_summary
    print("strata token ranges:", {h: (summ[h]["min"], summ[h]["max"]) for h in sorted(summ)})


if __name__ == "__main__":
    main()
