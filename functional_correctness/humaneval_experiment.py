#!/usr/bin/env python3
"""HumanEval+ functional-correctness experiment (Section 3.3 and Appendix E).

Target: pass@1 of deepseek-coder-6.7b-base on HumanEval+, theta = sum_h pi_h E(Y | stratum h), where
Y indicates that a completion passes the full HumanEval+ test suite. Predictors: the partial
evaluators Plus50, Plus25, Plus10, OriginalTests and StaticOK. Costs are test workloads normalized by
the full suite, and a query that runs several evaluators costs the most expensive one (the suites are
nested, so running a suite also yields every cheaper evaluator). Five prompt-length strata.

Each repetition: a stratified pilot is drawn without replacement and removed from the pool. Every
method computes its design (stratum budget split, allocation, coefficients and, for OMPPI, the active
predictor set) from one Ledoit-Wolf estimate of Cov(Y, F) per stratum of the pilot. For each budget,
each method draws its final samples with replacement from the pool within each stratum and reports a
95% interval whose variance uses the final-stage samples. Coverage and RMSE are measured against
theta_pool, the pi-weighted mean of Y over the pool. Repetitions are organized as outer x inner; each
outer repetition also draws a stratified 4500-row subset whose pi-weighted mean (theta_true) is
recorded in trials.csv.

Methods: Classical (LO), VectorPPI++, MultiPPI (full, joint and singleton blocks), OMPPI(Exhaustive)
and OMPPI(DAG), the DAG search of Appendix A (it selects the same design as exhaustive search).

Outputs in --out-dir: trials.csv, summary.csv, allocation_summary.csv, diagnostics_summary.csv,
outer_truth.csv and config.json.
Usage: python humaneval_experiment.py --n-pilot 800 --n-trials 500     (writes results/pilot800/)
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# One BLAS/OpenMP thread per process: repetitions run in parallel worker processes, and multithreaded
# BLAS on 6x6 matrices only oversubscribes the CPU. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import tiktoken
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "humaneval_plus" / "humaneval_plus_completions.csv"
TARGET_COL = "Y_full_plus"
PREDICTION_COLS = ["f_plus_50", "f_plus_25", "f_plus_10", "f_original_tests", "f_static_ok"]
PREDICTION_NAMES = ["Plus50", "Plus25", "Plus10", "OriginalTests", "StaticOK"]
Y_COST_COL = "cost_full_plus"
PREDICTION_COST_COLS = ["cost_plus_50", "cost_plus_25", "cost_plus_10", "cost_original_tests", "cost_static_ok"]
NUM_STRATA = 5
COST_FLOOR = 1e-4
COVARIANCE_METHOD = "ledoitwolf"
RIDGE = 1e-8
EPS_GAP = 1e-4
ALPHA = 0.05
STRATUM_MIN_LABELS = 2      # every stratum receives at least the cost of two full evaluations


# ============================================================
# Data
# ============================================================

@dataclass
class ExecutionPopulation:
    df: pd.DataFrame
    Y: np.ndarray
    S: np.ndarray
    X: np.ndarray                      # columns: Y, then the predictors
    prediction_names: List[str]
    costs_raw: Dict[str, float]
    costs_used: Dict[str, float]
    y_cost_raw: float
    y_cost_used: float
    theta_true: float
    strata_labels: np.ndarray
    strata_pi: Dict[int, float]
    strata_summary: Dict[int, Dict[str, float]]

@dataclass
class MethodSpec:
    name: str
    kind: str
    extra: List[str] = field(default_factory=list)


def count_tokens(texts: Sequence[str]) -> np.ndarray:
    enc = tiktoken.get_encoding("o200k_base")
    vals = ["" if x is None else str(x) for x in texts]
    return np.asarray([len(enc.encode(v)) for v in vals], dtype=int)

def make_quantile_strata(values: np.ndarray, num_strata: int) -> np.ndarray:
    ranks = pd.Series(np.asarray(values)).rank(method="first")
    labels = np.asarray(pd.qcut(ranks, q=num_strata, labels=False, duplicates="drop"), dtype=int)
    remap = {old: new for new, old in enumerate(sorted(np.unique(labels).tolist()))}
    return np.asarray([remap[int(x)] for x in labels], dtype=int)

def summarize_strata(values: np.ndarray, labels: np.ndarray) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    values = np.asarray(values, dtype=float)
    for h in sorted(np.unique(labels).tolist()):
        vals = values[labels == h]
        out[int(h)] = {"count": int(vals.size), "min": float(np.min(vals)), "max": float(np.max(vals)),
                       "mean": float(np.mean(vals)), "median": float(np.median(vals))}
    return out

def load_population(path: str | Path = DATA) -> ExecutionPopulation:
    """Read the completion table, form prompt-length strata and normalize the evaluator costs."""
    df = pd.read_csv(path)
    use_cols = [TARGET_COL] + PREDICTION_COLS
    work = df.copy()
    for col in use_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.loc[work[use_cols].notna().all(axis=1)].copy().reset_index(drop=True)

    token_counts = count_tokens(work["prompt"].astype(str).tolist())
    labels = make_quantile_strata(token_counts, NUM_STRATA)
    work["stratum"] = labels
    pi = {int(h): float(np.mean(labels == h)) for h in sorted(np.unique(labels).tolist())}

    y_cost_raw = max(float(pd.to_numeric(work[Y_COST_COL], errors="coerce").fillna(0.0).mean()), COST_FLOOR)
    costs_raw = {name: max(float(pd.to_numeric(work[col], errors="coerce").fillna(0.0).mean()), COST_FLOOR)
                 for name, col in zip(PREDICTION_NAMES, PREDICTION_COST_COLS)}
    denom = max(y_cost_raw, COST_FLOOR)
    costs_used = {m: max(float(costs_raw[m]) / denom, COST_FLOOR) for m in PREDICTION_NAMES}

    Y = work[TARGET_COL].astype(float).to_numpy()
    S = work[PREDICTION_COLS].astype(float).to_numpy()
    return ExecutionPopulation(
        df=work[[TARGET_COL] + PREDICTION_COLS + ["stratum"]].copy(), Y=Y, S=S, X=np.column_stack([Y, S]),
        prediction_names=list(PREDICTION_NAMES), costs_raw=costs_raw, costs_used=costs_used,
        y_cost_raw=float(y_cost_raw), y_cost_used=1.0, theta_true=float(np.mean(Y)),
        strata_labels=labels.astype(int), strata_pi=pi, strata_summary=summarize_strata(token_counts, labels),
    )

def subset_population_by_indices(pop: ExecutionPopulation, idx: np.ndarray) -> ExecutionPopulation:
    idx = np.asarray(idx, dtype=int)
    labels = pop.strata_labels[idx]
    y = pop.Y[idx]
    return ExecutionPopulation(
        df=pop.df.iloc[idx].copy().reset_index(drop=True), Y=y, S=pop.S[idx], X=pop.X[idx],
        prediction_names=list(pop.prediction_names), costs_raw=dict(pop.costs_raw), costs_used=dict(pop.costs_used),
        y_cost_raw=float(pop.y_cost_raw), y_cost_used=float(pop.y_cost_used),
        theta_true=float(np.mean(y)) if len(y) else float("nan"), strata_labels=labels.astype(int),
        strata_pi={int(h): float(np.mean(labels == h)) for h in sorted(np.unique(labels).tolist())}, strata_summary={},
    )


# ============================================================
# Utilities
# ============================================================

def parse_budgets(s: str) -> List[float]:
    start, stop, num = s.split(":")
    return [float(x) for x in np.linspace(float(start), float(stop), int(num))]

def _normal_z(alpha: float = 0.05) -> float:
    if abs(alpha - 0.05) < 1e-12:
        return 1.959963984540054
    from scipy.stats import norm
    return float(norm.ppf(1.0 - alpha / 2.0))

def seed_for_trial(base_seed: int, trial_id: int) -> int:
    ss = np.random.SeedSequence([int(base_seed), int(trial_id), 20260527])
    return int(ss.generate_state(1, dtype=np.uint64)[0] % np.uint64(2**63 - 1))

def safe_sample_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return 0.0
    return float(np.var(x, ddof=1))

def safe_sample_cov(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size <= 1:
        return 0.0
    return float(np.sum((x - x.mean()) * (y - y.mean())) / (x.size - 1))

def _regularize_covariance(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    return cov + eps * np.eye(cov.shape[0], dtype=float)

def _matrix_inverse(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mat = _regularize_covariance(mat, eps=eps)
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(mat)

def estimate_covariance(X: np.ndarray, method: str = "ledoitwolf", eps: float = 1e-8) -> np.ndarray:
    """Ledoit-Wolf ("ledoitwolf") or unshrunk sample covariance ("sample"), plus eps * I."""
    X = np.asarray(X, dtype=float)
    if method == "ledoitwolf":
        lw = LedoitWolf(store_precision=False, assume_centered=False)
        lw.fit(X)
        return _regularize_covariance(lw.covariance_, eps=eps)
    return _regularize_covariance(np.cov(X, rowvar=False, ddof=1), eps=eps)

def joint_prediction_cost(costs: Mapping[str, float], prediction_names: Sequence[str]) -> float:
    """Cost of a query that runs several partial evaluators: that of the most expensive one."""
    vals = [float(costs[m]) for m in prediction_names]
    return float(max(vals)) if vals else 0.0


# ============================================================
# Stratified sampling
# ============================================================

def sample_indices_stratified(
    labels: np.ndarray,
    pi_map: Mapping[int, float],
    n: int,
    rng: np.random.Generator,
    *,
    replace: bool,
    exclude: Optional[np.ndarray] = None,
    min_each: int = 0,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if n <= 0:
        return np.empty(0, dtype=int)
    all_idx = np.arange(labels.shape[0], dtype=int)
    if exclude is not None and len(exclude):
        mask = np.ones(labels.shape[0], dtype=bool)
        mask[np.asarray(exclude, dtype=int)] = False
        all_idx = all_idx[mask]
    keys = [int(h) for h in sorted(pi_map.keys())]
    probs = np.asarray([max(float(pi_map[h]), 0.0) for h in keys], dtype=float)
    probs = probs / probs.sum()
    counts = rng.multinomial(n, probs)
    if min_each > 0 and n >= min_each * len(keys):
        counts = np.maximum(counts, min_each)
        # Remove surplus greedily from largest counts.
        while counts.sum() > n:
            j = int(np.argmax(counts))
            if counts[j] > min_each:
                counts[j] -= 1
            else:
                break
        while counts.sum() < n:
            j = int(rng.choice(len(keys), p=probs))
            counts[j] += 1

    pieces: List[np.ndarray] = []
    for h, cnt in zip(keys, counts):
        if cnt <= 0:
            continue
        idx_h = all_idx[labels[all_idx] == h]
        if idx_h.size == 0:
            continue
        if not replace and cnt > idx_h.size:
            cnt = idx_h.size
        chosen = rng.choice(idx_h, size=int(cnt), replace=replace)
        pieces.append(np.asarray(chosen, dtype=int))
    if not pieces:
        return np.empty(0, dtype=int)
    out = np.concatenate(pieces)
    rng.shuffle(out)
    return out

def sample_theta_truth_subset_stratified(
    population: ExecutionPopulation,
    truth_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, Dict[int, int]]:
    """Stratified subset of truth_size rows and its pi-weighted mean (theta_true of an outer repetition)."""
    if not 0 < truth_size <= population.Y.shape[0]:
        raise ValueError(f"truth subset size {truth_size} must be in (0, {population.Y.shape[0]}]")
    truth_idx = sample_indices_stratified(
        population.strata_labels, population.strata_pi, truth_size, rng, replace=False, min_each=1,
    )
    counts: Dict[int, int] = {}
    theta_true = 0.0
    for h in sorted(population.strata_pi):
        mask_h = population.strata_labels[truth_idx] == int(h)
        counts[int(h)] = int(np.sum(mask_h))
        if counts[int(h)] <= 0:
            continue
        theta_true += float(population.strata_pi[h]) * float(np.mean(population.Y[truth_idx[mask_h]]))
    return np.asarray(truth_idx, dtype=int), float(theta_true), counts


# ============================================================
# Pilot statistics
# ============================================================

def compute_scalar_stats_from_cov(Sigma: np.ndarray, n: int, prediction_names: Sequence[str], costs: Mapping[str, float], y_cost: float) -> Dict[str, Any]:
    """Marginal moments of each predictor (OMPPI), read off a covariance estimate of (Y, S_1, ..., S_K)."""
    Sigma = np.asarray(Sigma, dtype=float)
    var_y = float(Sigma[0, 0])
    out: Dict[str, Any] = {"n": int(n), "mean_y": float("nan"), "var_y": var_y, "y_cost": float(y_cost), "per_model": {}}
    for j, name in enumerate(prediction_names, start=1):
        var_z = float(Sigma[j, j])
        cov_yz = float(Sigma[0, j])
        gamma = 0.0 if var_z <= 1e-15 else cov_yz / var_z
        tau2 = 0.0 if var_z <= 1e-15 else (cov_yz ** 2) / var_z
        tau2 = float(min(max(tau2, 0.0), max(var_y - 1e-12, 0.0)))
        out["per_model"][name] = {"var_z": var_z, "cov_yz": cov_yz, "gamma": float(gamma), "tau2": tau2, "cost": float(costs[name])}
    return out

def compute_vector_stats_from_cov(Sigma: np.ndarray, prediction_names: Sequence[str], costs: Mapping[str, float], y_cost: float) -> Dict[str, Any]:
    """Joint regression of Y on all predictors (VectorPPI++), from a covariance estimate of (Y, S)."""
    Sigma = np.asarray(Sigma, dtype=float)
    var_y = float(Sigma[0, 0])
    Sigma_zz = Sigma[1:, 1:]
    cov_zy = Sigma[1:, 0]
    try:
        gamma = np.linalg.solve(Sigma_zz, cov_zy)
    except np.linalg.LinAlgError:
        gamma = np.linalg.pinv(Sigma_zz) @ cov_zy
    tau2 = float(min(max(float(cov_zy @ gamma), 0.0), max(var_y - 1e-12, 0.0)))
    return {
        "var_y": var_y,
        "tau2": tau2,
        "gamma": np.asarray(gamma, dtype=float).tolist(),
        "Sigma_zz": Sigma_zz,
        "cov_zy": cov_zy,
        "joint_cost": joint_prediction_cost(costs, prediction_names),
        "y_cost": float(y_cost),
    }


# ============================================================
# Classical and VectorPPI++
# ============================================================

def ci_from_est(theta: float, var_hat: float, alpha: float) -> Dict[str, float]:
    var_hat = max(float(var_hat), 0.0)
    half = _normal_z(alpha) * math.sqrt(var_hat)
    return {
        "theta_hat": float(theta),
        "var_hat": float(var_hat),
        "ci_low": float(theta - half),
        "ci_high": float(theta + half),
        "width": float(2.0 * half),
    }

def run_classical_stratum(pop_h: ExecutionPopulation, budget_h: float, alpha: float, rng: np.random.Generator, stats_h: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    c0 = float(pop_h.y_cost_used)
    n = int(math.floor(max(budget_h, 0.0) / max(c0, 1e-12)))
    if n < 2:
        return {
            "theta_hat": float("nan"), "var_hat": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "width": float("nan"), "actual_cost": 0.0, "budget_left": float(budget_h)
        }, {"n_y": n, "warning": "budget too small for at least two labels"}
    idx = rng.integers(0, pop_h.X.shape[0], size=n)
    y = pop_h.Y[idx]
    theta = float(np.mean(y))
    var_hat = safe_sample_var(y) / n
    est = ci_from_est(theta, var_hat, alpha)
    actual_cost = n * c0
    est.update({"actual_cost": float(actual_cost), "budget_left": float(budget_h - actual_cost)})
    return est, {"n_y": int(n), "cost_y": c0}

def allocate_two_level(var_y: float, tau2: float, c0: float, c1: float, budget: float) -> Optional[Tuple[int, int, float]]:
    tau2 = float(max(tau2, 0.0))
    g0 = max(var_y - tau2, 0.0)
    g1 = max(tau2, 0.0)
    if budget <= 0 or c0 <= 0 or c1 <= 0 or g1 <= 1e-15:
        return None
    q = math.sqrt(g0 * c0) + math.sqrt(g1 * c1)
    if q <= 0:
        return None
    n0 = int(math.floor(budget / q * math.sqrt(g0 / c0))) if g0 > 0 else 2
    n1 = int(math.floor(budget / q * math.sqrt(g1 / c1)))
    n0 = max(n0, 2)
    n1 = max(n1, n0)
    # If rounding exceeds budget, shrink n1 first, then n0.
    while c0 * n0 + c1 * n1 > budget + 1e-12 and n1 > n0:
        n1 -= 1
    while c0 * n0 + c1 * n1 > budget + 1e-12 and n0 > 2:
        n0 -= 1
        n1 = max(n1, n0)
    if c0 * n0 + c1 * n1 > budget + 1e-12:
        return None
    return n0, n1, float(q)

def vector_ppi_level_costs(y_cost: float, joint_cost: float) -> Tuple[float, float]:
    """Level costs of the two-layer VectorPPI++ design (n0 <= n1): a labeled draw runs the full suite
    and yields every predictor, so the total c_Y n0 + c_F (n1 - n0) equals (c_Y - c_F) n0 + c_F n1."""
    return float(y_cost) - float(joint_cost), float(joint_cost)

def run_vector_ppi_stratum(pop_h: ExecutionPopulation, budget_h: float, alpha: float, rng: np.random.Generator, stats_h: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    vstats = stats_h["vector_stats"]
    var_y = float(vstats["var_y"])
    tau2 = float(vstats["tau2"])
    c0, c1 = vector_ppi_level_costs(float(pop_h.y_cost_used), float(vstats["joint_cost"]))
    if c0 <= 1e-12 or c1 <= 1e-12:
        return run_classical_stratum(pop_h, budget_h, alpha, rng, stats_h)
    alloc = allocate_two_level(var_y, tau2, c0, c1, float(budget_h))
    if alloc is None:
        return run_classical_stratum(pop_h, budget_h, alpha, rng, stats_h)
    n0, n1, q = alloc
    idx = rng.integers(0, pop_h.X.shape[0], size=n1)
    y = pop_h.Y[idx[:n0]]
    Z = pop_h.S[idx, :]
    gamma = np.asarray(vstats["gamma"], dtype=float)
    theta = float(np.mean(y) + gamma @ (Z.mean(axis=0) - Z[:n0, :].mean(axis=0)))
    # Final-stage moments of the combined prediction Z gamma: variance over all n1 draws,
    # covariance with Y over the n0 labeled draws.
    zg = Z @ gamma
    adj = safe_sample_var(zg) - 2.0 * safe_sample_cov(zg[:n0], y)
    var_hat = safe_sample_var(y) / n0 + (1.0 / n0 - 1.0 / n1) * adj
    est = ci_from_est(theta, var_hat, alpha)
    actual_cost = c0 * n0 + c1 * n1
    est.update({"actual_cost": float(actual_cost), "budget_left": float(budget_h - actual_cost)})
    return est, {"n_y": int(n0), "n_joint": int(n1), "q_hat": q, "tau2": tau2,
                 "joint_cost": float(vstats["joint_cost"]), "level_costs": [float(c0), float(c1)]}


# ============================================================
# OMPPI
# ============================================================

def _omppi_nested_incremental_costs(y_cost: float, route_costs: Sequence[float], eps: float = 1e-12) -> Optional[List[float]]:
    """Incremental level costs of a nested route.

    A sample evaluated at one fidelity level also has every cheaper evaluator of the route, so for
    cumulative costs c0 > c1 > ... > cL (c0 the full suite) the budget c0 n0 + c1 (n1 - n0) + ... +
    cL (nL - n_{L-1}) equals (c0 - c1) n0 + (c1 - c2) n1 + ... + cL nL.
    """
    cumulative = [float(y_cost)] + [float(c) for c in route_costs]
    if any((not np.isfinite(c)) or c <= 0 for c in cumulative):
        return None
    # The route is sorted by decreasing cost; reject non-nested cost orders.
    for j in range(len(cumulative) - 1):
        if cumulative[j] + eps < cumulative[j + 1]:
            return None
    incremental: List[float] = []
    for j in range(len(cumulative) - 1):
        incremental.append(float(cumulative[j] - cumulative[j + 1]))
    incremental.append(float(cumulative[-1]))
    # Equal costs would make the continuous allocation divide by zero.
    if any((not np.isfinite(c)) or c <= eps for c in incremental):
        return None
    return incremental

def _nested_actual_cost_from_cumulative_costs(counts: Sequence[int], cumulative_costs: Sequence[float]) -> float:
    """Nested evaluator cost for counts n0 <= n1 <= ... <= nL and cumulative costs c0, c1, ..., cL."""
    ns = [int(x) for x in counts]
    cs = [float(x) for x in cumulative_costs]
    if len(ns) != len(cs):
        raise ValueError("counts and cumulative_costs must have the same length")
    if len(ns) == 0:
        return 0.0
    total = cs[0] * ns[0]
    for j in range(1, len(ns)):
        total += cs[j] * max(ns[j] - ns[j - 1], 0)
    return float(total)

def route_q_and_alloc(route: Sequence[str], stats: Dict[str, Any], budget: float, eps_gap: float) -> Optional[Dict[str, Any]]:
    var_y = float(stats["var_y"])
    c0 = float(stats["y_cost"])
    if len(route) == 0:
        return None
    tau = [float(stats["per_model"][m]["tau2"]) for m in route]
    route_costs = [float(stats["per_model"][m]["cost"]) for m in route]

    # Need strictly decreasing explained variation along the selected route.
    if var_y - tau[0] <= eps_gap:
        return None
    for j in range(len(route) - 1):
        if tau[j] - tau[j + 1] <= eps_gap:
            return None
    if tau[-1] <= eps_gap:
        return None

    gaps = [var_y - tau[0]]
    gaps.extend(tau[j] - (tau[j + 1] if j + 1 < len(route) else 0.0) for j in range(len(route)))

    cumulative_costs = [c0] + route_costs
    level_costs = _omppi_nested_incremental_costs(c0, route_costs)
    if level_costs is None:
        return None

    if any(g <= eps_gap for g in gaps) or any(c <= 0 for c in level_costs):
        return None

    # Nested allocation requires n0 <= n1 <= ...; check continuous allocation.
    ratios = [g / c for g, c in zip(gaps, level_costs)]
    for j in range(len(ratios) - 1):
        if ratios[j] > ratios[j + 1] + 1e-12:
            return None

    q = float(np.sum(np.sqrt(np.asarray(gaps) * np.asarray(level_costs))))
    if q <= 0 or budget <= 0:
        return None
    n_cont = [budget / q * math.sqrt(g / c) for g, c in zip(gaps, level_costs)]
    n_int = [max(2, int(math.floor(n_cont[0])))]
    for val in n_cont[1:]:
        n_int.append(max(n_int[-1], int(math.floor(val))))

    def _cost(ns: Sequence[int]) -> float:
        return float(sum(c * n for c, n in zip(level_costs, ns)))

    # Reduce if rounding/enforcing monotone sample sizes slightly exceeds budget.
    while _cost(n_int) > budget + 1e-12 and n_int[-1] > n_int[-2]:
        n_int[-1] -= 1

    # If still over budget, shrink from the right while preserving monotonicity.
    guard = 0
    while _cost(n_int) > budget + 1e-12 and guard < 100000:
        guard += 1
        changed = False
        for j in range(len(n_int) - 1, -1, -1):
            lower = 2 if j == 0 else n_int[j - 1]
            if n_int[j] > lower:
                n_int[j] -= 1
                changed = True
                break
        if not changed:
            break
    if _cost(n_int) > budget + 1e-12:
        return None

    actual_cost = _cost(n_int)
    # Sanity check: this should match the direct nested cumulative-cost formula.
    direct_cost = _nested_actual_cost_from_cumulative_costs(n_int, cumulative_costs)
    if not np.isfinite(direct_cost) or abs(actual_cost - direct_cost) > 1e-6 * max(1.0, actual_cost):
        return None

    return {
        "route": list(route),
        "gaps": [float(x) for x in gaps],
        "costs": [float(x) for x in level_costs],
        "cumulative_costs": [float(x) for x in cumulative_costs],
        "q_hat": q,
        "counts": [int(x) for x in n_int],
        "actual_cost": float(actual_cost),
    }

def _select_omppi_route_exhaustive(prediction_names: Sequence[str], stats: Dict[str, Any], budget: float, eps_gap: float) -> Optional[Dict[str, Any]]:
    # The hierarchy is ordered by decreasing query cost and may skip redundant levels.
    ordered_by_cost = sorted(prediction_names, key=lambda m: (-float(stats["per_model"][m]["cost"]), m))
    best: Optional[Dict[str, Any]] = None
    for r in range(1, len(ordered_by_cost) + 1):
        for subset in itertools.combinations(ordered_by_cost, r):
            cand = route_q_and_alloc(list(subset), stats, budget, eps_gap)
            if cand is None:
                continue
            if best is None or cand["q_hat"] < best["q_hat"] - 1e-15:
                best = cand
    return best

def _dag_shortest_route(prediction_names: Sequence[str], stats: Dict[str, Any], eps_gap: float, *, include_lo: bool = True) -> Tuple[Optional[List[str]], float]:
    """Pair-state DAG of Appendix A with nested-evaluator incremental costs.

    Layers: 0 = outcome, 1..K = predictors by decreasing cost, K+1 = terminal (tau_{K+1} = 0,
    c_{K+1} = 0). A state (i, j) means layers i and j are consecutive. The level (i, j) has gap
    tau_i - tau_j and incremental cost c_i - c_j, so the path weight equals sum_j sqrt(gap_j * cost_j),
    the Q of route_q_and_alloc. Validity mirrors route_q_and_alloc: every gap > eps_gap, every
    incremental cost > 1e-12, and gap/cost nondecreasing along the path (monotone nested sample
    sizes). Returns (route, Q); route [] means labeled-only, None means nothing admissible.
    """
    ordered = sorted(prediction_names, key=lambda m: (-float(stats["per_model"][m]["cost"]), m))
    K = len(ordered)
    tau = [float(stats["var_y"])] + [float(stats["per_model"][m]["tau2"]) for m in ordered] + [0.0]
    cost = [float(stats["y_cost"])] + [float(stats["per_model"][m]["cost"]) for m in ordered] + [0.0]
    if any((not np.isfinite(c)) or c <= 0 for c in cost[:-1]):
        return None, float("inf")

    def level(i: int, j: int) -> Optional[Tuple[float, float]]:
        g = tau[i] - tau[j]
        lc = cost[i] - cost[j]
        if g <= eps_gap or lc <= 1e-12:
            return None
        return g, lc

    INF = float("inf")
    D: Dict[Tuple[int, int], float] = {}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
    for j in range(1, K + 1):
        lv = level(0, j)
        if lv is not None:
            D[(0, j)] = math.sqrt(lv[0] * lv[1])
            parent[(0, j)] = None
    for j in range(1, K + 1):  # states (i, j) are final once all i < j are processed
        for i in range(0, j):
            if (i, j) not in D:
                continue
            g_ij, c_ij = level(i, j)  # type: ignore[misc]
            ratio_ij = g_ij / c_ij
            for k in range(j + 1, K + 2):
                lv = level(j, k)
                if lv is None:
                    continue
                if ratio_ij > lv[0] / lv[1] + 1e-12:
                    continue
                val = D[(i, j)] + math.sqrt(lv[0] * lv[1])
                if val < D.get((j, k), INF) - 1e-15:
                    D[(j, k)] = val
                    parent[(j, k)] = (i, j)
    best_state: Optional[Tuple[int, int]] = None
    best_val = INF
    for j in range(1, K + 1):
        val = D.get((j, K + 1), INF)
        if val < best_val - 1e-15:
            best_val = val
            best_state = (j, K + 1)
    if include_lo:
        lo_val = math.sqrt(max(tau[0], 0.0) * cost[0])
        if lo_val < best_val - 1e-15:
            return [], float(lo_val)
    if best_state is None:
        return None, INF
    layers: List[int] = []
    state: Optional[Tuple[int, int]] = best_state
    while state is not None:
        layers.append(state[0])
        state = parent[state]
    route_layers = sorted(l for l in layers if l != 0)
    return [ordered[l - 1] for l in route_layers], float(best_val)

def _lo_plan(stats: Dict[str, Any]) -> Dict[str, Any]:
    q_lo = math.sqrt(max(float(stats["var_y"]), 0.0) * float(stats["y_cost"]))
    return {"route": [], "q_hat": q_lo, "lo_selected": True}

def select_omppi_route(prediction_names: Sequence[str], stats: Dict[str, Any], budget: float, eps_gap: float, *, search_mode: str) -> Optional[Dict[str, Any]]:
    """Return the OMPPI plan, a labeled-only plan ({"lo_selected": True}), or None.
    search_mode "dag" runs the DAG search and records its agreement with exhaustive search."""
    q_lo = math.sqrt(max(float(stats["var_y"]), 0.0) * float(stats["y_cost"]))
    if search_mode == "dag":
        route, q_dag = _dag_shortest_route(prediction_names, stats, eps_gap, include_lo=True)
        exh = _select_omppi_route_exhaustive(prediction_names, stats, budget, eps_gap)
        exh_route = list(exh["route"]) if exh is not None else None
        exh_q = float(exh["q_hat"]) if exh is not None else float("inf")
        if q_lo < exh_q - 1e-15:
            exh_route, exh_q = [], q_lo
        best: Optional[Dict[str, Any]]
        dag_fallback = False
        if route is None:
            best = None
        elif len(route) == 0:
            best = _lo_plan(stats)
        else:
            best = route_q_and_alloc(route, stats, budget, eps_gap)
            if best is None:
                # Integer rounding at a tiny budget can reject the continuous
                # optimum; exhaustive search then skips to the next route.
                dag_fallback = True
                best = exh if exh_route else (_lo_plan(stats) if exh_route == [] else None)
            elif abs(float(best["q_hat"]) - q_dag) > 1e-9 * max(1.0, q_dag):
                raise RuntimeError(f"DAG path weight {q_dag} != route Q {best['q_hat']}")
        if best is not None:
            best = dict(best)
            best["search_mode"] = "dag_graph"
            best["dag_route"] = route
            best["exhaustive_route"] = exh_route
            best["dag_matches_exhaustive"] = bool(route == exh_route)
            best["dag_rounding_fallback"] = dag_fallback
        return best

    best = _select_omppi_route_exhaustive(prediction_names, stats, budget, eps_gap)
    if best is None or q_lo < float(best["q_hat"]) - 1e-15:
        best = _lo_plan(stats)
    best["search_mode"] = search_mode
    return best

def run_omppi_stratum(pop_h: ExecutionPopulation, budget_h: float, alpha: float, rng: np.random.Generator, stats_h: Dict[str, Any], *, eps_gap: float, search_mode: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    plan = select_omppi_route(pop_h.prediction_names, stats_h, float(budget_h), eps_gap, search_mode=search_mode)
    if plan is None or plan.get("lo_selected"):
        est, det = run_classical_stratum(pop_h, budget_h, alpha, rng, stats_h)
        det["fallback"] = "labeled_only_selected" if plan is not None else "classical_no_admissible_omppi_route"
        if plan is not None:
            for key in ("dag_route", "exhaustive_route", "dag_matches_exhaustive", "dag_rounding_fallback"):
                if key in plan:
                    det[key] = plan[key]
        return est, det

    route = list(plan["route"])
    counts = [int(x) for x in plan["counts"]]
    n0 = counts[0]
    nmax = max(counts)
    idx = rng.integers(0, pop_h.X.shape[0], size=nmax)
    y = pop_h.Y[idx[:n0]]
    theta = float(np.mean(y))
    var_hat = safe_sample_var(y) / n0
    prev_n = n0
    for level, m in enumerate(route, start=1):
        n_cur = counts[level]
        col = pop_h.prediction_names.index(m)
        z = pop_h.S[idx[:n_cur], col]
        z_prev = pop_h.S[idx[:prev_n], col]
        gamma = float(stats_h["per_model"][m]["gamma"])
        # Final-stage moments: Var(f) over the n_cur rows that observe f,
        # Cov(Y, f) over the n0 rows that observe both.
        var_z = safe_sample_var(z)
        cov_yz = safe_sample_cov(y, pop_h.S[idx[:n0], col])
        theta += gamma * (float(np.mean(z)) - float(np.mean(z_prev)))
        var_hat += (1.0 / prev_n - 1.0 / n_cur) * (gamma ** 2 * var_z - 2.0 * gamma * cov_yz)
        prev_n = n_cur
    est = ci_from_est(theta, var_hat, alpha)
    actual_cost = float(plan["actual_cost"])
    est.update({"actual_cost": actual_cost, "budget_left": float(budget_h - actual_cost)})
    detail = {
        "route": route,
        "q_hat": float(plan["q_hat"]),
        "counts": {("Y" if j == 0 else route[j - 1]): int(counts[j]) for j in range(len(counts))},
        "gaps": plan["gaps"],
        "costs": plan["costs"],
        "selected_tau2": {m: float(stats_h["per_model"][m]["tau2"]) for m in route},
        "selected_gamma": {m: float(stats_h["per_model"][m]["gamma"]) for m in route},
        "search_mode": search_mode,
    }
    for key in ("dag_route", "exhaustive_route", "dag_matches_exhaustive", "dag_rounding_fallback"):
        if key in plan:
            detail[key] = plan[key]
    return est, detail


# ============================================================
# MultiPPI (full, joint and singleton blocks)
# ============================================================

def _embed_inverse_block(p: int, subset_idx: Sequence[int], cov_sub: np.ndarray) -> np.ndarray:
    out = np.zeros((p, p), dtype=float)
    out[np.ix_(subset_idx, subset_idx)] = _matrix_inverse(cov_sub)
    return out

def _objective_from_counts(counts: np.ndarray, blocks: Sequence[np.ndarray], a: np.ndarray) -> float:
    M = np.zeros_like(blocks[0], dtype=float)
    for n_i, block in zip(counts, blocks):
        if n_i > 0:
            M += float(n_i) * block
    M = _regularize_covariance(M, eps=1e-10)
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        Minv = np.linalg.pinv(M)
    return float(a @ Minv @ a)

def build_restricted_family(prediction_names: Sequence[str]) -> List[Tuple[str, List[int]]]:
    # Indices are in X = [Y, S1, ..., SK].
    all_proxy = list(range(1, len(prediction_names) + 1))
    fam: List[Tuple[str, List[int]]] = [("full", list(range(0, len(prediction_names) + 1)))]
    fam.append(("joint_all", all_proxy))
    for j, name in enumerate(prediction_names, start=1):
        fam.append((f"single__{name}", [j]))
    return fam

def _multippi_problem(
    X_pilot: np.ndarray,
    prediction_names: Sequence[str],
    costs: Mapping[str, float],
    y_cost: float,
    *,
    covariance_method: str,
):
    p = X_pilot.shape[1]
    Sigma = estimate_covariance(X_pilot, method=covariance_method, eps=1e-8)
    fam = build_restricted_family(prediction_names)
    blocks: List[np.ndarray] = []
    subset_costs: List[float] = []
    subset_indices: Dict[str, List[int]] = {}
    subset_cost_map: Dict[str, float] = {}
    for name, idx in fam:
        subset_indices[name] = list(idx)
        blocks.append(_embed_inverse_block(p, idx, Sigma[np.ix_(idx, idx)]))
        if name == "full":
            c = float(y_cost)        # the full suite also yields every partial evaluator
        elif name == "joint_all":
            c = joint_prediction_cost(costs, prediction_names)
        else:
            c = float(costs[name.split("single__", 1)[1]])
        subset_costs.append(c)
        subset_cost_map[name] = c
    return Sigma, fam, blocks, np.asarray(subset_costs, dtype=float), subset_indices, subset_cost_map

def _multippi_slsqp(blocks: Sequence[np.ndarray], subset_costs_arr: np.ndarray, a: np.ndarray, budget: float, *, maxiter: int, ftol: float) -> Tuple[np.ndarray, float, int]:
    """Multi-start SLSQP for min a'M(n)^{-1}a s.t. c'n <= budget."""
    nfam = len(blocks)
    starts: List[np.ndarray] = []
    for j in range(nfam):
        x = np.zeros(nfam, dtype=float)
        x[j] = budget / max(subset_costs_arr[j], 1e-12)
        starts.append(x)
    starts.append(np.full(nfam, budget / max(nfam, 1), dtype=float) / np.maximum(subset_costs_arr, 1e-12))

    def fun(x: np.ndarray) -> float:
        return _objective_from_counts(np.maximum(x, 0.0), blocks, a)

    constraints = [{"type": "ineq", "fun": lambda x, c=subset_costs_arr, b=budget: float(b - c @ np.maximum(x, 0.0))}]
    bounds = [(0.0, None) for _ in range(nfam)]
    best_x = starts[0]
    best_obj = fun(best_x)
    n_success = 0
    for x0 in starts:
        try:
            res = minimize(fun, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": maxiter, "ftol": ftol, "disp": False})
            n_success += int(bool(res.success))
            x = np.maximum(np.asarray(res.x if res.success else x0, dtype=float), 0.0)
            obj = fun(x)
            if obj < best_obj:
                best_x = x
                best_obj = obj
        except Exception:
            continue
    return np.asarray(best_x, dtype=float), float(best_obj), int(n_success)

_MULTIPPI_UNIT_CACHE: Dict[str, Dict[str, Any]] = {}

def multippi_unit_budget_solution(
    X_pilot: np.ndarray,
    prediction_names: Sequence[str],
    costs: Mapping[str, float],
    y_cost: float,
    *,
    covariance_method: str,
) -> Dict[str, Any]:
    """Continuous MultiPPI allocation at unit budget (cached per pilot).

    Returns x_unit (counts per unit budget) and obj_unit = min over c'x <= 1 of a'M(x)^{-1}a. The
    objective is homogeneous of degree -1 in x, so the optimum at budget B is B x_unit with variance
    obj_unit / B, and the stratum constant is Q = sqrt(obj_unit).
    """
    Xp = np.ascontiguousarray(np.asarray(X_pilot, dtype=float))
    key_src = [Xp.tobytes(), repr(tuple(prediction_names)).encode(),
               repr(tuple(float(costs[m]) for m in prediction_names)).encode(),
               repr((float(y_cost), covariance_method)).encode()]
    key = hashlib.sha1(b"|".join(key_src)).hexdigest()
    hit = _MULTIPPI_UNIT_CACHE.get(key)
    if hit is not None:
        return hit
    p = Xp.shape[1]
    _, _, blocks, subset_costs_arr, _, _ = _multippi_problem(Xp, prediction_names, costs, y_cost, covariance_method=covariance_method)
    a = np.zeros(p, dtype=float)
    a[0] = 1.0
    x_unit, obj_unit, n_success = _multippi_slsqp(blocks, subset_costs_arr, a, 1.0, maxiter=1000, ftol=1e-14)
    out = {"x_unit": x_unit, "obj_unit": float(obj_unit), "n_success": int(n_success)}
    if len(_MULTIPPI_UNIT_CACHE) > 4096:
        _MULTIPPI_UNIT_CACHE.clear()
    _MULTIPPI_UNIT_CACHE[key] = out
    return out

def solve_restricted_multippi_fullcost(
    X_pilot: np.ndarray,
    prediction_names: Sequence[str],
    costs: Mapping[str, float],
    y_cost: float,
    budget: float,
    *,
    covariance_method: str,
) -> Optional[Dict[str, Any]]:
    if budget <= 0:
        return None
    p = X_pilot.shape[1]
    Sigma, fam, blocks, subset_costs_arr, subset_indices, subset_cost_map = _multippi_problem(
        X_pilot, prediction_names, costs, y_cost, covariance_method=covariance_method
    )
    a = np.zeros(p, dtype=float)
    a[0] = 1.0

    unit = multippi_unit_budget_solution(X_pilot, prediction_names, costs, y_cost, covariance_method=covariance_method)
    best_x = float(budget) * np.asarray(unit["x_unit"], dtype=float)
    solver_info: Dict[str, Any] = {"unit_objective": float(unit["obj_unit"]), "n_starts_success": int(unit["n_success"])}

    counts = np.floor(best_x).astype(int)
    # Ensure at least two full labels if affordable; otherwise this method is not usable.
    full_idx = 0
    if counts[full_idx] < 2 and 2 * subset_costs_arr[full_idx] <= budget:
        counts[full_idx] = 2
    # Greedy fill leftover by objective gain.
    spent = float(subset_costs_arr @ counts)
    while spent <= budget + 1e-12:
        affordable = [j for j, c in enumerate(subset_costs_arr) if spent + c <= budget + 1e-12]
        if not affordable:
            break
        base = _objective_from_counts(counts.astype(float), blocks, a)
        best_j = None
        best_gain = 0.0
        for j in affordable:
            trial = counts.copy()
            trial[j] += 1
            gain = base - _objective_from_counts(trial.astype(float), blocks, a)
            if gain > best_gain + 1e-15:
                best_gain = gain
                best_j = j
        if best_j is None:
            break
        counts[best_j] += 1
        spent += subset_costs_arr[best_j]

    if counts[full_idx] < 2:
        return None

    M = np.zeros((p, p), dtype=float)
    for n_i, block in zip(counts, blocks):
        if n_i > 0:
            M += float(n_i) * block
    M = _regularize_covariance(M, eps=1e-10)
    Minv = _matrix_inverse(M)
    lambdas: Dict[str, List[float]] = {}
    count_map: Dict[str, int] = {}
    for (name, idx), n_i in zip(fam, counts):
        count_map[name] = int(n_i)
        if n_i <= 0:
            lambdas[name] = [0.0] * len(idx)
            continue
        inv_sub = _matrix_inverse(Sigma[np.ix_(idx, idx)])
        lamb = float(n_i) * inv_sub @ (Minv[np.asarray(idx, dtype=int), :] @ a)
        lambdas[name] = np.asarray(lamb, dtype=float).tolist()

    return {
        "family": [name for name, _ in fam],
        "subset_indices": subset_indices,
        "subset_costs": subset_cost_map,
        "counts": count_map,
        "lambdas": lambdas,
        "actual_cost": float(subset_costs_arr @ counts),
        "objective_value": float(a @ Minv @ a),
        "solver_info": solver_info,
    }

def run_multippi_stratum(pop_h: ExecutionPopulation, budget_h: float, alpha: float, rng: np.random.Generator, stats_h: Dict[str, Any], *, covariance_method: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    plan = solve_restricted_multippi_fullcost(
        stats_h["X_pilot"], pop_h.prediction_names, pop_h.costs_used, pop_h.y_cost_used, float(budget_h),
        covariance_method=covariance_method,
    )
    if plan is None:
        est, det = run_classical_stratum(pop_h, budget_h, alpha, rng, stats_h)
        det["fallback"] = "classical_no_multippi_plan"
        return est, det

    transformed_means: List[float] = []
    transformed_vars: List[float] = []
    for name in plan["family"]:
        n = int(plan["counts"][name])
        if n <= 0:
            continue
        idx_cols = plan["subset_indices"][name]
        lamb = np.asarray(plan["lambdas"][name], dtype=float)
        idx_rows = rng.integers(0, pop_h.X.shape[0], size=n)
        vals = np.asarray(pop_h.X[idx_rows][:, idx_cols] @ lamb, dtype=float)
        transformed_means.append(float(np.mean(vals)))
        transformed_vars.append(safe_sample_var(vals) / max(n, 1))
    theta = float(np.sum(transformed_means))
    var_hat = float(np.sum(transformed_vars))
    est = ci_from_est(theta, var_hat, alpha)
    actual_cost = float(plan["actual_cost"])
    est.update({"actual_cost": actual_cost, "budget_left": float(budget_h - actual_cost)})
    return est, plan


# ============================================================
# Stratified design and one repetition
# ============================================================

def make_methods() -> List[MethodSpec]:
    return [
        MethodSpec("Classical", "classical"),
        MethodSpec("VectorPPI++", "vector_ppi"),
        MethodSpec("MultiPPI", "restrictedmultippi"),
        MethodSpec("OMPPI(Exhaustive)", "omppi", ["exhaustive"]),
        MethodSpec("OMPPI(DAG)", "omppi", ["dag"]),
    ]

def build_stratum_stats(pop_h_pilot: ExecutionPopulation, *, covariance_method: str, ridge: float) -> Dict[str, Any]:
    """Every pilot quantity of every method is read off one covariance estimate of (Y, F)."""
    X = np.asarray(pop_h_pilot.X, dtype=float)
    names, costs, y_cost = pop_h_pilot.prediction_names, pop_h_pilot.costs_used, pop_h_pilot.y_cost_used
    Sigma = estimate_covariance(X, method=covariance_method, eps=ridge)
    stats = compute_scalar_stats_from_cov(Sigma, X.shape[0], names, costs, y_cost)
    stats["vector_stats"] = compute_vector_stats_from_cov(Sigma, names, costs, y_cost)
    stats["X_pilot"] = X
    stats["covariance_method"] = covariance_method
    return stats

def plan_q_for_method(method: MethodSpec, pop_h: ExecutionPopulation, stats_h: Dict[str, Any], budget_h: float, *, eps_gap: float) -> float:
    """Stratum constant Q_h of a method (its variance at stratum budget B_h is Q_h^2 / B_h)."""
    var_y = float(stats_h["var_y"])
    c0 = float(pop_h.y_cost_used)
    if method.kind == "classical":
        return math.sqrt(max(var_y, 0.0) * c0)
    if method.kind == "vector_ppi":
        tau2 = float(stats_h["vector_stats"]["tau2"])
        lc0, lc1 = vector_ppi_level_costs(c0, float(stats_h["vector_stats"]["joint_cost"]))
        if lc0 <= 1e-12 or lc1 <= 1e-12:
            return math.sqrt(max(var_y, 0.0) * c0)
        return math.sqrt(max(var_y - tau2, 0.0) * lc0) + math.sqrt(max(tau2, 0.0) * lc1)
    if method.kind == "omppi":
        route = select_omppi_route(pop_h.prediction_names, stats_h, max(float(budget_h), 1e-9), eps_gap, search_mode=method.extra[0])
        if route is None:
            return math.sqrt(max(var_y, 0.0) * c0)
        return float(route["q_hat"])
    if method.kind == "restrictedmultippi":
        unit = multippi_unit_budget_solution(
            stats_h["X_pilot"], pop_h.prediction_names, pop_h.costs_used, pop_h.y_cost_used,
            covariance_method=stats_h["covariance_method"],
        )
        return math.sqrt(max(float(unit["obj_unit"]), 0.0))
    return math.sqrt(max(var_y, 0.0) * c0)

def apply_stratum_budget_floor(budget_by_h: Mapping[int, float], total: float, floor: float) -> Tuple[Dict[int, float], bool]:
    """Give every stratum at least `floor`, rescaling the others to keep the total.
    Returns the split unchanged (and False) when no stratum is below the floor."""
    out = {h: float(v) for h, v in budget_by_h.items()}
    if floor <= 0 or not out or all(v >= floor for v in out.values()):
        return out, False
    if total <= floor * len(out):
        return {h: float(total) / len(out) for h in out}, True
    low = {h for h, v in out.items() if v < floor}
    while True:
        rest = float(total) - floor * len(low)
        others = {h: out[h] for h in out if h not in low}
        mass = sum(others.values())
        new = {h: (floor if h in low else (rest * out[h] / mass if mass > 0 else rest / len(others))) for h in out}
        newly_low = {h for h in others if new[h] < floor}
        if not newly_low:
            return new, True
        low |= newly_low

def aggregate_stratified(per_h: Mapping[int, Dict[str, Any]], pi_map: Mapping[int, float], *, alpha: float) -> Dict[str, Any]:
    theta = 0.0
    var_hat = 0.0
    actual_cost = 0.0
    for h, est_h in per_h.items():
        pi_h = float(pi_map[h])
        theta += pi_h * float(est_h["theta_hat"])
        var_hat += (pi_h ** 2) * float(est_h["var_hat"])
        actual_cost += float(est_h["actual_cost"])
    est = ci_from_est(theta, var_hat, alpha)
    est.update({"actual_cost": float(actual_cost)})
    return est

def run_method_stratified(
    method: MethodSpec,
    pilot_pop: ExecutionPopulation,
    final_pop: ExecutionPopulation,
    budget: float,
    *,
    covariance_method: str,
    ridge: float,
    eps_gap: float,
    alpha: float,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    strata_keys = sorted(final_pop.strata_pi.keys())
    stats_by_h: Dict[int, Dict[str, Any]] = {}
    final_by_h: Dict[int, ExecutionPopulation] = {}
    pilot_by_h: Dict[int, ExecutionPopulation] = {}
    q_by_h: Dict[int, float] = {}

    for h in strata_keys:
        idx_final = np.where(final_pop.strata_labels == h)[0]
        if len(idx_final) == 0:
            continue
        final_by_h[h] = subset_population_by_indices(final_pop, idx_final)
        idx_pilot = np.where(pilot_pop.strata_labels == h)[0]
        if len(idx_pilot) < 5:
            # Use all pilot rows if a stratum is too small; this does not happen with the pilot sizes used.
            idx_pilot = np.arange(pilot_pop.X.shape[0])
        pilot_by_h[h] = subset_population_by_indices(pilot_pop, idx_pilot)
        stats_by_h[h] = build_stratum_stats(pilot_by_h[h], covariance_method=covariance_method, ridge=ridge)
        q_by_h[h] = plan_q_for_method(method, final_by_h[h], stats_by_h[h], float(final_pop.strata_pi[h]) * budget, eps_gap=eps_gap)

    # Budget split across strata: B_h proportional to pi_h Q_h, with a floor of two full evaluations.
    denom = sum(float(final_pop.strata_pi[h]) * max(q_by_h.get(h, 0.0), 0.0) for h in q_by_h)
    if denom <= 1e-15:
        budget_by_h = {h: float(final_pop.strata_pi[h]) * budget for h in q_by_h}
        split_rule = "pi_fallback_zero_q"
    else:
        budget_by_h = {h: float(budget) * float(final_pop.strata_pi[h]) * max(q_by_h[h], 0.0) / denom for h in q_by_h}
        split_rule = "pi_q_fullcost"
    budget_by_h, floor_applied = apply_stratum_budget_floor(
        budget_by_h, float(budget), float(STRATUM_MIN_LABELS) * float(final_pop.y_cost_used)
    )

    per_h_est: Dict[int, Dict[str, Any]] = {}
    details: Dict[str, Any] = {
        "method_kind": method.kind,
        "budget_split_rule": split_rule,
        "budget_floor_applied": bool(floor_applied),
        "strata": {},
    }
    for h in sorted(q_by_h):
        pop_h = final_by_h[h]
        stats_h = stats_by_h[h]
        B_h = budget_by_h[h]
        if method.kind == "classical":
            est_h, det_h = run_classical_stratum(pop_h, B_h, alpha, rng, stats_h)
        elif method.kind == "vector_ppi":
            est_h, det_h = run_vector_ppi_stratum(pop_h, B_h, alpha, rng, stats_h)
        elif method.kind == "restrictedmultippi":
            est_h, det_h = run_multippi_stratum(pop_h, B_h, alpha, rng, stats_h, covariance_method=covariance_method)
        elif method.kind == "omppi":
            est_h, det_h = run_omppi_stratum(pop_h, B_h, alpha, rng, stats_h, eps_gap=eps_gap, search_mode=method.extra[0])
        else:
            raise ValueError(method.kind)
        per_h_est[h] = est_h
        details["strata"][str(h)] = {
            "pi_h": float(final_pop.strata_pi[h]),
            "q_hat_h": float(q_by_h[h]),
            "budget_h": float(B_h),
            "n_pilot_h": int(pilot_by_h[h].X.shape[0]),
            "n_final_pool_h": int(pop_h.X.shape[0]),
            "theta_hat_h": float(est_h["theta_hat"]) if np.isfinite(est_h["theta_hat"]) else None,
            "var_hat_h": float(est_h["var_hat"]) if np.isfinite(est_h["var_hat"]) else None,
            "actual_cost_h": float(est_h["actual_cost"]),
            "detail": det_h,
        }

    est = aggregate_stratified(per_h_est, final_pop.strata_pi, alpha=alpha)
    est["budget_left"] = float(budget - est["actual_cost"])
    return est, details

def run_one_trial(
    trial_id: int,
    population: ExecutionPopulation,
    budgets: Sequence[float],
    methods: Sequence[MethodSpec],
    *,
    n_pilot: int,
    seed: int,
    theta_true: float,
    outer_trial: int,
    trial_id_offset: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    local_seed = seed_for_trial(seed, trial_id_offset + trial_id)
    rng = np.random.default_rng(local_seed)

    pilot_idx = sample_indices_stratified(
        population.strata_labels,
        population.strata_pi,
        n_pilot,
        rng,
        replace=False,
        min_each=5 if n_pilot >= 5 * len(population.strata_pi) else 1,
    )
    if len(pilot_idx) < max(10, len(population.prediction_names) + 5):
        raise ValueError("Pilot sample too small after stratified sampling")
    pilot_pop = subset_population_by_indices(population, pilot_idx)
    final_pool_idx = np.setdiff1d(np.arange(population.X.shape[0], dtype=int), pilot_idx, assume_unique=False)
    if final_pool_idx.size < 10:
        raise ValueError("Final pool too small after removing pilot rows")
    final_pop = subset_population_by_indices(population, final_pool_idx)
    # Population stratum proportions for the target and the budget split.
    final_pop.strata_pi = dict(population.strata_pi)
    pilot_pop.strata_pi = dict(population.strata_pi)

    rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    global_trial_id = int(trial_id_offset + trial_id)
    # All final-stage draws are i.i.d. with replacement from the pool (all rows minus the pilot),
    # so every unbiased method estimates the pi-weighted pool mean.
    theta_pool = 0.0
    for h, pi_h in final_pop.strata_pi.items():
        mask_h = final_pop.strata_labels == int(h)
        if np.any(mask_h):
            theta_pool += float(pi_h) * float(np.mean(final_pop.Y[mask_h]))

    for b_idx, budget in enumerate(budgets):
        for method in methods:
            rng_m = np.random.default_rng(np.random.SeedSequence(
                [int(local_seed), int(b_idx), int(zlib.crc32(method.name.encode("utf-8"))), 20260926]
            ))
            start = time.perf_counter()
            est, detail = run_method_stratified(
                method, pilot_pop, final_pop, float(budget),
                covariance_method=COVARIANCE_METHOD, ridge=RIDGE, eps_gap=EPS_GAP, alpha=ALPHA, rng=rng_m,
            )
            elapsed = time.perf_counter() - start
            theta_hat = float(est["theta_hat"])
            rows.append({
                "trial": global_trial_id,
                "outer_trial": int(outer_trial),
                "inner_trial": int(trial_id),
                "method": method.name,
                "budget": float(budget),
                "theta_true": float(theta_true),
                "theta_hat": theta_hat,
                "bias": theta_hat - float(theta_true),
                "sq_error": (theta_hat - float(theta_true)) ** 2,
                "covered": float(est["ci_low"] <= theta_true <= est["ci_high"]),
                "theta_pool": float(theta_pool),
                "bias_pool": theta_hat - float(theta_pool),
                "sq_error_pool": (theta_hat - float(theta_pool)) ** 2,
                "covered_pool": float(est["ci_low"] <= theta_pool <= est["ci_high"]),
                "ci_low": float(est["ci_low"]),
                "ci_high": float(est["ci_high"]),
                "ci_width": float(est["width"]),
                "var_hat": float(est["var_hat"]),
                "actual_cost": float(est["actual_cost"]),
                "budget_left": float(est["budget_left"]),
                "algo_compute_time_sec": float(elapsed),
                "n_pilot": int(n_pilot),
                "num_strata": int(len(population.strata_pi)),
                "epsilon_gap": float(EPS_GAP),
                "covariance_method": COVARIANCE_METHOD,
            })
            detail_rows.append({
                "trial": global_trial_id,
                "outer_trial": int(outer_trial),
                "inner_trial": int(trial_id),
                "method": method.name,
                "budget": float(budget),
                "detail_json": json.dumps({"detail": detail}, ensure_ascii=False, sort_keys=True),
            })
    return rows, detail_rows

def _worker(args):
    trial_ids, population, budgets, methods, n_pilot, seed, theta_true, outer_trial, trial_id_offset = args
    all_rows: List[Dict[str, Any]] = []
    all_details: List[Dict[str, Any]] = []
    for trial_id in trial_ids:
        rows, details = run_one_trial(
            int(trial_id), population, budgets, methods, n_pilot=n_pilot, seed=seed,
            theta_true=theta_true, outer_trial=outer_trial, trial_id_offset=trial_id_offset,
        )
        all_rows.extend(rows)
        all_details.extend(details)
    return all_rows, all_details

def split_trials(n_trials: int, num_workers: int) -> List[List[int]]:
    num_workers = max(1, min(int(num_workers), int(n_trials)))
    chunks = [[] for _ in range(num_workers)]
    for t in range(n_trials):
        chunks[t % num_workers].append(t)
    return [c for c in chunks if c]


# ============================================================
# Summaries
# ============================================================

def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    aggs = dict(
        n_trials=("trial", "count"),
        n_nan=("theta_hat", lambda x: int(x.isna().sum())),
        coverage=("covered", "mean"),
        mse=("sq_error", "mean"),
        rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
        mean_bias=("bias", "mean"),
        mean_estimate=("theta_hat", "mean"),
        mean_theta_true=("theta_true", "mean"),
        mean_ci_width=("ci_width", "mean"),
        sd_ci_width=("ci_width", "std"),
        mean_var_hat=("var_hat", "mean"),
        actual_cost_mean=("actual_cost", "mean"),
        actual_cost_max=("actual_cost", "max"),
        budget_left_mean=("budget_left", "mean"),
        algo_compute_time_mean=("algo_compute_time_sec", "mean"),
        algo_compute_time_sd=("algo_compute_time_sec", "std"),
        coverage_pool=("covered_pool", "mean"),
        rmse_pool=("sq_error_pool", lambda x: float(np.sqrt(np.mean(x)))),
        mean_bias_pool=("bias_pool", "mean"),
        mean_theta_pool=("theta_pool", "mean"),
    )
    return (
        trials.groupby(["method", "budget"], as_index=False)
        .agg(**aggs)
        .sort_values(["method", "budget"])
        .reset_index(drop=True)
    )

def make_diagnostics_summary(details_df: pd.DataFrame) -> pd.DataFrame:
    """Per (method, budget): DAG/exhaustive agreement, labeled-only selections, budget-floor use,
    MultiPPI solver success and joint-block usage, averaged over strata x repetitions."""
    rows: List[Dict[str, Any]] = []
    for method, budget, detail_json in details_df[["method", "budget", "detail_json"]].itertuples(index=False, name=None):
        top = json.loads(detail_json).get("detail", {})
        floor_flag = float(bool(top.get("budget_floor_applied", False)))
        for h_obj in top.get("strata", {}).values():
            d = h_obj.get("detail", {})
            rec: Dict[str, Any] = {"method": method, "budget": float(budget), "budget_floor_applied": floor_flag}
            if not (method.startswith("OMPPI") or method == "MultiPPI"):
                rows.append(rec)
                continue
            if method.startswith("OMPPI"):
                rec["lo_selected"] = float(d.get("fallback") == "labeled_only_selected")
                rec["no_admissible_route"] = float(d.get("fallback") == "classical_no_admissible_omppi_route")
                if "dag_matches_exhaustive" in d:
                    rec["dag_matches_exhaustive"] = float(bool(d["dag_matches_exhaustive"]))
                    rec["dag_rounding_fallback"] = float(bool(d.get("dag_rounding_fallback", False)))
            else:
                info = d.get("solver_info", {})
                if "n_starts_success" in info:
                    rec["multippi_starts_success"] = float(info["n_starts_success"])
                counts = d.get("counts", {})
                costs = d.get("subset_costs", {})
                spent = sum(float(counts[k]) * float(costs.get(k, 0.0)) for k in counts)
                if spent > 0:
                    rec["multippi_cost_share_full"] = float(counts.get("full", 0)) * float(costs.get("full", 0.0)) / spent
                    rec["multippi_cost_share_joint"] = float(counts.get("joint_all", 0)) * float(costs.get("joint_all", 0.0)) / spent
                rec["multippi_classical_fallback"] = float(d.get("fallback") == "classical_no_multippi_plan")
            rows.append(rec)
    df = pd.DataFrame(rows)
    return df.groupby(["method", "budget"], as_index=False).mean(numeric_only=True).sort_values(["method", "budget"]).reset_index(drop=True)

def make_allocation_summary(details_df: pd.DataFrame) -> pd.DataFrame:
    """Mean query counts per (method, budget, stratum, source) for OMPPI and MultiPPI."""
    rows: List[Dict[str, Any]] = []
    for method, budget, detail_json in details_df[["method", "budget", "detail_json"]].itertuples(index=False, name=None):
        if method not in {"OMPPI(Exhaustive)", "OMPPI(DAG)", "MultiPPI"}:
            continue
        for h_str, h_obj in json.loads(detail_json).get("detail", {}).get("strata", {}).items():
            h_detail = h_obj.get("detail", {})
            counts = h_detail.get("counts", {})
            route = set(h_detail.get("route", []))
            for source, count in counts.items():
                if method.startswith("OMPPI"):
                    selected = 1.0 if source in route or source == "Y" else 0.0
                else:
                    selected = 1.0 if float(count) > 0 else 0.0
                rows.append({"method": method, "budget": float(budget), "stratum": int(h_str), "source": str(source),
                             "query_count": float(count), "selected": float(selected)})
    df = pd.DataFrame(rows)
    return (
        df.groupby(["method", "budget", "stratum", "source"], as_index=False)
        .agg(query_mean=("query_count", "mean"), selected_frequency=("selected", "mean"))
        .sort_values(["method", "budget", "stratum", "source"])
        .reset_index(drop=True)
    )


# ============================================================
# Experiment
# ============================================================

def run_outer_inner_experiment(
    population: ExecutionPopulation,
    *,
    budgets: Sequence[float],
    n_pilot: int,
    n_outer_trials: int,
    n_inner_trials: int,
    theta_truth_size: int,
    methods: Sequence[MethodSpec],
    seed: int,
    num_workers: int,
    trial_id_stride: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    outer_truth_rows: List[Dict[str, Any]] = []
    all_trial_dfs: List[pd.DataFrame] = []
    all_detail_dfs: List[pd.DataFrame] = []
    # Seeds depend on outer_trial * stride + inner_trial; the stride defaults to n_inner_trials.
    stride = int(n_inner_trials if trial_id_stride is None else trial_id_stride)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    for outer_trial in range(int(n_outer_trials)):
        outer_seed = seed_for_trial(seed, 10_000_000 + outer_trial)
        rng_outer = np.random.default_rng(outer_seed)
        truth_idx, theta_true_outer, truth_counts = sample_theta_truth_subset_stratified(population, int(theta_truth_size), rng_outer)
        outer_truth_rows.append({
            "outer_trial": int(outer_trial),
            "outer_seed": int(outer_seed),
            "theta_truth_size": int(theta_truth_size),
            "theta_true": float(theta_true_outer),
            "truth_subset_idx_json": json.dumps(truth_idx.tolist(), ensure_ascii=False, sort_keys=True),
            "truth_counts_json": json.dumps({str(int(h)): int(truth_counts[h]) for h in sorted(truth_counts)}, ensure_ascii=False, sort_keys=True),
        })
        worker_args = [
            (chunk, population, budgets, methods, n_pilot, outer_seed, theta_true_outer, outer_trial, outer_trial * stride)
            for chunk in split_trials(int(n_inner_trials), num_workers)
        ]
        trial_rows: List[Dict[str, Any]] = []
        detail_rows: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=len(worker_args), mp_context=ctx) as ex:
            for rows, details in ex.map(_worker, worker_args):
                trial_rows.extend(rows)
                detail_rows.extend(details)
        trial_rows = sorted(trial_rows, key=lambda r: (r["trial"], r["budget"], r["method"]))
        detail_rows = sorted(detail_rows, key=lambda r: (r["trial"], r["budget"], r["method"]))
        all_trial_dfs.append(pd.DataFrame(trial_rows))
        all_detail_dfs.append(pd.DataFrame(detail_rows))
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] outer repetition {outer_trial + 1}/{int(n_outer_trials)} done", flush=True)

    trials_df = pd.concat(all_trial_dfs, ignore_index=True)
    details_df = pd.concat(all_detail_dfs, ignore_index=True)
    return {
        "summary_df": summarize_trials(trials_df),
        "trials_df": trials_df,
        "allocation_summary_df": make_allocation_summary(details_df),
        "diagnostics_df": make_diagnostics_summary(details_df),
        "outer_truth_df": pd.DataFrame(outer_truth_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HumanEval+ functional-correctness experiment for one pilot size.")
    parser.add_argument("--n-pilot", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=500, help="inner repetitions per outer repetition")
    parser.add_argument("--n-outer-trials", type=int, default=100)
    parser.add_argument("--theta-truth-size", type=int, default=4500)
    parser.add_argument("--budgets", type=str, default="200:3000:10", help="start:stop:count (linspace)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--trial-id-stride", type=int, default=None,
                        help="seed stride per outer repetition (default --n-trials); use 500 (pilot 800) or 200 "
                             "(pilots 400, 200) to rerun part of the paper's runs")
    parser.add_argument("--data", type=str, default=str(DATA))
    parser.add_argument("--out-dir", type=str, default=None, help="default: results/pilot<n-pilot>")
    args = parser.parse_args()

    population = load_population(args.data)
    budgets = parse_budgets(args.budgets)
    methods = make_methods()
    print(f"Rows: {population.X.shape[0]}; predictors: {population.prediction_names}")
    print(f"Normalized costs: {population.costs_used}; joint predictor block: {joint_prediction_cost(population.costs_used, population.prediction_names):.4f}")
    print(f"Strata pi_h: {population.strata_pi}")
    print(f"Pilot {args.n_pilot}; {args.n_outer_trials} outer x {args.n_trials} inner repetitions; budgets {budgets}", flush=True)

    results = run_outer_inner_experiment(
        population, budgets=budgets, n_pilot=args.n_pilot, n_outer_trials=args.n_outer_trials,
        n_inner_trials=args.n_trials, theta_truth_size=args.theta_truth_size, methods=methods, seed=args.seed,
        num_workers=args.num_workers, trial_id_stride=args.trial_id_stride,
    )

    out_dir = Path(args.out_dir) if args.out_dir else HERE / "results" / f"pilot{args.n_pilot}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results["summary_df"].to_csv(out_dir / "summary.csv", index=False)
    results["trials_df"].to_csv(out_dir / "trials.csv", index=False)
    results["outer_truth_df"].to_csv(out_dir / "outer_truth.csv", index=False)
    results["diagnostics_df"].to_csv(out_dir / "diagnostics_summary.csv", index=False)
    results["allocation_summary_df"].to_csv(out_dir / "allocation_summary.csv", index=False)
    config = {
        "n_pilot": args.n_pilot, "n_trials": args.n_trials, "n_outer_trials": args.n_outer_trials,
        "theta_truth_size": args.theta_truth_size, "budgets": budgets, "seed": args.seed,
        "trial_id_stride": args.trial_id_stride, "num_workers": args.num_workers,
        "target_col": TARGET_COL, "prediction_cols": PREDICTION_COLS, "prediction_names": population.prediction_names,
        "num_strata": NUM_STRATA, "covariance_method": COVARIANCE_METHOD, "ridge": RIDGE, "eps_gap": EPS_GAP,
        "alpha": ALPHA, "stratum_min_labels": STRATUM_MIN_LABELS,
        "y_cost_raw": population.y_cost_raw, "y_cost_used": population.y_cost_used,
        "prediction_costs_raw": population.costs_raw, "prediction_costs_used": population.costs_used,
        "joint_prediction_cost_used": joint_prediction_cost(population.costs_used, population.prediction_names),
        "methods": [m.name for m in methods],
        "strata_pi": population.strata_pi, "strata_summary": population.strata_summary,
        "full_empirical_theta": population.theta_true,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results["summary_df"].to_string(index=False))
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
