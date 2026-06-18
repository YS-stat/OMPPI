#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    from sklearn.covariance import LedoitWolf
    _HAS_LEDOIT_WOLF = True
except Exception:
    LedoitWolf = None  # type: ignore
    _HAS_LEDOIT_WOLF = False

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    tiktoken = None  # type: ignore
    _HAS_TIKTOKEN = False


# ============================================================
# Data containers
# ============================================================

@dataclass
class ExecutionPopulation:
    df: pd.DataFrame
    Y: np.ndarray
    S: np.ndarray
    X: np.ndarray
    prediction_names: List[str]
    costs_raw: Dict[str, float]
    costs_used: Dict[str, float]
    times_raw: Dict[str, float]
    y_cost_raw: float
    y_cost_used: float
    y_time_raw: float
    theta_true: float
    prompt_col: str
    strata_labels: np.ndarray
    strata_pi: Dict[int, float]
    strata_summary: Dict[int, Dict[str, float]]

@dataclass
class MethodSpec:
    name: str
    kind: str
    extra: List[str] = field(default_factory=list)

@dataclass
class StratumPlan:
    method: str
    q_hat: float
    detail: Dict[str, Any]


# ============================================================
# Basic utilities
# ============================================================

def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table suffix: {suffix}")

def parse_csv_list(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]

def parse_budgets(s: str) -> List[float]:
    s = str(s).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError("For colon budget syntax use start:stop:num")
        start, stop, num = float(parts[0]), float(parts[1]), int(parts[2])
        return [float(x) for x in np.linspace(start, stop, num)]
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _normal_z(alpha: float = 0.05) -> float:
    if abs(alpha - 0.05) < 1e-12:
        return 1.959963984540054
    from scipy.stats import norm
    return float(norm.ppf(1.0 - alpha / 2.0))

def resolve_device(device_request: str = "auto") -> str:
    req = (device_request or "auto").lower()
    if req not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if req == "cpu":
        return "cpu"
    if req == "cuda":
        return "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
    return "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"

def set_deterministic(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed % (2**32 - 1))
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

def seed_for_trial(base_seed: int, trial_id: int) -> int:
    ss = np.random.SeedSequence([int(base_seed), int(trial_id), 20260527])
    return int(ss.generate_state(1, dtype=np.uint64)[0] % np.uint64(2**63 - 1))

def pop_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.mean((x - np.mean(x)) ** 2))

def pop_cov(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))))

def safe_sample_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return 0.0
    return float(np.var(x, ddof=1))

def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) if x.size else float("nan")

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
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 rows to estimate covariance")
    method = method.lower()
    if method == "ledoitwolf" and _HAS_LEDOIT_WOLF:
        lw = LedoitWolf(store_precision=False, assume_centered=False)
        lw.fit(X)
        return _regularize_covariance(lw.covariance_, eps=eps)
    cov = np.cov(X, rowvar=False, ddof=1)
    return _regularize_covariance(cov, eps=eps)

def _fallback_token_count(s: str) -> int:
    s = "" if s is None else str(s)
    return len(s.strip().split()) if s.strip() else 0

def count_tokens(texts: Sequence[str], encoding_name: str = "o200k_base") -> np.ndarray:
    vals = ["" if x is None else str(x) for x in texts]
    if _HAS_TIKTOKEN:
        try:
            enc = tiktoken.get_encoding(encoding_name)
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = None
        if enc is not None:
            return np.asarray([len(enc.encode(v)) for v in vals], dtype=int)
    return np.asarray([_fallback_token_count(v) for v in vals], dtype=int)

def make_quantile_strata(values: np.ndarray, num_strata: int) -> np.ndarray:
    values = np.asarray(values)
    if num_strata <= 1:
        return np.zeros(values.shape[0], dtype=int)
    ranks = pd.Series(values).rank(method="first")
    labels = pd.qcut(ranks, q=num_strata, labels=False, duplicates="drop")
    labels = np.asarray(labels, dtype=int)
    uniq = sorted(np.unique(labels).tolist())
    remap = {old: new for new, old in enumerate(uniq)}
    return np.asarray([remap[int(x)] for x in labels], dtype=int)

def summarize_strata(values: np.ndarray, labels: np.ndarray) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    labels = np.asarray(labels, dtype=int)
    values = np.asarray(values, dtype=float)
    for h in sorted(np.unique(labels).tolist()):
        vals = values[labels == h]
        out[int(h)] = {
            "count": int(vals.size),
            "min": float(np.min(vals)) if vals.size else 0.0,
            "max": float(np.max(vals)) if vals.size else 0.0,
            "mean": float(np.mean(vals)) if vals.size else 0.0,
            "median": float(np.median(vals)) if vals.size else 0.0,
        }
    return out

def build_strata(df: pd.DataFrame, *, prompt_col: str, strata_col: Optional[str], num_strata: int) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    if strata_col:
        if strata_col not in df.columns:
            raise ValueError(f"strata_col not found: {strata_col}")
        s = df[strata_col]
        if pd.api.types.is_numeric_dtype(s):
            values = s.astype(float).to_numpy()
            labels = make_quantile_strata(values, num_strata)
            summary = summarize_strata(values, labels)
        else:
            codes, uniques = pd.factorize(s.astype(str), sort=True)
            labels = np.asarray(codes, dtype=int)
            summary = {}
            for h, name in enumerate(uniques):
                summary[int(h)] = {"count": int(np.sum(labels == h)), "category": str(name)}
        return labels, summary

    if prompt_col not in df.columns:
        raise ValueError(f"prompt_col not found: {prompt_col}")
    token_counts = count_tokens(df[prompt_col].astype(str).tolist())
    labels = make_quantile_strata(token_counts, num_strata)
    summary = summarize_strata(token_counts, labels)
    return labels, summary

def read_costs_from_args(
    df: pd.DataFrame,
    prediction_names: Sequence[str],
    *,
    y_cost_col: Optional[str],
    y_cost: Optional[float],
    prediction_cost_cols: Sequence[str],
    prediction_costs: Sequence[float],
    cost_floor: float,
) -> Tuple[float, Dict[str, float]]:
    if y_cost_col:
        if y_cost_col not in df.columns:
            raise ValueError(f"y_cost_col not found: {y_cost_col}")
        y_cost_raw = float(pd.to_numeric(df[y_cost_col], errors="coerce").fillna(0.0).mean())
    elif y_cost is not None:
        y_cost_raw = float(y_cost)
    else:
        y_cost_raw = 1.0
    y_cost_raw = max(y_cost_raw, cost_floor)

    pred_cost_raw: Dict[str, float] = {}
    if prediction_cost_cols:
        if len(prediction_cost_cols) != len(prediction_names):
            raise ValueError("prediction_cost_cols must have the same length as prediction_cols")
        for name, col in zip(prediction_names, prediction_cost_cols):
            if col not in df.columns:
                raise ValueError(f"prediction cost column not found: {col}")
            pred_cost_raw[name] = max(float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).mean()), cost_floor)
    elif prediction_costs:
        if len(prediction_costs) != len(prediction_names):
            raise ValueError("prediction_costs must have the same length as prediction_cols")
        for name, val in zip(prediction_names, prediction_costs):
            pred_cost_raw[name] = max(float(val), cost_floor)
    else:
        # Fallback: progressively cheaper default hierarchy.
        defaults = np.geomspace(0.5, 0.01, num=len(prediction_names))
        pred_cost_raw = {name: max(float(val), cost_floor) for name, val in zip(prediction_names, defaults)}
    return y_cost_raw, pred_cost_raw

def build_execution_population(
    df: pd.DataFrame,
    *,
    target_col: str,
    prediction_cols: Sequence[str],
    prediction_names: Optional[Sequence[str]],
    prompt_col: str,
    strata_col: Optional[str],
    num_strata: int,
    y_cost_col: Optional[str],
    y_cost: Optional[float],
    prediction_cost_cols: Sequence[str],
    prediction_costs: Sequence[float],
    normalize_costs: bool,
    cost_floor: float,
) -> ExecutionPopulation:
    if target_col not in df.columns:
        raise ValueError(f"target_col not found: {target_col}")
    for col in prediction_cols:
        if col not in df.columns:
            raise ValueError(f"prediction_col not found: {col}")
    names = list(prediction_names) if prediction_names else list(prediction_cols)
    if len(names) != len(prediction_cols):
        raise ValueError("prediction_names must have the same length as prediction_cols")

    use_cols = [target_col] + list(prediction_cols)
    work = df.copy()
    for col in use_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.loc[work[use_cols].notna().all(axis=1)].copy().reset_index(drop=True)
    if work.empty:
        raise ValueError("No complete rows remain after dropping missing target/predictions")

    labels, strata_summary = build_strata(work, prompt_col=prompt_col, strata_col=strata_col, num_strata=num_strata)
    work["stratum"] = labels
    pi = {int(h): float(np.mean(labels == h)) for h in sorted(np.unique(labels).tolist())}

    y_cost_raw, pred_cost_raw = read_costs_from_args(
        work,
        names,
        y_cost_col=y_cost_col,
        y_cost=y_cost,
        prediction_cost_cols=prediction_cost_cols,
        prediction_costs=prediction_costs,
        cost_floor=cost_floor,
    )

    if normalize_costs:
        denom = max(y_cost_raw, cost_floor)
        y_cost_used = 1.0
        pred_cost_used = {m: max(float(pred_cost_raw[m]) / denom, cost_floor) for m in names}
    else:
        y_cost_used = y_cost_raw
        pred_cost_used = dict(pred_cost_raw)

    Y = work[target_col].astype(float).to_numpy()
    S = work[list(prediction_cols)].astype(float).to_numpy()
    X = np.column_stack([Y, S])

    return ExecutionPopulation(
        df=work,
        Y=Y,
        S=S,
        X=X,
        prediction_names=names,
        costs_raw=pred_cost_raw,
        costs_used=pred_cost_used,
        times_raw=dict(pred_cost_raw),
        y_cost_raw=float(y_cost_raw),
        y_cost_used=float(y_cost_used),
        y_time_raw=float(y_cost_raw),
        theta_true=float(np.mean(Y)),
        prompt_col=prompt_col,
        strata_labels=labels.astype(int),
        strata_pi=pi,
        strata_summary=strata_summary,
    )

def subset_population_by_indices(pop: ExecutionPopulation, idx: np.ndarray) -> ExecutionPopulation:
    idx = np.asarray(idx, dtype=int)
    df_sub = pop.df.iloc[idx].copy().reset_index(drop=True)
    y = pop.Y[idx]
    s = pop.S[idx]
    x = pop.X[idx]
    labels = pop.strata_labels[idx]
    # Preserve global stratum labels and pi inside the subset as local empirical pi.
    uniq = sorted(np.unique(labels).tolist())
    pi = {int(h): float(np.mean(labels == h)) for h in uniq}
    summary = summarize_strata(np.arange(len(labels), dtype=float), labels) if len(labels) else {}
    return ExecutionPopulation(
        df=df_sub,
        Y=y,
        S=s,
        X=x,
        prediction_names=list(pop.prediction_names),
        costs_raw=dict(pop.costs_raw),
        costs_used=dict(pop.costs_used),
        times_raw=dict(pop.times_raw),
        y_cost_raw=float(pop.y_cost_raw),
        y_cost_used=float(pop.y_cost_used),
        y_time_raw=float(pop.y_time_raw),
        theta_true=float(np.mean(y)) if len(y) else float("nan"),
        prompt_col=pop.prompt_col,
        strata_labels=labels.astype(int),
        strata_pi=pi,
        strata_summary=summary,
    )

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
    """Draw a stratified truth subset and compute the target by fixed stratum weights.

    This mirrors the LLM-preference experiment: the outer truth is not the full
    empirical mean. Instead, each outer repetition draws a large stratified subset,
    computes stratum means on that subset, and aggregates them using the population
    stratum proportions pi_h.
    """
    if truth_size <= 0:
        raise ValueError("theta truth subset size must be positive")
    if truth_size > population.Y.shape[0]:
        raise ValueError(
            f"theta truth subset size {truth_size} exceeds available rows {population.Y.shape[0]}"
        )
    truth_idx = sample_indices_stratified(
        population.strata_labels,
        population.strata_pi,
        truth_size,
        rng,
        replace=False,
        min_each=1,
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

def compute_scalar_stats(X_pilot: np.ndarray, prediction_names: Sequence[str], costs: Mapping[str, float], y_cost: float) -> Dict[str, Any]:
    y = np.asarray(X_pilot[:, 0], dtype=float)
    var_y = pop_var(y)
    out: Dict[str, Any] = {
        "n": int(len(y)),
        "mean_y": safe_mean(y),
        "var_y": float(var_y),
        "y_cost": float(y_cost),
        "per_model": {},
    }
    for j, name in enumerate(prediction_names, start=1):
        z = np.asarray(X_pilot[:, j], dtype=float)
        var_z = pop_var(z)
        cov_yz = pop_cov(y, z)
        gamma = 0.0 if var_z <= 1e-15 else cov_yz / var_z
        tau2 = 0.0 if var_z <= 1e-15 else (cov_yz ** 2) / var_z
        tau2 = float(min(max(tau2, 0.0), max(var_y - 1e-12, 0.0)))
        out["per_model"][name] = {
            "var_z": float(var_z),
            "cov_yz": float(cov_yz),
            "gamma": float(gamma),
            "tau2": float(tau2),
            "cost": float(costs[name]),
        }
    return out

def compute_vector_stats(X_pilot: np.ndarray, prediction_names: Sequence[str], costs: Mapping[str, float], y_cost: float, *, covariance_method: str, ridge: float) -> Dict[str, Any]:
    y = np.asarray(X_pilot[:, 0], dtype=float)
    Z = np.asarray(X_pilot[:, 1:1+len(prediction_names)], dtype=float)
    var_y = pop_var(y)
    if Z.shape[1] == 0:
        return {"var_y": var_y, "tau2": 0.0, "gamma": [], "joint_cost": 0.0, "y_cost": float(y_cost)}
    Zc = Z - Z.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    Sigma_zz = estimate_covariance(Z, method=covariance_method, eps=ridge)
    cov_zy = (Zc.T @ yc) / max(Z.shape[0] - 1, 1)
    try:
        gamma = np.linalg.solve(Sigma_zz, cov_zy)
    except np.linalg.LinAlgError:
        gamma = np.linalg.pinv(Sigma_zz) @ cov_zy
    tau2 = float(cov_zy.T @ gamma)
    tau2 = float(min(max(tau2, 0.0), max(var_y - 1e-12, 0.0)))
    return {
        "var_y": float(var_y),
        "tau2": tau2,
        "gamma": np.asarray(gamma, dtype=float).tolist(),
        "Sigma_zz": Sigma_zz,
        "cov_zy": cov_zy,
        "joint_cost": float(sum(costs[m] for m in prediction_names)),
        "y_cost": float(y_cost),
    }


# ============================================================
# Allocation and estimators
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

def run_vector_ppi_stratum(pop_h: ExecutionPopulation, budget_h: float, alpha: float, rng: np.random.Generator, stats_h: Dict[str, Any], *, covariance_method: str, ridge: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prediction_names = pop_h.prediction_names
    vstats = compute_vector_stats(pop_h.X_pilot_for_stats, prediction_names, pop_h.costs_used, pop_h.y_cost_used, covariance_method=covariance_method, ridge=ridge) if hasattr(pop_h, "X_pilot_for_stats") else None
    # The caller passes vstats through stats_h when available.
    vstats = stats_h.get("vector_stats", vstats)
    if vstats is None:
        raise RuntimeError("vector_stats missing")
    var_y = float(vstats["var_y"])
    tau2 = float(vstats["tau2"])
    c0 = float(pop_h.y_cost_used)
    c1 = float(vstats["joint_cost"])
    alloc = allocate_two_level(var_y, tau2, c0, c1, float(budget_h))
    if alloc is None:
        return run_classical_stratum(pop_h, budget_h, alpha, rng, stats_h)
    n0, n1, q = alloc
    idx = rng.integers(0, pop_h.X.shape[0], size=n1)
    y = pop_h.Y[idx[:n0]]
    Z = pop_h.S[idx, :]
    gamma = np.asarray(vstats["gamma"], dtype=float)
    theta = float(np.mean(y) + gamma @ (Z.mean(axis=0) - Z[:n0, :].mean(axis=0)))
    Sigma_zz = np.asarray(vstats["Sigma_zz"], dtype=float)
    cov_zy = np.asarray(vstats["cov_zy"], dtype=float)
    adj = float(gamma @ Sigma_zz @ gamma - 2.0 * gamma @ cov_zy)
    var_hat = var_y / n0 + (1.0 / n0 - 1.0 / n1) * adj
    est = ci_from_est(theta, var_hat, alpha)
    actual_cost = c0 * n0 + c1 * n1
    est.update({"actual_cost": float(actual_cost), "budget_left": float(budget_h - actual_cost)})
    return est, {"n_y": int(n0), "n_joint": int(n1), "q_hat": q, "tau2": tau2, "joint_cost": c1}

def _omppi_nested_incremental_costs(y_cost: float, route_costs: Sequence[float], eps: float = 1e-12) -> Optional[List[float]]:
    """Convert cumulative evaluator costs into OMPPI incremental level costs.

    This is specific to the HumanEval execution setting. If a sample is evaluated
    to a higher fidelity level, all lower-fidelity predictions in the selected
    route are already observed. For a route with cumulative costs
    c0 > c1 > ... > cL, where c0 is the full Y cost and c1,...,cL are the
    selected proxy costs ordered from high to low fidelity, the nested sampling
    budget is

        c0*n0 + c1*(n1-n0) + ... + cL*(nL-n_{L-1})

    equivalently

        (c0-c1)*n0 + (c1-c2)*n1 + ... + cL*nL.

    MultiPPI and VectorPPI++ are left unchanged; only OMPPI uses this conversion.
    """
    cumulative = [float(y_cost)] + [float(c) for c in route_costs]
    if any((not np.isfinite(c)) or c <= 0 for c in cumulative):
        return None
    # The selected route is sorted by decreasing cumulative evaluator cost.
    # Reject non-nested cost orders rather than silently double-counting.
    for j in range(len(cumulative) - 1):
        if cumulative[j] + eps < cumulative[j + 1]:
            return None

    incremental: List[float] = []
    for j in range(len(cumulative) - 1):
        incremental.append(float(cumulative[j] - cumulative[j + 1]))
    incremental.append(float(cumulative[-1]))

    # In degenerate equal-cost cases, the continuous allocation formula would
    # divide by zero. Keep the route only when all effective costs are positive.
    if any((not np.isfinite(c)) or c <= eps for c in incremental):
        return None
    return incremental


def _nested_actual_cost_from_cumulative_costs(counts: Sequence[int], cumulative_costs: Sequence[float]) -> float:
    """Actual nested evaluator cost for monotone counts.

    counts = (n0,n1,...,nL) with n0 <= n1 <= ... <= nL.
    cumulative_costs = (c0,c1,...,cL), where c0 is full Y cost and c1,...,cL are
    selected proxy cumulative costs.
    """
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

    # HumanEval execution proxies are nested fidelities. OMPPI alone uses nested
    # sampling, so its allocation should use incremental evaluator costs rather
    # than independently charging every level.
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
        # Equivalent to sum(d_j n_j), where d_j are incremental costs.
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
        "incremental_costs": [float(x) for x in level_costs],
        "cumulative_costs": [float(x) for x in cumulative_costs],
        "raw_route_costs": [float(x) for x in route_costs],
        "q_hat": q,
        "counts": [int(x) for x in n_int],
        "actual_cost": float(actual_cost),
        "cost_mode": "omppi_nested_evaluator_incremental",
    }

def select_omppi_route(prediction_names: Sequence[str], stats: Dict[str, Any], budget: float, eps_gap: float, *, search_mode: str) -> Optional[Dict[str, Any]]:
    # The selected hierarchy is ordered by decreasing query cost. It may skip redundant levels.
    ordered_by_cost = sorted(prediction_names, key=lambda m: (-float(stats["per_model"][m]["cost"]), m))
    import itertools
    best: Optional[Dict[str, Any]] = None
    for r in range(1, len(ordered_by_cost) + 1):
        for subset in itertools.combinations(ordered_by_cost, r):
            route = list(subset)
            cand = route_q_and_alloc(route, stats, budget, eps_gap)
            if cand is None:
                continue
            if best is None or cand["q_hat"] < best["q_hat"] - 1e-15:
                best = cand
    if best is not None:
        best["search_mode"] = search_mode
        # For K<=5 in this demo, DAG and exhaustive return the same exact finite-class minimizer.
        # We keep both method names so figures match the main human-preference experiment.
    return best

def run_omppi_stratum(pop_h: ExecutionPopulation, budget_h: float, alpha: float, rng: np.random.Generator, stats_h: Dict[str, Any], *, eps_gap: float, search_mode: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    plan = select_omppi_route(pop_h.prediction_names, stats_h, float(budget_h), eps_gap, search_mode=search_mode)
    if plan is None:
        est, det = run_classical_stratum(pop_h, budget_h, alpha, rng, stats_h)
        det["fallback"] = "classical_no_admissible_omppi_route"
        return est, det

    route = list(plan["route"])
    counts = [int(x) for x in plan["counts"]]
    n0 = counts[0]
    nmax = max(counts)
    idx = rng.integers(0, pop_h.X.shape[0], size=nmax)
    y = pop_h.Y[idx[:n0]]
    theta = float(np.mean(y))
    var_hat = float(stats_h["var_y"]) / n0
    prev_n = n0
    for level, m in enumerate(route, start=1):
        n_cur = counts[level]
        col = pop_h.prediction_names.index(m)
        z = pop_h.S[idx[:n_cur], col]
        z_prev = pop_h.S[idx[:prev_n], col]
        gamma = float(stats_h["per_model"][m]["gamma"])
        var_z = float(stats_h["per_model"][m]["var_z"])
        cov_yz = float(stats_h["per_model"][m]["cov_yz"])
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
    return est, detail


# ============================================================
# Restricted MultiPPI full-cost baseline
# ============================================================

def _embed_inverse_block(p: int, subset_idx: Sequence[int], cov_sub: np.ndarray) -> np.ndarray:
    out = np.zeros((p, p), dtype=float)
    inv_sub = _matrix_inverse(cov_sub)
    out[np.ix_(subset_idx, subset_idx)] = inv_sub
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
    # Indices are in X=[Y,S1,...,SK].
    all_proxy = list(range(1, len(prediction_names) + 1))
    fam: List[Tuple[str, List[int]]] = [("full", list(range(0, len(prediction_names) + 1)))]
    fam.append(("joint_all", all_proxy))
    for j, name in enumerate(prediction_names, start=1):
        fam.append((f"single__{name}", [j]))
    return fam

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
    Sigma = estimate_covariance(X_pilot, method=covariance_method, eps=1e-8)
    fam = build_restricted_family(prediction_names)
    blocks: List[np.ndarray] = []
    subset_costs: List[float] = []
    subset_indices: Dict[str, List[int]] = {}
    subset_cost_map: Dict[str, float] = {}
    for name, idx in fam:
        subset_indices[name] = list(idx)
        cov_sub = Sigma[np.ix_(idx, idx)]
        blocks.append(_embed_inverse_block(p, idx, cov_sub))
        if name == "full":
            c = float(y_cost)
        elif name == "joint_all":
            c = float(sum(costs[m] for m in prediction_names))
        else:
            m = name.split("single__", 1)[1]
            c = float(costs[m])
        subset_costs.append(c)
        subset_cost_map[name] = c

    subset_costs_arr = np.asarray(subset_costs, dtype=float)
    a = np.zeros(p, dtype=float)
    a[0] = 1.0

    starts: List[np.ndarray] = []
    for j in range(len(fam)):
        x = np.zeros(len(fam), dtype=float)
        x[j] = budget / max(subset_costs_arr[j], 1e-12)
        starts.append(x)
    starts.append(np.full(len(fam), budget / max(len(fam), 1), dtype=float) / np.maximum(subset_costs_arr, 1e-12))

    def fun(x: np.ndarray) -> float:
        return _objective_from_counts(np.maximum(x, 0.0), blocks, a)

    constraints = [{"type": "ineq", "fun": lambda x, c=subset_costs_arr, b=budget: float(b - c @ np.maximum(x, 0.0))}]
    bounds = [(0.0, None) for _ in fam]
    best_x = starts[0]
    best_obj = fun(best_x)
    for x0 in starts:
        try:
            res = minimize(fun, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500, "ftol": 1e-10, "disp": False})
            x = np.maximum(np.asarray(res.x if res.success else x0, dtype=float), 0.0)
            obj = fun(x)
            if obj < best_obj:
                best_x = x
                best_obj = obj
        except Exception:
            continue

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
        cov_sub = Sigma[np.ix_(idx, idx)]
        inv_sub = _matrix_inverse(cov_sub)
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
    }

def run_multippi_stratum(pop_h: ExecutionPopulation, budget_h: float, alpha: float, rng: np.random.Generator, stats_h: Dict[str, Any], *, covariance_method: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    plan = solve_restricted_multippi_fullcost(
        stats_h["X_pilot"],
        pop_h.prediction_names,
        pop_h.costs_used,
        pop_h.y_cost_used,
        float(budget_h),
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
        Xsub = pop_h.X[idx_rows][:, idx_cols]
        vals = np.asarray(Xsub @ lamb, dtype=float)
        transformed_means.append(float(np.mean(vals)))
        transformed_vars.append(safe_sample_var(vals) / max(n, 1))
    theta = float(np.sum(transformed_means))
    var_hat = float(np.sum(transformed_vars))
    est = ci_from_est(theta, var_hat, alpha)
    actual_cost = float(plan["actual_cost"])
    est.update({"actual_cost": actual_cost, "budget_left": float(budget_h - actual_cost)})
    return est, plan


# ============================================================
# Stratified planning and trial running
# ============================================================

def make_methods() -> List[MethodSpec]:
    return [
        MethodSpec("Classical", "classical"),
        MethodSpec("VectorPPI++", "vector_ppi"),
        MethodSpec("MultiPPI", "restrictedmultippi"),
        MethodSpec("OMPPI(Exhaustive)", "omppi", ["exhaustive"]),
        MethodSpec("OMPPI(DAG)", "omppi", ["dag"]),
    ]

def build_stratum_stats(
    pop_h_pilot: ExecutionPopulation,
    *,
    covariance_method: str,
    ridge: float,
) -> Dict[str, Any]:
    stats = compute_scalar_stats(pop_h_pilot.X, pop_h_pilot.prediction_names, pop_h_pilot.costs_used, pop_h_pilot.y_cost_used)
    stats["X_pilot"] = np.asarray(pop_h_pilot.X, dtype=float)
    stats["vector_stats"] = compute_vector_stats(
        pop_h_pilot.X,
        pop_h_pilot.prediction_names,
        pop_h_pilot.costs_used,
        pop_h_pilot.y_cost_used,
        covariance_method=covariance_method,
        ridge=ridge,
    )
    return stats

def plan_q_for_method(method: MethodSpec, pop_h: ExecutionPopulation, stats_h: Dict[str, Any], budget_h: float, *, eps_gap: float) -> float:
    var_y = float(stats_h["var_y"])
    c0 = float(pop_h.y_cost_used)
    if method.kind == "classical":
        return math.sqrt(max(var_y, 0.0) * c0)
    if method.kind == "vector_ppi":
        tau2 = float(stats_h["vector_stats"]["tau2"])
        c1 = float(stats_h["vector_stats"]["joint_cost"])
        return math.sqrt(max(var_y - tau2, 0.0) * c0) + math.sqrt(max(tau2, 0.0) * c1)
    if method.kind == "omppi":
        route = select_omppi_route(pop_h.prediction_names, stats_h, max(float(budget_h), 1e-9), eps_gap, search_mode=method.extra[0])
        if route is None:
            return math.sqrt(max(var_y, 0.0) * c0)
        return float(route["q_hat"])
    if method.kind == "restrictedmultippi":
        # Use the pilot objective scale as a rough Q for cross-stratum allocation.
        return math.sqrt(max(var_y, 0.0) * c0)
    return math.sqrt(max(var_y, 0.0) * c0)

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
    # Build local stratum populations from pilot and final pools.
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
            # Use all pilot rows if a stratum is too small; this should not happen with reasonable n_pilot.
            idx_pilot = np.arange(pilot_pop.X.shape[0])
        pilot_by_h[h] = subset_population_by_indices(pilot_pop, idx_pilot)
        stats_by_h[h] = build_stratum_stats(pilot_by_h[h], covariance_method=covariance_method, ridge=ridge)
        q_by_h[h] = plan_q_for_method(method, final_by_h[h], stats_by_h[h], float(final_pop.strata_pi[h]) * budget, eps_gap=eps_gap)

    denom = sum(float(final_pop.strata_pi[h]) * max(q_by_h.get(h, 0.0), 0.0) for h in q_by_h)
    if denom <= 1e-15:
        budget_by_h = {h: float(final_pop.strata_pi[h]) * budget for h in q_by_h}
        split_rule = "pi_fallback_zero_q"
    else:
        budget_by_h = {h: float(budget) * float(final_pop.strata_pi[h]) * max(q_by_h[h], 0.0) / denom for h in q_by_h}
        split_rule = "pi_q_fullcost"

    per_h_est: Dict[int, Dict[str, Any]] = {}
    details: Dict[str, Any] = {
        "method_kind": method.kind,
        "budget_split_rule": split_rule,
        "strata": {},
    }
    for h in sorted(q_by_h):
        pop_h = final_by_h[h]
        stats_h = stats_by_h[h]
        B_h = budget_by_h[h]
        if method.kind == "classical":
            est_h, det_h = run_classical_stratum(pop_h, B_h, alpha, rng, stats_h)
        elif method.kind == "vector_ppi":
            est_h, det_h = run_vector_ppi_stratum(pop_h, B_h, alpha, rng, stats_h, covariance_method=covariance_method, ridge=ridge)
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
    covariance_method: str,
    ridge: float,
    eps_gap: float,
    alpha: float,
    theta_true: Optional[float] = None,
    outer_trial: int = 0,
    trial_id_offset: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    local_seed = seed_for_trial(seed, trial_id_offset + trial_id)
    set_deterministic(local_seed)
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
    # Use original full-population pi for the final target and budget split.
    final_pop.strata_pi = dict(population.strata_pi)
    pilot_pop.strata_pi = dict(population.strata_pi)

    rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    theta_true_used = float(population.theta_true if theta_true is None else theta_true)
    global_trial_id = int(trial_id_offset + trial_id)

    for budget in budgets:
        for method in methods:
            start = time.perf_counter()
            est, detail = run_method_stratified(
                method,
                pilot_pop,
                final_pop,
                float(budget),
                covariance_method=covariance_method,
                ridge=ridge,
                eps_gap=eps_gap,
                alpha=alpha,
                rng=rng,
            )
            elapsed = time.perf_counter() - start
            theta_hat = float(est["theta_hat"])
            rows.append({
                "trial": global_trial_id,
                "outer_trial": int(outer_trial),
                "inner_trial": int(trial_id),
                "method": method.name,
                "budget": float(budget),
                "theta_true": theta_true_used,
                "theta_hat": theta_hat,
                "bias": theta_hat - theta_true_used,
                "sq_error": (theta_hat - theta_true_used) ** 2,
                "covered": float(est["ci_low"] <= theta_true_used <= est["ci_high"]),
                "ci_low": float(est["ci_low"]),
                "ci_high": float(est["ci_high"]),
                "ci_width": float(est["width"]),
                "var_hat": float(est["var_hat"]),
                "actual_cost": float(est["actual_cost"]),
                "budget_left": float(est["budget_left"]),
                "algo_compute_time_sec": float(elapsed),
                "n_pilot": int(n_pilot),
                "num_strata": int(len(population.strata_pi)),
                "epsilon_gap": float(eps_gap),
                "covariance_method": covariance_method,
            })
            detail_rows.append({
                "trial": global_trial_id,
                "outer_trial": int(outer_trial),
                "inner_trial": int(trial_id),
                "method": method.name,
                "budget": float(budget),
                "detail_json": json.dumps({
                    "pilot_idx": pilot_idx.tolist(),
                    "detail": detail,
                }, ensure_ascii=False, sort_keys=True),
            })
    return rows, detail_rows

def _worker(args):
    (
        trial_ids, population, budgets, methods, n_pilot, seed, covariance_method,
        ridge, eps_gap, alpha, theta_true, outer_trial, trial_id_offset
    ) = args
    all_rows: List[Dict[str, Any]] = []
    all_details: List[Dict[str, Any]] = []
    for trial_id in trial_ids:
        rows, details = run_one_trial(
            int(trial_id),
            population,
            budgets,
            methods,
            n_pilot=n_pilot,
            seed=seed,
            covariance_method=covariance_method,
            ridge=ridge,
            eps_gap=eps_gap,
            alpha=alpha,
            theta_true=theta_true,
            outer_trial=outer_trial,
            trial_id_offset=trial_id_offset,
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

def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    return (
        trials.groupby(["method", "budget"], as_index=False)
        .agg(
            n_trials=("trial", "count"),
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
            budget_left_mean=("budget_left", "mean"),
            algo_compute_time_mean=("algo_compute_time_sec", "mean"),
            algo_compute_time_sd=("algo_compute_time_sec", "std"),
        )
        .sort_values(["method", "budget"])
        .reset_index(drop=True)
    )

def make_allocation_summary(details_df: pd.DataFrame, prediction_names: Sequence[str]) -> pd.DataFrame:
    target_methods = {"OMPPI(Exhaustive)", "OMPPI(DAG)", "MultiPPI"}
    rows: List[Dict[str, Any]] = []
    for method, budget, detail_json in details_df[["method", "budget", "detail_json"]].itertuples(index=False, name=None):
        if method not in target_methods:
            continue
        try:
            obj = json.loads(detail_json)
        except Exception:
            continue
        detail = obj.get("detail", {})
        strata = detail.get("strata", {})
        for h_str, h_obj in strata.items():
            h_detail = h_obj.get("detail", {})
            if method.startswith("OMPPI"):
                counts = h_detail.get("counts", {})
                route = set(h_detail.get("route", []))
                for source, count in counts.items():
                    rows.append({
                        "method": method,
                        "budget": float(budget),
                        "stratum": int(h_str),
                        "source": str(source),
                        "query_count": float(count),
                        "selected": float(1.0 if source in route or source == "Y" else 0.0),
                    })
            elif method == "MultiPPI":
                counts = h_detail.get("counts", {})
                for source, count in counts.items():
                    rows.append({
                        "method": method,
                        "budget": float(budget),
                        "stratum": int(h_str),
                        "source": str(source),
                        "query_count": float(count),
                        "selected": float(1.0 if float(count) > 0 else 0.0),
                    })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = (
        df.groupby(["method", "budget", "stratum", "source"], as_index=False)
        .agg(query_mean=("query_count", "mean"), selected_frequency=("selected", "mean"))
        .sort_values(["method", "budget", "stratum", "source"])
        .reset_index(drop=True)
    )
    return out

def run_experiment(
    population: ExecutionPopulation,
    *,
    budgets: Sequence[float],
    n_pilot: int,
    n_trials: int,
    methods: Sequence[MethodSpec],
    covariance_method: str,
    ridge: float,
    eps_gap: float,
    alpha: float,
    seed: int,
    num_workers: int,
) -> Dict[str, pd.DataFrame]:
    if num_workers <= 1:
        all_rows: List[Dict[str, Any]] = []
        all_details: List[Dict[str, Any]] = []
        for t in range(n_trials):
            rows, details = run_one_trial(
                t,
                population,
                budgets,
                methods,
                n_pilot=n_pilot,
                seed=seed,
                covariance_method=covariance_method,
                ridge=ridge,
                eps_gap=eps_gap,
                alpha=alpha,
                theta_true=None,
                outer_trial=0,
                trial_id_offset=0,
            )
            all_rows.extend(rows)
            all_details.extend(details)
    else:
        import multiprocessing as mp
        chunks = split_trials(n_trials, num_workers)
        ctx = mp.get_context("spawn")
        worker_args = [
            (chunk, population, budgets, methods, n_pilot, seed, covariance_method, ridge, eps_gap, alpha, None, 0, 0)
            for chunk in chunks
        ]
        all_rows = []
        all_details = []
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as ex:
            for rows, details in ex.map(_worker, worker_args):
                all_rows.extend(rows)
                all_details.extend(details)
        all_rows = sorted(all_rows, key=lambda r: (r["trial"], r["budget"], r["method"]))
        all_details = sorted(all_details, key=lambda r: (r["trial"], r["budget"], r["method"]))

    trials_df = pd.DataFrame(all_rows)
    details_df = pd.DataFrame(all_details)
    summary_df = summarize_trials(trials_df)
    allocation_summary_df = make_allocation_summary(details_df, population.prediction_names)
    return {"summary_df": summary_df, "trials_df": trials_df, "details_df": details_df, "allocation_summary_df": allocation_summary_df}


def run_outer_inner_truth_experiment(
    population: ExecutionPopulation,
    *,
    budgets: Sequence[float],
    n_pilot: int,
    n_outer_trials: int,
    n_inner_trials: int,
    theta_truth_size: int,
    exclude_truth_from_inference: bool,
    methods: Sequence[MethodSpec],
    covariance_method: str,
    ridge: float,
    eps_gap: float,
    alpha: float,
    seed: int,
    num_workers: int,
) -> Dict[str, pd.DataFrame]:
    outer_truth_rows: List[Dict[str, Any]] = []
    all_trial_dfs: List[pd.DataFrame] = []
    all_detail_dfs: List[pd.DataFrame] = []

    for outer_trial in range(int(n_outer_trials)):
        outer_seed = seed_for_trial(seed, 10_000_000 + outer_trial)
        rng_outer = np.random.default_rng(outer_seed)
        truth_idx, theta_true_outer, truth_counts = sample_theta_truth_subset_stratified(
            population, int(theta_truth_size), rng_outer
        )
        outer_truth_rows.append({
            "outer_trial": int(outer_trial),
            "outer_seed": int(outer_seed),
            "theta_truth_size": int(theta_truth_size),
            "theta_true": float(theta_true_outer),
            "truth_subset_idx_json": json.dumps(truth_idx.tolist(), ensure_ascii=False, sort_keys=True),
            "truth_counts_json": json.dumps({str(int(h)): int(truth_counts[h]) for h in sorted(truth_counts)}, ensure_ascii=False, sort_keys=True),
        })

        if exclude_truth_from_inference:
            keep_idx = np.setdiff1d(np.arange(population.X.shape[0], dtype=int), truth_idx, assume_unique=False)
            inference_pop = subset_population_by_indices(population, keep_idx)
            # Keep the original target stratum proportions for budget splitting and aggregation.
            inference_pop.strata_pi = dict(population.strata_pi)
        else:
            inference_pop = population

        if num_workers <= 1:
            trial_rows: List[Dict[str, Any]] = []
            detail_rows: List[Dict[str, Any]] = []
            for inner_trial in range(int(n_inner_trials)):
                rows, details = run_one_trial(
                    inner_trial,
                    inference_pop,
                    budgets,
                    methods,
                    n_pilot=n_pilot,
                    seed=outer_seed,
                    covariance_method=covariance_method,
                    ridge=ridge,
                    eps_gap=eps_gap,
                    alpha=alpha,
                    theta_true=theta_true_outer,
                    outer_trial=outer_trial,
                    trial_id_offset=outer_trial * int(n_inner_trials),
                )
                trial_rows.extend(rows)
                detail_rows.extend(details)
        else:
            import multiprocessing as mp
            chunks = split_trials(int(n_inner_trials), num_workers)
            ctx = mp.get_context("spawn")
            worker_args = [
                (
                    chunk, inference_pop, budgets, methods, n_pilot, outer_seed,
                    covariance_method, ridge, eps_gap, alpha, theta_true_outer,
                    outer_trial, outer_trial * int(n_inner_trials)
                )
                for chunk in chunks
            ]
            trial_rows = []
            detail_rows = []
            with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as ex:
                for rows, details in ex.map(_worker, worker_args):
                    trial_rows.extend(rows)
                    detail_rows.extend(details)
            trial_rows = sorted(trial_rows, key=lambda r: (r["trial"], r["budget"], r["method"]))
            detail_rows = sorted(detail_rows, key=lambda r: (r["trial"], r["budget"], r["method"]))

        all_trial_dfs.append(pd.DataFrame(trial_rows))
        all_detail_dfs.append(pd.DataFrame(detail_rows))

    trials_df = pd.concat(all_trial_dfs, ignore_index=True) if all_trial_dfs else pd.DataFrame()
    details_df = pd.concat(all_detail_dfs, ignore_index=True) if all_detail_dfs else pd.DataFrame()
    summary_df = summarize_trials(trials_df)
    allocation_summary_df = make_allocation_summary(details_df, population.prediction_names)
    outer_truth_df = pd.DataFrame(outer_truth_rows)
    return {
        "summary_df": summary_df,
        "trials_df": trials_df,
        "details_df": details_df,
        "allocation_summary_df": allocation_summary_df,
        "outer_truth_df": outer_truth_df,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Full-cost stratified OMPPI demo for HumanEval / HumanEval+ pass@1 inference. "
            "This version does not reuse Y: pilot rows estimate routes/covariances only; "
            "fresh final rows are queried under each total budget."
        )
    )
    parser.add_argument("--final-table", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="./humaneval_execution_compare_outputs_fullcost_stratified")

    parser.add_argument("--target-col", type=str, default="Y_full_plus")
    parser.add_argument("--prediction-cols", type=str, required=True,
                        help="Comma-separated proxy columns, e.g. f_plus_50,f_plus_25,f_plus_10,f_original_tests,f_static_ok")
    parser.add_argument("--prediction-names", type=str, default=None,
                        help="Optional comma-separated display names for proxy columns.")
    parser.add_argument("--prompt-col", type=str, default="prompt")
    parser.add_argument("--strata-col", type=str, default=None,
                        help="Optional column used for stratification. Numeric columns are quantile-stratified; categorical columns are used as groups.")
    parser.add_argument("--num-strata", type=int, default=5)

    parser.add_argument("--y-cost-col", type=str, default=None)
    parser.add_argument("--y-cost", type=float, default=None)
    parser.add_argument("--prediction-cost-cols", type=str, default=None)
    parser.add_argument("--prediction-costs", type=str, default=None)
    parser.add_argument("--normalize-costs", type=str, choices=["yes", "no"], default="yes")
    parser.add_argument("--cost-floor", type=float, default=1e-4)

    parser.add_argument("--n-pilot", type=int, default=500)
    parser.add_argument("--n-trials", type=int, default=500, help="Inner trials per outer truth draw.")
    parser.add_argument("--n-outer-trials", type=int, default=100)
    parser.add_argument("--theta-truth-size", type=int, default=1500)
    parser.add_argument("--exclude-truth-from-inference", choices=["yes", "no"], default="yes")
    parser.add_argument("--budgets", type=str, default="200:2000:10")
    parser.add_argument("--covariance-method", type=str, default="ledoitwolf")
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--eps-gap", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", help="Accepted for consistency with other scripts; computation is CPU-based except random seeding.")
    parser.add_argument("--detail-save-mode", choices=["compact", "full", "gzip", "none"], default="compact")
    parser.add_argument("--save", choices=["yes", "no"], default="yes")
    args = parser.parse_args()

    _ = resolve_device(args.device)  # Keep consistent CLI behavior; this script mainly uses NumPy/SciPy.

    final_df = load_table(args.final_table)
    pred_cols = parse_csv_list(args.prediction_cols)
    pred_names = parse_csv_list(args.prediction_names) if args.prediction_names else None
    pred_cost_cols = parse_csv_list(args.prediction_cost_cols)
    pred_costs = [float(x) for x in parse_csv_list(args.prediction_costs)] if args.prediction_costs else []

    population = build_execution_population(
        final_df,
        target_col=args.target_col,
        prediction_cols=pred_cols,
        prediction_names=pred_names,
        prompt_col=args.prompt_col,
        strata_col=args.strata_col,
        num_strata=args.num_strata,
        y_cost_col=args.y_cost_col,
        y_cost=args.y_cost,
        prediction_cost_cols=pred_cost_cols,
        prediction_costs=pred_costs,
        normalize_costs=(args.normalize_costs == "yes"),
        cost_floor=args.cost_floor,
    )

    budgets = parse_budgets(args.budgets)
    methods = make_methods()

    print(f"Rows used: {population.X.shape[0]}")
    print(f"Target column: {args.target_col}")
    print(f"Prediction columns: {pred_cols}")
    print(f"Prediction names: {population.prediction_names}")
    print(f"Full empirical pass@1 mean over all rows (not used directly as theta_true): {population.theta_true:.6f}")
    print(f"Y cost raw: {population.y_cost_raw:.6g}; Y cost used: {population.y_cost_used:.6g}")
    print(f"Prediction costs used: {population.costs_used}")
    print("OMPPI cost mode: nested evaluator incremental costs (OMPPI only).")
    print(f"Number of strata: {len(population.strata_pi)}")
    print(f"Strata pi_h: {population.strata_pi}")
    print(f"Outer theta_true construction: {args.n_outer_trials} draws, subset size={args.theta_truth_size}")
    print(f"Inner inference repetitions per outer draw: {args.n_trials}")
    print(f"Exclude truth subset from inference pool: {args.exclude_truth_from_inference}")
    print(f"Pilot rows per inner trial: {args.n_pilot}. Pilot is not reused in the final estimator.")
    print(f"Budgets: {budgets}")
    print(f"Methods: {[m.name for m in methods]}")

    results = run_outer_inner_truth_experiment(
        population,
        budgets=budgets,
        n_pilot=args.n_pilot,
        n_outer_trials=args.n_outer_trials,
        n_inner_trials=args.n_trials,
        theta_truth_size=args.theta_truth_size,
        exclude_truth_from_inference=(args.exclude_truth_from_inference == "yes"),
        methods=methods,
        covariance_method=args.covariance_method,
        ridge=args.ridge,
        eps_gap=args.eps_gap,
        alpha=args.alpha,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    summary_df = results["summary_df"]
    trials_df = results["trials_df"]
    details_df = results["details_df"]
    allocation_summary_df = results["allocation_summary_df"]
    outer_truth_df = results["outer_truth_df"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save == "yes":
        summary_df.to_csv(out_dir / "summary.csv", index=False)
        trials_df.to_csv(out_dir / "trials.csv", index=False)
        outer_truth_df.to_csv(out_dir / "outer_truth.csv", index=False)
        if args.detail_save_mode == "full":
            details_df.to_csv(out_dir / "method_details.csv", index=False)
        elif args.detail_save_mode == "gzip":
            details_df.to_csv(out_dir / "method_details.csv.gz", index=False, compression="gzip")
        elif args.detail_save_mode == "compact":
            allocation_summary_df.to_csv(out_dir / "allocation_summary.csv", index=False)
            (out_dir / "method_details_NOT_SAVED.txt").write_text(
                "method_details.csv was not saved because detail_save_mode=compact.\n"
                "Use allocation_summary.csv for OMPPI/MultiPPI allocation diagnostics.\n",
                encoding="utf-8",
            )
        elif args.detail_save_mode == "none":
            (out_dir / "method_details_NOT_SAVED.txt").write_text(
                "method_details.csv was not saved because detail_save_mode=none.\n",
                encoding="utf-8",
            )

        config = {
            "final_table": args.final_table,
            "target_col": args.target_col,
            "prediction_cols": pred_cols,
            "prediction_names": population.prediction_names,
            "prompt_col": args.prompt_col,
            "strata_col": args.strata_col,
            "num_strata": args.num_strata,
            "n_pilot": args.n_pilot,
            "n_trials": args.n_trials,
            "n_outer_trials": args.n_outer_trials,
            "theta_truth_size": args.theta_truth_size,
            "exclude_truth_from_inference": args.exclude_truth_from_inference,
            "budgets": budgets,
            "covariance_method": args.covariance_method,
            "ridge": args.ridge,
            "eps_gap": args.eps_gap,
            "alpha": args.alpha,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "full_empirical_theta_true_not_used_directly": population.theta_true,
            "y_cost_raw": population.y_cost_raw,
            "y_cost_used": population.y_cost_used,
            "prediction_costs_raw": population.costs_raw,
            "prediction_costs_used": population.costs_used,
            "omppi_cost_mode": "nested_evaluator_incremental_omppi_only",
            "strata_pi": population.strata_pi,
            "strata_summary": population.strata_summary,
            "detail_save_mode": args.detail_save_mode,
            "no_reuse_y": True,
            "target": "pass@1 mean: theta*=E[Y], Y=1{completion passes full HumanEval+/execution tests}",
        }
        (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"Saved outputs to: {out_dir}")

if __name__ == "__main__":
    main()
