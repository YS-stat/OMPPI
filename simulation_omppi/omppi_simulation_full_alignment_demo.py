#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# Global constants
# ============================================================

THETA_STAR = np.array([1.0, 2.0, -1.5])
CHI2_95_D3 = 7.814727903251179


# ============================================================
# Command-line options
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OMPPI simulation with full alignment, exhaustive selection, and DAG selection."
    )
    parser.add_argument("--out-dir", type=str, default="./omppi_sim_outputs")
    parser.add_argument("--font-path", type=str, default="fonts/Helvetica.ttf")
    parser.add_argument("--nrep", type=int, default=300)
    parser.add_argument("--pop-n", type=int, default=200000)
    parser.add_argument("--eta", type=float, default=0.20)
    parser.add_argument("--budgets", type=str, default="5000:100000:20")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_budget_grid(s: str) -> List[int]:
    s = str(s).strip()
    if ":" in s:
        lo, hi, n = s.split(":")
        return [int(round(x)) for x in np.linspace(float(lo), float(hi), int(n))]
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


# ============================================================
# Plot style
# ============================================================

def configure_plot_style(font_path: str) -> None:
    path = Path(font_path)
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        font_name = font_manager.FontProperties(fname=str(path)).get_name()
    else:
        font_name = "DejaVu Sans"

    plt.rcParams.update({
        "font.family": font_name,
        "mathtext.fontset": "cm",
        "font.size": 25,
        "axes.titlesize": 25,
        "axes.labelsize": 25,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "figure.titlesize": 25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
    })


# ============================================================
# Data generating process
# ============================================================

def generate_X(n: int, rng: np.random.Generator) -> np.ndarray:
    cov12 = np.array([[1.0, 0.3], [0.3, 1.0]])
    idx = np.arange(4)
    cov3456 = 0.45 ** np.abs(np.subtract.outer(idx, idx))

    x12 = rng.multivariate_normal(np.zeros(2), cov12, size=n)
    x3456 = rng.multivariate_normal(np.zeros(4), cov3456, size=n)
    return np.hstack([x12, x3456])


def Zmat(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X[:, 0], X[:, 1]])


def components(X: np.ndarray) -> List[np.ndarray]:
    _, _, x3, x4, x5, x6 = X.T

    g0 = 1.6 * np.sin(x3)
    g1 = 1.25 * (x4 * x5 - 0.45)
    g2 = 1.05 * (x6 ** 2 - 1.0)
    g3 = 0.95 * (np.cos(x3 + 0.5 * x4) - np.exp(-0.85))
    g4 = 0.85 * (x5 * x6 - 0.45)
    return [g0, g1, g2, g3, g4]


def oracle_y(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mu = Zmat(X) @ THETA_STAR + sum(components(X))
    return mu + rng.normal(0.0, 0.45, size=len(X))


def predictors(X: np.ndarray, scenario: str) -> List[np.ndarray]:
    base = Zmat(X) @ THETA_STAR
    g0, g1, g2, g3, g4 = components(X)

    if scenario == "A":
        return [
            base + 0.78 * g0,
            base + g0 + 0.75 * g1,
            base + g0 + g1 + 0.55 * g2,
            base + g0 + g1 + g2 + 0.92 * g3,
            base + g0 + g1 + g2 + g3 + 0.96 * g4,
        ]

    if scenario == "B":
        return [
            base + 0.78 * g0,
            base + g0 + 0.76 * g1,
            base + g0 + g1 + 0.75 * g2,
            base + g0 + g1 + g2 + 0.05 * g3,
            base + g0 + g1 + g2 + g3 + 0.94 * g4,
        ]

    raise ValueError("scenario must be 'A' or 'B'.")


# ============================================================
# Score, covariance, and alignment
# ============================================================

def cov_emp(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    A = np.asarray(A)
    if A.ndim == 1:
        A = A[:, None]

    if B is None:
        B = A
    else:
        B = np.asarray(B)
        if B.ndim == 1:
            B = B[:, None]

    if len(A) == 0:
        raise ValueError("Empty sample in covariance computation.")

    A0 = A - A.mean(axis=0, keepdims=True)
    B0 = B - B.mean(axis=0, keepdims=True)
    return (A0.T @ B0) / len(A)


def solve_theta_labeled(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Z = Zmat(X)
    return np.linalg.solve(Z.T @ Z, Z.T @ y)


def score_matrix(X: np.ndarray, response: np.ndarray, theta: np.ndarray) -> np.ndarray:
    Z = Zmat(X)
    return Z * (response - Z @ theta)[:, None]


def apply_gamma(Gamma: np.ndarray, score: np.ndarray) -> np.ndarray:
    return score @ Gamma.T


def compute_alignment_profile(
    X: np.ndarray,
    y: np.ndarray,
    pred_list: Sequence[np.ndarray],
    theta_ref: np.ndarray | None = None,
    ridge: float = 1e-8,
) -> Tuple[Dict[int, float], Dict[int, np.ndarray], Dict[int, float]]:
    """Estimate full-alignment residual traces and alignment matrices.

    Returns:
        r[-1]: trace covariance of the true score.
        r[k]: residual trace after optimally aligning predictor k.
        r[5]: zero residual trace for the true-label level.
        gammas[k]: full alignment matrix for predictor k.
        tau2[k]: explained trace for predictor k.
    """
    if theta_ref is None:
        theta_ref = solve_theta_labeled(X, y)

    ell_y = score_matrix(X, y, theta_ref)
    Sigma00 = cov_emp(ell_y)
    tau0 = float(np.trace(Sigma00))

    r: Dict[int, float] = {-1: tau0}
    gammas: Dict[int, np.ndarray] = {}
    tau2: Dict[int, float] = {}

    d = ell_y.shape[1]
    eye = np.eye(d)

    for k, pred in enumerate(pred_list):
        ell_k = score_matrix(X, pred, theta_ref)
        Sigma0k = cov_emp(ell_y, ell_k)
        Sigmakk = cov_emp(ell_k)

        inv_kk = np.linalg.pinv(Sigmakk + ridge * eye)
        Gamma = Sigma0k @ inv_kk

        explained = float(np.trace(Sigma0k @ inv_kk @ Sigma0k.T))
        residual = float(tau0 - explained)

        gammas[k] = Gamma
        tau2[k] = explained
        r[k] = max(residual, 1e-12)

    r[5] = 0.0
    return r, gammas, tau2


def approximate_population_profile(
    scenario: str,
    n: int,
    seed: int,
    ridge: float,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    rng = np.random.default_rng(seed)
    X = generate_X(n, rng)
    y = oracle_y(X, rng)
    pred_list = predictors(X, scenario)

    r, _, tau2 = compute_alignment_profile(
        X,
        y,
        pred_list,
        theta_ref=THETA_STAR,
        ridge=ridge,
    )
    return r, tau2


# ============================================================
# Sub-hierarchy selection
# ============================================================

def path_score(path: Sequence[int], r: Dict[int, float], costs: np.ndarray) -> float:
    prev = -1
    total = 0.0
    for s in path:
        drop = r[prev] - r[s]
        if drop <= 0:
            return math.inf
        total += math.sqrt(drop * costs[s])
        prev = s
    return total


def exhaustive_select(r: Dict[int, float], costs: np.ndarray) -> Tuple[List[int], float]:
    """Enumerate all ordered sub-hierarchies and return the minimum path."""
    best_path: List[int] | None = None
    best_score = math.inf

    predictor_levels = list(range(5))
    for size in range(0, len(predictor_levels) + 1):
        for subset in itertools.combinations(predictor_levels, size):
            path = list(subset) + [5]
            score = path_score(path, r, costs)
            if score < best_score:
                best_score = score
                best_path = path

    if best_path is None:
        raise RuntimeError("No valid exhaustive path found.")

    return best_path, best_score


def dag_select(r: Dict[int, float], costs: np.ndarray) -> Tuple[List[int], float]:
    """Shortest path on the level DAG from the pre-prediction state -1 to level 5."""
    nodes = [-1, 0, 1, 2, 3, 4, 5]
    dist = {-1: 0.0}
    prev_node: Dict[int, int] = {}

    for v in nodes[1:]:
        best = math.inf
        best_u = None
        for u in nodes:
            if u >= v:
                continue
            if u not in dist:
                continue
            drop = r[u] - r[v]
            if drop <= 0:
                continue
            val = dist[u] + math.sqrt(drop * costs[v])
            if val < best:
                best = val
                best_u = u

        if best_u is not None:
            dist[v] = best
            prev_node[v] = best_u

    if 5 not in dist:
        raise RuntimeError("No valid DAG path found.")

    path = []
    v = 5
    while v != -1:
        path.append(v)
        v = prev_node[v]

    path.reverse()
    return path, dist[5]


# ============================================================
# Allocation
# ============================================================

def allocate_path(
    rhat: Dict[int, float],
    costs: np.ndarray,
    budget_remaining: float,
    selected_path: Sequence[int],
) -> Dict[int, int]:
    """Continuous OMPPI allocation followed by integer nesting and budget repair."""
    prev = -1
    drops = []
    cvals = []

    for s in selected_path:
        drops.append(max(rhat[prev] - rhat[s], 1e-12))
        cvals.append(costs[s])
        prev = s

    drops = np.asarray(drops, dtype=float)
    cvals = np.asarray(cvals, dtype=float)

    raw = np.sqrt(drops / cvals)
    n_cont = budget_remaining * raw / np.sum(cvals * raw)
    n = np.maximum(np.floor(n_cont).astype(int), 2)

    for j in range(len(n) - 2, -1, -1):
        n[j] = max(n[j], n[j + 1])

    while float(np.dot(cvals, n)) > budget_remaining:
        changed = False
        for j in range(len(n) - 1, -1, -1):
            lower = n[j + 1] if j < len(n) - 1 else 2
            if n[j] > lower:
                n[:j + 1] -= 1
                changed = True
                break

        if not changed:
            raise ValueError("Budget is too small to support a positive nested allocation.")

    return {int(s): int(nj) for s, nj in zip(selected_path, n)}


# ============================================================
# Estimation and inference
# ============================================================

def stable_inverse(A: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def fit_labeled(X: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    Xn = X[:n]
    yn = y[:n]
    Z = Zmat(Xn)

    A = (Z.T @ Z) / n
    b = (Z.T @ yn) / n
    theta_hat = np.linalg.solve(A, b)

    psi = score_matrix(Xn, yn, theta_hat)
    Vhat = cov_emp(psi)

    Ainv = stable_inverse(A)
    Sigma = Ainv @ (Vhat / n) @ Ainv.T
    return theta_hat, Sigma


def fit_omppi(
    X: np.ndarray,
    y: np.ndarray,
    pred_list: Sequence[np.ndarray],
    selected_path: Sequence[int],
    n_by_level: Dict[int, int],
    gammas: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit OMPPI for a selected path with full alignment matrices."""
    if selected_path == [5]:
        return fit_labeled(X, y, n_by_level[5])

    d = len(THETA_STAR)
    A = np.zeros((d, d))
    b = np.zeros(d)

    def szz(n: int) -> np.ndarray:
        Z = Zmat(X[:n])
        return (Z.T @ Z) / n

    def zresp(response: np.ndarray, n: int) -> np.ndarray:
        Z = Zmat(X[:n])
        return (Z.T @ response[:n]) / n

    pred_levels = [s for s in selected_path if s != 5]
    first = pred_levels[0]
    n_first = n_by_level[first]
    G_first = gammas[first]

    A += G_first @ szz(n_first)
    b += G_first @ zresp(pred_list[first], n_first)

    prev = first
    G_prev = G_first

    for s in pred_levels[1:]:
        n_s = n_by_level[s]
        G_s = gammas[s]

        A += G_s @ szz(n_s) - G_prev @ szz(n_s)
        b += G_s @ zresp(pred_list[s], n_s) - G_prev @ zresp(pred_list[prev], n_s)

        prev = s
        G_prev = G_s

    n_y = n_by_level[5]
    A += szz(n_y) - G_prev @ szz(n_y)
    b += zresp(y, n_y) - G_prev @ zresp(pred_list[prev], n_y)

    theta_hat = np.linalg.solve(A, b)

    # Build aligned score increments at theta_hat.
    increments: List[np.ndarray] = []
    sample_sizes: List[int] = []

    ell_first = score_matrix(X[:n_first], pred_list[first][:n_first], theta_hat)
    increments.append(apply_gamma(G_first, ell_first))
    sample_sizes.append(n_first)

    prev = first
    G_prev = G_first

    for s in pred_levels[1:]:
        n_s = n_by_level[s]
        G_s = gammas[s]

        ell_s = score_matrix(X[:n_s], pred_list[s][:n_s], theta_hat)
        ell_prev = score_matrix(X[:n_s], pred_list[prev][:n_s], theta_hat)

        inc = apply_gamma(G_s, ell_s) - apply_gamma(G_prev, ell_prev)
        increments.append(inc)
        sample_sizes.append(n_s)

        prev = s
        G_prev = G_s

    ell_y = score_matrix(X[:n_y], y[:n_y], theta_hat)
    ell_prev = score_matrix(X[:n_y], pred_list[prev][:n_y], theta_hat)
    inc_y = ell_y - apply_gamma(G_prev, ell_prev)
    increments.append(inc_y)
    sample_sizes.append(n_y)

    # Nested-sample covariance estimator.
    Vsum = np.zeros((d, d))
    for j, inc_j in enumerate(increments):
        n_j = sample_sizes[j]
        Vhat_j = cov_emp(inc_j)

        for t in range(j + 1, len(increments)):
            n_t = sample_sizes[t]
            Vhat_j += 2.0 * cov_emp(inc_j[:n_t], increments[t][:n_t])

        Vsum += Vhat_j / n_j

    Ainv = stable_inverse(A)
    Sigma = Ainv @ Vsum @ Ainv.T
    return theta_hat, Sigma


def compute_metrics(theta_hat: np.ndarray, Sigma: np.ndarray) -> Tuple[float, float, float]:
    S = 0.5 * (Sigma + Sigma.T)
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.clip(eigvals, 1e-12, None)
    S = eigvecs @ np.diag(eigvals) @ eigvecs.T

    diff = THETA_STAR - theta_hat
    coverage = float(diff.T @ stable_inverse(S) @ diff <= CHI2_95_D3)
    volume = float((4.0 * np.pi / 3.0) * (CHI2_95_D3 ** 1.5) * np.sqrt(np.linalg.det(S)))
    sqerr = float(np.sum(diff ** 2))
    return sqerr, coverage, volume


# ============================================================
# One Monte Carlo replication
# ============================================================

def run_one(
    scenario: str,
    costs: np.ndarray,
    budget: float,
    eta: float,
    rtrue: Dict[int, float],
    oracle_path: Sequence[int],
    seed: int,
    ridge: float,
) -> Tuple[List[dict], dict]:
    rng = np.random.default_rng(seed)

    n_pilot = int(np.floor(eta * budget / np.sum(costs)))
    if n_pilot < 10:
        raise ValueError("Pilot sample is too small. Increase budget or eta.")

    budget_remaining = float((1.0 - eta) * budget)

    # Pilot sample for covariance estimation and route selection.
    Xp = generate_X(n_pilot, rng)
    yp = oracle_y(Xp, rng)
    predp = predictors(Xp, scenario)
    rhat, gammas, _ = compute_alignment_profile(Xp, yp, predp, theta_ref=None, ridge=ridge)

    path_exh, _ = exhaustive_select(rhat, costs)
    path_dag, _ = dag_select(rhat, costs)

    n_exh = allocate_path(rhat, costs, budget_remaining, path_exh)
    n_dag = allocate_path(rhat, costs, budget_remaining, path_dag)

    n_lab = int(np.floor(budget / costs[5]))
    n_lab = max(n_lab, 10)

    n_need = max(max(n_exh.values()), max(n_dag.values()), n_lab) + 5

    # Fresh main sample for estimation.
    Xm = generate_X(n_need, rng)
    ym = oracle_y(Xm, rng)
    predm = predictors(Xm, scenario)

    rows = []

    theta_hat, Sigma = fit_labeled(Xm, ym, n_lab)
    sqerr, cover, volume = compute_metrics(theta_hat, Sigma)
    rows.append({
        "method": "Classical",
        "sqerr": sqerr,
        "coverage": cover,
        "volume": volume,
    })

    theta_hat, Sigma = fit_omppi(Xm, ym, predm, path_exh, n_exh, gammas)
    sqerr, cover, volume = compute_metrics(theta_hat, Sigma)
    rows.append({
        "method": "OMPPI(Exhaustive)",
        "sqerr": sqerr,
        "coverage": cover,
        "volume": volume,
    })

    theta_hat, Sigma = fit_omppi(Xm, ym, predm, path_dag, n_dag, gammas)
    sqerr, cover, volume = compute_metrics(theta_hat, Sigma)
    rows.append({
        "method": "OMPPI(DAG)",
        "sqerr": sqerr,
        "coverage": cover,
        "volume": volume,
    })

    selection_info = {
        "path_exhaustive": tuple(path_exh),
        "path_dag": tuple(path_dag),
        "path_oracle": tuple(oracle_path),
        "select_oracle_exhaustive": int(tuple(path_exh) == tuple(oracle_path)),
        "select_oracle_dag": int(tuple(path_dag) == tuple(oracle_path)),
        "dag_matches_exhaustive": int(tuple(path_dag) == tuple(path_exh)),
        "n_pilot": n_pilot,
        "n_lab": n_lab,
    }

    return rows, selection_info


# ============================================================
# Simulation driver
# ============================================================

def summarize_results(raw_df: pd.DataFrame, sel_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        raw_df.groupby(["scenario", "cost_profile", "budget", "method"])
        .agg(
            rmse=("sqerr", lambda x: float(np.sqrt(np.mean(x)))),
            coverage=("coverage", "mean"),
            volume=("volume", "mean"),
        )
        .reset_index()
    )

    base = (
        summary[summary["method"] == "Classical"]
        [["scenario", "cost_profile", "budget", "rmse", "volume"]]
        .rename(columns={"rmse": "rmse_classical", "volume": "volume_classical"})
    )

    summary = summary.merge(base, on=["scenario", "cost_profile", "budget"], how="left")
    summary["rmse_ratio"] = summary["rmse"] / summary["rmse_classical"]
    summary["volume_ratio"] = summary["volume"] / summary["volume_classical"]

    sel_summary = (
        sel_df.groupby(["scenario", "cost_profile", "budget"])
        .agg(
            p_select_oracle_exhaustive=("select_oracle_exhaustive", "mean"),
            p_select_oracle_dag=("select_oracle_dag", "mean"),
            p_dag_matches_exhaustive=("dag_matches_exhaustive", "mean"),
            mode_exhaustive_path=("path_exhaustive", lambda s: s.mode().iloc[0]),
            mode_dag_path=("path_dag", lambda s: s.mode().iloc[0]),
            oracle_path=("path_oracle", lambda s: s.iloc[0]),
        )
        .reset_index()
    )

    return summary, sel_summary


# ============================================================
# Plotting
# ============================================================

def setting_label(scenario: str, cost_profile: str) -> str:
    cost_map = {
        "Balanced": "Balanced",
        "Redundant-mid-expensive": "Redundant mid-cost",
    }
    return f"Scenario {scenario}\n{cost_map.get(cost_profile, cost_profile)}"


def plot_main_figure(summary: pd.DataFrame, sel_summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    settings = [
        ("A", "Balanced"),
        ("A", "Redundant-mid-expensive"),
        ("B", "Balanced"),
        ("B", "Redundant-mid-expensive"),
    ]

    method_order = ["Classical", "OMPPI(Exhaustive)", "OMPPI(DAG)"]
    method_colors = {
        "Classical": "#7f7f7f",
        "OMPPI(Exhaustive)": plt.get_cmap("Blues")(0.82),
        "OMPPI(DAG)": plt.get_cmap("Blues")(0.55),
    }
    method_styles = {
        "Classical": dict(lw=2.4, ls=":", marker=None, alpha=0.95),
        "OMPPI(Exhaustive)": dict(lw=2.6, ls="-", marker="o", ms=4.8, alpha=0.95),
        "OMPPI(DAG)": dict(lw=0.0, ls="None", marker="o", ms=5.2, alpha=0.95),
    }

    fig, axes = plt.subplots(4, 4, figsize=(25.0, 18.0), sharex="col")
    row_titles = ["Coverage", "RMSE / Classical", "CI volume / Classical", "Selection probability"]

    for col_idx, (scenario, cost_profile) in enumerate(settings):
        axes[0, col_idx].set_title(setting_label(scenario, cost_profile))

        sub = summary[
            (summary["scenario"] == scenario)
            & (summary["cost_profile"] == cost_profile)
        ].copy()

        sel = sel_summary[
            (sel_summary["scenario"] == scenario)
            & (sel_summary["cost_profile"] == cost_profile)
        ].copy()

        # Row 1: coverage.
        ax = axes[0, col_idx]
        for method in method_order:
            g = sub[sub["method"] == method].sort_values("budget")
            ax.plot(
                g["budget"],
                g["coverage"],
                color=method_colors[method],
                label=method,
                **method_styles[method],
            )
        ax.axhline(0.95, color="black", lw=1.2, ls=":")
        ax.set_ylim(0.85, 1.00)

        # Row 2: RMSE ratio.
        ax = axes[1, col_idx]
        for method in method_order:
            g = sub[sub["method"] == method].sort_values("budget")
            ax.plot(
                g["budget"],
                g["rmse_ratio"],
                color=method_colors[method],
                label=method,
                **method_styles[method],
            )
        ax.axhline(1.0, color="black", lw=1.2, ls=":")
        vals = sub["rmse_ratio"].to_numpy(dtype=float)
        ax.set_ylim(max(0.0, np.nanmin(vals) - 0.04), min(1.08, np.nanmax(vals) + 0.04))

        # Row 3: CI volume ratio.
        ax = axes[2, col_idx]
        for method in method_order:
            g = sub[sub["method"] == method].sort_values("budget")
            ax.plot(
                g["budget"],
                g["volume_ratio"],
                color=method_colors[method],
                label=method,
                **method_styles[method],
            )
        ax.axhline(1.0, color="black", lw=1.2, ls=":")
        vals = sub["volume_ratio"].to_numpy(dtype=float)
        ax.set_ylim(max(0.0, np.nanmin(vals) - 0.04), min(1.08, np.nanmax(vals) + 0.04))

        # Row 4: selection probability and DAG consistency.
        ax = axes[3, col_idx]
        sel = sel.sort_values("budget")
        ax.plot(
            sel["budget"],
            sel["p_select_oracle_exhaustive"],
            color=method_colors["OMPPI(Exhaustive)"],
            lw=2.6,
            marker="o",
            ms=4.8,
            label="Exhaustive selects oracle",
        )
        ax.plot(
            sel["budget"],
            sel["p_select_oracle_dag"],
            color=method_colors["OMPPI(DAG)"],
            lw=0.0,
            marker="o",
            ms=5.2,
            label="DAG selects oracle",
        )
        ax.plot(
            sel["budget"],
            sel["p_dag_matches_exhaustive"],
            color="#2ca02c",
            lw=2.2,
            ls="--",
            marker="s",
            ms=4.5,
            label="DAG = Exhaustive",
        )
        ax.set_ylim(0.40, 1.02)
        ax.set_xlabel("Budget")

        for row_idx in range(4):
            axes[row_idx, col_idx].grid(alpha=0.25, linewidth=0.8)
            axes[row_idx, col_idx].spines["top"].set_visible(False)
            axes[row_idx, col_idx].spines["right"].set_visible(False)
            axes[row_idx, col_idx].ticklabel_format(style="plain", axis="x")

    for row_idx, title in enumerate(row_titles):
        axes[row_idx, 0].set_ylabel(title)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    handles_sel, labels_sel = axes[3, 0].get_legend_handles_labels()

    fig.legend(
        handles + handles_sel,
        labels + labels_sel,
        loc="lower center",
        ncol=6,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.0,
        handletextpad=0.5,
        bbox_to_anchor=(0.5, 0.07),
    )

    fig.tight_layout(rect=[0.0, 0.10, 1.0, 1.0])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_plot_style(args.font_path)

    budgets = parse_budget_grid(args.budgets)
    cost_profiles = {
        "Balanced": np.array([0.25, 0.5, 0.9, 1.6, 8.0, 40.0], dtype=float),
        "Redundant-mid-expensive": np.array([0.25, 0.5, 0.9, 12.0, 14.0, 40.0], dtype=float),
    }

    for name, costs in cost_profiles.items():
        if not np.all(np.diff(costs) > 0):
            raise ValueError(f"Cost profile {name} must be strictly increasing.")

    print("=" * 100)
    print("Approximating population full-alignment profiles")
    print("=" * 100)
    rtrue_by_scenario = {}
    oracle_paths = {}

    for scenario, seed_offset in [("A", 10), ("B", 20)]:
        rtrue, tau2 = approximate_population_profile(
            scenario=scenario,
            n=args.pop_n,
            seed=args.seed + seed_offset,
            ridge=args.ridge,
        )
        rtrue_by_scenario[scenario] = rtrue

        print(f"\nScenario {scenario}")
        print("Residual traces r:", rtrue)
        print("Explained traces tau2:", tau2)

        for cname, costs in cost_profiles.items():
            oracle_path, oracle_score = exhaustive_select(rtrue, costs)
            oracle_paths[(scenario, cname)] = oracle_path
            print(f"Oracle path for scenario={scenario}, cost={cname}: {oracle_path}, score={oracle_score:.4f}")

    raw_rows = []
    selection_rows = []

    rep_id = 0
    for scenario in ["A", "B"]:
        rtrue = rtrue_by_scenario[scenario]

        for cname, costs in cost_profiles.items():
            oracle_path = oracle_paths[(scenario, cname)]

            for budget in budgets:
                print(f"Running scenario={scenario}, cost_profile={cname}, budget={budget}, nrep={args.nrep}")

                for _ in range(args.nrep):
                    rep_id += 1
                    rows, sel = run_one(
                        scenario=scenario,
                        costs=costs,
                        budget=float(budget),
                        eta=args.eta,
                        rtrue=rtrue,
                        oracle_path=oracle_path,
                        seed=args.seed + rep_id,
                        ridge=args.ridge,
                    )

                    for row in rows:
                        raw_rows.append({
                            "scenario": scenario,
                            "cost_profile": cname,
                            "budget": float(budget),
                            **row,
                        })

                    selection_rows.append({
                        "scenario": scenario,
                        "cost_profile": cname,
                        "budget": float(budget),
                        **sel,
                    })

    raw_df = pd.DataFrame(raw_rows)
    sel_df = pd.DataFrame(selection_rows)

    summary, sel_summary = summarize_results(raw_df, sel_df)

    print("\n" + "=" * 100)
    print("Simulation summary")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("Selection summary")
    print("=" * 100)
    print(sel_summary.to_string(index=False))

    pdf_path = out_dir / "omppi_simulation_main.pdf"
    plot_main_figure(summary, sel_summary, pdf_path, dpi=args.dpi)
    print(f"\nSaved PDF to: {pdf_path}")


if __name__ == "__main__":
    main()
