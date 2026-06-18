"""
OMPPI vs VectorPPI++ vs MultiPPI geometry simulation.

Output:
    omppi_vectorppi_multippi_sim_outputs/simulation_summary.csv
    omppi_vectorppi_multippi_sim_outputs/omppi_vectorppi_multippi_geometry_combined.pdf

Notes:
    - OMPPI uses the full nested chain with scalar marginal alignments.
    - VectorPPI++ uses the full predictor vector as a single multivariate surrogate.
    - MultiPPI uses the original continuous single-budget allocation over
      one fully labeled subset and all nonempty predictor-only subsets.
    - The perturbation severity q is restricted to [0, 0.6] to keep the
      worst-direction path in the local perturbation regime.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import minimize


warnings.filterwarnings(
    "ignore",
    message="Values in x were outside bounds during a minimize step",
    category=RuntimeWarning,
)

Z975 = 1.959963984540054
FULL_COV_NOISE_NORM = 0.02
HELVETICA_PATH = Path("fonts/Helvetica.ttf")


def configure_matplotlib_style() -> None:
    if HELVETICA_PATH.exists():
        font_manager.fontManager.addfont(str(HELVETICA_PATH))
        plt.rcParams["font.family"] = "Helvetica"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"

    plt.rcParams.update(
        {
            "mathtext.fontset": "cm",
            "font.size": 22,
            "axes.titlesize": 22,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "figure.titlesize": 22,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def coverage_from_variances(V_true: float, V_report: float) -> float:
    V_true = max(float(V_true), 1e-14)
    V_report = max(float(V_report), 1e-14)
    return 2.0 * normal_cdf(Z975 * math.sqrt(V_report / V_true)) - 1.0


def width_from_variance(V_report: float) -> float:
    V_report = max(float(V_report), 1e-14)
    return 2.0 * Z975 * math.sqrt(V_report)


def equicorr(K: int, eta: float) -> np.ndarray:
    rho = 1.0 - eta
    return np.full((K, K), rho) + np.eye(K) * (1.0 - rho)


def make_true_sigma(K: int = 5, eta: float = 0.05) -> np.ndarray:
    r0 = np.array([0.850, 0.835, 0.820, 0.805, 0.790], dtype=float)[:K]
    R = equicorr(K, eta)
    scale = min(1.0, math.sqrt(0.90 / float(r0 @ np.linalg.inv(R) @ r0)))
    r = scale * r0

    Sigma = np.zeros((K + 1, K + 1))
    Sigma[0, 0] = 1.0
    Sigma[0, 1:] = r
    Sigma[1:, 0] = r
    Sigma[1:, 1:] = R

    if np.linalg.eigvalsh(Sigma).min() <= 0:
        raise RuntimeError("Constructed Sigma is not positive definite.")
    return Sigma


def least_stable_signal_direction(R: np.ndarray, cov_yf: np.ndarray) -> tuple[np.ndarray, float]:
    vals, vecs = np.linalg.eigh(R)
    eta_min = float(vals[0])
    mask = vals <= vals[0] + 1e-8
    U = vecs[:, mask]
    v = U @ (U.T @ cov_yf)

    if np.linalg.norm(v) < 1e-12:
        v = vecs[:, 0]
    else:
        v = v / np.linalg.norm(v)

    if float(v @ cov_yf) < 0:
        v = -v

    return v, eta_min


def make_R_delta(
    Sigma_true: np.ndarray,
    q: float,
    direction: str = "worst",
    zero_diagonal: bool = True,
    full_psd_margin: float = 1e-6,
) -> np.ndarray:
    R = Sigma_true[1:, 1:]
    cov_yf = Sigma_true[0, 1:]

    v, eta_min = least_stable_signal_direction(R, cov_yf)
    M = np.outer(v, v)

    if zero_diagonal:
        M = M - np.diag(np.diag(M))

    op = np.linalg.norm(M, 2)
    if op > 0:
        M = M / op

    sign = -1.0 if direction == "worst" else 1.0
    Delta = sign * q * eta_min * M

    alpha = 1.0
    while True:
        Sigma_hat = Sigma_true.copy()
        Sigma_hat[1:, 1:] = R + alpha * Delta
        if np.linalg.eigvalsh(Sigma_hat).min() > full_psd_margin:
            return alpha * Delta
        alpha *= 0.8
        if alpha < 1e-8:
            return alpha * Delta


def add_fixed_full_covariance_noise(
    Sigma_hat: np.ndarray,
    scale: float = FULL_COV_NOISE_NORM,
    seed: int = 777,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    E = rng.normal(size=Sigma_hat.shape)
    E = (E + E.T) / 2.0
    E = E / np.linalg.norm(E, "fro") * scale

    alpha = 1.0
    while True:
        candidate = Sigma_hat + alpha * E
        if np.linalg.eigvalsh(candidate).min() > 1e-7 and np.all(np.diag(candidate) > 1e-7):
            return candidate
        alpha *= 0.8
        if alpha < 1e-8:
            return Sigma_hat


def omppi_variance(n: np.ndarray, gamma: np.ndarray, Sigma: np.ndarray) -> float:
    K = Sigma.shape[0] - 1
    var_y = Sigma[0, 0]
    var_f = np.diag(Sigma)[1:]
    cov_yf = Sigma[0, 1:]

    V = var_y / n[0]
    for k in range(K):
        coeff = 1.0 / n[k] - 1.0 / n[k + 1]
        V += coeff * (gamma[k] ** 2 * var_f[k] - 2.0 * gamma[k] * cov_yf[k])

    return float(max(V, 1e-14))


def omppi_eval(Sigma_hat: np.ndarray, Sigma_true: np.ndarray, costs: np.ndarray, B: float) -> dict:
    K = Sigma_hat.shape[0] - 1
    var_y_hat = Sigma_hat[0, 0]
    var_f_hat = np.diag(Sigma_hat)[1:]
    cov_yf_hat = Sigma_hat[0, 1:]

    tau = np.empty(K + 2)
    tau[0] = var_y_hat
    tau[1 : K + 1] = cov_yf_hat ** 2 / var_f_hat
    tau[K + 1] = 0.0

    Delta = np.maximum(tau[:-1] - tau[1:], 1e-8)
    Q = float(np.sum(np.sqrt(Delta * costs)))
    n = B / Q * np.sqrt(Delta / costs)
    gamma = cov_yf_hat / var_f_hat

    V_report = omppi_variance(n, gamma, Sigma_hat)
    V_true = omppi_variance(n, gamma, Sigma_true)

    return {
        "method": "OMPPI",
        "V_report": V_report,
        "V_true": V_true,
        "coverage": coverage_from_variances(V_true, V_report),
        "width": width_from_variance(V_report),
        "active_sets": "full nested chain",
        "n_active_sets": 6,
    }


def vectorppi_variance(n0: float, nF: float, lam: np.ndarray, Sigma: np.ndarray) -> float:
    """Variance of nested VectorPPI++ for mean estimation."""
    var_y = float(Sigma[0, 0])
    cov_yf = Sigma[0, 1:]
    Sigma_ff = Sigma[1:, 1:]

    quad = float(lam @ Sigma_ff @ lam)
    lin = float(lam @ cov_yf)
    V = var_y / n0 + (1.0 / n0 - 1.0 / nF) * (quad - 2.0 * lin)
    return float(max(V, 1e-14))


def vectorppi_eval(
    Sigma_hat: np.ndarray,
    Sigma_true: np.ndarray,
    marginal_costs: np.ndarray,
    B: float,
) -> dict:
    """Evaluate VectorPPI++ using the full predictor vector as one surrogate."""
    K = Sigma_hat.shape[0] - 1
    var_y_hat = float(Sigma_hat[0, 0])
    cov_yf_hat = Sigma_hat[0, 1:]
    Sigma_ff_hat = Sigma_hat[1:, 1:]

    inv_ff_hat = np.linalg.inv(Sigma_ff_hat + 1e-10 * np.eye(K))
    lam = inv_ff_hat @ cov_yf_hat
    tau_joint = float(cov_yf_hat @ inv_ff_hat @ cov_yf_hat)
    tau_joint = min(max(tau_joint, 1e-10), var_y_hat - 1e-10)

    c0 = float(marginal_costs[0])
    c_vec = float(np.sum(marginal_costs[1:]))

    Delta = np.array([var_y_hat - tau_joint, tau_joint], dtype=float)
    c_two = np.array([c0, c_vec], dtype=float)
    Q = float(np.sum(np.sqrt(Delta * c_two)))
    n0, nF = B / Q * np.sqrt(Delta / c_two)

    # The current simulation regime satisfies n0 <= nF. This guard keeps the
    # variance well-defined if future costs violate the nested ordering.
    if nF < n0:
        n0 = nF = B / (c0 + c_vec)

    V_report = vectorppi_variance(n0, nF, lam, Sigma_hat)
    V_true = vectorppi_variance(n0, nF, lam, Sigma_true)

    return {
        "method": "VectorPPI++",
        "V_report": V_report,
        "V_true": V_true,
        "coverage": coverage_from_variances(V_true, V_report),
        "width": width_from_variance(V_report),
        "active_sets": "joint predictor vector",
        "n_active_sets": 2,
    }


def build_multipppi_collection(K: int) -> list[tuple[int, ...]]:
    full = tuple(range(K + 1))
    predictor_sets = []
    for r in range(1, K + 1):
        for subset in itertools.combinations(range(1, K + 1), r):
            predictor_sets.append(tuple(subset))
    return [full] + predictor_sets


def collection_costs(collection: list[tuple[int, ...]], marginal_costs: np.ndarray) -> np.ndarray:
    return np.array([sum(marginal_costs[list(I)]) for I in collection], dtype=float)


def multipppi_original_eval(
    Sigma_hat: np.ndarray,
    Sigma_true: np.ndarray,
    marginal_costs: np.ndarray,
    B: float,
) -> dict:
    d = Sigma_hat.shape[0]
    K = d - 1
    a = np.zeros(d)
    a[0] = 1.0

    collection = build_multipppi_collection(K)
    cI = collection_costs(collection, marginal_costs)
    m = len(collection)

    G_blocks = []
    inv_blocks_hat = []
    for I in collection:
        block = Sigma_hat[np.ix_(I, I)]
        inv_block = np.linalg.inv(block + 1e-10 * np.eye(len(I)))
        inv_blocks_hat.append(inv_block)

        G = np.zeros((d, d))
        for local_i, global_i in enumerate(I):
            for local_j, global_j in enumerate(I):
                G[global_i, global_j] = inv_block[local_i, local_j]
        G_blocks.append(G)

    def M0_from_b(b: np.ndarray) -> np.ndarray:
        M0 = np.zeros((d, d))
        for bi, ci, Gi in zip(b, cI, G_blocks):
            if bi > 0:
                M0 += (bi / ci) * Gi
        return M0

    def objective_and_grad(b: np.ndarray) -> tuple[float, np.ndarray]:
        M0 = M0_from_b(b) + 1e-10 * np.eye(d)
        x = np.linalg.solve(M0, a)
        val = float(a @ x)

        grad = np.empty_like(b)
        for idx, (ci, Gi) in enumerate(zip(cI, G_blocks)):
            grad[idx] = -(x @ Gi @ x) / ci
        return val, grad

    def fun(b: np.ndarray) -> float:
        return objective_and_grad(b)[0]

    def jac(b: np.ndarray) -> np.ndarray:
        return objective_and_grad(b)[1]

    b0 = np.ones(m) * (0.5 / (m - 1))
    b0[0] = 0.5
    b0 = b0 / b0.sum()

    constraints = [
        {
            "type": "eq",
            "fun": lambda b: np.sum(b) - 1.0,
            "jac": lambda b: np.ones_like(b),
        }
    ]
    bounds = [(1e-8, 1.0)] + [(0.0, 1.0)] * (m - 1)

    res = minimize(
        fun,
        b0,
        jac=jac,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-11, "maxiter": 1000, "disp": False},
    )

    b = np.maximum(res.x, 0.0)
    b = b / b.sum()
    n = B * b / cI

    M = np.zeros((d, d))
    for ni, Gi in zip(n, G_blocks):
        if ni > 0:
            M += ni * Gi
    S = np.linalg.inv(M + 1e-10 * np.eye(d))
    x_full = S @ a

    q_hat = []
    q_true = []
    active_sets = []
    for I, ni, inv_hat in zip(collection, n, inv_blocks_hat):
        x_I = x_full[list(I)]
        lam = ni * (inv_hat @ x_I)

        qh = float(lam @ Sigma_hat[np.ix_(I, I)] @ lam)
        qt = float(lam @ Sigma_true[np.ix_(I, I)] @ lam)
        q_hat.append(max(qh, 0.0))
        q_true.append(max(qt, 0.0))

        if ni > 1e-5 and qh > 1e-10:
            active_sets.append(I)

    q_hat = np.array(q_hat)
    q_true = np.array(q_true)
    active = n > 1e-8

    V_report = float(np.sum(q_hat[active] / n[active]))
    V_true = float(np.sum(q_true[active] / n[active]))

    return {
        "method": "MultiPPI",
        "V_report": max(V_report, 1e-14),
        "V_true": max(V_true, 1e-14),
        "coverage": coverage_from_variances(V_true, V_report),
        "width": width_from_variance(V_report),
        "active_sets": ";".join(str(I) for I in active_sets),
        "n_active_sets": int(len(active_sets)),
    }


def run_one(
    eta: float,
    q: float,
    setting: str,
    direction: str,
    costs: np.ndarray,
    B: float,
    full_noise_scale: float = FULL_COV_NOISE_NORM,
) -> list[dict]:
    Sigma_true = make_true_sigma(K=5, eta=eta)

    Delta_R = make_R_delta(
        Sigma_true,
        q=q,
        direction=direction,
        zero_diagonal=True,
    )

    Sigma_hat = Sigma_true.copy()
    Sigma_hat[1:, 1:] = Sigma_true[1:, 1:] + Delta_R

    if setting == "full_covariance_perturbation":
        Sigma_hat = add_fixed_full_covariance_noise(
            Sigma_hat,
            scale=full_noise_scale,
            seed=777,
        )

    rows = []
    for out in [
        omppi_eval(Sigma_hat, Sigma_true, costs, B),
        vectorppi_eval(Sigma_hat, Sigma_true, costs, B),
        multipppi_original_eval(Sigma_hat, Sigma_true, costs, B),
    ]:
        out.update(
            {
                "setting": setting,
                "direction": direction,
                "eta_min_R": eta,
                "q": q,
                "eig_min_R_hat": float(np.linalg.eigvalsh(Sigma_hat[1:, 1:]).min()),
                "variance_ratio_report_over_true": out["V_report"] / out["V_true"],
                "full_noise_norm": full_noise_scale,
            }
        )
        rows.append(out)

    return rows


def print_table(rows: list[dict], metric: str, setting: str) -> None:
    etas = sorted({r["eta_min_R"] for r in rows})
    qs = sorted({r["q"] for r in rows})

    print(f"\n{metric} | {setting}")
    print("lambda_min  q      OMPPI   VectorPPI++   MultiPPI")

    for eta in etas:
        for q in qs:
            ss = [
                r
                for r in rows
                if r["setting"] == setting
                and r["eta_min_R"] == eta
                and abs(r["q"] - q) < 1e-12
            ]
            om = next(r for r in ss if r["method"] == "OMPPI")
            vp = next(r for r in ss if r["method"] == "VectorPPI++")
            mp = next(r for r in ss if r["method"] == "MultiPPI")
            print(
                f"{eta:<7.3g} {q:<6.3g} "
                f"{om[metric]:>7.3f} {vp[metric]:>13.3f} {mp[metric]:>10.3f}"
            )


def method_color(method: str) -> str:
    """Use one fixed color per method."""
    return {
        "OMPPI": plt.get_cmap("Blues")(0.82),        # blue
        "VectorPPI++": plt.get_cmap("Reds")(0.62),  # red
        "MultiPPI": plt.get_cmap("Oranges")(0.92),     
    }[method]


def eta_style_map(etas: list[float]) -> dict[float, tuple]:
    """Encode eta by line style instead of color."""
    etas_sorted = sorted(etas, reverse=True)
    styles = [
        ("-", 0.98),
        ("--", 0.98),
        ("-.", 0.98),
        ((0, (3, 1, 1, 1)), 0.98),
        ((0, (1, 1)), 0.98),
        ((0, (5, 1, 1, 1, 1, 1)), 0.98),
    ]
    return {eta: styles[i % len(styles)] for i, eta in enumerate(etas_sorted)}

def eta_legend_label(method: str, eta: float) -> str:
    return rf"{method}, $\lambda_{{\min}}(R)={eta:.3g}$"

def save_combined_figure(df: pd.DataFrame, out_path: Path) -> None:
    configure_matplotlib_style()

    metric_info = [
        ("coverage", "Coverage"),
        ("width", "CI width"),
        ("variance_ratio_report_over_true", "Variance ratio"),
    ]
    settings = [
        ("pure_R_perturbation", r"Pure $R$ perturbation"),
        ("full_covariance_perturbation", "Full covariance perturbation"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20.5, 10.2), sharex=True)

    for row, (setting, row_title) in enumerate(settings):
        dset = df[df["setting"] == setting].copy()
        etas = sorted(dset["eta_min_R"].unique(), reverse=True)
        eta_styles = eta_style_map(etas)

        method_markers = {
            "OMPPI": "o",
            "VectorPPI++": "^",
            "MultiPPI": "s",
        }
        method_linewidths = {
            "OMPPI": 2.8,
            "VectorPPI++": 2.6,
            "MultiPPI": 2.6,
        }

        for col, (metric, ylabel) in enumerate(metric_info):
            ax = axes[row, col]

            for method in ["OMPPI", "VectorPPI++", "MultiPPI"]:
                for eta in etas:
                    sub = dset[
                        (dset["method"] == method)
                        & (dset["eta_min_R"] == eta)
                    ].sort_values("q")

                    linestyle, alpha = eta_styles[eta]
                    ax.plot(
                        sub["q"],
                        sub[metric],
                        marker=method_markers[method],
                        linestyle=linestyle,
                        linewidth=method_linewidths[method],
                        markersize=6.0,
                        color=method_color(method),
                        alpha=alpha,
                        label=eta_legend_label(method, eta),
                    )

            if metric == "coverage":
                ax.axhline(0.95, linestyle=":", linewidth=1.5, color="black")

                coverage_min = float(dset["coverage"].min())
                coverage_max = float(dset["coverage"].max())
                lower = max(0.0, coverage_min - 0.03)
                upper = min(1.01, coverage_max + 0.015)
                if upper - lower < 0.08:
                    lower = max(0.0, upper - 0.08)
                ax.set_ylim(lower, upper)

            if metric == "variance_ratio_report_over_true":
                ax.axhline(1.0, linestyle=":", linewidth=1.5, color="black")

            ax.set_title(row_title)
            ax.set_xlabel(r"Perturbation severity $q$")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25, linewidth=0.7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=15,
        columnspacing=1.15,
        handlelength=2.8,
        handletextpad=0.5,
        borderaxespad=0.2,
        bbox_to_anchor=(0.5, 0.1),
    )

    fig.tight_layout(rect=[0.0, 0.18, 1.0, 1.0])
    fig.savefig(out_path / "omppi_vectorppi_multippi_geometry_combined.pdf", bbox_inches="tight")
    plt.close(fig)


def run_all(out_dir: str = "omppi_vectorppi_multippi_sim_outputs") -> list[dict]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    costs = np.array([1000.0, 20.0, 4.0, 0.8, 0.16, 0.03])
    B = 5000.0

    eta_grid = [0.10, 0.08, 0.06, 0.04, 0.02]
    #eta_grid = [0.10, 0.05, 0.02]
    q_grid = list(np.linspace(0.0, 0.6, 10))
    settings = ["pure_R_perturbation", "full_covariance_perturbation"]
    direction = "worst"

    rows: list[dict] = []
    for setting in settings:
        for eta in eta_grid:
            for q in q_grid:
                rows.extend(
                    run_one(
                        eta=eta,
                        q=q,
                        setting=setting,
                        direction=direction,
                        costs=costs,
                        B=B,
                    )
                )

    csv_path = out_path / "simulation_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    save_combined_figure(df, out_path)

    for setting in settings:
        print_table(rows, "coverage", setting)
        print_table(rows, "width", setting)
        print_table(rows, "variance_ratio_report_over_true", setting)

    print(f"\nSaved CSV to: {csv_path.resolve()}")
    print(
        f"Saved combined PDF to: "
        f"{(out_path / 'omppi_vectorppi_multippi_geometry_combined.pdf').resolve()}"
    )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="omppi_vectorppi_multippi_sim_outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(out_dir=args.out_dir)
