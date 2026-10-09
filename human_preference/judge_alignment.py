#!/usr/bin/env python3
"""Judge alignment in the Chatbot Arena data (Figure D.2 and Table D.1).

For each LLM judge f: correlation with Y, alignment gamma = Cov(Y, f) / Var(f), explained variance
tau^2 = Cov(Y, f)^2 / Var(f), and the single-predictor leading variance ratio V_OMPPI / V_LO with
n0 labeled and n1 judged comparisons, together with the normalized cost and mean API time
(Table D.1). Three empirical bias directions b(X) are standardized: Position (GPT-4-1106-Preview
shown first), Length (log length ratio of the two responses) and Consensus (mean vote share of the
other judges). Figure D.2 shows Corr{Y, b(X)} and V_OMPPI / V_LO for the base judge perturbed on
the log-odds scale, f_lambda = sigmoid(logit f + lambda b(X)).

Writes results/judge_alignment/: judge_table.csv, alignment_table.csv, perturbation_curve.csv,
run_metadata.json and judge_alignment.pdf.
Usage: python judge_alignment.py
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import arena_data as ad

HERE = Path(__file__).resolve().parent
SHORT = {"gemini-2.5-pro": "G2.5-Pro", "gemini-2.5-flash": "G2.5-Flash", "gemini-2.5-flash-lite": "G2.5-Lite",
         "gemini-3.1-flash-preview": "G3.1-Flash", "gemini-3.1-flash-lite-preview": "G3.1-Lite",
         "qwen3-next-80b-instruct": "Qwen-Next", "qwen3-235b-a22b-instruct": "Qwen-235B",
         "qwen3-coder-480b-a35b-instruct": "Qwen-Coder", "gpt-oss-120b": "OSS-120B", "gpt-oss-20b": "OSS-20B"}


def pop_var(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean((x - np.mean(x)) ** 2))


def pop_cov(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))))


def pop_corr(x, y):
    vx, vy = pop_var(x), pop_var(y)
    if vx <= 1e-15 or vy <= 1e-15:
        return float("nan")
    return float(pop_cov(x, y) / np.sqrt(vx * vy))


def alignment_stats(y, f) -> Dict[str, float]:
    var_y, var_f, cov_yf = pop_var(y), pop_var(f), pop_cov(y, f)
    gamma = 0.0 if var_f <= 1e-15 else cov_yf / var_f
    tau2 = 0.0 if var_f <= 1e-15 else cov_yf ** 2 / var_f
    return {"corr_yf": pop_corr(y, f), "cov_yf": cov_yf, "gamma": gamma, "tau2": tau2,
            "R2_tau_over_varY": 0.0 if var_y <= 1e-15 else tau2 / var_y}


def v_ratio(var_y, tau2, n0, n1):
    v_lo = var_y / n0
    v_omppi = v_lo - (1.0 / n0 - 1.0 / n1) * tau2
    return v_lo, v_omppi, v_omppi / v_lo


def standardize(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    return np.zeros_like(x) if not np.isfinite(sd) or sd <= 1e-15 else (x - np.nanmean(x)) / sd


def sigmoid(x):
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def logit_clip(p, eps):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def configure_style():
    font = os.environ.get("OMPPI_FONT")       # optional TrueType font, e.g. Helvetica as in the paper
    if font and Path(font).exists():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"]
    plt.rcParams.update({"font.size": 21, "axes.titlesize": 22.5, "axes.labelsize": 21, "xtick.labelsize": 18,
                         "ytick.labelsize": 18, "legend.fontsize": 16.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "mathtext.fontset": "cm", "axes.facecolor": "white", "figure.facecolor": "white",
                         "savefig.facecolor": "white"})


def plot(alignment_df, curve_df, base_model, out_pdf):
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.0), gridspec_kw={"width_ratios": [3, 7]})
    ax = axes[0]
    df = alignment_df.copy()
    df["direction"] = pd.Categorical(df["direction"], categories=["Position", "Length", "Consensus"], ordered=True)
    df = df.sort_values("direction")
    x = np.arange(len(df)); vals = df["corr_Y_btilde"].to_numpy(dtype=float)
    ax.bar(x, vals, width=0.40, color="0.65", edgecolor="0.35", linewidth=0.8)
    ax.axhline(0.0, color="black", lw=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(df["direction"].astype(str).tolist(), rotation=18, ha="right")
    ax.set_ylabel(r"$\widehat{\mathrm{Corr}}\{Y,\widetilde b(X)\}$")
    ax.set_title("Alignment")
    ax.set_ylim(min(-0.05, float(np.nanmin(vals)) - 0.05), max(0.05, float(np.nanmax(vals)) + 0.06))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    style = {"Position": {"marker": "o", "ls": "-"}, "Length": {"marker": "s", "ls": "--"}, "Consensus": {"marker": "^", "ls": "-."}}
    for direction, g in curve_df.groupby("direction", sort=False):
        g = g.sort_values("perturb_strength")
        ax.plot(g["perturb_strength"].to_numpy(dtype=float), g["V_ratio_OMPPI_over_LO"].to_numpy(dtype=float),
                label=direction, lw=2.8, ms=6.5, **style[direction])
    ax.axhline(1.0, color="black", lw=1.3, ls=":")
    ax.axvline(0.0, color="black", lw=1.1, ls=":", alpha=0.8)
    ax.set_xlabel(r"Perturbation strength $\lambda$")
    ax.set_ylabel(r"$\widehat V_{\mathrm{OMPPI}}/\widehat V_{\mathrm{LO}}$")
    ax.set_title(f"Perturbed {SHORT.get(base_model, base_model)} judge")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.88, wspace=0.35)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-model", default="gpt-oss-20b")
    ap.add_argument("--n0", type=int, default=600)
    ap.add_argument("--n1", type=int, default=1600)
    ap.add_argument("--lambdas", default="-2:2:17", help="start:stop:count")
    ap.add_argument("--logit-eps", type=float, default=1e-4)
    a = ap.parse_args()
    out = HERE / "results" / "judge_alignment"
    out.mkdir(parents=True, exist_ok=True)

    pop, names = ad.load_population()
    y = pop.X[:, 0]
    score = {m: pop.X[:, 1 + j] for j, m in enumerate(names)}
    var_y = pop_var(y)

    rows = []
    for m in names:
        st = alignment_stats(y, score[m])
        v_lo, v_om, ratio = v_ratio(var_y, st["tau2"], a.n0, a.n1)
        rows.append({"model": m, "display_model": SHORT.get(m, m), "corr_Y_f": st["corr_yf"], "cov_Y_f": st["cov_yf"],
                     "gamma_hat": st["gamma"], "tau2_hat": st["tau2"], "R2_tau_over_varY": st["R2_tau_over_varY"],
                     "V_LO": v_lo, "V_OMPPI_single": v_om, "V_ratio_OMPPI_over_LO": ratio, "variance_reduction": 1.0 - ratio,
                     "cost_normalized": pop.costs_used[m], "cost_raw_mean_usd": pop.costs_raw[m], "time_mean_sec": pop.times[m]})
    judge_table = pd.DataFrame(rows).sort_values(["R2_tau_over_varY", "cost_normalized"], ascending=[False, True]).reset_index(drop=True)

    gpt4_is_A = pop.df["gpt4_is_A"].to_numpy(dtype=bool)
    resp_a = pop.df["response_a"].fillna("").astype(str).to_numpy()
    resp_b = pop.df["response_b"].fillna("").astype(str).to_numpy()
    len_gpt4 = np.asarray([len(s) for s in np.where(gpt4_is_A, resp_a, resp_b)], dtype=float)
    len_claude = np.asarray([len(s) for s in np.where(gpt4_is_A, resp_b, resp_a)], dtype=float)
    directions = {
        "Position": standardize(np.where(gpt4_is_A, 1.0, -1.0)),
        "Length": standardize(np.log1p(len_gpt4) - np.log1p(len_claude)),
        "Consensus": standardize(np.mean(np.column_stack([score[m] for m in names if m != a.base_model]), axis=1)),
    }
    alignment_df = pd.DataFrame([{"direction": d, "n": int(len(y)), "corr_Y_btilde": pop_corr(y, b), "cov_Y_btilde": pop_cov(y, b),
                                  "var_btilde": pop_var(b), "mean_btilde_if_Y1": float(np.mean(b[y == 1])),
                                  "mean_btilde_if_Y0": float(np.mean(b[y == 0]))} for d, b in directions.items()])

    lo, hi, k = a.lambdas.split(":")
    lambdas = np.linspace(float(lo), float(hi), int(k))
    base_logit = logit_clip(score[a.base_model], a.logit_eps)
    curve = []
    for d, b in directions.items():
        for lam in lambdas:
            st = alignment_stats(y, sigmoid(base_logit + float(lam) * b))
            v_lo, v_om, ratio = v_ratio(var_y, st["tau2"], a.n0, a.n1)
            curve.append({"base_model": a.base_model, "base_display_model": SHORT.get(a.base_model, a.base_model), "direction": d,
                          "perturb_strength": float(lam), "corr_Y_f_lambda": st["corr_yf"], "cov_Y_f_lambda": st["cov_yf"],
                          "gamma_lambda": st["gamma"], "tau2_lambda": st["tau2"], "R2_lambda": st["R2_tau_over_varY"],
                          "V_LO": v_lo, "V_OMPPI_lambda": v_om, "V_ratio_OMPPI_over_LO": ratio, "variance_reduction": 1.0 - ratio})
    curve_df = pd.DataFrame(curve)

    judge_table.to_csv(out / "judge_table.csv", index=False)
    alignment_df.to_csv(out / "alignment_table.csv", index=False)
    curve_df.to_csv(out / "perturbation_curve.csv", index=False)
    meta = {"n_comparisons": int(len(y)), "theta_hat": float(np.mean(y)), "var_y": var_y, "n0": a.n0, "n1": a.n1,
            "base_model": a.base_model, "lambda_grid": [float(x) for x in lambdas], "judges": names}
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    configure_style()
    plot(alignment_df, curve_df, a.base_model, out / "judge_alignment.pdf")

    print(f"n = {len(y)}, theta_hat = {np.mean(y):.4f}, var_y = {var_y:.4f}")
    print("\nTable D.1 (tau2_hat, cost_normalized, time_mean_sec), with the correlation corr_Y_f:")
    print(judge_table[["display_model", "corr_Y_f", "tau2_hat", "cost_normalized", "time_mean_sec"]].round(4).to_string(index=False))
    print("\nAlignment of the bias directions:")
    print(alignment_df[["direction", "corr_Y_btilde", "cov_Y_btilde"]].to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
