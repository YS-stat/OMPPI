#!/usr/bin/env python3
"""Figures for the Chatbot Arena experiment, written to results/figures/.

arena_combined_share.pdf (Figure D.3)
  One row per pilot size (50, 100, 200, 400). Columns: coverage | RMSE | CI width. RMSE and CI width
  are shown as reduction / oracle reduction, where the reduction of a design is 1 - (its RMSE or CI
  width) / (that of Classical) and the oracle reduction at budget B is that of the best linear
  unbiased design over all 1023 judge subsets with the population covariance (results/oracle.csv).
  Dotted curves: each method's own design with the population covariance. Curves are centered
  3-point moving averages over budgets.
arena_oracle_reduction.pdf (Figure D.5)
  Reduction relative to Classical (%) against the budget, all with the population covariance: the
  oracle (solid), each method's own design (dotted) and the oracle as B -> infinity (dashed).
arena_combined_allocation.pdf (Figure D.6)
  OMPPI and MultiPPI sample and cost allocations of the extra judge queries, in percent per stratum
  at B = 1500, averaged over the pilots of 200 repetitions (results/allocation.csv).
arena_correlation.pdf (Figure D.4)
  Correlation of Y and the ten judges over all 823 comparisons; the matrix is also saved as
  results/arena_correlation.csv.

OMPPI is the OMPPI(DAG) run (exhaustive search selects the same route in every repetition).
Usage: python arena_figures.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

import arena_data as ad

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "figures"
PILOTS = [50, 100, 200, 400]
SMOOTH = 3
METHODS = ["Classical", "VectorPPI++", "MultiPPI", "OMPPI"]
COLORS = {"Classical": "#7f7f7f", "VectorPPI++": "#d62728",
          "MultiPPI": plt.get_cmap("Oranges")(0.78), "OMPPI": plt.get_cmap("Blues")(0.82)}
STYLES = {"Classical": dict(lw=2.4, ls="-", marker="^", ms=7),   # all methods solid; dotted = oracle
          "VectorPPI++": dict(lw=2.4, ls="-", marker="D", ms=6.5),
          "MultiPPI": dict(lw=2.6, ls="-", marker="s", ms=7),
          "OMPPI": dict(lw=2.8, ls="-", marker="o", ms=7)}
ORACLE_LS = (0, (1.2, 1.8))
JUDGES = {"gemini-2.5-pro": "G2.5-Pro", "gemini-3.1-flash-preview": "G3.1-Flash",
          "gemini-3.1-flash-lite-preview": "G3.1-Lite", "gemini-2.5-flash": "G2.5-Flash",
          "gemini-2.5-flash-lite": "G2.5-Lite", "qwen3-next-80b-instruct": "Qwen-Next",
          "qwen3-235b-a22b-instruct": "Qwen-235B", "gpt-oss-120b": "OSS-120B",
          "qwen3-coder-480b-a35b-instruct": "Qwen-Coder", "gpt-oss-20b": "OSS-20B"}


def configure_style():
    font = os.environ.get("OMPPI_FONT")       # optional TrueType font, e.g. Helvetica as in the paper
    if font and Path(font).exists():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"]
    plt.rcParams.update({
        "mathtext.fontset": "cm", "font.size": 20, "axes.titlesize": 22, "axes.labelsize": 22,
        "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 18,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def smooth(y):
    return pd.Series(np.asarray(y, float)).rolling(SMOOTH, min_periods=1, center=True).mean().to_numpy()


def load(p, orc):
    s = pd.read_csv(RES / f"pilot{p}" / "summary.csv")
    s["method"] = s["method"].replace({"OMPPI(DAG)": "OMPPI"})
    s = s[s["method"].isin(METHODS)].copy()
    s["key"] = s["budget"].round(4)
    s = s.merge(orc[["key", "ratio_ceiling"]], on="key")
    lo = s[s["method"] == "Classical"].set_index("budget")
    for col, key in (("rmse_pool", "rmse"), ("mean_ci_width", "width")):
        s[f"{key}_share"] = 100.0 * (1.0 - s[col] / s["budget"].map(lo[col])) / (1.0 - s["ratio_ceiling"])
    return s.sort_values(["method", "budget"])


def finish(ax):
    ax.grid(alpha=0.25, lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 1600)
    ax.set_xticks([0, 500, 1000, 1500])


def label_rows(fig, rows):
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for ax, text in rows:
        bb = ax.yaxis.label.get_window_extent().transformed(inv)
        fig.text(bb.x0 - 0.008, 0.5 * (bb.y0 + bb.y1), text, rotation=90, ha="right", va="center", fontsize=23)


# ============================================================
# Coverage, RMSE and CI width (Figure D.3)
# ============================================================
def make_share_figure(orc):
    data = {p: load(p, orc) for p in PILOTS}
    vals = [data[p][data[p]["method"] != "Classical"][f"{k}_share"].to_numpy() for p in PILOTS for k in ("rmse", "width")]
    lo_y = min(-10.0, 10 * np.floor((np.min(np.concatenate(vals)) - 3) / 10))
    width, height = 20.0, 16.5
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(len(PILOTS), 3, left=0.11, right=0.985, top=1 - 0.45 / height, bottom=1.85 / height, hspace=0.20, wspace=0.30)
    rows = []
    for i, p in enumerate(PILOTS):
        s = data[p]
        cov = fig.add_subplot(grid[i, 0])
        for m in METHODS:
            g = s[s["method"] == m]
            cov.plot(g["budget"], smooth(g["coverage_pool"]), color=COLORS[m], zorder=3, **STYLES[m])
        cov.axhline(0.95, color="black", lw=1.8, ls=":")
        cov.set_ylim(0.90, 1.00); cov.set_yticks([0.90, 0.95, 1.00]); finish(cov)
        cov.set_ylabel("Coverage"); rows.append((cov, f"Pilot size {p}"))
        for j, key in ((1, "rmse"), (2, "width")):
            ax = fig.add_subplot(grid[i, j])
            for m, okey in (("VectorPPI++", "share_VectorPPI++"), ("MultiPPI", "share_MultiPPI"), ("OMPPI", "share_OMPPI")):
                ax.plot(orc["budget"], 100 * orc[okey], color=COLORS[m], lw=2.0, ls=ORACLE_LS, zorder=2)
                g = s[s["method"] == m]
                ax.plot(g["budget"], smooth(g[f"{key}_share"]), color=COLORS[m], zorder=3, **STYLES[m])
            ax.set_ylim(lo_y, 105); finish(ax)
            if j == 1:
                ax.set_ylabel("Reduction /\noracle reduction (%)")
            if i == 0:
                ax.set_title({"rmse": "RMSE", "width": "CI width"}[key])
            if i == len(PILOTS) - 1:
                ax.set_xlabel("Budget")
        if i == 0:
            cov.set_title("Coverage")
        if i == len(PILOTS) - 1:
            cov.set_xlabel("Budget")
    h = {m: Line2D([0], [0], color=COLORS[m], label=m if m == "Classical" else f"{m} (empirical)", **STYLES[m]) for m in METHODS}
    o = {m: Line2D([0], [0], color=COLORS[m], lw=2.0, ls=ORACLE_LS, label=f"{m} (oracle)") for m in ("VectorPPI++", "MultiPPI", "OMPPI")}
    blank = Line2D([0], [0], color="none", label=" ")
    fig.legend(handles=[h["Classical"], blank, h["VectorPPI++"], o["VectorPPI++"], h["MultiPPI"], o["MultiPPI"], h["OMPPI"], o["OMPPI"]],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=2.4, columnspacing=1.6, handletextpad=0.5)
    label_rows(fig, rows)
    fig.savefig(OUT / "arena_combined_share.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Oracle reductions against the budget (Figure D.5)
# ============================================================
def make_oracle_figure(orc):
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    ax.plot(orc["budget"], 100 * (1 - orc["ratio_ceiling"]), color="black", lw=2.8, label="Oracle", zorder=3)
    for m in ("OMPPI", "MultiPPI", "VectorPPI++"):
        ax.plot(orc["budget"], 100 * (1 - orc[f"ratio_{m}"]), color=COLORS[m], lw=2.6, ls=ORACLE_LS, label=f"{m} (oracle)", zorder=2)
    ax.axhline(100 * (1 - orc["ratio_ceiling_inf"].iloc[0]), color="#7f7f7f", lw=1.8, ls="--", label=r"Oracle, $B\to\infty$", zorder=1)
    ax.set_ylim(0, 12)
    ax.set_yticks([0, 3, 6, 9, 12])
    finish(ax)
    ax.set_xlabel("Budget")
    ax.set_ylabel("Reduction (%)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, handlelength=2.4)
    fig.savefig(OUT / "arena_oracle_reduction.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Allocation at the largest budget (Figure D.6)
# ============================================================
def allocation_tables(a, p, strata):
    a = a[a["pilot"] == p].copy()
    a["src"] = a["source"].map({"Joint": "Joint all", **JUDGES})
    judges = list(JUDGES.values())

    def pct(method, value, order):
        t = (a[a["method"] == method].pivot_table(index="stratum", columns="src", values=value, aggfunc="sum")
             .reindex(index=strata, columns=order).fillna(0.0).astype(float))
        return 100.0 * t.div(t.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    return [pct("OMPPI", "mean_extra_queries", judges), pct("MultiPPI", "mean_extra_queries", ["Joint all"] + judges),
            pct("OMPPI", "mean_cost", judges), pct("MultiPPI", "mean_cost", ["Joint all"] + judges)]


def heatmap(ax, df, cmap, row_labels, show_y, show_x):
    mat = df.to_numpy(dtype=float)
    vmax = max(1e-8, float(np.nanmax(mat)))
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(row_labels if show_y else [])
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(list(df.columns) if show_x else [], rotation=45, ha="right", rotation_mode="anchor", fontsize=17)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=15,
                    color="white" if mat[i, j] > 0.6 * vmax else "black")


def make_allocation_figure(pop):
    a = pd.read_csv(RES / "allocation.csv")
    strata = sorted(int(h) for h in pop.strata_token_summary)
    labels = [f"S{h}: {int(pop.strata_token_summary[h]['min'])}-{int(pop.strata_token_summary[h]['max'])}" for h in strata]
    width, height = 20.0, 14.0
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(len(PILOTS), 4, width_ratios=[len(JUDGES), len(JUDGES) + 1] * 2, left=0.16, right=0.995,
                            top=1 - 0.5 / height, bottom=2.0 / height, hspace=0.10, wspace=0.06)
    titles = ["OMPPI: sample allocation", "MultiPPI: sample allocation", "OMPPI: cost allocation", "MultiPPI: cost allocation"]
    cmaps = ["Blues", "Oranges", "Blues", "Oranges"]
    rows = []
    for i, p in enumerate(PILOTS):
        tables = allocation_tables(a, p, strata)
        for j in range(4):
            ax = fig.add_subplot(grid[i, j])
            heatmap(ax, tables[j], cmaps[j], labels, show_y=(j == 0), show_x=(i == len(PILOTS) - 1))
            if i == 0:
                ax.set_title(titles[j])
            if j == 0:
                ax.set_ylabel("Prompt-length\nstratum")
                rows.append((ax, f"Pilot size {p}"))
            if i == len(PILOTS) - 1:
                ax.set_xlabel("LLM judge" if j in (0, 2) else "Queried block")
    label_rows(fig, rows)
    fig.savefig(OUT / "arena_combined_allocation.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Correlations (Figure D.4)
# ============================================================
def make_correlation_figure(pop, names):
    cols = [0] + [1 + names.index(m) for m in JUDGES]
    mat = np.corrcoef(np.asarray(pop.X, float)[:, cols], rowvar=False)
    labels = ["Y"] + list(JUDGES.values())
    corr = pd.DataFrame(mat, index=labels, columns=labels)
    corr.round(4).to_csv(RES / "arena_correlation.csv")

    fig, ax = plt.subplots(figsize=(13.0, 11.0))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_title("Correlation among Human Label and LLM Judges")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=15,
                    color="white" if abs(mat[i, j]) >= 0.7 else "black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")
    fig.tight_layout()
    fig.savefig(OUT / "arena_correlation.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    orc = pd.read_csv(RES / "oracle.csv")
    orc["key"] = orc["budget"].round(4)
    pop, names = ad.load_population()
    make_share_figure(orc)
    make_oracle_figure(orc)
    make_allocation_figure(pop)
    make_correlation_figure(pop, names)
    print("saved:", sorted(x.name for x in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
