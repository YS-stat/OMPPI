#!/usr/bin/env python3
"""Figures for the HumanEval+ experiment, written to results/figures/.

humaneval_combined_share.pdf (Figure 4)
  One row per pilot size (200, 400, 800). Columns: coverage | RMSE | CI width. RMSE and CI width are
  shown as reduction / oracle reduction, where the reduction of a design is 1 - (its RMSE or CI width)
  / (that of Classical) and the oracle is the best linear unbiased design over all 63 query blocks
  with the population covariance (results/oracle.csv). Dotted lines: each method's own design with
  the population covariance. 0% = Classical, 100% = oracle. The y-axis is broken (66-102% above,
  -95-25% below). Curves are centered 3-point moving averages over budgets.
humaneval_combined_allocation.pdf (Figure E.1)
  OMPPI and MultiPPI sample and cost allocations in percent per stratum at the largest budget. OMPPI
  counts are nested (a sample evaluated at one level also has every cheaper evaluator) and its cost is
  charged incrementally; MultiPPI blocks are priced by their most expensive evaluator.

OMPPI is the OMPPI(DAG) run (exhaustive search selects the same design in every repetition).
Usage: python humaneval_figures.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "figures"
PILOTS = [200, 400, 800]
SMOOTH = 3
METHODS = ["Classical", "VectorPPI++", "MultiPPI", "OMPPI"]
COLORS = {"Classical": "#7f7f7f", "VectorPPI++": "#d62728",
          "MultiPPI": plt.get_cmap("Oranges")(0.78), "OMPPI": plt.get_cmap("Blues")(0.82)}
STYLES = {"Classical": dict(lw=2.4, ls="-", marker="^", ms=7),   # all methods solid; dotted = oracle
          "VectorPPI++": dict(lw=2.4, ls="-", marker="D", ms=6.5),
          "MultiPPI": dict(lw=2.6, ls="-", marker="s", ms=7),
          "OMPPI": dict(lw=2.8, ls="-", marker="o", ms=7)}
ORACLE_LS = (0, (1.2, 1.8))
TOP_LIM, TOP_TICKS = (66, 102), [70, 80, 90, 100]
BOT_LIM, BOT_TICKS = (-95, 25), [-80, 0]
PRED_NAMES = ["Plus50", "Plus25", "Plus10", "OriginalTests", "StaticOK"]
OMPPI_SRC = ["Y"] + PRED_NAMES
MULTI_SRC = ["Y+Joint all", "Joint all"] + PRED_NAMES

_orc = pd.read_csv(RES / "oracle.csv").set_index("method")["width_ratio"]
CEIL = float(_orc["Oracle (all 63 blocks)"])
ORACLE_SHARE = {m: 100.0 * (1.0 - float(_orc[m])) / (1.0 - CEIL) for m in ("VectorPPI++", "MultiPPI", "OMPPI")}


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


def label_rows(fig, rows):
    """Row labels just left of each row's y-axis label."""
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for ax, text in rows:
        bb = ax.yaxis.label.get_window_extent().transformed(inv)
        fig.text(bb.x0 - 0.008, 0.5 * (bb.y0 + bb.y1), text, rotation=90, ha="right", va="center", fontsize=23)


# ============================================================
# Coverage, RMSE and CI width (Figure 4)
# ============================================================
def smooth(y):
    return pd.Series(np.asarray(y, float)).rolling(SMOOTH, min_periods=1, center=True).mean().to_numpy()


def load(p):
    s = pd.read_csv(RES / f"pilot{p}" / "summary.csv")
    s["method"] = s["method"].replace({"OMPPI(DAG)": "OMPPI"})
    s = s[s["method"].isin(METHODS)].copy()
    lo = s[s["method"] == "Classical"].set_index("budget")
    s["rmse_share"] = 100.0 * (1.0 - s["rmse_pool"] / s["budget"].map(lo["rmse_pool"])) / (1.0 - CEIL)
    s["width_share"] = 100.0 * (1.0 - s["mean_ci_width"] / s["budget"].map(lo["mean_ci_width"])) / (1.0 - CEIL)
    return s.sort_values(["method", "budget"])


def curve(ax, s, m, col):
    g = s[s["method"] == m]
    ax.plot(g["budget"], smooth(g[col]), color=COLORS[m], zorder=3, **STYLES[m])


def finish(ax):
    ax.grid(alpha=0.25, lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 3100)
    ax.set_xticks([0, 1000, 2000, 3000])


def coverage_panel(ax, s):
    for m in METHODS:
        curve(ax, s, m, "coverage_pool")
    ax.axhline(0.95, color="black", lw=1.8, ls=":")
    ax.set_ylim(0.90, 1.00)
    ax.set_yticks([0.90, 0.95, 1.00])
    finish(ax)


def share_panel(fig, spec, s, key):
    sub = spec.subgridspec(2, 1, height_ratios=[3.0, 1.3], hspace=0.14)
    top = fig.add_subplot(sub[0])
    bot = fig.add_subplot(sub[1], sharex=top)
    for ax in (top, bot):
        for m in ("VectorPPI++", "MultiPPI", "OMPPI"):
            ax.axhline(ORACLE_SHARE[m], color=COLORS[m], lw=2.0, ls=ORACLE_LS, zorder=2)
            curve(ax, s, m, f"{key}_share")
        finish(ax)
    top.set_ylim(*TOP_LIM)
    top.set_yticks(TOP_TICKS)
    bot.set_ylim(*BOT_LIM)
    bot.set_yticks(BOT_TICKS)
    top.spines["bottom"].set_visible(False)
    top.tick_params(axis="x", bottom=False, labelbottom=False)
    # break marks: two slashes on the y-axis
    kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=13, linestyle="none", color="black", mew=1.4, clip_on=False)
    top.plot([0], [0], transform=top.transAxes, **kw)
    bot.plot([0], [1], transform=bot.transAxes, **kw)
    return top, bot


def legend(fig):
    h = {m: Line2D([0], [0], color=COLORS[m], label=m if m == "Classical" else f"{m} (empirical)", **STYLES[m])
         for m in METHODS}
    o = {m: Line2D([0], [0], color=COLORS[m], lw=2.0, ls=ORACLE_LS, label=f"{m} (oracle) {ORACLE_SHARE[m]:.0f}%")
         for m in ORACLE_SHARE}
    blank = Line2D([0], [0], color="none", label=" ")
    # ncol=4 fills column-wise: row 1 = methods, row 2 = oracle levels
    handles = [h["Classical"], blank, h["VectorPPI++"], o["VectorPPI++"], h["MultiPPI"], o["MultiPPI"], h["OMPPI"], o["OMPPI"]]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.4, columnspacing=1.6, handletextpad=0.5)


def make_performance_figure():
    width, height = 20.0, 12.5
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(3, 3, left=0.11, right=0.985, top=1 - 0.45 / height, bottom=1.85 / height,
                            hspace=0.20, wspace=0.30)
    rows = []
    for i, p in enumerate(PILOTS):
        s = load(p)
        cov = fig.add_subplot(grid[i, 0])
        coverage_panel(cov, s)
        cov.set_ylabel("Coverage")
        rows.append((cov, f"Pilot size {p}"))
        for j, key in ((1, "rmse"), (2, "width")):
            top, bot = share_panel(fig, grid[i, j], s, key)
            if j == 1:          # centred on the whole row
                top.set_ylabel("Reduction /\noracle reduction (%)", y=0.233)
            if i == 0:
                top.set_title({"rmse": "RMSE", "width": "CI width"}[key])
            if i == len(PILOTS) - 1:
                bot.set_xlabel("Budget")
        if i == 0:
            cov.set_title("Coverage")
        if i == len(PILOTS) - 1:
            cov.set_xlabel("Budget")
    legend(fig)
    label_rows(fig, rows)
    fig.savefig(OUT / "humaneval_combined_share.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Allocation at the largest budget (Figure E.1)
# ============================================================
def allocation_tables(p, costs, strata):
    a = pd.read_csv(RES / f"pilot{p}" / "allocation_summary.csv")
    a = a[np.isclose(a["budget"], a["budget"].max())].copy()
    a["src"] = [("Y+Joint all" if s == "full" else "Joint all" if s == "joint_all"
                 else s.split("single__", 1)[1] if s.startswith("single__") else s) for s in a["source"]]
    hs = sorted(a["stratum"].unique())
    labels = [f"S{h}: {int(strata[str(h)]['min'])}-{int(strata[str(h)]['max'])}" for h in hs]

    def counts(method, order):
        return (a[a["method"] == method].pivot_table(index="stratum", columns="src", values="query_mean", aggfunc="sum")
                .reindex(index=hs, columns=order))

    q_o = counts("OMPPI(DAG)", OMPPI_SRC)
    q_o["Y"] = q_o["Y"].fillna(0.0)
    prev = q_o["Y"]
    for src in PRED_NAMES:      # nested sampling: each level also has every cheaper evaluator
        q_o[src] = np.maximum(q_o[src].fillna(prev), prev)
        prev = q_o[src]
    q_m = counts("MultiPPI", MULTI_SRC).fillna(0.0)

    c = {"Y": 1.0, **costs}
    incremental = {s: c[s] - (c[OMPPI_SRC[k + 1]] if k + 1 < len(OMPPI_SRC) else 0.0) for k, s in enumerate(OMPPI_SRC)}
    price = {"Y+Joint all": 1.0, "Joint all": max(costs.values()), **{s: costs[s] for s in PRED_NAMES}}

    def pct(d):
        d = d.fillna(0.0).astype(float)
        return 100.0 * d.div(d.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    tables = [pct(q_o), pct(q_m), pct(q_o * pd.Series(incremental)), pct(q_m * pd.Series(price))]
    return tables, labels


def heatmap(ax, df, cmap, row_labels, show_y, show_x):
    mat = df.to_numpy(dtype=float)
    vmax = max(1e-8, float(np.nanmax(mat)))
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(row_labels if show_y else [])
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(list(df.columns) if show_x else [], rotation=45, ha="right")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=16,
                    color="white" if mat[i, j] > 0.6 * vmax else "black")


def make_allocation_figure():
    cfg = json.loads((RES / "pilot800" / "config.json").read_text())
    costs = {str(k): float(v) for k, v in cfg["prediction_costs_used"].items()}
    width, height = 20.0, 11.0
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(3, 4, width_ratios=[len(OMPPI_SRC), len(MULTI_SRC)] * 2, left=0.16, right=0.995,
                            top=1 - 0.5 / height, bottom=1.9 / height, hspace=0.10, wspace=0.08)
    titles = ["OMPPI: sample allocation", "MultiPPI: sample allocation", "OMPPI: cost allocation", "MultiPPI: cost allocation"]
    cmaps = ["Blues", "Oranges", "Blues", "Oranges"]
    rows = []
    for i, p in enumerate(PILOTS):
        tables, labels = allocation_tables(p, costs, cfg["strata_summary"])
        for j in range(4):
            ax = fig.add_subplot(grid[i, j])
            heatmap(ax, tables[j], cmaps[j], labels, show_y=(j == 0), show_x=(i == len(PILOTS) - 1))
            if i == 0:
                ax.set_title(titles[j])
            if j == 0:
                ax.set_ylabel("Prompt-length\nstratum")
                rows.append((ax, f"Pilot size {p}"))
            if i == len(PILOTS) - 1:
                ax.set_xlabel("Source" if j in (0, 2) else "Queried block")
    label_rows(fig, rows)
    fig.savefig(OUT / "humaneval_combined_allocation.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    make_performance_figure()
    make_allocation_figure()
    print("saved:", sorted(x.name for x in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
