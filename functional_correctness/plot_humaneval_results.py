#!/usr/bin/env python3
# ============================================================
# HumanEval plotting notebook cell
# Matches LLM-preference plotting template as closely as possible
#
# Outputs:
#   1. prediction_correlation_heatmap.pdf
#   2. performance_absolute_coverage_rmse_width.pdf
#   3. performance_relative_coverage_rmse_width.pdf
#   4. allocation_omppi_only_max_budget.pdf
#   5. allocation_query_cost_2x2_max_budget.pdf
#   6. auxiliary_oracle_refcost_table.csv
#
# No VectorPPI++, no refline drawn on figures.
# Refline/refcost diagnostic is printed and saved as a table.
# ============================================================

from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

print("SCRIPT VERSION: HumanEval nested-cost plots, no plotted refline, with refcost diagnostics")

# ============================================================
# Paths
# Run from:
#   OMPPI/functional_correctness
# ============================================================
OUT_DIR = Path("./outputs/main")
SUMMARY_PATH = OUT_DIR / "summary.csv"
CONFIG_PATH = OUT_DIR / "config.json"
ALLOC_PATH = OUT_DIR / "allocation_summary.csv"
FINAL_TABLE_PATH = Path("./data/humaneval_plus_generated_outputs_final.csv")

PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = Path("./fonts/Helvetica.ttf")
SMOOTH_WINDOW = 3

CORR_PDF = PLOTS_DIR / "prediction_correlation_heatmap.pdf"
ABS_PDF = PLOTS_DIR / "performance_absolute_coverage_rmse_width.pdf"
REL_PDF = PLOTS_DIR / "performance_relative_coverage_rmse_width.pdf"
ALLOC_OMPPI_ONLY_PDF = PLOTS_DIR / "allocation_omppi_only_max_budget.pdf"
ALLOC_2X2_PDF = PLOTS_DIR / "allocation_query_cost_2x2_max_budget.pdf"
REF_TABLE_CSV = PLOTS_DIR / "auxiliary_oracle_refcost_table.csv"

ALLOCATION_BUDGET_MODE = "max"
LEGEND_FONTSIZE = 18

# ============================================================
# Column names
# ============================================================
TARGET_COL = "Y_full_plus"
PROMPT_COL = "prompt"

PREDICTION_COLS = [
    "f_plus_50",
    "f_plus_25",
    "f_plus_10",
    "f_original_tests",
    "f_static_ok",
]

PREDICTION_NAMES = [
    "Plus50",
    "Plus25",
    "Plus10",
    "OriginalTests",
    "StaticOK",
]

SOURCE_ORDER = ["Y", "Plus50", "Plus25", "Plus10", "OriginalTests", "StaticOK"]

COST_COLS = {
    "Y": "cost_full_plus",
    "Plus50": "cost_plus_50",
    "Plus25": "cost_plus_25",
    "Plus10": "cost_plus_10",
    "OriginalTests": "cost_original_tests",
    "StaticOK": "cost_static_ok",
}

# ============================================================
# Style: copied to match LLM preference plot template
# ============================================================
def configure_style_large():
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        try:
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
        except Exception:
            plt.rcParams["font.family"] = "DejaVu Sans"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"

    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": LEGEND_FONTSIZE,
        "figure.titlesize": 22,
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

def smooth_series(y, window=3):
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < window:
        return y.copy()
    return pd.Series(y).rolling(window=window, min_periods=1, center=True).mean().to_numpy()

def normalize_method_name(x):
    mapping = {
        "classical": "Classical",
        "Classical": "Classical",
        "ppi_vector__all_models": "VectorPPI++",
        "PPI++ vector": "VectorPPI++",
        "VectorPPI++": "VectorPPI++",
        "restrictedmultippi": "MultiPPI",
        "Restricted MultiPPI": "MultiPPI",
        "MultiPPI": "MultiPPI",
        "nf1_exh": "OMPPI(Exhaustive)",
        "NF-I exhaustive": "OMPPI(Exhaustive)",
        "OMPPI(Exhaustive)": "OMPPI(Exhaustive)",
        "nf1_dag": "OMPPI(DAG)",
        "NF-I DAG": "OMPPI(DAG)",
        "OMPPI(DAG)": "OMPPI(DAG)",
    }
    return mapping.get(str(x), str(x))

configure_style_large()

# ============================================================
# Method display settings: same template, VectorPPI++ removed
# ============================================================
METHOD_ORDER = [
    "Classical",
    "MultiPPI",
    "OMPPI(Exhaustive)",
    "OMPPI(DAG)",
]

METHOD_COLORS = {
    "Classical": "#7f7f7f",
    "MultiPPI": plt.get_cmap("Oranges")(0.78),
    "OMPPI(Exhaustive)": plt.get_cmap("Blues")(0.82),
    "OMPPI(DAG)": plt.get_cmap("Blues")(0.55),
}

METHOD_STYLES = {
    "Classical": dict(lw=2.5, ls=":", marker=None, alpha=0.95),
    "MultiPPI": dict(lw=2.8, ls="--", marker="s", ms=6.5, alpha=0.95),
    "OMPPI(Exhaustive)": dict(lw=3.0, ls="-", marker="o", ms=6.5, alpha=0.95),
    "OMPPI(DAG)": dict(lw=0.0, ls="None", marker="o", ms=7.0, alpha=0.95),
}

# ============================================================
# Load data
# ============================================================
summary = pd.read_csv(SUMMARY_PATH)
summary["method"] = summary["method"].map(normalize_method_name)
summary = summary[summary["method"].isin(METHOD_ORDER)].copy()
summary = summary.sort_values(["method", "budget"]).reset_index(drop=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

alloc = pd.read_csv(ALLOC_PATH) if ALLOC_PATH.exists() else pd.DataFrame()
if not alloc.empty:
    alloc["method"] = alloc["method"].map(normalize_method_name)

final_df = pd.read_csv(FINAL_TABLE_PATH)

print("Loaded summary rows:", summary.shape[0])
print("Loaded final table rows:", final_df.shape[0])
print("Methods:", summary["method"].unique().tolist())

# ============================================================
# Costs
# ============================================================
if "prediction_costs_used" in config:
    costs = {str(k): float(v) for k, v in config["prediction_costs_used"].items()}
else:
    y_cost_raw = float(final_df[COST_COLS["Y"]].mean())
    costs = {name: float(final_df[COST_COLS[name]].mean()) / y_cost_raw for name in PREDICTION_NAMES}

cost_map = {"Y": 1.0}
cost_map.update(costs)

print("Normalized costs:", cost_map)

# ============================================================
# Stratum labels
# ============================================================
def make_stratum_label(h):
    h_int = int(h)
    summary_obj = config.get("strata_summary", {})
    info = summary_obj.get(str(h_int), summary_obj.get(h_int, None))
    if isinstance(info, dict) and "min" in info and "max" in info:
        return f"S{h_int}: {int(info['min'])}-{int(info['max'])}"
    return f"S{h_int}"

# ============================================================
# Reconstruct prompt-length strata for ref diagnostic
# Same rank-qcut logic as main experiment
# ============================================================
def count_prompt_tokens(texts, encoding_name="o200k_base"):
    vals = ["" if x is None else str(x) for x in texts]
    try:
        import tiktoken
        try:
            enc = tiktoken.get_encoding(encoding_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return np.asarray([len(enc.encode(v)) for v in vals], dtype=int)
    except Exception:
        return np.asarray([len(v.strip().split()) if v.strip() else 0 for v in vals], dtype=int)

def make_quantile_strata(values, num_strata=5):
    values = np.asarray(values)
    if num_strata <= 1:
        return np.zeros(values.shape[0], dtype=int)
    ranks = pd.Series(values).rank(method="first")
    labels = pd.qcut(ranks, q=num_strata, labels=False, duplicates="drop")
    labels = np.asarray(labels, dtype=int)
    uniq = sorted(np.unique(labels).tolist())
    remap = {old: new for new, old in enumerate(uniq)}
    return np.asarray([remap[int(x)] for x in labels], dtype=int)

num_strata = int(config.get("num_strata", 5))
token_counts = count_prompt_tokens(final_df[PROMPT_COL].fillna("").astype(str).tolist())
final_df["stratum_ref"] = make_quantile_strata(token_counts, num_strata=num_strata)
strata_weights = final_df["stratum_ref"].value_counts(normalize=True).sort_index().to_dict()

# ============================================================
# Figure 0: correlation heatmap
# ============================================================
corr_cols = [TARGET_COL] + PREDICTION_COLS
corr_names = ["Y"] + PREDICTION_NAMES

corr_df = final_df[corr_cols].astype(float).copy()
corr_mat = corr_df.corr(method="pearson")
corr_mat.index = corr_names
corr_mat.columns = corr_names

print("\nCorrelation matrix:")
print(corr_mat.round(3))

fig, ax = plt.subplots(figsize=(9.4, 7.8))
mat = corr_mat.to_numpy(dtype=float)

im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
ax.set_title("Correlation among Full Label and Predictors")

ax.set_xticks(np.arange(len(corr_names)))
ax.set_yticks(np.arange(len(corr_names)))
ax.set_xticklabels(corr_names, rotation=45, ha="right")
ax.set_yticklabels(corr_names)

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(
            j, i, f"{mat[i, j]:.2f}",
            ha="center", va="center",
            fontsize=16,
            color="black",
        )

for spine in ax.spines.values():
    spine.set_visible(False)

ax.tick_params(axis="both", length=0)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Correlation")

fig.tight_layout()
fig.savefig(CORR_PDF, bbox_inches="tight")
plt.close(fig)
print("Saved:", CORR_PDF)

# ============================================================
# Auxiliary-oracle refline/refcost diagnostic
# Not drawn on figures
# ============================================================
def safe_var(x):
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return 0.0
    return float(np.var(x, ddof=1))

def safe_cov(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size <= 1:
        return 0.0
    return float(np.cov(x, y, ddof=1)[0, 1])

def tau2_from_scalar_predictor(y, f):
    vy = safe_var(y)
    vf = safe_var(f)
    if vy <= 1e-12 or vf <= 1e-12:
        return 0.0
    cov_yf = safe_cov(y, f)
    val = (cov_yf ** 2) / vf
    return float(max(0.0, min(val, vy)))

def monotone_nonincreasing(vals):
    out = []
    cur = None
    for v in vals:
        if cur is None:
            cur = float(v)
        else:
            cur = min(cur, float(v))
        out.append(cur)
    return out

def all_ordered_routes(names):
    routes = []
    k = len(names)
    for r in range(1, k + 1):
        for idxs in itertools.combinations(range(k), r):
            routes.append([names[i] for i in idxs])
    return routes

def route_q_nested(var_y, tau_map, route, cost_map):
    tau_raw = [var_y] + [tau_map[name] for name in route]
    tau_proj = monotone_nonincreasing(tau_raw)
    tau_proj = tau_proj + [0.0]

    deltas = []
    for j in range(len(tau_proj) - 1):
        deltas.append(max(0.0, tau_proj[j] - tau_proj[j + 1]))

    c_route = [cost_map[name] for name in route]
    d_costs = [1.0 - c_route[0]]
    for j in range(len(c_route) - 1):
        d_costs.append(c_route[j] - c_route[j + 1])
    d_costs.append(c_route[-1])

    q = float(sum(np.sqrt(max(0.0, d) * max(0.0, dc)) for d, dc in zip(deltas, d_costs)))
    top_cost = c_route[0]
    return q, top_cost

routes = all_ordered_routes(PREDICTION_NAMES)

ref_rows = []
for h in sorted(strata_weights.keys()):
    df_h = final_df[final_df["stratum_ref"] == h].copy()
    y = df_h[TARGET_COL].astype(float).to_numpy()
    var_y = safe_var(y)
    classical_q = np.sqrt(max(0.0, var_y))

    tau_map = {}
    for col, name in zip(PREDICTION_COLS, PREDICTION_NAMES):
        f = df_h[col].astype(float).to_numpy()
        tau_map[name] = tau2_from_scalar_predictor(y, f)

    best = None
    for route in routes:
        q, top_cost = route_q_nested(var_y, tau_map, route, costs)
        if best is None or q < best["q"]:
            best = {
                "route": route,
                "q": q,
                "top_cost": top_cost,
            }

    ref_rows.append({
        "stratum": int(h),
        "pi_h": float(strata_weights[h]),
        "var_y": float(var_y),
        "classical_q": float(classical_q),
        "best_q": float(best["q"]),
        "best_top_cost": float(best["top_cost"]),
        "best_route": " -> ".join(best["route"]),
    })

ref_df = pd.DataFrame(ref_rows)

r_aux = float((ref_df["pi_h"] * ref_df["best_q"]).sum() / (ref_df["pi_h"] * ref_df["classical_q"]).sum())
C_aux = float(len(final_df) * (ref_df["pi_h"] * ref_df["best_top_cost"]).sum())

label_budgets = np.asarray([3000, 6000, 9000, 10000, 15000], dtype=float)
ref_table = pd.DataFrame({
    "label_budget_B": label_budgets,
    "auxiliary_cost": C_aux,
})
ref_table["total_cost_B_plus_Caux"] = ref_table["label_budget_B"] + ref_table["auxiliary_cost"]
ref_table["raw_auxiliary_ref_ratio"] = r_aux
ref_table["same_total_cost_ref_ratio"] = r_aux * np.sqrt(
    ref_table["total_cost_B_plus_Caux"] / ref_table["label_budget_B"]
)

print("\n================ Auxiliary-oracle diagnostic ================")
print(ref_df[["stratum", "pi_h", "classical_q", "best_q", "best_top_cost", "best_route"]])
print(f"\nRaw auxiliary refline r_aux = {r_aux:.4f}")
print(f"Upfront auxiliary cost C_aux = {C_aux:.2f}")
print("\nSame-total-cost correction table:")
print(ref_table.to_string(index=False, formatters={
    "label_budget_B": "{:.0f}".format,
    "auxiliary_cost": "{:.0f}".format,
    "total_cost_B_plus_Caux": "{:.0f}".format,
    "raw_auxiliary_ref_ratio": "{:.3f}".format,
    "same_total_cost_ref_ratio": "{:.3f}".format,
}))
print("=============================================================\n")

ref_table.to_csv(REF_TABLE_CSV, index=False)
print("Saved:", REF_TABLE_CSV)

# ============================================================
# Shared legend handles
# ============================================================
method_handles = []
for method in METHOD_ORDER:
    style = METHOD_STYLES[method]
    handle = Line2D(
        [0], [0],
        color=METHOD_COLORS[method],
        lw=style.get("lw", 2.5),
        ls=style.get("ls", "-"),
        marker=style.get("marker", None),
        markersize=style.get("ms", 6.0),
        alpha=style.get("alpha", 0.95),
        label=method,
    )
    method_handles.append(handle)

# ============================================================
# Performance plot helper
# ============================================================
def add_method_curves(ax, df, metric, ylabel, smooth_window=3, y_lim=None, draw_nominal=False, draw_one=False):
    for method in METHOD_ORDER:
        g = df[df["method"] == method].sort_values("budget")
        if g.empty:
            continue

        x = g["budget"].to_numpy(dtype=float)
        y_raw = g[metric].to_numpy(dtype=float)
        y = np.array(smooth_series(y_raw, window=smooth_window), copy=True)

        ax.plot(
            x,
            y,
            color=METHOD_COLORS[method],
            label=method,
            zorder=3,
            **METHOD_STYLES[method],
        )

    ax.set_xlabel("Budget")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if y_lim is not None:
        ax.set_ylim(*y_lim)

    if draw_nominal:
        ax.axhline(0.95, color="black", lw=1.8, ls=":")

    if draw_one:
        ax.axhline(1.0, color="black", lw=1.8, ls=":")

# ============================================================
# Prepare relative metrics
# ============================================================
plot_df = summary.copy()

classical = (
    plot_df[plot_df["method"] == "Classical"][["budget", "rmse", "mean_ci_width"]]
    .rename(columns={"rmse": "rmse_classical", "mean_ci_width": "width_classical"})
)
plot_df = plot_df.merge(classical, on="budget", how="left")

plot_df["rmse_ratio_classical"] = plot_df["rmse"] / plot_df["rmse_classical"]
plot_df["width_ratio_classical"] = plot_df["mean_ci_width"] / plot_df["width_classical"]

# ============================================================
# PDF 1: absolute performance
# Same figure size and legend template as LLM preference code
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20.0, 5.4), sharex=True)

for ax in axes:
    ax.set_xlim(0, 3100)
    ax.set_xticks([0, 1000, 2000, 3000])

add_method_curves(
    axes[0],
    plot_df,
    metric="coverage",
    ylabel="Coverage",
    smooth_window=SMOOTH_WINDOW,
    y_lim=(0.70, 1.00),
    draw_nominal=True,
)

add_method_curves(
    axes[1],
    plot_df,
    metric="rmse",
    ylabel="RMSE",
    smooth_window=SMOOTH_WINDOW,
)

add_method_curves(
    axes[2],
    plot_df,
    metric="mean_ci_width",
    ylabel="CI width",
    smooth_window=SMOOTH_WINDOW,
)

fig.legend(
    method_handles,
    METHOD_ORDER,
    loc="lower center",
    ncol=4,
    frameon=False,
    columnspacing=1.1,
    handlelength=2.2,
    handletextpad=0.5,
    bbox_to_anchor=(0.5, 0.05),
)

fig.tight_layout(rect=[0.0, 0.13, 1.0, 1.0])
fig.savefig(ABS_PDF, bbox_inches="tight")
plt.close(fig)
print("Saved:", ABS_PDF)

# ============================================================
# PDF 2: relative performance
# Coverage absolute; RMSE and width are ratios to Classical
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20.0, 5.4), sharex=True)

add_method_curves(
    axes[0],
    plot_df,
    metric="coverage",
    ylabel="Coverage",
    smooth_window=SMOOTH_WINDOW,
    y_lim=(0.70, 1.00),
    draw_nominal=True,
)

add_method_curves(
    axes[1],
    plot_df,
    metric="rmse_ratio_classical",
    ylabel="RMSE / Classical",
    smooth_window=SMOOTH_WINDOW,
    y_lim=(0.80, 1.02),
    draw_one=True,
)

add_method_curves(
    axes[2],
    plot_df,
    metric="width_ratio_classical",
    ylabel="CI width / Classical",
    smooth_window=SMOOTH_WINDOW,
    y_lim=(0.80, 1.02),
    draw_one=True,
)

axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
axes[1].set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
axes[2].set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])

for ax in axes:
    ax.set_xlim(0, 3100)
    ax.set_xticks([0, 1000, 2000, 3000])

fig.legend(
    method_handles,
    METHOD_ORDER,
    loc="lower center",
    ncol=4,
    frameon=False,
    columnspacing=1.1,
    handlelength=2.2,
    handletextpad=0.5,
    bbox_to_anchor=(0.5, 0.05),
)

fig.tight_layout(rect=[0.0, 0.13, 1.0, 1.0])
fig.savefig(REL_PDF, bbox_inches="tight")
plt.close(fig)
print("Saved:", REL_PDF)

# ============================================================
# Allocation helpers
# Use only the largest budget, matching ALLOCATION_BUDGET_MODE="max"
# ============================================================
if alloc.empty:
    print("[warning] allocation_summary.csv not found. Skip allocation plots.")
else:
    alloc = alloc[alloc["budget"] > 0].copy()
    max_budget = float(alloc["budget"].max())
    alloc_use = alloc[np.isclose(alloc["budget"], max_budget)].copy()
    print(f"Using allocation at max budget = {max_budget}")

    def display_source_name(source):
        s = str(source)
        if s == "full":
            return "Y+Joint all"
        if s == "joint_all":
            return "Joint all"
        if s.startswith("single__"):
            return s.split("single__", 1)[1]
        return s

    alloc_use["source_display"] = alloc_use["source"].map(display_source_name)
    alloc_use["stratum_label"] = alloc_use["stratum"].map(make_stratum_label)

    omppi_order = ["Y", "Plus50", "Plus25", "Plus10", "OriginalTests", "StaticOK"]
    multippi_order = ["Y+Joint all", "Joint all", "Plus50", "Plus25", "Plus10", "OriginalTests", "StaticOK"]

    def build_raw_query_matrix(method_name, source_order):
        dfm = alloc_use[alloc_use["method"] == method_name].copy()
        if dfm.empty:
            raise ValueError(f"No allocation rows for method={method_name}")

        mat = (
            dfm.pivot_table(
                index="stratum_label",
                columns="source_display",
                values="query_mean",
                aggfunc="sum",
                fill_value=np.nan,
            )
        )

        all_strata = sorted(alloc_use["stratum"].unique().tolist())
        row_labels = [make_stratum_label(h) for h in all_strata]
        mat = mat.reindex(index=row_labels, columns=source_order)

        return mat

    q_omppi_raw = build_raw_query_matrix("OMPPI(DAG)", omppi_order)
    q_multi_raw = build_raw_query_matrix("MultiPPI", multippi_order)

    def fill_omppi_cumulative_counts(q_raw):
        out = q_raw.copy()

        if "Y" in out.columns:
            out["Y"] = out["Y"].fillna(0.0)
        else:
            out["Y"] = 0.0

        prev = out["Y"].copy()
        for src in ["Plus50", "Plus25", "Plus10", "OriginalTests", "StaticOK"]:
            if src not in out.columns:
                out[src] = np.nan
            out[src] = out[src].where(out[src].notna(), prev)
            out[src] = np.maximum(out[src].astype(float), prev.astype(float))
            prev = out[src].copy()

        return out[omppi_order]

    q_omppi_cum = fill_omppi_cumulative_counts(q_omppi_raw)

    def row_percent(mat):
        mat2 = mat.fillna(0.0).astype(float)
        denom = mat2.sum(axis=1).replace(0, np.nan)
        return 100.0 * mat2.div(denom, axis=0).fillna(0.0)

    q_omppi_pct = row_percent(q_omppi_cum)
    q_multi_pct = row_percent(q_multi_raw)

    def omppi_nested_cost_matrix(q_cum):
        c = cost_map
        dcost = {
            "Y": c["Y"] - c["Plus50"],
            "Plus50": c["Plus50"] - c["Plus25"],
            "Plus25": c["Plus25"] - c["Plus10"],
            "Plus10": c["Plus10"] - c["OriginalTests"],
            "OriginalTests": c["OriginalTests"] - c["StaticOK"],
            "StaticOK": c["StaticOK"],
        }
        out = q_cum.copy().astype(float)
        for src in omppi_order:
            out[src] = out[src] * dcost[src]
        return out

    def multippi_independent_cost_matrix(q_raw):
        out = q_raw.fillna(0.0).astype(float).copy()
        source_cost = {
            "Y+Joint all": 1.0,
            "Joint all": sum(costs.values()),
            "Plus50": costs["Plus50"],
            "Plus25": costs["Plus25"],
            "Plus10": costs["Plus10"],
            "OriginalTests": costs["OriginalTests"],
            "StaticOK": costs["StaticOK"],
        }
        for src in out.columns:
            out[src] = out[src] * source_cost.get(src, 0.0)
        return out

    c_omppi_pct = row_percent(omppi_nested_cost_matrix(q_omppi_cum))
    c_multi_pct = row_percent(multippi_independent_cost_matrix(q_multi_raw))

    def draw_heatmap(
        ax,
        values,
        row_labels,
        col_labels,
        title,
        cmap,
        show_y=True,
        show_xlabels=True,
        xlabel=None,
        ylabel=None,
        annot_size=15,
    ):
        mat = values.to_numpy(dtype=float)
        vmax = max(1e-8, float(np.nanmax(mat)))

        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(title)

        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels if show_y else [""] * len(row_labels))

        ax.set_xticks(np.arange(len(col_labels)))
        if show_xlabels:
            ax.set_xticklabels(col_labels, rotation=45, ha="right")
        else:
            ax.set_xticklabels([])

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(
                    j, i, f"{mat[i, j]:.0f}",
                    ha="center", va="center",
                    fontsize=annot_size,
                    color="black",
                )

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Percent")

    row_labels_omppi = list(q_omppi_pct.index)
    row_labels_multi = list(q_multi_pct.index)

    # ========================================================
    # PDF 3a: OMPPI only allocation
    # ========================================================
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 4.5),
        gridspec_kw={"wspace": 0.26},
    )

    draw_heatmap(
        axes[0],
        q_omppi_pct,
        row_labels_omppi,
        list(q_omppi_pct.columns),
        title="OMPPI(DAG): sample allocation",
        cmap="Blues",
        show_y=True,
        show_xlabels=True,
        xlabel="Source",
        ylabel="Prompt-length stratum",
        annot_size=15,
    )

    draw_heatmap(
        axes[1],
        c_omppi_pct,
        row_labels_omppi,
        list(c_omppi_pct.columns),
        title="OMPPI(DAG): cost allocation",
        cmap="Blues",
        show_y=False,
        show_xlabels=True,
        xlabel="Source",
        ylabel=None,
        annot_size=15,
    )

    fig.tight_layout()
    fig.savefig(ALLOC_OMPPI_ONLY_PDF, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", ALLOC_OMPPI_ONLY_PDF)

    # ========================================================
    # PDF 3b: OMPPI + MultiPPI allocation
    # ========================================================
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        gridspec_kw={"wspace": 0.26, "hspace": 0.18},
    )

    draw_heatmap(
        axes[0, 0],
        q_omppi_pct,
        row_labels_omppi,
        list(q_omppi_pct.columns),
        title="OMPPI(DAG): sample allocation",
        cmap="Blues",
        show_y=True,
        show_xlabels=False,
        xlabel=None,
        ylabel="Prompt-length stratum",
        annot_size=15,
    )

    draw_heatmap(
        axes[0, 1],
        q_multi_pct,
        row_labels_multi,
        list(q_multi_pct.columns),
        title="MultiPPI: sample allocation",
        cmap="Oranges",
        show_y=False,
        show_xlabels=False,
        xlabel=None,
        ylabel=None,
        annot_size=15,
    )

    draw_heatmap(
        axes[1, 0],
        c_omppi_pct,
        row_labels_omppi,
        list(c_omppi_pct.columns),
        title="OMPPI(DAG): cost allocation",
        cmap="Blues",
        show_y=True,
        show_xlabels=True,
        xlabel="Source",
        ylabel="Prompt-length stratum",
        annot_size=15,
    )

    draw_heatmap(
        axes[1, 1],
        c_multi_pct,
        row_labels_multi,
        list(c_multi_pct.columns),
        title="MultiPPI: cost allocation",
        cmap="Oranges",
        show_y=False,
        show_xlabels=True,
        xlabel="Queried block",
        ylabel=None,
        annot_size=15,
    )

    fig.tight_layout()
    fig.savefig(ALLOC_2X2_PDF, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", ALLOC_2X2_PDF)

print("\nDone.")
print("Plots saved to:", PLOTS_DIR)
