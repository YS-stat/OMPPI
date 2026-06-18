#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

print("SCRIPT VERSION: cost-annotated reference line (no pkl reload; existing JSON + reference cost)")

# ============================================================
# Paths
# Run this script from:
#   OMPPI/human_preference
# ============================================================
OUT_DIR = Path("./outputs/main")
SUMMARY_PATH = OUT_DIR / "summary.csv"
CONFIG_PATH = OUT_DIR / "config.json"
ALLOC_PATH = OUT_DIR / "allocation_summary.csv"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = Path("fonts/Helvetica.ttf")
SMOOTH_WINDOW = 3

PERF_PDF = PLOTS_DIR / "performance_coverage_rmse_width.pdf"
ALLOC_PDF = PLOTS_DIR / "allocation_query_cost_2x2.pdf"

# Allocation aggregation:
#   "all_positive" = aggregate all positive budgets, closest to the previous full-detail trend plot
#   "max" = only use the largest budget
ALLOCATION_BUDGET_MODE = "all_positive"

GPT4_NAME = "gpt-4-1106-preview"
CLAUDE_NAME = "claude-2.1"
LEGEND_FONTSIZE = 18
REF_COLOR = "0.30"
REF_BAND_COLOR = "0.92"

# Control distance between text and dashed reference line
REF_TEXT_OFFSET_FRAC = 0.025
REF_TEXT_OFFSET_MIN = 0.004

# Control extra space below Ref. cost text
REF_BOTTOM_PAD_FRAC = 0.035
REF_BOTTOM_PAD_MIN = 0.010
CODE_VERSION = "existing_json_reference_cost"
REFERENCE_JSON_PATH = OUT_DIR / "stratified_joint_reference_all_predictions.json"
REFERENCE_JSON_WITH_COST_PATH = OUT_DIR / "stratified_joint_reference_all_predictions_with_cost.json"
# Fallback only for the current Chatbot Arena non-tie experiment, used when the
# existing reference JSON is unavailable. It avoids re-reading the original pkl.
FALLBACK_REFERENCE_VALUES = {
    "r2_stratified_joint": 0.2053,
    "reference_ratio": 0.8915,
    "attainable_gain": 0.1085,
    "n_reference_rows": 823,
    "status": "fallback_no_json",
}


# Normalized costs from the LLM preference experiment table.
# These are used as a fallback if config.json does not contain normalized costs.
DEFAULT_NORMALIZED_COSTS = {
    "gemini-2.5-pro": 1.0000,
    "gemini-3.1-flash-preview": 0.3849,
    "gemini-3.1-flash": 0.3849,
    "gemini-3.1-flash-lite-preview": 0.2041,
    "gemini-3.1-flash-lite": 0.2041,
    "gemini-2.5-flash": 0.2366,
    "gemini-2.5-flash-lite": 0.0772,
    "qwen3-next-80b-instruct": 0.1135,
    "qwen-next": 0.1135,
    "qwen3-235b-a22b-instruct": 0.1611,
    "qwen3-235b-a22b-instruct-2507": 0.1611,
    "qwen-235b": 0.1611,
    "gpt-oss-120b": 0.1500,
    "gpt-oss-120b-maas": 0.1500,
    "oss-120b": 0.1500,
    "qwen3-coder-480b-a35b-instruct": 0.1686,
    "qwen-coder": 0.1686,
    "gpt-oss-20b": 0.1156,
    "gpt-oss-20b-maas": 0.1156,
    "oss-20b": 0.1156,
}

# ============================================================
# Style
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
# Method display settings
# ============================================================
METHOD_ORDER = [
    "Classical",
    "VectorPPI++",
    "MultiPPI",
    "OMPPI(Exhaustive)",
    "OMPPI(DAG)",
]

METHOD_COLORS = {
    "Classical": "#7f7f7f",
    "VectorPPI++": "#d62728",
    "MultiPPI": plt.get_cmap("Oranges")(0.78),
    "OMPPI(Exhaustive)": plt.get_cmap("Blues")(0.82),
    "OMPPI(DAG)": plt.get_cmap("Blues")(0.55),
}

METHOD_STYLES = {
    "Classical": dict(lw=2.5, ls=":", marker=None, alpha=0.95),
    "VectorPPI++": dict(lw=2.6, ls="-.", marker="^", ms=6.5, alpha=0.95),
    "MultiPPI": dict(lw=2.8, ls="--", marker="s", ms=6.5, alpha=0.95),
    "OMPPI(Exhaustive)": dict(lw=3.0, ls="-", marker="o", ms=6.5, alpha=0.95),
    "OMPPI(DAG)": dict(lw=0.0, ls="None", marker="o", ms=7.0, alpha=0.95),
}

# ============================================================
# Display names for allocation heatmaps
# ============================================================
DISPLAY_NAME = {
    "gemini-2.5-flash": "G2.5-Flash",
    "gemini-2.5-flash-lite": "G2.5-Lite",
    "gemini-2.5-pro": "G2.5-Pro",
    "gemini-3.1-flash-lite-preview": "G3.1-Lite",
    "gemini-3.1-flash-preview": "G3.1-Flash",
    "gpt-oss-120b": "OSS-120B",
    "gpt-oss-120b-maas": "OSS-120B",
    "gpt-oss-20b": "OSS-20B",
    "gpt-oss-20b-maas": "OSS-20B",
    "qwen3-235b-a22b-instruct": "Qwen-235B",
    "qwen3-235b-a22b-instruct-2507": "Qwen-235B",
    "qwen3-coder-480b-a35b-instruct": "Qwen-Coder",
    "qwen3-next-80b-instruct": "Qwen-Next",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
}

# ============================================================
# Pickle compatibility fallback
# ============================================================
def _read_pickle_with_stringdtype_compat(path: Path) -> pd.DataFrame:
    """
    Fallback for pandas pickle files created with a newer StringDtype.
    Handles errors such as:
        TypeError: issubclass() arg 1 must be a class
    """
    try:
        from pandas import StringDtype
    except Exception:
        raise

    old_init = getattr(StringDtype, "__init__", None)
    if old_init is None:
        return pd.read_pickle(path)

    def _patched_init(self, *args, **kwargs):
        storage = kwargs.get("storage", None)
        if len(args) >= 1:
            storage = args[0]

        try:
            old_init(self, storage=storage)
        except TypeError:
            try:
                old_init(self, storage)
            except TypeError:
                old_init(self)

        na_value = kwargs.get("na_value", None)
        if len(args) >= 2:
            na_value = args[1]

        if na_value is not None:
            for attr in ("_na_value", "na_value"):
                try:
                    object.__setattr__(self, attr, na_value)
                    break
                except Exception:
                    pass

    StringDtype.__init__ = _patched_init
    try:
        return pd.read_pickle(path)
    finally:
        StringDtype.__init__ = old_init

def load_final_table(path):
    path = Path(path)

    candidates = [
        path,
        Path(".") / path,
        OUT_DIR / path,
        OUT_DIR.parent / path,
        Path("./data/final_aligned_with_vertex_online.pkl"),
    ]

    for p in candidates:
        if p.exists():
            if p.suffix.lower() in {".pkl", ".pickle"}:
                try:
                    return pd.read_pickle(p)
                except TypeError as e:
                    msg = str(e)
                    if (
                        "StringDtype.__init__" in msg
                        or "issubclass() arg 1 must be a class" in msg
                        or "StringDtype" in msg
                    ):
                        print("[load_final_table] pandas pickle compatibility fallback triggered.")
                        return _read_pickle_with_stringdtype_compat(p)
                    raise
            if p.suffix.lower() == ".csv":
                return pd.read_csv(p)
            raise ValueError(f"Unsupported final table type: {p}")

    raise FileNotFoundError(f"Final table not found from config path: {path}")

# ============================================================
# Helpers for stratified reference line
# ============================================================
def find_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns exist: {candidates}")

def detect_prompt_column(df, prompt_col=None):
    if prompt_col is not None and prompt_col in df.columns:
        return prompt_col
    candidates = [
        "prompt", "question", "original_prompt", "user_prompt", "instruction",
        "full_prompt", "raw_prompt", "query", "text", "input",
    ]
    return find_first_existing_col(df, candidates)

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
        return np.asarray([max(1, len(v.strip().split())) if v.strip() else 0 for v in vals], dtype=int)

def make_quantile_strata(token_counts, num_strata=5):
    token_counts = np.asarray(token_counts, dtype=int)
    if num_strata <= 1:
        return np.zeros(token_counts.shape[0], dtype=int)
    ranks = pd.Series(token_counts).rank(method="first")
    labels = pd.qcut(ranks, q=num_strata, labels=False, duplicates="drop")
    labels = np.asarray(labels, dtype=int)
    uniq = sorted(np.unique(labels).tolist())
    remap = {old: new for new, old in enumerate(uniq)}
    return np.asarray([remap[int(x)] for x in labels], dtype=int)

def pop_var(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.mean((x - np.mean(x)) ** 2))

def compute_joint_tau2(y, Z, ridge=1e-10):
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if y.size <= 2 or Z.ndim != 2 or Z.shape[0] <= 2 or Z.shape[1] == 0:
        return 0.0, pop_var(y)

    var_y = pop_var(y)
    if var_y <= 1e-15:
        return 0.0, var_y

    yc = y - np.mean(y)
    Zc = Z - np.mean(Z, axis=0, keepdims=True)

    cov_yf = np.mean(yc[:, None] * Zc, axis=0)
    cov_ff = (Zc.T @ Zc) / max(Zc.shape[0], 1)
    cov_ff = 0.5 * (cov_ff + cov_ff.T)
    cov_ff = cov_ff + ridge * np.eye(cov_ff.shape[0])

    tau2 = float(cov_yf @ np.linalg.pinv(cov_ff) @ cov_yf)
    tau2 = min(max(tau2, 0.0), max(var_y, 0.0))
    return tau2, var_y


# ============================================================
# Cost helpers for reference-line annotation
# ============================================================
def _as_float_dict(obj):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            try:
                val = float(v)
                if np.isfinite(val):
                    out[str(k)] = val
            except Exception:
                pass
    return out


def lookup_cost(model, cost_map):
    keys = [
        str(model),
        str(model).lower(),
        str(model).replace("-maas", ""),
        str(model).replace("-2507", ""),
        str(model).lower().replace("-maas", ""),
        str(model).lower().replace("-2507", ""),
    ]
    for k in keys:
        if k in cost_map:
            return float(cost_map[k])
    return np.nan


def get_model_cost_map(config, usable_models):
    """Return normalized costs in the same budget units as the plot."""
    candidate_keys = [
        "costs_used",
        "model_costs_used",
        "costs",
        "model_costs",
        "costs_raw",
        "times",
    ]

    cost_map = {}
    for key in candidate_keys:
        cost_map.update(_as_float_dict(config.get(key, {})))

    normalized_map = {}
    for k, v in cost_map.items():
        normalized_map[k] = v
        normalized_map[k.lower()] = v
        normalized_map[k.replace("-maas", "")] = v
        normalized_map[k.replace("-2507", "")] = v
        normalized_map[k.lower().replace("-maas", "")] = v
        normalized_map[k.lower().replace("-2507", "")] = v

    # Fill any missing normalized costs using the fixed table from the experiment.
    for k, v in DEFAULT_NORMALIZED_COSTS.items():
        normalized_map.setdefault(k, float(v))
        normalized_map.setdefault(k.lower(), float(v))

    final_costs = {}
    missing = []
    for model in usable_models:
        c = lookup_cost(model, normalized_map)
        if np.isfinite(c):
            final_costs[model] = float(c)
        else:
            missing.append(model)

    if missing:
        print("[warning] Missing normalized costs for models:", missing)

    return final_costs

def build_llm_stratified_reference_matrix(config):
    """
    Reconstruct the same scalar mean population used in the stratified experiment.

    Y is the human label after dropping ties, aligned to GPT-4 win vs Claude.
    Predictors are all available judge vote fractions aligned to GPT-4.
    Strata are prompt-token quantile strata, matching the base plotting experiment.
    """
    final_table = config.get("final_table", None)
    if final_table is None:
        raise ValueError("config.json does not contain final_table.")

    df0 = load_final_table(final_table).copy()

    model_names = config.get("models", None)
    if not isinstance(model_names, list) or len(model_names) == 0:
        suffix_a = "__judge_num_A"
        suffix_b = "__judge_num_B"
        cand_a = {c[:-len(suffix_a)] for c in df0.columns if c.endswith(suffix_a)}
        cand_b = {c[:-len(suffix_b)] for c in df0.columns if c.endswith(suffix_b)}
        model_names = sorted(cand_a & cand_b)

    model_a_col = find_first_existing_col(df0, ["model_a", "model_a_name", "left_model", "response_a_model"])
    model_b_col = find_first_existing_col(df0, ["model_b", "model_b_name", "right_model", "response_b_model"])

    a_name = df0[model_a_col].astype(str)
    b_name = df0[model_b_col].astype(str)

    gpt4_name = config.get("gpt4_name", GPT4_NAME)
    claude_name = config.get("claude_name", CLAUDE_NAME)

    mask_pair = (
        ((a_name == gpt4_name) & (b_name == claude_name)) |
        ((a_name == claude_name) & (b_name == gpt4_name))
    )

    df = df0.loc[mask_pair].copy().reset_index(drop=True)

    prompt_col = config.get("prompt_col", None)
    prompt_col = detect_prompt_column(df, prompt_col)
    token_counts = count_prompt_tokens(df[prompt_col].astype(str).fillna("").tolist())

    num_strata = int(config.get("num_strata", 5))
    strata_labels_full = make_quantile_strata(token_counts, num_strata=num_strata)

    df["stratum"] = strata_labels_full
    df["gpt4_is_A"] = df[model_a_col].astype(str) == gpt4_name
    df["gpt4_is_B"] = df[model_b_col].astype(str) == gpt4_name

    if not (df["gpt4_is_A"] ^ df["gpt4_is_B"]).all():
        raise ValueError("Each retained row must place GPT-4 on exactly one side.")

    required = {"winner_model_a", "winner_model_b", "winner_tie"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing human winner columns: {required - set(df.columns)}")

    label_mode = str(config.get("label_mode", "drop_ties")).lower()

    def row_to_y(row):
        if float(row.get("winner_tie", 0)) == 1:
            return 0.5 if label_mode == "half_ties" else np.nan
        if float(row.get("winner_model_a", 0)) == 1:
            return 1.0 if row["gpt4_is_A"] else 0.0
        if float(row.get("winner_model_b", 0)) == 1:
            return 1.0 if row["gpt4_is_B"] else 0.0
        return np.nan

    df["Y"] = df.apply(row_to_y, axis=1)
    df = df.loc[df["Y"].notna()].copy().reset_index(drop=True)

    gpt4_is_A = df["gpt4_is_A"].to_numpy(dtype=bool)

    usable_models = []
    score_cols = []

    for model in model_names:
        a_col = f"{model}__judge_num_A"
        b_col = f"{model}__judge_num_B"
        if a_col not in df.columns or b_col not in df.columns:
            continue

        a_votes = pd.to_numeric(df[a_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        b_votes = pd.to_numeric(df[b_col], errors="coerce").fillna(0).to_numpy(dtype=float)

        gpt4_votes = np.where(gpt4_is_A, a_votes, b_votes)
        claude_votes = np.where(gpt4_is_A, b_votes, a_votes)

        total = gpt4_votes + claude_votes
        score = np.full_like(total, 0.5, dtype=float)
        mask = total > 0
        score[mask] = gpt4_votes[mask] / total[mask]

        usable_models.append(model)
        score_cols.append(score)

    if len(score_cols) == 0:
        raise ValueError("No usable judge score columns found for reference line.")

    out = pd.DataFrame({
        "Y": df["Y"].astype(float).to_numpy(),
        "stratum": df["stratum"].astype(int).to_numpy(),
    })

    for j, model in enumerate(usable_models):
        out[f"score__{model}"] = score_cols[j]

    return out, usable_models

def compute_stratified_joint_reference_all_predictions(config):
    """
    Stratified joint reference using all LLM predictions jointly.

    For scalar mean, within stratum h:
      oracle residual variance = Var_h(Y) - tau_joint,h^2

    With n_h approximately proportional to pi_h, the reference line on
    CI width / Classical is:
      sqrt( sum_h pi_h [Var_h(Y) - tau_joint,h^2] / sum_h pi_h Var_h(Y) ).

    We also report the full auxiliary cost needed to observe all usable judge
    predictions for all non-tie examples in this empirical population.
    """
    try:
        df_ref, usable_models = build_llm_stratified_reference_matrix(config)
    except Exception as exc:
        return {
            "r2_stratified_joint": np.nan,
            "reference_ratio": np.nan,
            "attainable_gain": np.nan,
            "reference_aux_cost": np.nan,
            "reference_oracle_cost": np.nan,
            "n_reference_rows": 0,
            "sum_model_cost": np.nan,
            "per_stratum": {},
            "usable_models": [],
            "status": f"failed: {exc}",
        }

    predictor_cols = [c for c in df_ref.columns if c.startswith("score__")]

    config_pi = config.get("strata_pi", {})
    pi_map = {}
    if isinstance(config_pi, dict):
        for k, v in config_pi.items():
            try:
                pi_map[int(k)] = float(v)
            except Exception:
                pass

    weighted_tau = 0.0
    weighted_var = 0.0
    per_stratum = {}

    for h, g in df_ref.groupby("stratum", sort=True):
        h_int = int(h)
        pi_h = float(pi_map.get(h_int, len(g) / len(df_ref)))

        y_h = g["Y"].astype(float).to_numpy()
        Z_h = g[predictor_cols].astype(float).to_numpy()

        tau2_h, var_h = compute_joint_tau2(y_h, Z_h)

        weighted_tau += pi_h * tau2_h
        weighted_var += pi_h * var_h

        per_stratum[str(h_int)] = {
            "n": int(len(g)),
            "pi": pi_h,
            "var_y": float(var_h),
            "tau2_joint": float(tau2_h),
            "r2_joint": float(tau2_h / var_h) if var_h > 1e-15 else None,
        }

    if weighted_var <= 1e-15:
        r2 = np.nan
        ratio = np.nan
        gain = np.nan
    else:
        r2 = float(weighted_tau / weighted_var)
        r2 = min(max(r2, 0.0), 0.999999)
        ratio = float(np.sqrt(1.0 - r2))
        gain = float(1.0 - ratio)

    model_costs = get_model_cost_map(config, usable_models)
    sum_model_cost = float(sum(model_costs.get(m, 0.0) for m in usable_models))
    n_rows = int(len(df_ref))
    reference_aux_cost = float(n_rows * sum_model_cost)
    reference_oracle_cost = float(n_rows * (1.0 + sum_model_cost))

    result = {
        "r2_stratified_joint": r2,
        "reference_ratio": ratio,
        "attainable_gain": gain,
        "reference_aux_cost": reference_aux_cost,
        "reference_oracle_cost": reference_oracle_cost,
        "n_reference_rows": n_rows,
        "sum_model_cost": sum_model_cost,
        "model_costs": model_costs,
        "per_stratum": per_stratum,
        "usable_models": usable_models,
        "status": "ok",
    }

    ref_path = OUT_DIR / "stratified_joint_reference_all_predictions.json"
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ============================================================
# Reference-line loader that avoids re-reading the original pkl
# ============================================================
def load_reference_without_pkl(config):
    """
    Load the already computed stratified joint reference if available.
    This deliberately avoids re-reading final_aligned_with_vertex_online.pkl,
    which can trigger pandas StringDtype pickle compatibility errors.
    """
    if REFERENCE_JSON_PATH.exists():
        with open(REFERENCE_JSON_PATH, "r", encoding="utf-8") as f:
            reference = json.load(f)
        reference["status"] = reference.get("status", "ok")
        print(f"Loaded existing reference JSON: {REFERENCE_JSON_PATH}")
        return reference

    reference = dict(FALLBACK_REFERENCE_VALUES)
    if "models" in config and isinstance(config["models"], list):
        reference["usable_models"] = list(config["models"])
    else:
        reference["usable_models"] = []
    print(
        "[warning] Existing reference JSON not found. "
        "Using hard-coded fallback values for the current Chatbot Arena run."
    )
    return reference


def _infer_n_reference_rows(reference):
    n_rows = reference.get("n_reference_rows", None)
    try:
        if n_rows is not None and np.isfinite(float(n_rows)) and float(n_rows) > 0:
            return int(round(float(n_rows)))
    except Exception:
        pass

    per_stratum = reference.get("per_stratum", {})
    if isinstance(per_stratum, dict) and per_stratum:
        total = 0
        for v in per_stratum.values():
            if isinstance(v, dict) and "n" in v:
                try:
                    total += int(v["n"])
                except Exception:
                    pass
        if total > 0:
            return total

    return None


def attach_reference_costs_without_pkl(reference, config, summary_df):
    """
    Attach reference auxiliary/oracle costs to a reference dict without reading pkl.

    Auxiliary reference cost is the cost of querying all usable judge predictions
    on all reference rows: n_reference_rows * sum_k c_k.

    Oracle-info cost additionally counts a unit true-label cost for all reference rows:
    n_reference_rows * (1 + sum_k c_k). This is only a diagnostic cost, because the
    reference line itself is an oracle diagnostic rather than a finite-budget estimator.
    """
    usable_models = reference.get("usable_models", [])
    if not usable_models and isinstance(config.get("models", None), list):
        usable_models = list(config["models"])

    n_rows = _infer_n_reference_rows(reference)
    if n_rows is None:
        n_rows = int(FALLBACK_REFERENCE_VALUES["n_reference_rows"])
        print(f"[warning] Could not infer n_reference_rows; using fallback n={n_rows}.")

    model_costs = get_model_cost_map(config, usable_models)
    sum_model_cost = float(sum(model_costs.get(m, 0.0) for m in usable_models))

    # If usable_models are absent or names did not match, fall back to the fixed table.
    if sum_model_cost <= 0:
        sum_model_cost = float(sum(DEFAULT_NORMALIZED_COSTS[k] for k in [
            "gemini-2.5-pro",
            "gemini-3.1-flash-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "qwen3-next-80b-instruct",
            "qwen3-235b-a22b-instruct",
            "gpt-oss-120b",
            "qwen3-coder-480b-a35b-instruct",
            "gpt-oss-20b",
        ]))
        print("[warning] Model names did not match cost map; used fixed ten-judge cost sum.")

    max_plot_budget = float(np.nanmax(summary_df["budget"].to_numpy(dtype=float)))

    reference_aux_cost = float(n_rows * sum_model_cost)
    reference_oracle_cost = float(n_rows * (1.0 + sum_model_cost))
    aux_cost_over_max_budget = (
        float(reference_aux_cost / max_plot_budget)
        if np.isfinite(reference_aux_cost) and max_plot_budget > 0
        else np.nan
    )

    reference["n_reference_rows"] = int(n_rows)
    reference["usable_models"] = usable_models
    reference["model_costs"] = model_costs
    reference["sum_model_cost"] = float(sum_model_cost)
    reference["reference_aux_cost"] = reference_aux_cost
    reference["reference_oracle_cost"] = reference_oracle_cost
    reference["max_plotted_budget"] = max_plot_budget
    reference["aux_cost_over_max_budget"] = aux_cost_over_max_budget

    try:
        with open(REFERENCE_JSON_WITH_COST_PATH, "w", encoding="utf-8") as f:
            json.dump(reference, f, indent=2, ensure_ascii=False)
        print(f"Saved reference-with-cost JSON to: {REFERENCE_JSON_WITH_COST_PATH}")
    except Exception as exc:
        print(f"[warning] Failed to save reference-with-cost JSON: {exc}")

    return reference

# ============================================================
# Figure 1: coverage, RMSE/Classical, CI width/Classical
# ============================================================
def plot_performance_with_reference():
    summary = pd.read_csv(SUMMARY_PATH)
    summary["method"] = summary["method"].map(normalize_method_name)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load reference line without re-reading the original pkl.
    reference = load_reference_without_pkl(config)
    reference_ratio = reference.get("reference_ratio", np.nan)
    reference_r2 = reference.get("r2_stratified_joint", np.nan)
    reference_gain = reference.get("attainable_gain", np.nan)

    plot_df = summary[summary["method"].isin(METHOD_ORDER)].copy()

    classical_df = plot_df[plot_df["method"] == "Classical"].copy()
    if classical_df.empty:
        raise ValueError("Classical method is missing from summary.csv.")

    if (classical_df["budget"] == 0).any():
        anchor = classical_df.loc[classical_df["budget"] == 0].iloc[0].copy()
    else:
        anchor = classical_df.loc[classical_df["budget"].idxmin()].copy()

    zero_rows = []
    for method in METHOD_ORDER:
        if not ((plot_df["method"] == method) & (plot_df["budget"] == 0)).any():
            row = anchor.copy()
            row["method"] = method
            row["budget"] = 0.0
            zero_rows.append(row)
    if zero_rows:
        plot_df = pd.concat([plot_df, pd.DataFrame(zero_rows)], ignore_index=True)

    anchor_values = {
        "coverage": float(anchor["coverage"]),
        "rmse": float(anchor["rmse"]),
        "mean_ci_width": float(anchor["mean_ci_width"]),
    }

    for method in METHOD_ORDER:
        mask0 = (plot_df["method"] == method) & (plot_df["budget"] == 0)
        plot_df.loc[mask0, "coverage"] = anchor_values["coverage"]
        plot_df.loc[mask0, "rmse"] = anchor_values["rmse"]
        plot_df.loc[mask0, "mean_ci_width"] = anchor_values["mean_ci_width"]

    classical_by_budget = plot_df[plot_df["method"] == "Classical"][
        ["budget", "rmse", "mean_ci_width"]
    ].rename(columns={
        "rmse": "rmse_classical",
        "mean_ci_width": "width_classical",
    })

    plot_df = plot_df.merge(classical_by_budget, on="budget", how="left")
    plot_df["rmse_classical"] = plot_df["rmse_classical"].fillna(anchor_values["rmse"])
    plot_df["width_classical"] = plot_df["width_classical"].fillna(anchor_values["mean_ci_width"])

    plot_df["rmse_ratio"] = plot_df["rmse"] / plot_df["rmse_classical"]
    plot_df["width_ratio"] = plot_df["mean_ci_width"] / plot_df["width_classical"]

    plot_df.loc[plot_df["budget"] == 0, "rmse_ratio"] = 1.0
    plot_df.loc[plot_df["budget"] == 0, "width_ratio"] = 1.0

    reference = attach_reference_costs_without_pkl(reference, config, plot_df)
    reference_aux_cost = reference.get("reference_aux_cost", np.nan)
    reference_oracle_cost = reference.get("reference_oracle_cost", np.nan)
    max_plot_budget = reference.get("max_plotted_budget", np.nan)
    reference_cost_ratio_to_max_budget = reference.get("aux_cost_over_max_budget", np.nan)

    if np.isfinite(reference_ratio):
        print(
            f"Stratified joint reference using all predictions: "
            f"R2={reference_r2:.4f}, reference ratio={reference_ratio:.4f}, "
            f"attainable gain={reference_gain:.2%}"
        )
        print(
            f"Reference auxiliary cost = {reference_aux_cost:.2f}; "
            f"reference oracle-info cost = {reference_oracle_cost:.2f}; "
            f"max plotted budget = {max_plot_budget:.2f}; "
            f"aux-cost/max-budget = {reference_cost_ratio_to_max_budget:.2f}x"
        )
    else:
        print("Reference line unavailable:", reference.get("status"))

    fig, axes = plt.subplots(1, 3, figsize=(20.0, 5.4))

    metrics = [
        ("coverage", "Coverage"),
        ("rmse_ratio", "RMSE / Classical"),
        ("width_ratio", "CI width / Classical"),
    ]

    handles = []
    labels = []

    for ax, (metric, ylabel) in zip(axes, metrics):
        draw_reference = (
            metric == "width_ratio"
            and np.isfinite(reference_ratio)
            and reference_ratio < 0.999999
        )

        if draw_reference:
            ax.axhspan(reference_ratio, 1.0, color=REF_BAND_COLOR, alpha=0.65, zorder=0)
            ax.axhline(reference_ratio, color=REF_COLOR, lw=2.0, ls=(0, (4, 2)), zorder=1)

        for method in METHOD_ORDER:
            g = plot_df[plot_df["method"] == method].sort_values("budget")
            if g.empty:
                continue

            x = g["budget"].to_numpy(dtype=float)
            y_raw = g[metric].to_numpy(dtype=float)
            y = np.array(smooth_series(y_raw, window=SMOOTH_WINDOW), copy=True)

            zero_idx = np.where(x == 0)[0]
            if len(zero_idx) > 0:
                y[zero_idx[0]] = anchor_values["coverage"] if metric == "coverage" else 1.0

            line, = ax.plot(
                x,
                y,
                color=METHOD_COLORS[method],
                label=method,
                zorder=3,
                **METHOD_STYLES[method],
            )

            if method not in labels:
                handles.append(line)
                labels.append(method)

        ax.set_xlabel("Budget")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if metric == "coverage":
            ax.set_ylim(0.7, 1.00)
            ax.axhline(0.95, color="black", lw=1.8, ls=":")
        else:
            vals = []
            for method in METHOD_ORDER:
                g = plot_df[plot_df["method"] == method].sort_values("budget")
                if g.empty:
                    continue
                y_tmp = smooth_series(g[metric].to_numpy(dtype=float), window=SMOOTH_WINDOW)
                vals.extend(y_tmp[np.isfinite(y_tmp)].tolist())

            if draw_reference:
                vals.append(reference_ratio)

            vals = np.asarray(vals, dtype=float)
            ymin = float(np.nanmin(vals))
            ymax = float(np.nanmax(vals))

            pad_low = 0.003
            pad_high = 0.004
            lower = max(0.0, ymin - pad_low)
            upper = max(1.0 + pad_high, ymax + pad_high)
            ax.set_ylim(lower, upper)
            ax.axhline(1.0, color="black", lw=1.8, ls=":")

    if np.isfinite(reference_ratio) and reference_ratio < 0.999999:
        ax_ref = axes[2]
        ymin, ymax = ax_ref.get_ylim()
        yrange = ymax - ymin

        # Put the two labels symmetrically around the dashed reference line in
        # data coordinates, and extend the lower y-limit so the lower label is
        # visibly below the line rather than squeezed onto the x-axis.
        text_offset = max(REF_TEXT_OFFSET_MIN, REF_TEXT_OFFSET_FRAC * yrange)
        label_y = reference_ratio + text_offset
        cost_y = reference_ratio - text_offset

        bottom_pad = max(REF_BOTTOM_PAD_MIN, REF_BOTTOM_PAD_FRAC * yrange)
        new_ymin = min(ymin, cost_y - bottom_pad)
        ax_ref.set_ylim(new_ymin, ymax)

        ax_ref.text(
            0.035,
            label_y,
            "Ref. line",
            transform=ax_ref.get_yaxis_transform(),  # x in axes coords, y in data coords
            ha="left",
            va="bottom",
            fontsize=LEGEND_FONTSIZE,
            color=REF_COLOR,
            zorder=10,
        )

        if np.isfinite(reference_aux_cost):
            cost_text = f"Ref. cost: {reference_aux_cost:.0f}"

            ax_ref.text(
                0.035,
                cost_y,
                cost_text,
                transform=ax_ref.get_yaxis_transform(),  # x in axes coords, y in data coords
                ha="left",
                va="top",
                fontsize=LEGEND_FONTSIZE,
                color=REF_COLOR,
                zorder=10,
            )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        columnspacing=1.1,
        handlelength=2.2,
        handletextpad=0.5,
        bbox_to_anchor=(0.5, 0.05),
    )

    fig.tight_layout(rect=[0.0, 0.13, 1.0, 1.0])
    fig.savefig(PERF_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {PERF_PDF}")

# ============================================================
# Figure 2: allocation heatmaps from allocation_summary.csv
# ============================================================
def plot_allocation_heatmaps():
    alloc = pd.read_csv(ALLOC_PATH)
    alloc["method"] = alloc["method"].map(normalize_method_name)
    alloc = alloc[alloc["budget"] > 0].copy()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    models = config.get("models", [])
    strata_token_summary = config.get("strata_token_summary", {})

    def display_source_name(x):
        x = str(x)
        if x == "joint_all_surrogates":
            return "Joint all"
        if x.startswith("single__"):
            x = x.split("single__", 1)[1]
        return DISPLAY_NAME.get(x, x)

    def make_stratum_label(h):
        h = int(h)
        info = strata_token_summary.get(str(h), strata_token_summary.get(h, None))
        if isinstance(info, dict) and "min" in info and "max" in info:
            return f"S{h}: {int(info['min'])}-{int(info['max'])}"
        return f"S{h}"

    if ALLOCATION_BUDGET_MODE == "max":
        chosen_budget = float(alloc["budget"].max())
        alloc_use = alloc[np.isclose(alloc["budget"], chosen_budget)].copy()
        print(f"Using allocation at max budget = {chosen_budget}")
    else:
        alloc_use = (
            alloc.groupby(["method", "stratum", "source_type", "source"], as_index=False)
            .agg(
                query_total=("query_total", "sum"),
                cost_total=("cost_total", "sum"),
                selected_total=("selected_total", "sum"),
                n_detail_rows=("n_detail_rows", "sum"),
            )
        )
        query_den = alloc_use.groupby(["method", "stratum"])["query_total"].transform("sum")
        cost_den = alloc_use.groupby(["method", "stratum"])["cost_total"].transform("sum")
        alloc_use["query_share"] = np.where(query_den > 0, alloc_use["query_total"] / query_den, 0.0)
        alloc_use["cost_share"] = np.where(cost_den > 0, alloc_use["cost_total"] / cost_den, 0.0)
        print("Using allocation aggregated over all positive budgets.")

    def allocation_matrix(method, value_col, source_order):
        df = alloc_use[alloc_use["method"] == method].copy()
        if df.empty:
            raise ValueError(f"No allocation rows found for method={method}.")

        df["source_display"] = df["source"].map(display_source_name)
        df["stratum_label"] = df["stratum"].map(make_stratum_label)

        all_strata = sorted(df["stratum"].unique().tolist())
        row_labels = [make_stratum_label(h) for h in all_strata]

        mat = (
            df.pivot_table(
                index="stratum_label",
                columns="source_display",
                values=value_col,
                aggfunc="sum",
                fill_value=0.0,
            )
            .reindex(index=row_labels, columns=source_order, fill_value=0.0)
        )
        return row_labels, mat

    model_cols = [DISPLAY_NAME.get(m, m) for m in models]
    multippi_cols = ["Joint all"] + model_cols

    row_labels, omppi_query = allocation_matrix("OMPPI(DAG)", "query_share", model_cols)
    _, multippi_query = allocation_matrix("MultiPPI", "query_share", multippi_cols)
    _, omppi_cost = allocation_matrix("OMPPI(DAG)", "cost_share", model_cols)
    _, multippi_cost = allocation_matrix("MultiPPI", "cost_share", multippi_cols)

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
                    j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=annot_size,
                )

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Percent")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(21, 11),
        gridspec_kw={"wspace": 0.26, "hspace": 0.18},
    )

    draw_heatmap(
        axes[0, 0],
        omppi_query,
        row_labels,
        list(omppi_query.columns),
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
        multippi_query,
        row_labels,
        list(multippi_query.columns),
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
        omppi_cost,
        row_labels,
        list(omppi_cost.columns),
        title="OMPPI(DAG): cost allocation",
        cmap="Blues",
        show_y=True,
        show_xlabels=True,
        xlabel="LLM judge",
        ylabel="Prompt-length stratum",
        annot_size=15,
    )

    draw_heatmap(
        axes[1, 1],
        multippi_cost,
        row_labels,
        list(multippi_cost.columns),
        title="MultiPPI: cost allocation",
        cmap="Oranges",
        show_y=False,
        show_xlabels=True,
        xlabel="Queried block",
        ylabel=None,
        annot_size=15,
    )

    fig.tight_layout()
    fig.savefig(ALLOC_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {ALLOC_PDF}")

if __name__ == "__main__":
    plot_performance_with_reference()
    plot_allocation_heatmaps()
