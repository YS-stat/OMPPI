#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False


# ============================================================
# Basic helpers
# ============================================================

def read_pickle_with_compat(path: Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_pickle(path)
    except TypeError as e:
        msg = str(e)
        if "StringDtype" not in msg and "issubclass() arg 1 must be a class" not in msg:
            raise

        from pandas import StringDtype
        old_init = getattr(StringDtype, "__init__", None)
        if old_init is None:
            raise

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


def find_first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns exists: {list(candidates)}")


def detect_model_names_from_final_df(df: pd.DataFrame) -> List[str]:
    suffix_a = "__judge_num_A"
    suffix_b = "__judge_num_B"
    cand_a = {c[:-len(suffix_a)] for c in df.columns if c.endswith(suffix_a)}
    cand_b = {c[:-len(suffix_b)] for c in df.columns if c.endswith(suffix_b)}
    models = sorted(cand_a & cand_b)
    if not models:
        raise ValueError("Could not detect judge model names from __judge_num_A/B columns.")
    return models


def resolve_model_name(requested: str, model_names: Sequence[str]) -> str:
    if requested in model_names:
        return requested

    req = requested.lower()
    aliases = {
        "oss-20b": "gpt-oss-20b",
        "oss-120b": "gpt-oss-120b",
        "qwen-next": "qwen3-next-80b-instruct",
        "qwen-235b": "qwen3-235b-a22b-instruct",
        "qwen-coder": "qwen3-coder-480b-a35b-instruct",
        "gemini-pro": "gemini-2.5-pro",
        "gemini-flash": "gemini-2.5-flash",
        "gemini-lite": "gemini-2.5-flash-lite",
    }
    if req in aliases and aliases[req] in model_names:
        return aliases[req]

    contains = [m for m in model_names if req in m.lower()]
    if len(contains) == 1:
        return contains[0]

    raise ValueError(
        f"Requested base model {requested!r} not found. "
        f"Available models are:\n{json.dumps(list(model_names), indent=2)}"
    )


def resolve_device(device_request: str = "auto") -> str:
    req = (device_request or "auto").lower()
    if req not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    if req == "cpu":
        return "cpu"

    if req == "cuda":
        if _HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        print("[warning] --device cuda requested but CUDA is unavailable. Falling back to CPU.")
        return "cpu"

    if _HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pop_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.mean((x - np.mean(x)) ** 2))


def pop_cov(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))))


def pop_corr(x: np.ndarray, y: np.ndarray) -> float:
    vx = pop_var(x)
    vy = pop_var(y)
    if not np.isfinite(vx) or not np.isfinite(vy) or vx <= 1e-15 or vy <= 1e-15:
        return float("nan")
    return float(pop_cov(x, y) / math.sqrt(vx * vy))


def compute_alignment_stats(y: np.ndarray, f: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    f = np.asarray(f, dtype=float)

    var_y = pop_var(y)
    var_f = pop_var(f)
    cov_yf = pop_cov(y, f)
    corr_yf = pop_corr(y, f)

    gamma = 0.0 if var_f <= 1e-15 else cov_yf / var_f
    tau2 = 0.0 if var_f <= 1e-15 else (cov_yf ** 2) / var_f
    r2 = 0.0 if var_y <= 1e-15 else tau2 / var_y

    return {
        "var_y": float(var_y),
        "var_f": float(var_f),
        "cov_yf": float(cov_yf),
        "corr_yf": float(corr_yf),
        "gamma": float(gamma),
        "tau2": float(tau2),
        "R2_tau_over_varY": float(r2),
    }


def v_ratio_from_tau2(var_y: float, tau2: float, n0: int, n1: int) -> Tuple[float, float, float]:
    if n1 < n0:
        raise ValueError("n1 must be >= n0.")

    v_lo = var_y / max(n0, 1)
    v_omppi = v_lo - (1.0 / max(n0, 1) - 1.0 / max(n1, 1)) * tau2
    ratio = float("nan") if v_lo <= 1e-15 else v_omppi / v_lo

    return float(v_lo), float(v_omppi), float(ratio)


def safe_score_from_votes(gpt4_votes: np.ndarray, claude_votes: np.ndarray) -> np.ndarray:
    gpt4_votes = np.asarray(gpt4_votes, dtype=float)
    claude_votes = np.asarray(claude_votes, dtype=float)

    total = gpt4_votes + claude_votes
    out = np.full(total.shape, 0.5, dtype=float)
    mask = total > 0
    out[mask] = gpt4_votes[mask] / total[mask]
    return out


def standardize_direction(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)

    if not np.isfinite(sd) or sd <= 1e-15:
        return np.zeros_like(x, dtype=float)

    return (x - mu) / sd


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)

    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))

    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)

    return out


def logit_clip(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def parse_lambda_grid(s: str) -> np.ndarray:
    s = str(s).strip()

    if ":" in s:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError("Use start:stop:num for lambda grid, e.g. -2:2:17.")
        start, stop, num = float(parts[0]), float(parts[1]), int(parts[2])
        return np.linspace(start, stop, num)

    return np.asarray([float(x.strip()) for x in s.split(",") if x.strip()], dtype=float)


def configure_matplotlib(font_path: Optional[str]) -> None:
    if font_path:
        fp = Path(font_path)
        if fp.exists():
            font_manager.fontManager.addfont(str(fp))
            try:
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(fp)).get_name()
            except Exception:
                plt.rcParams["font.family"] = "DejaVu Sans"
        else:
            print(f"[warning] Font path not found: {fp}. Falling back to DejaVu Sans.")
            plt.rcParams["font.family"] = "DejaVu Sans"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"

    # All figure fonts are 1.5x larger than the first version.
    plt.rcParams.update({
        "font.size": 21,
        "axes.titlesize": 22.5,
        "axes.labelsize": 21,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "cm",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def short_model_name(model: str) -> str:
    mapping = {
        "gemini-2.5-pro": "G2.5-Pro",
        "gemini-2.5-flash": "G2.5-Flash",
        "gemini-2.5-flash-lite": "G2.5-Lite",
        "gemini-3.1-flash-preview": "G3.1-Flash",
        "gemini-3.1-flash-lite-preview": "G3.1-Lite",
        "qwen3-next-80b-instruct": "Qwen-Next",
        "qwen3-235b-a22b-instruct": "Qwen-235B",
        "qwen3-coder-480b-a35b-instruct": "Qwen-Coder",
        "gpt-oss-120b": "OSS-120B",
        "gpt-oss-20b": "OSS-20B",
    }
    return mapping.get(model, model)


# ============================================================
# Build aligned empirical population
# ============================================================

def build_aligned_population(
    df0: pd.DataFrame,
    *,
    model_names: Sequence[str],
    gpt4_name: str,
    claude_name: str,
    label_mode: str,
) -> Dict[str, Any]:
    model_a_col = find_first_existing_col(
        df0,
        ["model_a", "model_a_name", "left_model", "response_a_model"],
    )
    model_b_col = find_first_existing_col(
        df0,
        ["model_b", "model_b_name", "right_model", "response_b_model"],
    )

    a_name = df0[model_a_col].astype(str)
    b_name = df0[model_b_col].astype(str)

    mask_pair = (
        ((a_name == gpt4_name) & (b_name == claude_name)) |
        ((a_name == claude_name) & (b_name == gpt4_name))
    )

    df = df0.loc[mask_pair].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows found for {gpt4_name} vs {claude_name}.")

    df["gpt4_is_A"] = df[model_a_col].astype(str) == gpt4_name
    df["gpt4_is_B"] = df[model_b_col].astype(str) == gpt4_name

    if not (df["gpt4_is_A"] ^ df["gpt4_is_B"]).all():
        raise ValueError("Each retained row must place GPT-4 on exactly one side.")

    required = {"winner_model_a", "winner_model_b", "winner_tie"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing human winner columns: {missing}")

    label_mode = label_mode.lower()
    if label_mode not in {"drop_ties", "half_ties"}:
        raise ValueError("--label-mode must be drop_ties or half_ties.")

    def row_to_y(row: pd.Series) -> float:
        tie = float(row.get("winner_tie", 0))
        if tie == 1:
            return 0.5 if label_mode == "half_ties" else np.nan

        if float(row.get("winner_model_a", 0)) == 1:
            return 1.0 if bool(row["gpt4_is_A"]) else 0.0

        if float(row.get("winner_model_b", 0)) == 1:
            return 1.0 if bool(row["gpt4_is_B"]) else 0.0

        return np.nan

    df["Y"] = df.apply(row_to_y, axis=1)
    df = df.loc[df["Y"].notna()].copy().reset_index(drop=True)

    y = df["Y"].astype(float).to_numpy()
    gpt4_is_A = df["gpt4_is_A"].to_numpy(dtype=bool)

    score_map: Dict[str, np.ndarray] = {}
    raw_cost_map: Dict[str, float] = {}
    raw_time_map: Dict[str, float] = {}
    usable_models: List[str] = []

    for model in model_names:
        a_col = f"{model}__judge_num_A"
        b_col = f"{model}__judge_num_B"

        if a_col not in df.columns or b_col not in df.columns:
            print(f"[warning] Skipping model with missing vote columns: {model}")
            continue

        a_votes = pd.to_numeric(df[a_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        b_votes = pd.to_numeric(df[b_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        gpt4_votes = np.where(gpt4_is_A, a_votes, b_votes)
        claude_votes = np.where(gpt4_is_A, b_votes, a_votes)

        score_map[model] = safe_score_from_votes(gpt4_votes, claude_votes)
        usable_models.append(model)

        cost_col = f"{model}__judge_total_cost_usd"
        time_col = f"{model}__judge_total_api_time_sec"

        raw_cost_map[model] = (
            float(pd.to_numeric(df[cost_col], errors="coerce").fillna(0.0).mean())
            if cost_col in df.columns else float("nan")
        )
        raw_time_map[model] = (
            float(pd.to_numeric(df[time_col], errors="coerce").fillna(0.0).mean())
            if time_col in df.columns else float("nan")
        )

    finite_costs = [v for v in raw_cost_map.values() if np.isfinite(v) and v > 0]
    max_cost = max(finite_costs) if finite_costs else 1.0

    norm_cost_map = {
        m: (raw_cost_map[m] / max_cost if np.isfinite(raw_cost_map[m]) else np.nan)
        for m in usable_models
    }

    return {
        "df": df,
        "Y": y,
        "gpt4_is_A": gpt4_is_A,
        "model_names": usable_models,
        "score_map": score_map,
        "raw_cost_map": raw_cost_map,
        "norm_cost_map": norm_cost_map,
        "raw_time_map": raw_time_map,
    }


# ============================================================
# Empirical bias directions: b_tilde(X)
# ============================================================

def get_response_texts_for_gpt4_claude(pop: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    df = pop["df"]
    gpt4_is_A = pop["gpt4_is_A"]

    response_a_col = find_first_existing_col(
        df,
        ["response_a", "answer_a", "output_a", "text_a"],
    )
    response_b_col = find_first_existing_col(
        df,
        ["response_b", "answer_b", "output_b", "text_b"],
    )

    resp_a = df[response_a_col].fillna("").astype(str).to_numpy()
    resp_b = df[response_b_col].fillna("").astype(str).to_numpy()

    gpt4_resp = np.where(gpt4_is_A, resp_a, resp_b)
    claude_resp = np.where(gpt4_is_A, resp_b, resp_a)

    return gpt4_resp, claude_resp


def build_btilde_directions(pop: Dict[str, Any], *, base_model: str) -> Dict[str, np.ndarray]:
    y = pop["Y"]
    gpt4_is_A = pop["gpt4_is_A"]
    score_map = pop["score_map"]

    # Position-like empirical bias direction.
    btilde_pos_raw = np.where(gpt4_is_A, 1.0, -1.0)

    # Length / verbosity empirical bias direction.
    try:
        gpt4_resp, claude_resp = get_response_texts_for_gpt4_claude(pop)
        len_gpt4 = np.asarray([len(str(x)) for x in gpt4_resp], dtype=float)
        len_claude = np.asarray([len(str(x)) for x in claude_resp], dtype=float)
        btilde_len_raw = np.log1p(len_gpt4) - np.log1p(len_claude)
    except Exception as exc:
        print(f"[warning] Failed to construct length direction: {exc}")
        btilde_len_raw = np.zeros_like(y, dtype=float)

    # Judge-consensus positive-control direction.
    consensus_models = [m for m in pop["model_names"] if m != base_model]
    if consensus_models:
        btilde_cons_raw = np.mean(
            np.column_stack([score_map[m] for m in consensus_models]),
            axis=1,
        )
    else:
        btilde_cons_raw = np.mean(
            np.column_stack([score_map[m] for m in pop["model_names"]]),
            axis=1,
        )

    return {
        "Position": standardize_direction(btilde_pos_raw),
        "Length": standardize_direction(btilde_len_raw),
        "Consensus": standardize_direction(btilde_cons_raw),
    }


def make_alignment_table(y: np.ndarray, directions: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []

    for name, btilde in directions.items():
        y1 = btilde[y == 1]
        y0 = btilde[y == 0]

        rows.append({
            "direction": name,
            "n": int(len(y)),
            "corr_Y_btilde": pop_corr(y, btilde),
            "cov_Y_btilde": pop_cov(y, btilde),
            "var_btilde": pop_var(btilde),
            "mean_btilde_if_Y1": float(np.mean(y1)) if y1.size else np.nan,
            "mean_btilde_if_Y0": float(np.mean(y0)) if y0.size else np.nan,
        })

    return pd.DataFrame(rows)


def make_judge_alignment_table(pop: Dict[str, Any], *, n0: int, n1: int) -> pd.DataFrame:
    y = pop["Y"]
    var_y = pop_var(y)

    rows = []
    for model in pop["model_names"]:
        f = pop["score_map"][model]
        stats = compute_alignment_stats(y, f)
        v_lo, v_omppi, v_ratio = v_ratio_from_tau2(var_y, stats["tau2"], n0, n1)

        norm_cost = pop["norm_cost_map"].get(model, np.nan)

        rows.append({
            "model": model,
            "display_model": short_model_name(model),
            "corr_Y_f": stats["corr_yf"],
            "cov_Y_f": stats["cov_yf"],
            "gamma_hat": stats["gamma"],
            "tau2_hat": stats["tau2"],
            "R2_tau_over_varY": stats["R2_tau_over_varY"],
            "V_LO": v_lo,
            "V_OMPPI_single": v_omppi,
            "V_ratio_OMPPI_over_LO": v_ratio,
            "variance_reduction": 1.0 - v_ratio,
            "cost_normalized": norm_cost,
            "cost_raw_mean_usd": pop["raw_cost_map"].get(model, np.nan),
            "time_mean_sec": pop["raw_time_map"].get(model, np.nan),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["R2_tau_over_varY", "cost_normalized"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return out


def make_perturbation_curve(
    pop: Dict[str, Any],
    *,
    base_model: str,
    directions: Dict[str, np.ndarray],
    lambdas: np.ndarray,
    n0: int,
    n1: int,
    logit_eps: float,
) -> pd.DataFrame:
    y = pop["Y"]
    var_y = pop_var(y)

    base_f = pop["score_map"][base_model]
    base_logit = logit_clip(base_f, eps=logit_eps)

    rows = []
    for direction_name, btilde in directions.items():
        for lam in lambdas:
            # Empirical log-odds-scale bias:
            # b_lambda(X) = lambda * b_tilde(X).
            f_lam = sigmoid(base_logit + float(lam) * btilde)

            stats = compute_alignment_stats(y, f_lam)
            v_lo, v_omppi, v_ratio = v_ratio_from_tau2(var_y, stats["tau2"], n0, n1)

            rows.append({
                "base_model": base_model,
                "base_display_model": short_model_name(base_model),
                "direction": direction_name,
                "perturb_strength": float(lam),
                "corr_Y_f_lambda": stats["corr_yf"],
                "cov_Y_f_lambda": stats["cov_yf"],
                "gamma_lambda": stats["gamma"],
                "tau2_lambda": stats["tau2"],
                "R2_lambda": stats["R2_tau_over_varY"],
                "V_LO": v_lo,
                "V_OMPPI_lambda": v_omppi,
                "V_ratio_OMPPI_over_LO": v_ratio,
                "variance_reduction": 1.0 - v_ratio,
            })

    return pd.DataFrame(rows)


# ============================================================
# Plot: exactly two panels
# ============================================================

def plot_demo(
    *,
    alignment_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    base_model: str,
    out_pdf: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.5, 6.0),
        gridspec_kw={"width_ratios": [3, 7]}
    )

    # --------------------------
    # Panel 1: Alignment
    # --------------------------
    ax = axes[0]

    order = ["Position", "Length", "Consensus"]
    df = alignment_df.copy()
    df["direction"] = pd.Categorical(df["direction"], categories=order, ordered=True)
    df = df.sort_values("direction")

    x = np.arange(len(df))
    vals = df["corr_Y_btilde"].to_numpy(dtype=float)

    ax.bar(x, vals, width=0.40, color="0.65", edgecolor="0.35", linewidth=0.8)
    ax.axhline(0.0, color="black", lw=1.3)

    ax.set_xticks(x)
    ax.set_xticklabels(df["direction"].astype(str).tolist(), rotation=18, ha="right")
    ax.set_ylabel(r"$\widehat{\mathrm{Corr}}\{Y,\widetilde b(X)\}$")
    ax.set_title("Alignment")

    ymin = min(-0.05, float(np.nanmin(vals)) - 0.05)
    ymax = max(0.05, float(np.nanmax(vals)) + 0.06)
    ax.set_ylim(ymin, ymax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --------------------------
    # Panel 2: Biased OSS-20B judge
    # --------------------------
    ax = axes[1]

    style_map = {
        "Position": {"marker": "o", "ls": "-"},
        "Length": {"marker": "s", "ls": "--"},
        "Consensus": {"marker": "^", "ls": "-."},
    }

    for direction, g in curve_df.groupby("direction", sort=False):
        g = g.sort_values("perturb_strength")
        st = style_map.get(direction, {"marker": "o", "ls": "-"})

        ax.plot(
            g["perturb_strength"].to_numpy(dtype=float),
            g["V_ratio_OMPPI_over_LO"].to_numpy(dtype=float),
            label=direction,
            lw=2.8,
            ms=6.5,
            **st,
        )

    ax.axhline(1.0, color="black", lw=1.3, ls=":")
    ax.axvline(0.0, color="black", lw=1.1, ls=":", alpha=0.8)

    ax.set_xlabel(r"Perturbation strength $\lambda$")
    ax.set_ylabel(r"$\widehat V_{\mathrm{OMPPI}}/\widehat V_{\mathrm{LO}}$")
    ax.set_title(f"Perturbed {short_model_name(base_model)} judge")

    ax.legend(frameon=False, loc="best")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.22,
        top=0.88,
        wspace=0.35
    )
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Empirical judge-bias alignment demo: alignment -> tau^2 -> OMPPI variance."
    )

    parser.add_argument("--final-table", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="./llm_preferences_alignment_bias_demo")
    parser.add_argument("--label-mode", type=str, default="drop_ties", choices=["drop_ties", "half_ties"])
    parser.add_argument("--gpt4-name", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--claude-name", type=str, default="claude-2.1")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--base-model", type=str, default="gpt-oss-20b")
    parser.add_argument("--n0", type=int, default=600)
    parser.add_argument("--n1", type=int, default=1600)
    parser.add_argument("--lambdas", type=str, default="-2:2:17")
    parser.add_argument("--logit-eps", type=float, default=1e-4)
    parser.add_argument("--font-path", type=str, default="fonts/Helvetica.ttf")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()

    device = resolve_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_path = Path(args.final_table)
    if final_path.suffix.lower() in {".pkl", ".pickle"}:
        df0 = read_pickle_with_compat(final_path)
    elif final_path.suffix.lower() == ".csv":
        df0 = pd.read_csv(final_path)
    else:
        raise ValueError(f"Unsupported final table type: {final_path.suffix}")

    if args.models.strip():
        model_names = [x.strip() for x in args.models.split(",") if x.strip()]
    else:
        model_names = detect_model_names_from_final_df(df0)

    pop = build_aligned_population(
        df0,
        model_names=model_names,
        gpt4_name=args.gpt4_name,
        claude_name=args.claude_name,
        label_mode=args.label_mode,
    )

    base_model = resolve_model_name(args.base_model, pop["model_names"])
    lambdas = parse_lambda_grid(args.lambdas)

    configure_matplotlib(args.font_path)

    judge_table = make_judge_alignment_table(pop, n0=args.n0, n1=args.n1)

    directions = build_btilde_directions(pop, base_model=base_model)
    alignment_df = make_alignment_table(pop["Y"], directions)

    curve_df = make_perturbation_curve(
        pop,
        base_model=base_model,
        directions=directions,
        lambdas=lambdas,
        n0=args.n0,
        n1=args.n1,
        logit_eps=args.logit_eps,
    )

    judge_table_path = out_dir / "judge_alignment_variance_table.csv"
    alignment_table_path = out_dir / "alignment_table.csv"
    curve_path = out_dir / "bias_perturbation_variance_curve.csv"
    metadata_path = out_dir / "run_metadata.json"
    pdf_path = out_dir / "judge_alignment_bias_variance_demo.pdf"

    judge_table.to_csv(judge_table_path, index=False)
    alignment_df.to_csv(alignment_table_path, index=False)
    curve_df.to_csv(curve_path, index=False)

    metadata = {
        "final_table": str(final_path),
        "n_raw_rows": int(df0.shape[0]),
        "n_used_rows": int(len(pop["Y"])),
        "label_mode": args.label_mode,
        "gpt4_name": args.gpt4_name,
        "claude_name": args.claude_name,
        "theta_hat": float(np.mean(pop["Y"])),
        "var_y": float(pop_var(pop["Y"])),
        "n0": int(args.n0),
        "n1": int(args.n1),
        "base_model_requested": args.base_model,
        "base_model_used": base_model,
        "lambda_grid": [float(x) for x in lambdas],
        "device_requested": args.device,
        "device_used": device,
        "models_used": list(pop["model_names"]),
        "outputs": {
            "judge_alignment_variance_table": str(judge_table_path),
            "alignment_table": str(alignment_table_path),
            "bias_perturbation_variance_curve": str(curve_path),
            "figure_pdf": str(pdf_path),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    plot_demo(
        alignment_df=alignment_df,
        curve_df=curve_df,
        base_model=base_model,
        out_pdf=pdf_path,
    )

    print("Saved outputs:")
    print(f"  {judge_table_path}")
    print(f"  {alignment_table_path}")
    print(f"  {curve_path}")
    print(f"  {metadata_path}")
    print(f"  {pdf_path}")

    print("\nKey summary:")
    print(f"  n used = {len(pop['Y'])}")
    print(f"  theta_hat = {np.mean(pop['Y']):.4f}")
    print(f"  var_y = {pop_var(pop['Y']):.4f}")
    print(f"  base model = {base_model}")
    print(f"  device used = {device}")

    print("\nAlignment:")
    print(alignment_df[["direction", "corr_Y_btilde", "cov_Y_btilde"]].to_string(index=False))


if __name__ == "__main__":
    main()