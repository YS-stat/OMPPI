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
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    sns = None  # type: ignore
    _HAS_SEABORN = False

import matplotlib.pyplot as plt
from matplotlib import font_manager

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
# Helpers
# ============================================================

@dataclass
class EmpiricalPopulation:
    df: pd.DataFrame
    model_names: List[str]
    Y: np.ndarray
    S: np.ndarray
    X: np.ndarray
    costs_raw: Dict[str, float]
    costs_used: Dict[str, float]
    times: Dict[str, float]
    theta_true: float
    label_mode: str
    gpt4_name: str
    claude_name: str
    prompt_col: str
    prompt_token_counts: np.ndarray
    strata_labels: np.ndarray
    num_strata: int
    strata_pi: Dict[int, float]
    strata_token_summary: Dict[int, Dict[str, float]]

@dataclass
class MethodSpec:
    name: str
    kind: str
    models: List[str] = field(default_factory=list)

def find_first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns exist: {list(candidates)}")

def _read_pickle_with_stringdtype_compat(path: Path) -> pd.DataFrame:
    """Best-effort fallback for pandas pickle files created with a newer StringDtype.

    Some environments pickle pandas StringDtype with two constructor arguments
    (storage, na_value), while older pandas versions only accept storage. This
    shim retries unpickling after temporarily widening StringDtype.__init__.
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


def load_final_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() in {".pkl", ".pickle"}:
        try:
            return pd.read_pickle(path)
        except TypeError as e:
            msg = str(e)
            if "StringDtype.__init__" in msg or "issubclass() arg 1 must be a class" in msg:
                print("[load_final_table] pandas pickle compatibility fallback triggered for StringDtype.")
                return _read_pickle_with_stringdtype_compat(path)
            raise
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")

def detect_model_names_from_final_df(df: pd.DataFrame) -> List[str]:
    suffix_a = "__judge_num_A"
    suffix_b = "__judge_num_B"
    cand_a = {c[: -len(suffix_a)] for c in df.columns if c.endswith(suffix_a)}
    cand_b = {c[: -len(suffix_b)] for c in df.columns if c.endswith(suffix_b)}
    models = sorted(cand_a & cand_b)
    if not models:
        raise ValueError("Could not auto-detect judge model names from final table columns.")
    return models

def json_dumps_stable(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)

def safe_sample_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return 0.0
    return float(np.var(x, ddof=1))

def pop_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.mean((x - np.mean(x)) ** 2))

def pop_cov(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))))

def pop_corr(x: np.ndarray, y: np.ndarray) -> float:
    vx = pop_var(x)
    vy = pop_var(y)
    if vx <= 1e-15 or vy <= 1e-15:
        return float("nan")
    return pop_cov(x, y) / math.sqrt(vx * vy)

def _normal_z(alpha: float = 0.05) -> float:
    if abs(alpha - 0.05) < 1e-12:
        return 1.959963984540054
    from scipy.stats import norm
    return float(norm.ppf(1.0 - alpha / 2.0))

def _safe_score_from_votes(gpt4_votes: np.ndarray, claude_votes: np.ndarray) -> np.ndarray:
    total = gpt4_votes + claude_votes
    out = np.full(total.shape, 0.5, dtype=float)
    mask = total > 0
    out[mask] = gpt4_votes[mask] / total[mask]
    return out

def _sample_rows(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.empty((0, X.shape[1]), dtype=float)
    idx = rng.integers(0, X.shape[0], size=n)
    return X[idx]

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
        raise ValueError("Need at least 2 samples to estimate covariance")
    method = method.lower()
    if method == "ledoitwolf" and _HAS_LEDOIT_WOLF:
        lw = LedoitWolf(store_precision=False, assume_centered=False)
        lw.fit(X)
        return _regularize_covariance(lw.covariance_, eps=eps)
    cov = np.cov(X, rowvar=False, ddof=1)
    return _regularize_covariance(cov, eps=eps)

def parse_budgets(s: str) -> List[float]:
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError("For colon budget syntax use start:stop:num")
        start, stop, num = float(parts[0]), float(parts[1]), int(parts[2])
        return [float(x) for x in np.linspace(start, stop, num)]
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_models(s: Optional[str], df: pd.DataFrame) -> List[str]:
    if s is None or not s.strip():
        return detect_model_names_from_final_df(df)
    return [x.strip() for x in s.split(",") if x.strip()]

def resolve_device(device_request: str = "auto") -> str:
    req = (device_request or "auto").lower()
    if req not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if req == "cpu":
        return "cpu"
    if req == "cuda":
        if _HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if _HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _torch_dtype() -> "torch.dtype":
    return torch.float64

def _torch_tensor(x: np.ndarray, device: str):
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=_torch_dtype(), device=device)

def safe_sample_var_device(x: np.ndarray, *, device: str = "cpu") -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return 0.0
    if device == "cuda" and _HAS_TORCH and torch.cuda.is_available():
        xt = _torch_tensor(x, device)
        return float(torch.var(xt, unbiased=True).detach().cpu().item())
    return float(np.var(x, ddof=1))

def pop_var_cov_corr_device(x: np.ndarray, y: Optional[np.ndarray] = None, *, device: str = "cpu"):
    if device == "cuda" and _HAS_TORCH and torch.cuda.is_available():
        xt = _torch_tensor(x, device)
        xc = xt - xt.mean()
        vx = torch.mean(xc * xc)
        if y is None:
            return float(vx.detach().cpu().item())
        yt = _torch_tensor(y, device)
        yc = yt - yt.mean()
        vy = torch.mean(yc * yc)
        cov = torch.mean(xc * yc)
        if vx.item() <= 1e-15 or vy.item() <= 1e-15:
            corr = float("nan")
        else:
            corr = float((cov / torch.sqrt(vx * vy)).detach().cpu().item())
        return float(vx.detach().cpu().item()), float(cov.detach().cpu().item()), corr
    vx = pop_var(x)
    if y is None:
        return vx
    cov = pop_cov(x, y)
    corr = pop_corr(x, y)
    return vx, cov, corr

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
    ss = np.random.SeedSequence([int(base_seed), int(trial_id), 20260406])
    return int(ss.generate_state(1, dtype=np.uint64)[0] % np.uint64(2**63 - 1))

def detect_prompt_column(df: pd.DataFrame, prompt_col: Optional[str] = None) -> str:
    if prompt_col is not None:
        if prompt_col not in df.columns:
            raise ValueError(f"Prompt column override not found: {prompt_col}")
        return prompt_col
    candidates = [
        "prompt", "question", "original_prompt", "user_prompt", "instruction",
        "full_prompt", "raw_prompt", "query", "text", "input",
    ]
    return find_first_existing_col(df, candidates)

def _fallback_prompt_token_count(s: str) -> int:
    s = str(s or "")
    return max(1, len(s.strip().split())) if s.strip() else 0

def count_prompt_tokens(texts: Sequence[str], encoding_name: str = "o200k_base") -> np.ndarray:
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
    return np.asarray([_fallback_prompt_token_count(v) for v in vals], dtype=int)

def make_quantile_strata(token_counts: np.ndarray, num_strata: int = 5) -> np.ndarray:
    token_counts = np.asarray(token_counts, dtype=int)
    if num_strata <= 1:
        return np.zeros(token_counts.shape[0], dtype=int)
    ranks = pd.Series(token_counts).rank(method="first")
    labels = pd.qcut(ranks, q=num_strata, labels=False, duplicates="drop")
    labels = np.asarray(labels, dtype=int)
    uniq = sorted(np.unique(labels).tolist())
    remap = {old: new for new, old in enumerate(uniq)}
    return np.asarray([remap[int(x)] for x in labels], dtype=int)

def summarize_strata(token_counts: np.ndarray, strata_labels: np.ndarray) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    token_counts = np.asarray(token_counts, dtype=int)
    strata_labels = np.asarray(strata_labels, dtype=int)
    for h in sorted(np.unique(strata_labels).tolist()):
        vals = token_counts[strata_labels == h]
        out[int(h)] = {
            "count": int(vals.size),
            "min": int(np.min(vals)) if vals.size else 0,
            "max": int(np.max(vals)) if vals.size else 0,
            "mean": float(np.mean(vals)) if vals.size else 0.0,
            "median": float(np.median(vals)) if vals.size else 0.0,
        }
    return out

def build_empirical_population_from_final_df(
    final_df: pd.DataFrame,
    model_names: Sequence[str],
    *,
    gpt4_name: str = "gpt-4-1106-preview",
    claude_name: str = "claude-2.1",
    label_mode: str = "drop_ties",
    normalize_costs: bool = True,
    num_strata: int = 5,
    prompt_col_override: Optional[str] = None,
) -> EmpiricalPopulation:
    label_mode = label_mode.lower()
    if label_mode not in {"drop_ties", "half_ties"}:
        raise ValueError("label_mode must be 'drop_ties' or 'half_ties'")

    df = final_df.copy()
    model_a_col = find_first_existing_col(df, ["model_a", "model_a_name", "left_model", "response_a_model"])
    model_b_col = find_first_existing_col(df, ["model_b", "model_b_name", "right_model", "response_b_model"])

    a_name = df[model_a_col].astype(str)
    b_name = df[model_b_col].astype(str)
    mask_pair = (((a_name == gpt4_name) & (b_name == claude_name)) | ((a_name == claude_name) & (b_name == gpt4_name)))
    df = df.loc[mask_pair].copy().reset_index(drop=True)

    prompt_col = detect_prompt_column(df, prompt_col_override)
    prompt_text = df[prompt_col].astype(str).fillna("")
    prompt_token_counts_full = count_prompt_tokens(prompt_text.tolist())
    strata_labels_full = make_quantile_strata(prompt_token_counts_full, num_strata=num_strata)
    strata_pi = {int(h): float(np.mean(strata_labels_full == h)) for h in sorted(np.unique(strata_labels_full).tolist())}
    strata_token_summary = summarize_strata(prompt_token_counts_full, strata_labels_full)

    df["prompt_token_count"] = prompt_token_counts_full
    df["stratum"] = strata_labels_full
    df["gpt4_is_A"] = df[model_a_col].astype(str) == gpt4_name
    df["gpt4_is_B"] = df[model_b_col].astype(str) == gpt4_name
    if not (df["gpt4_is_A"] ^ df["gpt4_is_B"]).all():
        raise ValueError("Each retained row must place GPT-4 on exactly one side.")

    def _row_y(row: pd.Series) -> float:
        if row.get("winner_tie", 0) == 1:
            return 0.5 if label_mode == "half_ties" else np.nan
        if row.get("winner_model_a", 0) == 1:
            return 1.0 if row["gpt4_is_A"] else 0.0
        if row.get("winner_model_b", 0) == 1:
            return 1.0 if row["gpt4_is_B"] else 0.0
        return np.nan

    df["Y"] = df.apply(_row_y, axis=1)
    df = df.loc[df["Y"].notna()].copy().reset_index(drop=True)

    gpt4_is_A = df["gpt4_is_A"].to_numpy(dtype=bool)
    score_cols: List[np.ndarray] = []
    costs_raw: Dict[str, float] = {}
    times: Dict[str, float] = {}

    for model in model_names:
        a_col = f"{model}__judge_num_A"
        b_col = f"{model}__judge_num_B"
        cost_col = f"{model}__judge_total_cost_usd"
        time_col = f"{model}__judge_total_api_time_sec"
        missing = [c for c in (a_col, b_col, cost_col, time_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for model {model}: {missing}")

        a_votes = df[a_col].fillna(0).astype(float).to_numpy()
        b_votes = df[b_col].fillna(0).astype(float).to_numpy()
        gpt4_votes = np.where(gpt4_is_A, a_votes, b_votes)
        claude_votes = np.where(gpt4_is_A, b_votes, a_votes)
        score_cols.append(_safe_score_from_votes(gpt4_votes, claude_votes))

        costs_raw[model] = float(df[cost_col].fillna(0.0).mean())
        times[model] = float(df[time_col].fillna(0.0).mean())

    Y = df["Y"].astype(float).to_numpy()
    S = np.column_stack(score_cols) if score_cols else np.empty((len(Y), 0), dtype=float)
    X = np.column_stack([Y, S])

    if normalize_costs:
        max_cost = max(costs_raw.values())
        costs_used = {m: costs_raw[m] / max_cost for m in model_names}
    else:
        costs_used = dict(costs_raw)

    return EmpiricalPopulation(
        df=df,
        model_names=list(model_names),
        Y=Y,
        S=S,
        X=X,
        costs_raw=costs_raw,
        costs_used=costs_used,
        times=times,
        theta_true=float(np.mean(Y)),
        label_mode=label_mode,
        gpt4_name=gpt4_name,
        claude_name=claude_name,
        prompt_col=prompt_col,
        prompt_token_counts=df["prompt_token_count"].to_numpy(dtype=int),
        strata_labels=df["stratum"].to_numpy(dtype=int),
        num_strata=int(len(strata_pi)),
        strata_pi=strata_pi,
        strata_token_summary=strata_token_summary,
    )

def scalar_lambda_hat(y: np.ndarray, x: np.ndarray, n_extra: int, *, device: str = "cpu") -> float:
    if n_extra <= 0:
        return 0.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size <= 1:
        return 0.0
    if device == "cuda" and _HAS_TORCH and torch.cuda.is_available():
        xt = _torch_tensor(x, device)
        yt = _torch_tensor(y, device)
        xc = xt - xt.mean()
        yc = yt - yt.mean()
        var_x = torch.mean(xc * xc)
        if var_x.item() <= 1e-12:
            return 0.0
        cov_xy = torch.mean(xc * yc)
        lam = (n_extra / (len(y) + n_extra)) * cov_xy / var_x
        return float(lam.detach().cpu().item())
    var_x = pop_var(x)
    if var_x <= 1e-12:
        return 0.0
    cov_xy = pop_cov(x, y)
    return float(n_extra / (len(y) + n_extra) * cov_xy / var_x)

def vector_lambda_hat(y: np.ndarray, Z: np.ndarray, n_extra: int, ridge: float = 1e-8, *, device: str = "cpu") -> np.ndarray:
    if n_extra <= 0:
        return np.zeros(Z.shape[1], dtype=float)
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    if Z.shape[0] <= 1:
        return np.zeros(Z.shape[1], dtype=float)
    if device == "cuda" and _HAS_TORCH and torch.cuda.is_available():
        Zt = _torch_tensor(Z, device)
        yt = _torch_tensor(y, device)
        Zc = Zt - Zt.mean(dim=0, keepdim=True)
        yc = yt - yt.mean()
        denom = max(Z.shape[0] - 1, 1)
        Sigma_zz = (Zc.T @ Zc) / denom
        Sigma_zz = Sigma_zz + ridge * torch.eye(Z.shape[1], dtype=_torch_dtype(), device=device)
        cov_zy = (Zc.T @ yc) / denom
        try:
            lam = torch.linalg.solve(Sigma_zz, cov_zy)
        except Exception:
            lam = torch.linalg.pinv(Sigma_zz) @ cov_zy
        lam = (n_extra / (len(y) + n_extra)) * lam
        return lam.detach().cpu().numpy().astype(float)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    Sigma_zz = (Zc.T @ Zc) / max(Z.shape[0] - 1, 1)
    cov_zy = (Zc.T @ yc) / max(Z.shape[0] - 1, 1)
    Sigma_zz = _regularize_covariance(Sigma_zz, eps=ridge)
    lam = np.linalg.solve(Sigma_zz, cov_zy)
    lam = (n_extra / (len(y) + n_extra)) * lam
    return np.asarray(lam, dtype=float)

def run_classical_trial(X_labeled: np.ndarray, *, alpha: float = 0.05, device: str = "cpu") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    y = X_labeled[:, 0]
    if device == "cuda" and _HAS_TORCH and torch.cuda.is_available():
        yt = _torch_tensor(y, device)
        theta_hat = float(yt.mean().detach().cpu().item())
    else:
        theta_hat = float(np.mean(y))
    var_hat = safe_sample_var_device(y, device=device) / max(len(y), 1)
    half = _normal_z(alpha) * math.sqrt(max(var_hat, 0.0))
    return {
        "theta_hat": theta_hat,
        "ci_low": theta_hat - half,
        "ci_high": theta_hat + half,
        "width": 2.0 * half,
        "var_hat": var_hat,
        "actual_cost": 0.0,
        "actual_time": 0.0,
        "budget_left": np.nan,
    }, {"n_labeled": int(len(y))}

def run_vector_ppi_trial(
    X_labeled: np.ndarray,
    X_population: np.ndarray,
    group_model_indices: Sequence[int],
    group_model_names: Sequence[str],
    group_cost: float,
    group_time: float,
    budget: float,
    *,
    alpha: float = 0.05,
    rng: np.random.Generator,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    y = X_labeled[:, 0]
    Z_lab = X_labeled[:, list(group_model_indices)]
    n_extra = int(math.floor(budget / group_cost)) if group_cost > 0 else 0
    lam = vector_lambda_hat(y, Z_lab, n_extra, device=device)

    term_lab = y - Z_lab @ lam
    var_lab = safe_sample_var_device(term_lab, device=device) / max(len(term_lab), 1)
    if n_extra > 0:
        Z_extra = _sample_rows(X_population[:, list(group_model_indices)], n_extra, rng)
        term_extra = Z_extra @ lam
        var_extra = safe_sample_var_device(term_extra, device=device) / n_extra
        mean_extra = float(np.mean(term_extra))
    else:
        var_extra = 0.0
        mean_extra = 0.0

    theta_hat = float(np.mean(term_lab) + mean_extra)
    var_hat = float(var_lab + var_extra)
    half = _normal_z(alpha) * math.sqrt(max(var_hat, 0.0))
    return {
        "theta_hat": theta_hat,
        "ci_low": theta_hat - half,
        "ci_high": theta_hat + half,
        "width": 2.0 * half,
        "var_hat": var_hat,
        "actual_cost": float(n_extra * group_cost),
        "actual_time": float(n_extra * group_time),
        "budget_left": float(budget - n_extra * group_cost),
    }, {"models": list(group_model_names), "lambda": lam.tolist(), "n_extra": int(n_extra)}

def build_restricted_subset_family(model_names: Sequence[str]) -> List[Tuple[str, List[int]]]:
    all_sur = list(range(1, len(model_names) + 1))
    subsets: List[Tuple[str, List[int]]] = [("joint_all_surrogates", all_sur)]
    subsets.extend((f"single__{m}", [i + 1]) for i, m in enumerate(model_names))
    return subsets

def _embed_inverse_block(p: int, subset_idx: Sequence[int], cov_sub: np.ndarray) -> np.ndarray:
    out = np.zeros((p, p), dtype=float)
    inv_sub = _matrix_inverse(cov_sub)
    out[np.ix_(subset_idx, subset_idx)] = inv_sub
    return out

def _objective_from_blocks(n_aux: np.ndarray, full_block: np.ndarray, aux_blocks: Sequence[np.ndarray], a: np.ndarray) -> float:
    M = full_block.copy()
    for n_i, block in zip(n_aux, aux_blocks):
        if n_i > 0:
            M += float(n_i) * block
    M = _regularize_covariance(M, eps=1e-10)
    M_inv = np.linalg.inv(M)
    return float(a @ M_inv @ a)

def _continuous_optimize_allocations(full_block: np.ndarray, aux_blocks: Sequence[np.ndarray], aux_costs: np.ndarray, budget: float, a: np.ndarray) -> np.ndarray:
    m = len(aux_blocks)
    if budget <= 0 or m == 0:
        return np.zeros(m, dtype=float)
    cheapest = int(np.argmin(aux_costs))
    x0_uniform = np.full(m, budget / max(m, 1), dtype=float) / np.maximum(aux_costs, 1e-12)
    x0_cheapest = np.zeros(m, dtype=float)
    x0_cheapest[cheapest] = budget / max(aux_costs[cheapest], 1e-12)
    x0_joint = np.zeros(m, dtype=float)
    x0_joint[0] = budget / max(aux_costs[0], 1e-12)
    starts = [x0_uniform, x0_cheapest, x0_joint]
    bounds = [(0.0, None) for _ in range(m)]
    constraints = [{"type": "ineq", "fun": lambda x, c=aux_costs, b=budget: float(b - c @ x)}]

    best_x = np.zeros(m, dtype=float)
    best_obj = _objective_from_blocks(best_x, full_block, aux_blocks, a)

    def fun(x: np.ndarray) -> float:
        return _objective_from_blocks(x, full_block, aux_blocks, a)

    for x0 in starts:
        try:
            res = minimize(fun, x0=np.asarray(x0, dtype=float), method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500, "ftol": 1e-10, "disp": False})
            x = np.asarray(res.x if res.success else x0, dtype=float)
            x = np.maximum(x, 0.0)
            obj = fun(x)
            if obj < best_obj:
                best_obj = obj
                best_x = x
        except Exception:
            continue
    return np.asarray(best_x, dtype=float)

def _round_and_greedy_fill(n_cont: np.ndarray, aux_costs: np.ndarray, budget: float, full_block: np.ndarray, aux_blocks: Sequence[np.ndarray], a: np.ndarray) -> np.ndarray:
    n_int = np.floor(np.maximum(n_cont, 0.0)).astype(int)
    spent = float(aux_costs @ n_int)
    left = float(budget - spent)
    if left < np.min(aux_costs) - 1e-12:
        return n_int
    while True:
        affordable = [i for i, c in enumerate(aux_costs) if c <= left + 1e-12]
        if not affordable:
            break
        base_obj = _objective_from_blocks(n_int.astype(float), full_block, aux_blocks, a)
        best_idx = None
        best_gain = 0.0
        for i in affordable:
            trial = n_int.copy()
            trial[i] += 1
            obj = _objective_from_blocks(trial.astype(float), full_block, aux_blocks, a)
            gain = base_obj - obj
            if gain > best_gain + 1e-15:
                best_gain = gain
                best_idx = i
        if best_idx is None:
            break
        n_int[best_idx] += 1
        left -= aux_costs[best_idx]
    return n_int

def solve_restricted_multippi(Sigma_hat: np.ndarray, model_names: Sequence[str], model_costs: Mapping[str, float], model_times: Mapping[str, float], *, n_labeled: int, budget: float) -> Dict[str, Any]:
    p = Sigma_hat.shape[0]
    a = np.zeros(p, dtype=float)
    a[0] = 1.0
    full_inv = _matrix_inverse(Sigma_hat)
    full_block = n_labeled * full_inv
    full_name = "full_labeled"

    subset_family = build_restricted_subset_family(model_names)
    aux_blocks: List[np.ndarray] = []
    aux_costs: List[float] = []
    aux_times: List[float] = []
    subset_names: List[str] = []
    for subset_name, subset_idx in subset_family:
        subset_names.append(subset_name)
        cov_sub = Sigma_hat[np.ix_(subset_idx, subset_idx)]
        aux_blocks.append(_embed_inverse_block(p, subset_idx, cov_sub))
        if subset_name == "joint_all_surrogates":
            aux_costs.append(float(sum(model_costs[m] for m in model_names)))
            aux_times.append(float(sum(model_times[m] for m in model_names)))
        else:
            m = subset_name.split("single__", 1)[1]
            aux_costs.append(float(model_costs[m]))
            aux_times.append(float(model_times[m]))

    aux_costs_arr = np.asarray(aux_costs, dtype=float)
    n_cont = _continuous_optimize_allocations(full_block, aux_blocks, aux_costs_arr, float(budget), a)
    n_int = _round_and_greedy_fill(n_cont, aux_costs_arr, float(budget), full_block, aux_blocks, a)

    M = full_block.copy()
    for n_i, block in zip(n_int, aux_blocks):
        if n_i > 0:
            M += float(n_i) * block
    M = _regularize_covariance(M, eps=1e-10)
    M_inv = np.linalg.inv(M)

    lambda_full = n_labeled * full_inv @ M_inv @ a
    lambdas: Dict[str, np.ndarray] = {full_name: lambda_full}
    counts: Dict[str, int] = {full_name: int(n_labeled)}
    subset_indices: Dict[str, List[int]] = {full_name: list(range(p))}
    subset_costs: Dict[str, float] = {full_name: 0.0}
    subset_times: Dict[str, float] = {full_name: 0.0}

    for (subset_name, subset_idx), n_i, subset_cost, subset_time in zip(subset_family, n_int, aux_costs, aux_times):
        subset_indices[subset_name] = list(subset_idx)
        subset_costs[subset_name] = float(subset_cost)
        subset_times[subset_name] = float(subset_time)
        counts[subset_name] = int(n_i)
        if n_i > 0:
            cov_sub = Sigma_hat[np.ix_(subset_idx, subset_idx)]
            inv_sub = _matrix_inverse(cov_sub)
            P_Minv_a = M_inv[np.asarray(subset_idx, dtype=int), :] @ a
            lambdas[subset_name] = float(n_i) * inv_sub @ P_Minv_a
        else:
            lambdas[subset_name] = np.zeros(len(subset_idx), dtype=float)

    actual_cost = float(sum(counts[name] * subset_costs[name] for name in subset_names))
    actual_time = float(sum(counts[name] * subset_times[name] for name in subset_names))
    return {
        "counts": counts,
        "lambdas": {k: np.asarray(v, dtype=float).tolist() for k, v in lambdas.items()},
        "subset_indices": subset_indices,
        "actual_cost": actual_cost,
        "actual_time": actual_time,
        "budget_left": float(budget - actual_cost),
        "objective_value": float(a @ M_inv @ a),
        "continuous_counts": {name: float(x) for x, name in zip(n_cont, subset_names)},
        "selected_subsets": [name for name, ct in counts.items() if ct > 0 and name != full_name],
    }

def run_multippi_restricted_trial(X_labeled: np.ndarray, X_population: np.ndarray, model_names: Sequence[str], model_costs: Mapping[str, float], model_times: Mapping[str, float], budget: float, *, covariance_method: str = "ledoitwolf", alpha: float = 0.05, rng: np.random.Generator) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    Sigma_hat = estimate_covariance(X_labeled, method=covariance_method)
    alloc = solve_restricted_multippi(Sigma_hat, model_names, model_costs, model_times, n_labeled=X_labeled.shape[0], budget=float(budget))
    transformed_means: List[float] = []
    transformed_vars: List[float] = []
    for subset_name, count in alloc["counts"].items():
        idx = alloc["subset_indices"][subset_name]
        lam = np.asarray(alloc["lambdas"][subset_name], dtype=float)
        if subset_name == "full_labeled":
            X_sub = X_labeled[:, idx]
        else:
            if count <= 0:
                continue
            X_sub = _sample_rows(X_population[:, idx], count, rng)
        values = np.asarray(X_sub @ lam, dtype=float)
        transformed_means.append(float(np.mean(values)))
        transformed_vars.append(safe_sample_var(values) / max(len(values), 1))
    theta_hat = float(np.sum(transformed_means))
    var_hat = float(np.sum(transformed_vars))
    half = _normal_z(alpha) * math.sqrt(max(var_hat, 0.0))
    return {
        "theta_hat": theta_hat,
        "ci_low": theta_hat - half,
        "ci_high": theta_hat + half,
        "width": 2.0 * half,
        "var_hat": var_hat,
        "actual_cost": float(alloc["actual_cost"]),
        "actual_time": float(alloc["actual_time"]),
        "budget_left": float(alloc["budget_left"]),
    }, alloc

def compute_stage1_stats(X_labeled: np.ndarray, model_names: Sequence[str], model_costs: Mapping[str, float], model_times: Mapping[str, float], *, device: str = "cpu") -> Dict[str, Any]:
    y = np.asarray(X_labeled[:, 0], dtype=float)
    mean_y = float(np.mean(y)) if not (device == "cuda" and _HAS_TORCH and torch.cuda.is_available()) else float(_torch_tensor(y, device).mean().detach().cpu().item())
    out: Dict[str, Any] = {"n": int(len(y)), "mean_y": mean_y, "var_y": pop_var_cov_corr_device(y, device=device), "per_model": {}}
    for j, model in enumerate(model_names, start=1):
        z = np.asarray(X_labeled[:, j], dtype=float)
        var_z, cov_yz, corr_yz = pop_var_cov_corr_device(z, y, device=device)
        gamma = 0.0 if var_z <= 1e-15 else cov_yz / var_z
        tau2 = 0.0 if var_z <= 1e-15 else (cov_yz ** 2) / var_z
        out["per_model"][model] = {
            "var_z": float(var_z), "cov_yz": float(cov_yz), "corr_yz": float(corr_yz),
            "gamma": float(gamma), "tau2": float(tau2),
            "cost": float(model_costs[model]), "time": float(model_times[model]),
        }
    return out

def _ours_route_score(selected_models_desc_tau: Sequence[str], stats: Dict[str, Any], budget: float, n_labeled: int, eps_gap: float) -> float:
    var_y = float(stats["var_y"])
    if len(selected_models_desc_tau) == 0:
        return var_y / max(n_labeled, 1)
    tau = [float(stats["per_model"][m]["tau2"]) for m in selected_models_desc_tau]
    if var_y - tau[0] <= eps_gap:
        return float("inf")
    for i in range(len(tau) - 1):
        if tau[i] - tau[i + 1] <= eps_gap:
            return float("inf")
    if tau[-1] <= eps_gap:
        return float("inf")
    fixed_term = (var_y - tau[0]) / max(n_labeled, 1)
    path_sum = 0.0
    for i, m in enumerate(selected_models_desc_tau):
        tau_cur = tau[i]
        tau_next = tau[i + 1] if i + 1 < len(tau) else 0.0
        path_sum += math.sqrt(max(tau_cur - tau_next, 0.0) * float(stats["per_model"][m]["cost"]))
    if budget <= 0:
        return float("inf")
    return fixed_term + (path_sum ** 2) / budget

def select_active_models_ours_exhaustive(model_names: Sequence[str], stats: Dict[str, Any], budget: float, n_labeled: int, eps_gap: float) -> Tuple[List[str], float]:
    import itertools
    best_subset: List[str] = []
    best_score = _ours_route_score([], stats, budget, n_labeled, eps_gap)
    for r in range(1, len(model_names) + 1):
        for subset in itertools.combinations(model_names, r):
            ordered = sorted(subset, key=lambda m: (-float(stats["per_model"][m]["tau2"]), m))
            score = _ours_route_score(ordered, stats, budget, n_labeled, eps_gap)
            if score < best_score - 1e-15:
                best_score = score
                best_subset = list(ordered)
    return best_subset, float(best_score)

def select_active_models_ours_dag(model_names: Sequence[str], stats: Dict[str, Any], budget: float, n_labeled: int, eps_gap: float) -> Tuple[List[str], float]:
    var_y = float(stats["var_y"])
    tau_map = {m: float(stats["per_model"][m]["tau2"]) for m in model_names}
    ordered = sorted(model_names, key=lambda m: (-tau_map[m], m))
    k = len(ordered)
    dp = np.full(k, np.inf, dtype=float)
    nxt = [-1] * k
    for i in range(k - 1, -1, -1):
        m_i = ordered[i]
        tau_i = tau_map[m_i]
        best = float("inf")
        best_next = -1
        base_gap = tau_i
        if base_gap > eps_gap:
            best = math.sqrt(base_gap * float(stats["per_model"][m_i]["cost"]))
        for j in range(i + 1, k):
            gap = tau_i - tau_map[ordered[j]]
            if gap <= eps_gap:
                continue
            cand = math.sqrt(gap * float(stats["per_model"][m_i]["cost"])) + dp[j]
            if cand < best - 1e-15:
                best = cand
                best_next = j
        dp[i] = best
        nxt[i] = best_next
    best_route: List[str] = []
    best_score = _ours_route_score([], stats, budget, n_labeled, eps_gap)
    for i, m in enumerate(ordered):
        top_gap = var_y - tau_map[m]
        if top_gap <= eps_gap or not np.isfinite(dp[i]):
            continue
        score = top_gap / max(n_labeled, 1) + (dp[i] ** 2) / max(budget, 1e-12)
        if score < best_score - 1e-15:
            route = [m]
            cur = i
            while nxt[cur] != -1:
                cur = nxt[cur]
                route.append(ordered[cur])
            best_route = route
            best_score = float(score)
    return best_route, float(best_score)

def allocate_ours_nested_samples(active_models_desc_tau: Sequence[str], stats: Dict[str, Any], budget: float, n_labeled: int) -> Dict[str, Any]:
    ordered = list(active_models_desc_tau)
    if len(ordered) == 0:
        return {"ordered_models": [], "n_labeled": int(n_labeled), "total_counts": {}, "extra_counts": {}, "increments": {}, "actual_cost": 0.0, "actual_time": 0.0, "budget_left": float(budget)}
    costs = {m: float(stats["per_model"][m]["cost"]) for m in ordered}
    times = {m: float(stats["per_model"][m]["time"]) for m in ordered}
    tau2 = {m: float(stats["per_model"][m]["tau2"]) for m in ordered}
    gaps, block_costs = [], []
    for t in range(len(ordered)):
        tau_cur = tau2[ordered[t]]
        tau_next = tau2[ordered[t + 1]] if t + 1 < len(ordered) else 0.0
        gaps.append(max(0.0, tau_cur - tau_next))
        block_costs.append(sum(costs[ordered[j]] for j in range(t, len(ordered))))
    gaps_arr = np.asarray(gaps, dtype=float)
    block_costs_arr = np.asarray(block_costs, dtype=float)
    increments = np.zeros(len(ordered), dtype=int)
    positive = (gaps_arr > 0) & (block_costs_arr > 0)
    if budget > 0 and positive.any():
        denom = float(np.sum(np.sqrt(gaps_arr[positive] * block_costs_arr[positive])))
        if denom > 0:
            d_cont = np.zeros(len(ordered), dtype=float)
            d_cont[positive] = budget * np.sqrt(gaps_arr[positive] / block_costs_arr[positive]) / denom
            increments = np.floor(d_cont).astype(int)
            spent = float(np.sum(increments * block_costs_arr))
            left = float(budget - spent)
            priorities = np.where(positive, np.sqrt(gaps_arr / np.maximum(block_costs_arr, 1e-12)), -np.inf)
            while left >= np.min(block_costs_arr[positive]) - 1e-12:
                affordable = [idx for idx, c in enumerate(block_costs_arr) if c <= left + 1e-12 and positive[idx]]
                if not affordable:
                    break
                best_idx = max(affordable, key=lambda idx: priorities[idx])
                increments[best_idx] += 1
                left -= block_costs_arr[best_idx]
    total_counts, extra_counts, increments_dict = {}, {}, {}
    cum = 0
    for j, m in enumerate(ordered):
        cum += int(increments[j])
        total_counts[m] = int(n_labeled + cum)
        extra_counts[m] = int(cum)
        increments_dict[m] = int(increments[j])
    actual_cost = float(sum(extra_counts[m] * costs[m] for m in ordered))
    actual_time = float(sum(extra_counts[m] * times[m] for m in ordered))
    return {"ordered_models": ordered, "n_labeled": int(n_labeled), "total_counts": total_counts, "extra_counts": extra_counts, "increments": increments_dict, "actual_cost": actual_cost, "actual_time": actual_time, "budget_left": float(budget - actual_cost)}

def estimate_ours_once_with_ci(X_labeled: np.ndarray, S_population: Dict[str, np.ndarray], allocation: Dict[str, Any], stage1_stats: Dict[str, Any], model_to_col: Mapping[str, int], *, alpha: float, rng: np.random.Generator) -> Dict[str, Any]:
    ordered = allocation["ordered_models"]
    n0 = int(allocation["n_labeled"])
    zcrit = _normal_z(alpha)
    y_stage1 = np.asarray(X_labeled[:, 0], dtype=float)
    y_mean = float(np.mean(y_stage1))
    var_y = float(stage1_stats["var_y"])
    if len(ordered) == 0:
        var_hat = var_y / max(n0, 1)
        half = zcrit * math.sqrt(max(var_hat, 0.0))
        return {"theta_hat": y_mean, "var_hat": float(var_hat), "ci_low": y_mean - half, "ci_high": y_mean + half, "width": 2.0 * half}
    total_counts = allocation["total_counts"]
    extra_counts = allocation["extra_counts"]
    nmax_extra = max(extra_counts[m] for m in ordered)
    idx_extra = rng.integers(0, len(next(iter(S_population.values()))), size=nmax_extra) if nmax_extra > 0 else np.empty(0, dtype=int)
    theta_hat = y_mean
    var_hat = var_y / max(n0, 1)
    n_prev_total = n0
    for m in ordered:
        gamma = float(stage1_stats["per_model"][m]["gamma"])
        var_z = float(stage1_stats["per_model"][m]["var_z"])
        cov_yz = float(stage1_stats["per_model"][m]["cov_yz"])
        z_stage1 = np.asarray(X_labeled[:, model_to_col[m]], dtype=float)
        z_pool = np.asarray(S_population[m], dtype=float)
        extra_prev = n_prev_total - n0
        n_cur_total = int(total_counts[m])
        extra_cur = n_cur_total - n0
        stage1_sum = float(np.sum(z_stage1))
        extra_prev_sum = float(np.sum(z_pool[idx_extra[:extra_prev]])) if extra_prev > 0 else 0.0
        extra_cur_sum = float(np.sum(z_pool[idx_extra[:extra_cur]])) if extra_cur > 0 else 0.0
        mean_prev = (stage1_sum + extra_prev_sum) / max(n_prev_total, 1)
        mean_cur = (stage1_sum + extra_cur_sum) / max(n_cur_total, 1)
        theta_hat += gamma * (mean_cur - mean_prev)
        var_hat += (1.0 / max(n_prev_total, 1) - 1.0 / max(n_cur_total, 1)) * (gamma ** 2 * var_z - 2.0 * gamma * cov_yz)
        n_prev_total = n_cur_total
    var_hat = max(float(var_hat), 0.0)
    half = zcrit * math.sqrt(var_hat)
    return {"theta_hat": float(theta_hat), "var_hat": float(var_hat), "ci_low": float(theta_hat - half), "ci_high": float(theta_hat + half), "width": float(2.0 * half)}

def run_ours_trial(X_labeled: np.ndarray, population: EmpiricalPopulation, budget: float, *, search_mode: str, eps_gap: float, alpha: float = 0.05, rng: np.random.Generator, device: str = "cpu") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_names = list(population.model_names)
    stage1_stats = compute_stage1_stats(X_labeled, model_names, population.costs_used, population.times, device=device)
    n_labeled = X_labeled.shape[0]
    if search_mode == "exhaustive":
        selected, route_obj = select_active_models_ours_exhaustive(model_names, stage1_stats, float(budget), n_labeled, eps_gap)
    elif search_mode == "dag":
        selected, route_obj = select_active_models_ours_dag(model_names, stage1_stats, float(budget), n_labeled, eps_gap)
    else:
        raise ValueError(search_mode)
    allocation = allocate_ours_nested_samples(selected, stage1_stats, float(budget), n_labeled)
    model_to_col = {m: i + 1 for i, m in enumerate(model_names)}
    S_population = {m: population.X[:, model_to_col[m]] for m in model_names}
    est = estimate_ours_once_with_ci(X_labeled, S_population, allocation, stage1_stats, model_to_col, alpha=alpha, rng=rng)
    est.update({"actual_cost": float(allocation["actual_cost"]), "actual_time": float(allocation["actual_time"]), "budget_left": float(allocation["budget_left"])})
    detail = {
        "search_mode": search_mode,
        "epsilon_gap": float(eps_gap),
        "route_objective": float(route_obj),
        "selected_models": list(selected),
        "selected_tau2": {m: float(stage1_stats["per_model"][m]["tau2"]) for m in selected},
        "selected_gamma": {m: float(stage1_stats["per_model"][m]["gamma"]) for m in selected},
        "total_counts": allocation["total_counts"],
        "extra_counts": allocation["extra_counts"],
        "increments": allocation["increments"],
    }
    return est, detail


def compute_ours_auxiliary_path_q(active_models_desc_tau: Sequence[str], stats: Dict[str, Any]) -> float:
    """Return the auxiliary path criterion used for cross-stratum budget splitting.

    The labeled examples are fixed in this experiment. The extra auxiliary budget is
    allocated over nested prediction levels. For a selected route, the relevant
    auxiliary path criterion is the square-root cost-weighted sum over the nested
    prediction increments. This is the quantity used in the pi_h * Q_h budget split.
    """
    ordered = list(active_models_desc_tau)
    if not ordered:
        return 0.0
    tau2 = {m: float(stats["per_model"][m]["tau2"]) for m in ordered}
    costs = {m: float(stats["per_model"][m]["cost"]) for m in ordered}
    q = 0.0
    for t, m in enumerate(ordered):
        tau_cur = tau2[m]
        tau_next = tau2[ordered[t + 1]] if t + 1 < len(ordered) else 0.0
        gap = max(0.0, tau_cur - tau_next)
        # In the implemented nested query design, an increment at level t purchases
        # all predictions from t through the end of the selected route.
        block_cost = sum(costs[ordered[j]] for j in range(t, len(ordered)))
        if gap > 0.0 and block_cost > 0.0:
            q += math.sqrt(gap * block_cost)
    return float(q)


def plan_ours_trial_fixed_route(
    X_labeled: np.ndarray,
    population: EmpiricalPopulation,
    planning_budget: float,
    *,
    search_mode: str,
    eps_gap: float,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Plan the OMPPI route once and return the fixed route and its Qhat.

    This separates route planning from the final stratum budget B_h, so the
    selected route used to compute Qhat is the same route used in the final
    estimator after the pi_h * Qhat_h cross-stratum budget split.
    """
    model_names = list(population.model_names)
    stage1_stats = compute_stage1_stats(X_labeled, model_names, population.costs_used, population.times, device=device)
    n_labeled = X_labeled.shape[0]
    if search_mode == "exhaustive":
        selected, route_obj = select_active_models_ours_exhaustive(
            model_names, stage1_stats, float(planning_budget), n_labeled, eps_gap
        )
    elif search_mode == "dag":
        selected, route_obj = select_active_models_ours_dag(
            model_names, stage1_stats, float(planning_budget), n_labeled, eps_gap
        )
    else:
        raise ValueError(search_mode)
    q_hat = compute_ours_auxiliary_path_q(selected, stage1_stats)
    return {
        "stage1_stats": stage1_stats,
        "selected": list(selected),
        "route_objective": float(route_obj),
        "q_hat": float(q_hat),
        "planning_budget": float(planning_budget),
        "search_mode": search_mode,
        "epsilon_gap": float(eps_gap),
    }


def run_ours_trial_with_fixed_route(
    X_labeled: np.ndarray,
    population: EmpiricalPopulation,
    budget: float,
    plan: Mapping[str, Any],
    *,
    alpha: float,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run OMPPI using a pre-selected stratum-specific route."""
    selected = list(plan["selected"])
    stage1_stats = dict(plan["stage1_stats"])
    n_labeled = X_labeled.shape[0]
    allocation = allocate_ours_nested_samples(selected, stage1_stats, float(budget), n_labeled)
    model_names = list(population.model_names)
    model_to_col = {m: i + 1 for i, m in enumerate(model_names)}
    S_population = {m: population.X[:, model_to_col[m]] for m in model_names}
    est = estimate_ours_once_with_ci(
        X_labeled, S_population, allocation, stage1_stats, model_to_col, alpha=alpha, rng=rng
    )
    est.update({
        "actual_cost": float(allocation["actual_cost"]),
        "actual_time": float(allocation["actual_time"]),
        "budget_left": float(budget - allocation["actual_cost"]),
    })
    detail = {
        "search_mode": str(plan.get("search_mode", "")),
        "epsilon_gap": float(plan.get("epsilon_gap", np.nan)),
        "route_objective": float(plan.get("route_objective", np.nan)),
        "q_hat": float(plan.get("q_hat", 0.0)),
        "planning_budget": float(plan.get("planning_budget", np.nan)),
        "selected_models": list(selected),
        "selected_tau2": {m: float(stage1_stats["per_model"][m]["tau2"]) for m in selected},
        "selected_gamma": {m: float(stage1_stats["per_model"][m]["gamma"]) for m in selected},
        "total_counts": allocation["total_counts"],
        "extra_counts": allocation["extra_counts"],
        "increments": allocation["increments"],
    }
    return est, detail

def split_trials(n_trials: int, num_workers: int) -> List[List[int]]:
    num_workers = max(1, min(num_workers, n_trials))
    buckets = [[] for _ in range(num_workers)]
    for i in range(n_trials):
        buckets[i % num_workers].append(i)
    return [b for b in buckets if b]

def _summarize_trial_df(trial_df: pd.DataFrame) -> pd.DataFrame:
    return (
        trial_df.groupby(["method", "budget"], as_index=False)
        .agg(
            n_trials=("trial", "count"),
            coverage=("covered", "mean"),
            mse=("sq_error", "mean"),
            rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            mean_bias=("bias", "mean"),
            mean_estimate=("theta_hat", "mean"),
            mean_theta_true=("theta_true", "mean"),
            sd_theta_true=("theta_true", "std"),
            mean_ci_width=("ci_width", "mean"),
            sd_ci_width=("ci_width", "std"),
            mean_var_hat=("var_hat", "mean"),
            actual_cost_mean=("actual_cost", "mean"),
            actual_time_mean=("actual_time", "mean"),
            budget_left_mean=("budget_left", "mean"),
            algo_compute_time_mean=("algo_compute_time_sec", "mean"),
            algo_compute_time_sd=("algo_compute_time_sec", "std"),
        )
        .sort_values(["method", "budget"])
        .reset_index(drop=True)
    )

def sample_theta_truth_subset(y: np.ndarray, truth_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    y = np.asarray(y, dtype=float)
    if truth_size <= 0:
        raise ValueError("theta truth subset size must be positive")
    if truth_size > y.shape[0]:
        raise ValueError(f"theta truth subset size {truth_size} exceeds available rows {y.shape[0]}")
    idx = rng.choice(y.shape[0], size=truth_size, replace=False)
    theta_true = float(np.mean(y[idx]))
    return np.asarray(idx, dtype=int), theta_true

def _allocate_counts_by_pi(
    total: int,
    pi_map: Mapping[int, float],
    rng: np.random.Generator,
    *,
    min_each: int = 0,
    capacities: Optional[Mapping[int, int]] = None,
) -> Dict[int, int]:
    keys = [int(k) for k in sorted(pi_map)]
    pis = np.asarray([max(float(pi_map[k]), 0.0) for k in keys], dtype=float)
    if total < 0:
        raise ValueError("total must be nonnegative")
    if total == 0:
        return {k: 0 for k in keys}
    positive = [k for k, p in zip(keys, pis) if p > 0]
    base = {k: 0 for k in keys}
    if min_each > 0 and len(positive) * min_each > total:
        raise ValueError("total is too small to allocate min_each across positive strata")
    if min_each > 0:
        for k in positive:
            base[k] = int(min_each)
    remaining = int(total - sum(base.values()))
    if remaining <= 0:
        return base

    if capacities is None:
        probs = np.asarray([float(pi_map[k]) for k in keys], dtype=float)
        probs = probs / probs.sum()
        add = rng.multinomial(remaining, probs)
        return {k: int(base[k] + add[i]) for i, k in enumerate(keys)}

    cap_left = {int(k): int(capacities[k]) - int(base[k]) for k in keys}
    if any(v < 0 for v in cap_left.values()):
        raise ValueError("capacities are too small for requested min_each allocation")
    out = dict(base)
    for _ in range(remaining):
        avail = [k for k in keys if cap_left[k] > 0 and float(pi_map[k]) > 0]
        if not avail:
            raise ValueError("capacities exhausted before completing stratified allocation")
        probs = np.asarray([float(pi_map[k]) for k in avail], dtype=float)
        probs = probs / probs.sum()
        chosen = int(rng.choice(avail, p=probs))
        out[chosen] += 1
        cap_left[chosen] -= 1
    return out

def subset_population_by_mask(population: EmpiricalPopulation, mask: np.ndarray) -> EmpiricalPopulation:
    mask = np.asarray(mask, dtype=bool)
    df_sub = population.df.loc[mask].copy().reset_index(drop=True)
    y_sub = np.asarray(population.Y[mask], dtype=float)
    s_sub = np.asarray(population.S[mask], dtype=float)
    x_sub = np.asarray(population.X[mask], dtype=float)
    token_counts = np.asarray(population.prompt_token_counts[mask], dtype=int)
    local_labels = np.zeros(int(mask.sum()), dtype=int)
    token_summary = summarize_strata(token_counts, local_labels) if local_labels.size > 0 else {0: {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0}}
    return EmpiricalPopulation(
        df=df_sub,
        model_names=list(population.model_names),
        Y=y_sub,
        S=s_sub,
        X=x_sub,
        costs_raw=dict(population.costs_raw),
        costs_used=dict(population.costs_used),
        times=dict(population.times),
        theta_true=float(np.mean(y_sub)),
        label_mode=population.label_mode,
        gpt4_name=population.gpt4_name,
        claude_name=population.claude_name,
        prompt_col=population.prompt_col,
        prompt_token_counts=token_counts,
        strata_labels=local_labels,
        num_strata=1,
        strata_pi={0: 1.0},
        strata_token_summary=token_summary,
    )

def sample_theta_truth_subset_stratified(
    population: EmpiricalPopulation,
    truth_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, Dict[int, int]]:
    if truth_size <= 0:
        raise ValueError("theta truth subset size must be positive")
    if truth_size > population.Y.shape[0]:
        raise ValueError(f"theta truth subset size {truth_size} exceeds available rows {population.Y.shape[0]}")
    capacities = {int(h): int(np.sum(population.strata_labels == h)) for h in sorted(population.strata_pi)}
    counts = _allocate_counts_by_pi(truth_size, population.strata_pi, rng, min_each=1, capacities=capacities)
    pieces: List[np.ndarray] = []
    theta_true = 0.0
    for h in sorted(population.strata_pi):
        idx_h = np.where(population.strata_labels == h)[0]
        m_h = int(counts[h])
        if m_h <= 0:
            continue
        chosen = rng.choice(idx_h, size=m_h, replace=False)
        pieces.append(np.asarray(chosen, dtype=int))
        theta_true += float(population.strata_pi[h]) * float(np.mean(population.Y[chosen]))
    truth_idx = np.concatenate(pieces) if pieces else np.empty(0, dtype=int)
    return truth_idx, float(theta_true), counts

def sample_labeled_rows_stratified(
    population: EmpiricalPopulation,
    n_labeled: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    if n_labeled <= 0:
        raise ValueError("n_labeled must be positive")
    min_each = 2 if n_labeled >= 2 * max(len(population.strata_pi), 1) else 1
    counts = _allocate_counts_by_pi(n_labeled, population.strata_pi, rng, min_each=min_each, capacities=None)
    idx_parts: List[np.ndarray] = []
    for h in sorted(population.strata_pi):
        n_h = int(counts[h])
        idx_h = np.where(population.strata_labels == h)[0]
        if n_h <= 0:
            continue
        chosen = rng.choice(idx_h, size=n_h, replace=True)
        idx_parts.append(np.asarray(chosen, dtype=int))
    labeled_idx = np.concatenate(idx_parts) if idx_parts else np.empty(0, dtype=int)
    rng.shuffle(labeled_idx)
    return (
        np.asarray(population.X[labeled_idx], dtype=float),
        np.asarray(population.strata_labels[labeled_idx], dtype=int),
        np.asarray(labeled_idx, dtype=int),
        counts,
    )

def compute_stratified_theta_var(
    per_stratum_estimates: Mapping[int, Dict[str, Any]],
    pi_map: Mapping[int, float],
    *,
    alpha: float,
) -> Dict[str, Any]:
    theta_hat = 0.0
    var_hat = 0.0
    actual_cost = 0.0
    actual_time = 0.0
    for h in sorted(pi_map):
        pi_h = float(pi_map[h])
        est_h = per_stratum_estimates[int(h)]
        theta_hat += pi_h * float(est_h["theta_hat"])
        var_hat += (pi_h ** 2) * float(est_h["var_hat"])
        actual_cost += float(est_h["actual_cost"])
        actual_time += float(est_h["actual_time"])
    var_hat = max(float(var_hat), 0.0)
    half = _normal_z(alpha) * math.sqrt(var_hat)
    return {
        "theta_hat": float(theta_hat),
        "ci_low": float(theta_hat - half),
        "ci_high": float(theta_hat + half),
        "width": float(2.0 * half),
        "var_hat": float(var_hat),
        "actual_cost": float(actual_cost),
        "actual_time": float(actual_time),
    }

def run_stratified_method_trial(
    method: MethodSpec,
    X_labeled: np.ndarray,
    H_labeled: np.ndarray,
    population: EmpiricalPopulation,
    budget: float,
    *,
    covariance_method: str,
    ours_top_ridge: float,
    eps_gap: float,
    alpha: float,
    rng: np.random.Generator,
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    per_stratum_estimates: Dict[int, Dict[str, Any]] = {}
    detail: Dict[str, Any] = {
        "stratified": True,
        "method_kind": method.kind,
        "pi_h": {str(int(h)): float(population.strata_pi[h]) for h in sorted(population.strata_pi)},
        "stratified_budget_rule": "pi_qhat" if method.kind == "nf1" and budget > 0 else "pi",
        "strata": {},
    }
    model_names = list(population.model_names)
    model_to_col = {m: i + 1 for i, m in enumerate(model_names)}

    # Build per-stratum views once so OMPPI can use a clean two-pass budget split:
    # first select/fix the route and compute Qhat_h in every stratum, then allocate
    # B_h proportional to pi_h * Qhat_h and run the fixed route.
    stratum_inputs: Dict[int, Dict[str, Any]] = {}
    for h in sorted(population.strata_pi):
        pi_h = float(population.strata_pi[h])
        mask_h_lab = np.asarray(H_labeled == h, dtype=bool)
        X_h = np.asarray(X_labeled[mask_h_lab], dtype=float)
        if X_h.shape[0] <= 0:
            raise ValueError(f"Stratum {h} has zero labeled samples in this trial; increase n_labeled or reduce num_strata.")
        mask_h_pop = np.asarray(population.strata_labels == h, dtype=bool)
        pop_h = subset_population_by_mask(population, mask_h_pop)
        stratum_inputs[int(h)] = {"pi_h": pi_h, "X_h": X_h, "pop_h": pop_h}

    if method.kind == "nf1" and budget > 0:
        plans: Dict[int, Dict[str, Any]] = {}
        q_denom = 0.0
        for h in sorted(stratum_inputs):
            pi_h = float(stratum_inputs[h]["pi_h"])
            # The planning budget only selects the route. The final B_h below is
            # computed after all Qhat_h values are known and the route is kept fixed.
            planning_budget_h = float(pi_h * budget)
            plans[h] = plan_ours_trial_fixed_route(
                stratum_inputs[h]["X_h"],
                stratum_inputs[h]["pop_h"],
                planning_budget_h,
                search_mode=method.models[0],
                eps_gap=eps_gap,
                device=device,
            )
            q_denom += pi_h * float(plans[h]["q_hat"])

        if q_denom <= 1e-15:
            # Degenerate case: no selected route has positive auxiliary Qhat.
            # Fall back to proportional allocation; the fixed routes will typically
            # be empty, so no auxiliary budget will actually be spent.
            budget_h_map = {h: float(stratum_inputs[h]["pi_h"] * budget) for h in sorted(stratum_inputs)}
            detail["stratified_budget_rule"] = "pi_fallback_zero_qhat"
        else:
            budget_h_map = {
                h: float(budget * stratum_inputs[h]["pi_h"] * float(plans[h]["q_hat"]) / q_denom)
                for h in sorted(stratum_inputs)
            }

        detail["pi_qhat_denominator"] = float(q_denom)
        for h in sorted(stratum_inputs):
            pi_h = float(stratum_inputs[h]["pi_h"])
            X_h = stratum_inputs[h]["X_h"]
            pop_h = stratum_inputs[h]["pop_h"]
            budget_h = float(budget_h_map[h])
            est_h, detail_h = run_ours_trial_with_fixed_route(
                X_h,
                pop_h,
                budget_h,
                plans[h],
                alpha=alpha,
                rng=rng,
            )
            per_stratum_estimates[int(h)] = est_h
            detail["strata"][str(int(h))] = {
                "pi_h": float(pi_h),
                "q_hat_h": float(plans[h]["q_hat"]),
                "planning_budget_h": float(plans[h]["planning_budget"]),
                "budget_h": float(budget_h),
                "n_labeled_h": int(X_h.shape[0]),
                "n_population_h": int(pop_h.X.shape[0]),
                "theta_hat_h": float(est_h["theta_hat"]),
                "var_hat_h": float(est_h["var_hat"]),
                "actual_cost_h": float(est_h["actual_cost"]),
                "actual_time_h": float(est_h["actual_time"]),
                "detail": detail_h,
            }

        est = compute_stratified_theta_var(per_stratum_estimates, population.strata_pi, alpha=alpha)
        est["budget_left"] = float(budget - est["actual_cost"])
        return est, detail

    for h in sorted(stratum_inputs):
        pi_h = float(stratum_inputs[h]["pi_h"])
        budget_h = float(pi_h * budget)
        X_h = stratum_inputs[h]["X_h"]
        pop_h = stratum_inputs[h]["pop_h"]

        if method.kind == "classical":
            est_h, detail_h = run_classical_trial(X_h, alpha=alpha, device=device)
        elif method.kind == "vector_ppi":
            group_names = list(model_names)
            group_idx = [model_to_col[m] for m in group_names]
            est_h, detail_h = run_vector_ppi_trial(
                X_h,
                pop_h.X,
                group_model_indices=group_idx,
                group_model_names=group_names,
                group_cost=float(sum(pop_h.costs_used[m] for m in group_names)),
                group_time=float(sum(pop_h.times[m] for m in group_names)),
                budget=budget_h,
                alpha=alpha,
                rng=rng,
                device=device,
            )
        elif method.kind == "restrictedmultippi":
            est_h, detail_h = run_multippi_restricted_trial(
                X_h,
                pop_h.X,
                model_names=model_names,
                model_costs=pop_h.costs_used,
                model_times=pop_h.times,
                budget=budget_h,
                covariance_method=covariance_method,
                alpha=alpha,
                rng=rng,
            )
        elif method.kind == "nf1":
            est_h, detail_h = run_ours_trial(
                X_h,
                pop_h,
                budget_h,
                search_mode=method.models[0],
                eps_gap=eps_gap,
                alpha=alpha,
                rng=rng,
                device=device,
            )
        elif method.kind == "nf2":
            est_h, detail_h = run_nf2_trial(
                X_h,
                pop_h,
                budget_h,
                covariance_method=covariance_method,
                top_ridge=ours_top_ridge,
                eps_gap=eps_gap,
                alpha=alpha,
                rng=rng,
                device=device,
            )
        elif method.kind == "nf3":
            est_h, detail_h = run_prefix_trial(
                X_h,
                pop_h,
                budget_h,
                search_mode=method.models[0],
                covariance_method=covariance_method,
                top_ridge=ours_top_ridge,
                prefix_order=method.models[1],
                eps_gap=eps_gap,
                alpha=alpha,
                rng=rng,
                device=device,
            )
        else:
            raise ValueError(method.kind)

        per_stratum_estimates[int(h)] = est_h
        detail["strata"][str(int(h))] = {
            "pi_h": float(pi_h),
            "budget_h": float(budget_h),
            "n_labeled_h": int(X_h.shape[0]),
            "n_population_h": int(pop_h.X.shape[0]),
            "theta_hat_h": float(est_h["theta_hat"]),
            "var_hat_h": float(est_h["var_hat"]),
            "actual_cost_h": float(est_h["actual_cost"]),
            "actual_time_h": float(est_h["actual_time"]),
            "detail": detail_h,
        }

    est = compute_stratified_theta_var(per_stratum_estimates, population.strata_pi, alpha=alpha)
    est["budget_left"] = float(budget - est["actual_cost"])
    return est, detail

def compute_stage1_stats_toplinear(
    X_labeled: np.ndarray,
    model_names: Sequence[str],
    model_costs: Mapping[str, float],
    model_times: Mapping[str, float],
    *,
    covariance_method: str = "ledoitwolf",
    top_ridge: float = 1e-8,
    device: str = "cpu",
) -> Dict[str, Any]:
    y = np.asarray(X_labeled[:, 0], dtype=float)
    Z_all = np.asarray(X_labeled[:, 1 : 1 + len(model_names)], dtype=float)
    mean_y = float(np.mean(y)) if not (device == "cuda" and _HAS_TORCH and torch.cuda.is_available()) else float(_torch_tensor(y, device).mean().detach().cpu().item())
    var_y = float(pop_var_cov_corr_device(y, device=device))
    out: Dict[str, Any] = {"n": int(len(y)), "mean_y": mean_y, "var_y": var_y, "per_model": {}}
    for j, model in enumerate(model_names, start=1):
        z = np.asarray(X_labeled[:, j], dtype=float)
        var_z, cov_yz, corr_yz = pop_var_cov_corr_device(z, y, device=device)
        gamma = 0.0 if var_z <= 1e-15 else cov_yz / var_z
        tau2 = 0.0 if var_z <= 1e-15 else (cov_yz ** 2) / var_z
        tau2 = float(min(max(tau2, 0.0), max(var_y - 1e-12, 0.0)))
        out["per_model"][model] = {
            "var_z": float(var_z),
            "cov_yz": float(cov_yz),
            "corr_yz": float(corr_yz),
            "gamma": float(gamma),
            "tau2": float(tau2),
            "cost": float(model_costs[model]),
            "time": float(model_times[model]),
        }

    if len(model_names) == 0:
        out["top_linear"] = {
            "weights": [],
            "var_u": 0.0,
            "cov_yu": 0.0,
            "gamma": 0.0,
            "tau2": 0.0,
            "all_models_cost": 0.0,
            "all_models_time": 0.0,
            "top_ridge": float(top_ridge),
            "covariance_method": covariance_method,
        }
        return out

    denom = max(len(y) - 1, 1)
    Zc = Z_all - Z_all.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    Sigma_zz = estimate_covariance(Z_all, method=covariance_method)
    Sigma_zy = (Zc.T @ yc) / denom
    solve_mat = _regularize_covariance(Sigma_zz, eps=max(1e-12, float(top_ridge)))
    try:
        weights = np.linalg.solve(solve_mat, Sigma_zy)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(solve_mat) @ Sigma_zy
    u = Z_all @ weights
    var_u = pop_var(u)
    cov_yu = pop_cov(y, u)
    gamma_u = 0.0 if var_u <= 1e-15 else cov_yu / var_u
    tau_u = 0.0 if var_u <= 1e-15 else (cov_yu ** 2) / var_u
    tau_u = float(min(max(tau_u, 0.0), max(var_y - 1e-12, 0.0)))
    out["top_linear"] = {
        "weights": np.asarray(weights, dtype=float).tolist(),
        "var_u": float(var_u),
        "cov_yu": float(cov_yu),
        "gamma": float(gamma_u),
        "tau2": float(tau_u),
        "all_models_cost": float(sum(model_costs[m] for m in model_names)),
        "all_models_time": float(sum(model_times[m] for m in model_names)),
        "top_ridge": float(top_ridge),
        "covariance_method": covariance_method,
    }
    return out

def _top_route_components(selected_models_desc_tau: Sequence[str], stats: Dict[str, Any], eps_gap: float) -> Optional[Dict[str, Any]]:
    top = stats["top_linear"]
    tau_top = float(top["tau2"])
    all_cost = float(top["all_models_cost"])
    all_time = float(top["all_models_time"])
    if len(selected_models_desc_tau) == 0:
        if tau_top <= eps_gap or all_cost <= 0:
            return None
        return {
            "ordered": [],
            "gaps": [tau_top],
            "block_costs": [all_cost],
            "block_times": [all_time],
            "top_extra_cost": all_cost,
            "top_extra_time": all_time,
            "top_shared_with_first": False,
        }
    tau = [float(stats["per_model"][m]["tau2"]) for m in selected_models_desc_tau]
    if tau_top - tau[0] <= eps_gap:
        return None
    for i in range(len(tau) - 1):
        if tau[i] - tau[i + 1] <= eps_gap:
            return None
    if tau[-1] <= eps_gap:
        return None
    raw_suffix_costs = [float(sum(stats["per_model"][selected_models_desc_tau[j]]["cost"] for j in range(i, len(selected_models_desc_tau)))) for i in range(len(selected_models_desc_tau))]
    raw_suffix_times = [float(sum(stats["per_model"][selected_models_desc_tau[j]]["time"] for j in range(i, len(selected_models_desc_tau)))) for i in range(len(selected_models_desc_tau))]
    top_extra_cost = float(all_cost - sum(stats["per_model"][m]["cost"] for m in selected_models_desc_tau))
    top_extra_time = float(all_time - sum(stats["per_model"][m]["time"] for m in selected_models_desc_tau))
    top_shared = top_extra_cost <= 1e-12
    gaps = [max(tau_top - tau[0], 0.0)]
    gaps.extend(max(tau[i] - (tau[i + 1] if i + 1 < len(tau) else 0.0), 0.0) for i in range(len(tau)))
    if top_shared:
        block_costs = [raw_suffix_costs[0]] + raw_suffix_costs
        block_times = [raw_suffix_times[0]] + raw_suffix_times
    else:
        block_costs = [max(top_extra_cost, 0.0)] + raw_suffix_costs
        block_times = [max(top_extra_time, 0.0)] + raw_suffix_times
    return {
        "ordered": list(selected_models_desc_tau),
        "gaps": [float(x) for x in gaps],
        "block_costs": [float(x) for x in block_costs],
        "block_times": [float(x) for x in block_times],
        "top_extra_cost": float(max(top_extra_cost, 0.0)),
        "top_extra_time": float(max(top_extra_time, 0.0)),
        "top_shared_with_first": bool(top_shared),
    }


def _top_route_score(selected_models_desc_tau: Sequence[str], stats: Dict[str, Any], budget: float, n_labeled: int, eps_gap: float) -> float:
    classical_score = float(stats["var_y"]) / max(n_labeled, 1)
    if budget <= 0:
        return classical_score
    comp = _top_route_components(selected_models_desc_tau, stats, eps_gap)
    if comp is None:
        return classical_score
    tau_top = float(stats["top_linear"]["tau2"])
    fixed_term = (float(stats["var_y"]) - tau_top) / max(n_labeled, 1)
    path_sum = float(np.sum(np.sqrt(np.asarray(comp["gaps"], dtype=float) * np.asarray(comp["block_costs"], dtype=float))))
    return min(classical_score, fixed_term + (path_sum ** 2) / max(budget, 1e-12))


def select_active_models_top_exhaustive(model_names: Sequence[str], stats: Dict[str, Any], budget: float, n_labeled: int, eps_gap: float) -> Tuple[bool, List[str], float]:
    import itertools
    best_use_top = False
    best_subset: List[str] = []
    best_score = float(stats["var_y"]) / max(n_labeled, 1)
    score_top_only = _top_route_score([], stats, budget, n_labeled, eps_gap)
    if score_top_only < best_score - 1e-15:
        best_use_top = True
        best_subset = []
        best_score = score_top_only
    for r in range(1, len(model_names) + 1):
        for subset in itertools.combinations(model_names, r):
            ordered = sorted(subset, key=lambda m: (-float(stats["per_model"][m]["tau2"]), m))
            score = _top_route_score(ordered, stats, budget, n_labeled, eps_gap)
            if score < best_score - 1e-15:
                best_use_top = True
                best_subset = list(ordered)
                best_score = score
    return best_use_top, best_subset, float(best_score)

def allocate_top_nested_samples(use_top_linear: bool, active_models_desc_tau: Sequence[str], stats: Dict[str, Any], budget: float, n_labeled: int, eps_gap: float) -> Dict[str, Any]:
    if not use_top_linear:
        return {"use_top_linear": False, "ordered_models": [], "n_labeled": int(n_labeled), "top_total_count": int(n_labeled), "top_increment": 0, "top_shared_with_first": False, "total_counts": {}, "extra_counts": {}, "increments": {}, "actual_cost": 0.0, "actual_time": 0.0, "budget_left": float(budget), "top_extra_cost_per_sample": 0.0, "top_extra_time_per_sample": 0.0, "nmax_extra": 0}
    comp = _top_route_components(active_models_desc_tau, stats, eps_gap)
    if comp is None:
        return {"use_top_linear": False, "ordered_models": [], "n_labeled": int(n_labeled), "top_total_count": int(n_labeled), "top_increment": 0, "top_shared_with_first": False, "total_counts": {}, "extra_counts": {}, "increments": {}, "actual_cost": 0.0, "actual_time": 0.0, "budget_left": float(budget), "top_extra_cost_per_sample": 0.0, "top_extra_time_per_sample": 0.0, "nmax_extra": 0}
    gaps_arr = np.asarray(comp["gaps"], dtype=float)
    block_costs_arr = np.asarray(comp["block_costs"], dtype=float)
    block_times_arr = np.asarray(comp["block_times"], dtype=float)
    increments_arr = np.zeros(len(gaps_arr), dtype=int)
    positive = (gaps_arr > 0) & (block_costs_arr > 0)
    if budget > 0 and positive.any():
        denom = float(np.sum(np.sqrt(gaps_arr[positive] * block_costs_arr[positive])))
        if denom > 0:
            d_cont = np.zeros(len(gaps_arr), dtype=float)
            d_cont[positive] = budget * np.sqrt(gaps_arr[positive] / block_costs_arr[positive]) / denom
            increments_arr = np.floor(d_cont).astype(int)
            spent = float(np.sum(increments_arr * block_costs_arr))
            left = float(budget - spent)
            priorities = np.where(positive, np.sqrt(gaps_arr / np.maximum(block_costs_arr, 1e-12)), -np.inf)
            while left >= np.min(block_costs_arr[positive]) - 1e-12:
                affordable = [idx for idx, c in enumerate(block_costs_arr) if c <= left + 1e-12 and positive[idx]]
                if not affordable:
                    break
                best_idx = max(affordable, key=lambda idx: priorities[idx])
                increments_arr[best_idx] += 1
                left -= block_costs_arr[best_idx]
    ordered = list(active_models_desc_tau)
    top_inc = int(increments_arr[0]) if len(increments_arr) > 0 else 0
    raw_increments = [int(x) for x in increments_arr[1:1+len(ordered)]]
    total_counts: Dict[str, int] = {}
    extra_counts: Dict[str, int] = {}
    increments_dict: Dict[str, int] = {}
    cum = 0
    for j, m in enumerate(ordered):
        cum += raw_increments[j] if j < len(raw_increments) else 0
        total_counts[m] = int(n_labeled + cum)
        extra_counts[m] = int(cum)
        increments_dict[m] = int(raw_increments[j] if j < len(raw_increments) else 0)
    if comp["top_shared_with_first"] and len(ordered) > 0:
        top_total = int(total_counts[ordered[0]])
    else:
        top_total = int(n_labeled + top_inc)
    actual_cost = float(np.sum(increments_arr * block_costs_arr))
    actual_time = float(np.sum(increments_arr * block_times_arr))
    nmax_extra = int(max([top_total - n_labeled] + [extra_counts[m] for m in ordered] + [0]))
    return {
        "use_top_linear": True,
        "ordered_models": ordered,
        "n_labeled": int(n_labeled),
        "top_total_count": int(top_total),
        "top_increment": int(top_inc),
        "top_shared_with_first": bool(comp["top_shared_with_first"]),
        "total_counts": total_counts,
        "extra_counts": extra_counts,
        "increments": increments_dict,
        "actual_cost": actual_cost,
        "actual_time": actual_time,
        "budget_left": float(budget - actual_cost),
        "top_extra_cost_per_sample": float(comp["top_extra_cost"]),
        "top_extra_time_per_sample": float(comp["top_extra_time"]),
        "nmax_extra": int(nmax_extra),
    }

def estimate_top_once_with_ci(
    X_labeled: np.ndarray,
    S_population_matrix: np.ndarray,
    allocation: Dict[str, Any],
    stage1_stats: Dict[str, Any],
    model_to_col: Mapping[str, int],
    *,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    use_top_linear = bool(allocation.get("use_top_linear", False))
    ordered = list(allocation["ordered_models"])
    n0 = int(allocation["n_labeled"])
    zcrit = _normal_z(alpha)
    y_stage1 = np.asarray(X_labeled[:, 0], dtype=float)
    y_mean = float(np.mean(y_stage1))
    var_y = float(stage1_stats["var_y"])
    if not use_top_linear:
        var_hat = var_y / max(n0, 1)
        half = zcrit * math.sqrt(max(var_hat, 0.0))
        return {"theta_hat": y_mean, "var_hat": float(var_hat), "ci_low": y_mean - half, "ci_high": y_mean + half, "width": 2.0 * half}
    nmax_extra = int(allocation.get("nmax_extra", 0))
    idx_extra = rng.integers(0, S_population_matrix.shape[0], size=nmax_extra) if nmax_extra > 0 else np.empty(0, dtype=int)
    Z_extra = np.asarray(S_population_matrix[idx_extra], dtype=float) if nmax_extra > 0 else np.empty((0, S_population_matrix.shape[1]), dtype=float)
    theta_hat = y_mean
    var_hat = var_y / max(n0, 1)
    top = stage1_stats["top_linear"]
    weights = np.asarray(top["weights"], dtype=float)
    gamma_u = float(top["gamma"])
    var_u = float(top["var_u"])
    cov_yu = float(top["cov_yu"])
    u_stage1 = np.asarray(X_labeled[:, 1 : 1 + len(weights)], dtype=float) @ weights
    n_top = int(allocation.get("top_total_count", n0))
    if n_top > n0:
        u_extra = Z_extra @ weights
        mean_prev = float(np.mean(u_stage1))
        mean_cur = float((np.sum(u_stage1) + np.sum(u_extra[: (n_top - n0)])) / max(n_top, 1))
        theta_hat += gamma_u * (mean_cur - mean_prev)
        var_hat += (1.0 / max(n0, 1) - 1.0 / max(n_top, 1)) * (gamma_u ** 2 * var_u - 2.0 * gamma_u * cov_yu)
    # If top is shared with the first lower level, keep prev_total = n0 so the first lower raw level
    # is also updated on its own sample size. Otherwise start from the top level sample size.
    prev_total = n0 if bool(allocation.get("top_shared_with_first", False)) else n_top
    prev_extra = prev_total - n0
    for m in ordered:
        gamma = float(stage1_stats["per_model"][m]["gamma"])
        var_z = float(stage1_stats["per_model"][m]["var_z"])
        cov_yz = float(stage1_stats["per_model"][m]["cov_yz"])
        cur_total = int(allocation["total_counts"][m])
        cur_extra = int(allocation["extra_counts"][m])
        if cur_total <= prev_total:
            continue
        col = int(model_to_col[m]) - 1
        z_stage1 = np.asarray(X_labeled[:, col + 1], dtype=float)
        z_extra = np.asarray(Z_extra[:, col], dtype=float) if cur_extra > 0 else np.empty(0, dtype=float)
        mean_prev = float((np.sum(z_stage1) + np.sum(z_extra[:prev_extra])) / max(prev_total, 1))
        mean_cur = float((np.sum(z_stage1) + np.sum(z_extra[:cur_extra])) / max(cur_total, 1))
        theta_hat += gamma * (mean_cur - mean_prev)
        var_hat += (1.0 / max(prev_total, 1) - 1.0 / max(cur_total, 1)) * (gamma ** 2 * var_z - 2.0 * gamma * cov_yz)
        prev_total = cur_total
        prev_extra = cur_extra
    var_hat = max(float(var_hat), 0.0)
    half = zcrit * math.sqrt(var_hat)
    return {"theta_hat": float(theta_hat), "var_hat": float(var_hat), "ci_low": float(theta_hat - half), "ci_high": float(theta_hat + half), "width": float(2.0 * half)}

def compute_stage1_stats_prefix(
    X_labeled: np.ndarray,
    model_names: Sequence[str],
    model_costs: Mapping[str, float],
    model_times: Mapping[str, float],
    *,
    covariance_method: str = "ledoitwolf",
    top_ridge: float = 1e-8,
    prefix_order: str = "cost",
    device: str = "cpu",
) -> Dict[str, Any]:
    y = np.asarray(X_labeled[:, 0], dtype=float)
    mean_y = float(np.mean(y)) if not (device == "cuda" and _HAS_TORCH and torch.cuda.is_available()) else float(_torch_tensor(y, device).mean().detach().cpu().item())
    var_y = float(pop_var_cov_corr_device(y, device=device))
    out: Dict[str, Any] = {"n": int(len(y)), "mean_y": mean_y, "var_y": var_y, "per_model": {}}

    for j, model in enumerate(model_names, start=1):
        z = np.asarray(X_labeled[:, j], dtype=float)
        var_z, cov_yz, corr_yz = pop_var_cov_corr_device(z, y, device=device)
        gamma = 0.0 if var_z <= 1e-15 else cov_yz / var_z
        tau2 = 0.0 if var_z <= 1e-15 else (cov_yz ** 2) / var_z
        tau2 = float(min(max(tau2, 0.0), max(var_y - 1e-12, 0.0)))
        out["per_model"][model] = {
            "var_z": float(var_z),
            "cov_yz": float(cov_yz),
            "corr_yz": float(corr_yz),
            "gamma": float(gamma),
            "tau2": float(tau2),
            "cost": float(model_costs[model]),
            "time": float(model_times[model]),
        }

    prefix_order = str(prefix_order).lower()
    if prefix_order == "cost":
        base_order = sorted(
            model_names,
            key=lambda m: (
                float(out["per_model"][m]["cost"]),
                -float(out["per_model"][m]["tau2"]),
                m,
            ),
        )
    elif prefix_order == "tau":
        base_order = sorted(
            model_names,
            key=lambda m: (
                -float(out["per_model"][m]["tau2"]),
                float(out["per_model"][m]["cost"]),
                m,
            ),
        )
    else:
        raise ValueError(f"Unsupported prefix_order: {prefix_order}")
    out["prefix_order_basis"] = prefix_order
    out["prefix_base_order"] = list(base_order)
    out["prefix_nodes"] = {}
    out["prefix_tau2"] = {}
    out["prefix_cost"] = {}
    out["prefix_time"] = {}

    if len(base_order) == 0:
        return out

    y_centered = y - y.mean()
    denom = max(len(y) - 1, 1)
    cumulative_cost = 0.0
    cumulative_time = 0.0
    for r in range(1, len(base_order) + 1):
        subset_models = list(base_order[:r])
        subset_cols = [model_names.index(m) + 1 for m in subset_models]
        Z_sub = np.asarray(X_labeled[:, subset_cols], dtype=float)
        Zc = Z_sub - Z_sub.mean(axis=0, keepdims=True)
        Sigma_zz = estimate_covariance(Z_sub, method=covariance_method)
        Sigma_zy = (Zc.T @ y_centered) / denom
        solve_mat = _regularize_covariance(Sigma_zz, eps=max(1e-12, float(top_ridge)))
        try:
            weights = np.linalg.solve(solve_mat, Sigma_zy)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(solve_mat) @ Sigma_zy
        u = Z_sub @ weights
        var_u = pop_var(u)
        cov_yu = pop_cov(y, u)
        gamma_u = 0.0 if var_u <= 1e-15 else cov_yu / var_u
        tau_u = 0.0 if var_u <= 1e-15 else (cov_yu ** 2) / var_u
        tau_u = float(min(max(tau_u, 0.0), max(var_y - 1e-12, 0.0)))
        cumulative_cost += float(model_costs[subset_models[-1]])
        cumulative_time += float(model_times[subset_models[-1]])
        node = {
            "size": int(r),
            "models": subset_models,
            "weights": np.asarray(weights, dtype=float).tolist(),
            "var_u": float(var_u),
            "cov_yu": float(cov_yu),
            "gamma": float(gamma_u),
            "tau2": float(tau_u),
            "cost": float(cumulative_cost),
            "time": float(cumulative_time),
            "ridge": float(top_ridge),
            "covariance_method": covariance_method,
        }
        out["prefix_nodes"][int(r)] = node
        out["prefix_tau2"][int(r)] = float(tau_u)
        out["prefix_cost"][int(r)] = float(cumulative_cost)
        out["prefix_time"][int(r)] = float(cumulative_time)
    return out

def _ours_prefix_route_score_prefix(selected_prefix_sizes_desc: Sequence[int], stats: Dict[str, Any], budget: float, n_labeled: int, eps_gap: float) -> float:
    var_y = float(stats["var_y"])
    classical_score = var_y / max(n_labeled, 1)
    if budget <= 0 or len(selected_prefix_sizes_desc) == 0:
        return classical_score
    route = [int(r) for r in selected_prefix_sizes_desc]
    tau_map = stats["prefix_tau2"]
    cost_map = stats["prefix_cost"]
    tau_first = float(tau_map[route[0]])
    if tau_first <= eps_gap:
        return classical_score
    fixed_term = (var_y - tau_first) / max(n_labeled, 1)
    path_sum = 0.0
    for i, r in enumerate(route):
        tau_cur = float(tau_map[r])
        tau_next = float(tau_map[route[i + 1]]) if i + 1 < len(route) else 0.0
        if tau_cur - tau_next <= eps_gap:
            return classical_score
        c_cur = float(cost_map[r])
        if c_cur <= 0:
            return classical_score
        path_sum += math.sqrt(max(tau_cur - tau_next, 0.0) * c_cur)
    return min(classical_score, fixed_term + (path_sum ** 2) / max(budget, 1e-12))

def select_active_models_prefix_exhaustive(model_names: Sequence[str], stats: Dict[str, Any], budget: float, n_labeled: int, eps_gap: float) -> Tuple[bool, List[int], float]:
    import itertools
    var_y = float(stats["var_y"])
    classical_score = var_y / max(n_labeled, 1)
    if budget <= 0:
        return False, [], float(classical_score)
    sizes = list(range(1, len(stats.get("prefix_base_order", [])) + 1))
    best_route: List[int] = []
    best_score = classical_score
    for r in range(1, len(sizes) + 1):
        for subset in itertools.combinations(sizes, r):
            route = sorted((int(x) for x in subset), reverse=True)
            score = _ours_prefix_route_score_prefix(route, stats, budget, n_labeled, eps_gap)
            if score < best_score - 1e-15:
                best_score = score
                best_route = list(route)
    return bool(best_route), best_route, float(best_score)

def select_active_models_prefix_dag(model_names: Sequence[str], stats: Dict[str, Any], budget: float, n_labeled: int, eps_gap: float) -> Tuple[bool, List[int], float]:
    var_y = float(stats["var_y"])
    classical_score = var_y / max(n_labeled, 1)
    if budget <= 0:
        return False, [], float(classical_score)
    sizes = list(range(1, len(stats.get("prefix_base_order", [])) + 1))
    tau_map = {int(r): float(stats["prefix_tau2"][int(r)]) for r in sizes}
    cost_map = {int(r): float(stats["prefix_cost"][int(r)]) for r in sizes}
    dp: Dict[int, float] = {}
    nxt: Dict[int, Optional[int]] = {}
    for r in sizes:
        tau_r = tau_map[r]
        cost_r = cost_map[r]
        best = float("inf")
        best_next: Optional[int] = None
        if tau_r > eps_gap and cost_r > 0:
            best = math.sqrt(tau_r * cost_r)
        for s in range(1, r):
            tau_s = tau_map[s]
            gap = tau_r - tau_s
            if gap <= eps_gap or not np.isfinite(dp.get(s, np.inf)):
                continue
            cand = math.sqrt(gap * cost_r) + dp[s]
            if cand < best - 1e-15:
                best = cand
                best_next = s
        dp[r] = float(best)
        nxt[r] = best_next

    best_route: List[int] = []
    best_score = classical_score
    for r in sizes:
        tau_r = tau_map[r]
        if tau_r <= eps_gap or not np.isfinite(dp[r]):
            continue
        score = (var_y - tau_r) / max(n_labeled, 1) + (dp[r] ** 2) / max(budget, 1e-12)
        if score < best_score - 1e-15:
            route = [int(r)]
            cur = r
            while nxt[cur] is not None:
                cur = int(nxt[cur])
                route.append(cur)
            best_score = float(score)
            best_route = route
    return bool(best_route), best_route, float(best_score)

def allocate_prefix_nested_samples(use_prefix_linear: bool, active_prefix_sizes_desc: Sequence[int], stats: Dict[str, Any], budget: float, n_labeled: int) -> Dict[str, Any]:
    route = [int(r) for r in active_prefix_sizes_desc]
    if not use_prefix_linear or len(route) == 0:
        return {
            "use_prefix_linear": False,
            "route_prefix_sizes": [],
            "n_labeled": int(n_labeled),
            "total_counts": {},
            "extra_counts": {},
            "increments": {},
            "actual_cost": 0.0,
            "actual_time": 0.0,
            "budget_left": float(budget),
            "nmax_extra": 0,
        }

    tau_map = stats["prefix_tau2"]
    cost_map = stats["prefix_cost"]
    time_map = stats["prefix_time"]
    gaps = []
    block_costs = []
    block_times = []
    for i, r in enumerate(route):
        tau_cur = float(tau_map[r])
        tau_next = float(tau_map[route[i + 1]]) if i + 1 < len(route) else 0.0
        gaps.append(max(tau_cur - tau_next, 0.0))
        block_costs.append(float(cost_map[r]))
        block_times.append(float(time_map[r]))

    gaps_arr = np.asarray(gaps, dtype=float)
    block_costs_arr = np.asarray(block_costs, dtype=float)
    increments_arr = np.zeros(len(route), dtype=int)
    positive = (gaps_arr > 0) & (block_costs_arr > 0)
    if budget > 0 and positive.any():
        denom = float(np.sum(np.sqrt(gaps_arr[positive] * block_costs_arr[positive])))
        if denom > 0:
            d_cont = np.zeros(len(route), dtype=float)
            d_cont[positive] = budget * np.sqrt(gaps_arr[positive] / block_costs_arr[positive]) / denom
            increments_arr = np.floor(d_cont).astype(int)
            spent = float(np.sum(increments_arr * block_costs_arr))
            left = float(budget - spent)
            priorities = np.where(positive, np.sqrt(gaps_arr / np.maximum(block_costs_arr, 1e-12)), -np.inf)
            while left >= np.min(block_costs_arr[positive]) - 1e-12:
                affordable = [idx for idx, c in enumerate(block_costs_arr) if c <= left + 1e-12 and positive[idx]]
                if not affordable:
                    break
                best_idx = max(affordable, key=lambda idx: priorities[idx])
                increments_arr[best_idx] += 1
                left -= block_costs_arr[best_idx]

    increments_dict: Dict[int, int] = {}
    total_counts: Dict[int, int] = {}
    extra_counts: Dict[int, int] = {}
    cum = 0
    for i, r in enumerate(route):
        inc = int(increments_arr[i])
        cum += inc
        increments_dict[int(r)] = inc
        total_counts[int(r)] = int(n_labeled + cum)
        extra_counts[int(r)] = int(cum)

    actual_cost = float(np.sum(increments_arr * block_costs_arr))
    actual_time = float(np.sum(increments_arr * np.asarray(block_times, dtype=float)))
    return {
        "use_prefix_linear": True,
        "route_prefix_sizes": route,
        "n_labeled": int(n_labeled),
        "total_counts": total_counts,
        "extra_counts": extra_counts,
        "increments": increments_dict,
        "actual_cost": actual_cost,
        "actual_time": actual_time,
        "budget_left": float(budget - actual_cost),
        "nmax_extra": int(cum),
    }

def estimate_prefix_once_with_ci(
    X_labeled: np.ndarray,
    S_population_matrix: np.ndarray,
    allocation: Dict[str, Any],
    stage1_stats: Dict[str, Any],
    model_to_col: Mapping[str, int],
    *,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    use_prefix_linear = bool(allocation.get("use_prefix_linear", False))
    route = [int(r) for r in allocation.get("route_prefix_sizes", [])]
    n0 = int(allocation["n_labeled"])
    zcrit = _normal_z(alpha)
    y_stage1 = np.asarray(X_labeled[:, 0], dtype=float)
    y_mean = float(np.mean(y_stage1))
    var_y = float(stage1_stats["var_y"])
    if not use_prefix_linear or len(route) == 0:
        var_hat = var_y / max(n0, 1)
        half = zcrit * math.sqrt(max(var_hat, 0.0))
        return {"theta_hat": y_mean, "var_hat": float(var_hat), "ci_low": y_mean - half, "ci_high": y_mean + half, "width": 2.0 * half}

    nmax_extra = int(allocation.get("nmax_extra", 0))
    idx_extra = rng.integers(0, S_population_matrix.shape[0], size=nmax_extra) if nmax_extra > 0 else np.empty(0, dtype=int)
    Z_extra = np.asarray(S_population_matrix[idx_extra], dtype=float) if nmax_extra > 0 else np.empty((0, S_population_matrix.shape[1]), dtype=float)

    theta_hat = y_mean
    var_hat = var_y / max(n0, 1)
    prev_total = n0
    prev_extra = 0
    stage1_sum_cache: Dict[int, float] = {}
    extra_prefix_cache: Dict[int, np.ndarray] = {}

    for r in route:
        cur_total = int(allocation["total_counts"][r])
        cur_extra = int(allocation["extra_counts"][r])
        if cur_total <= prev_total:
            continue
        node = stage1_stats["prefix_nodes"][int(r)]
        models = list(node["models"])
        weights = np.asarray(node["weights"], dtype=float)
        gamma = float(node["gamma"])
        var_u = float(node["var_u"])
        cov_yu = float(node["cov_yu"])
        if int(r) not in stage1_sum_cache:
            lab_cols = [int(model_to_col[m]) for m in models]
            u_stage1 = np.asarray(X_labeled[:, lab_cols], dtype=float) @ weights
            stage1_sum_cache[int(r)] = float(np.sum(u_stage1))
        if int(r) not in extra_prefix_cache:
            extra_cols = [int(model_to_col[m]) - 1 for m in models]
            extra_prefix_cache[int(r)] = np.asarray(Z_extra[:, extra_cols], dtype=float) @ weights if nmax_extra > 0 else np.empty(0, dtype=float)
        u_extra = extra_prefix_cache[int(r)]
        prev_sum = stage1_sum_cache[int(r)] + (float(np.sum(u_extra[:prev_extra])) if prev_extra > 0 else 0.0)
        cur_sum = stage1_sum_cache[int(r)] + (float(np.sum(u_extra[:cur_extra])) if cur_extra > 0 else 0.0)
        mean_prev = prev_sum / max(prev_total, 1)
        mean_cur = cur_sum / max(cur_total, 1)
        theta_hat += gamma * (mean_cur - mean_prev)
        var_hat += (1.0 / max(prev_total, 1) - 1.0 / max(cur_total, 1)) * (gamma ** 2 * var_u - 2.0 * gamma * cov_yu)
        prev_total = cur_total
        prev_extra = cur_extra

    var_hat = max(float(var_hat), 0.0)
    half = zcrit * math.sqrt(var_hat)
    return {"theta_hat": float(theta_hat), "var_hat": float(var_hat), "ci_low": float(theta_hat - half), "ci_high": float(theta_hat + half), "width": float(2.0 * half)}

def run_prefix_trial(
    X_labeled: np.ndarray,
    population: EmpiricalPopulation,
    budget: float,
    *,
    search_mode: str,
    covariance_method: str,
    top_ridge: float,
    prefix_order: str,
    eps_gap: float,
    alpha: float = 0.05,
    rng: np.random.Generator,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_names = list(population.model_names)
    stage1_stats = compute_stage1_stats_prefix(
        X_labeled,
        model_names,
        population.costs_used,
        population.times,
        covariance_method=covariance_method,
        top_ridge=top_ridge,
        prefix_order=prefix_order,
        device=device,
    )
    n_labeled = X_labeled.shape[0]
    if search_mode == "exhaustive":
        use_prefix_linear, selected, route_obj = select_active_models_prefix_exhaustive(model_names, stage1_stats, float(budget), n_labeled, eps_gap)
    elif search_mode == "dag":
        use_prefix_linear, selected, route_obj = select_active_models_prefix_dag(model_names, stage1_stats, float(budget), n_labeled, eps_gap)
    else:
        raise ValueError(search_mode)
    allocation = allocate_prefix_nested_samples(use_prefix_linear, selected, stage1_stats, float(budget), n_labeled)
    model_to_col = {m: i + 1 for i, m in enumerate(model_names)}
    est = estimate_prefix_once_with_ci(X_labeled, population.S, allocation, stage1_stats, model_to_col, alpha=alpha, rng=rng)
    est.update({"actual_cost": float(allocation["actual_cost"]), "actual_time": float(allocation["actual_time"]), "budget_left": float(allocation["budget_left"])})
    detail = {
        "search_mode": search_mode,
        "epsilon_gap": float(eps_gap),
        "route_objective": float(route_obj),
        "use_prefix_linear": bool(use_prefix_linear),
        "prefix_base_order": list(stage1_stats.get("prefix_base_order", [])),
        "prefix_order_basis": str(stage1_stats.get("prefix_order_basis", prefix_order)),
        "prefix_top_ridge": float(top_ridge),
        "selected_prefix_sizes": list(selected),
        "selected_prefix_models": {int(r): list(stage1_stats["prefix_nodes"][int(r)]["models"]) for r in selected},
        "selected_prefix_tau2": {int(r): float(stage1_stats["prefix_nodes"][int(r)]["tau2"]) for r in selected},
        "selected_prefix_gamma": {int(r): float(stage1_stats["prefix_nodes"][int(r)]["gamma"]) for r in selected},
        "all_prefix_tau2": {int(r): float(stage1_stats["prefix_nodes"][int(r)]["tau2"]) for r in stage1_stats.get("prefix_nodes", {})},
        "all_prefix_cost": {int(r): float(stage1_stats["prefix_nodes"][int(r)]["cost"]) for r in stage1_stats.get("prefix_nodes", {})},
        "all_prefix_models": {int(r): list(stage1_stats["prefix_nodes"][int(r)]["models"]) for r in stage1_stats.get("prefix_nodes", {})},
        "total_counts": allocation["total_counts"],
        "extra_counts": allocation["extra_counts"],
        "increments": allocation["increments"],
    }
    return est, detail

# ============================================================
# Unified method menu and trial runner
# ============================================================

def make_methods() -> List[MethodSpec]:
    # Keep only Classical, VectorPPI++, MultiPPI, and the first OMPPI construction.
    # NF-II and NF-III are no longer run in this experiment.
    return [
        MethodSpec("Classical", "classical"),
        MethodSpec("VectorPPI++", "vector_ppi"),
        MethodSpec("MultiPPI", "restrictedmultippi"),
        MethodSpec("OMPPI(Exhaustive)", "nf1", ["exhaustive"]),
        MethodSpec("OMPPI(DAG)", "nf1", ["dag"]),
    ]


def run_nf2_trial(
    X_labeled: np.ndarray,
    population: EmpiricalPopulation,
    budget: float,
    *,
    covariance_method: str,
    top_ridge: float,
    eps_gap: float,
    alpha: float,
    rng: np.random.Generator,
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_names = list(population.model_names)
    stage1_stats = compute_stage1_stats_toplinear(
        X_labeled,
        model_names,
        population.costs_used,
        population.times,
        covariance_method=covariance_method,
        top_ridge=top_ridge,
        device=device,
    )
    n_labeled = X_labeled.shape[0]
    use_top, selected, route_obj = select_active_models_top_exhaustive(
        model_names, stage1_stats, float(budget), n_labeled, eps_gap
    )
    allocation = allocate_top_nested_samples(
        use_top, selected, stage1_stats, float(budget), n_labeled, eps_gap
    )
    model_to_col = {m: i + 1 for i, m in enumerate(model_names)}
    est = estimate_top_once_with_ci(
        X_labeled,
        population.S,
        allocation,
        stage1_stats,
        model_to_col,
        alpha=alpha,
        rng=rng,
    )
    est.update(
        {
            "actual_cost": float(allocation["actual_cost"]),
            "actual_time": float(allocation["actual_time"]),
            "budget_left": float(allocation["budget_left"]),
        }
    )
    detail = {
        "construction": "IIexact",
        "search_mode": "exhaustive",
        "route_objective": float(route_obj),
        "use_top_linear": bool(use_top),
        "selected_models": list(selected),
        "top_linear_tau2": float(stage1_stats["top_linear"]["tau2"]),
        "top_linear_weights": stage1_stats["top_linear"]["weights"],
        "top_total_count": int(allocation.get("top_total_count", n_labeled)),
        "top_increment": int(allocation.get("top_increment", 0)),
        "top_shared_with_first": bool(allocation.get("top_shared_with_first", False)),
        "top_extra_cost_per_sample": float(allocation.get("top_extra_cost_per_sample", 0.0)),
        "selected_tau2": {m: float(stage1_stats["per_model"][m]["tau2"]) for m in selected},
        "total_counts": allocation["total_counts"],
        "extra_counts": allocation["extra_counts"],
        "increments": allocation["increments"],
    }
    return est, detail


def run_one_trial(
    trial_id: int,
    population: EmpiricalPopulation,
    budgets: Sequence[float],
    n_labeled: int,
    methods: Sequence[MethodSpec],
    covariance_method: str,
    ours_top_ridge: float,
    base_seed: int,
    alpha: float,
    eps_gap: float,
    device: str,
    *,
    theta_true: Optional[float] = None,
    outer_trial: int = 0,
    trial_id_offset: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    local_seed = seed_for_trial(base_seed, trial_id_offset + trial_id)
    set_deterministic(local_seed)
    rng = np.random.default_rng(local_seed)
    X_labeled, H_labeled, labeled_idx, labeled_counts = sample_labeled_rows_stratified(population, n_labeled, rng)
    theta_true_used = float(population.theta_true if theta_true is None else theta_true)
    global_trial_id = int(trial_id_offset + trial_id)
    trial_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    for budget in budgets:
        budget = float(budget)
        classical_est = classical_detail = None
        if budget <= 0:
            classical_est, classical_detail = run_stratified_method_trial(
                MethodSpec("classical", "classical"),
                X_labeled,
                H_labeled,
                population,
                budget,
                covariance_method=covariance_method,
                ours_top_ridge=ours_top_ridge,
                eps_gap=eps_gap,
                alpha=alpha,
                rng=rng,
                device=device,
            )
        for method in methods:
            start = time.perf_counter()
            if budget <= 0:
                est, detail = classical_est, classical_detail
            else:
                est, detail = run_stratified_method_trial(
                    method,
                    X_labeled,
                    H_labeled,
                    population,
                    budget,
                    covariance_method=covariance_method,
                    ours_top_ridge=ours_top_ridge,
                    eps_gap=eps_gap,
                    alpha=alpha,
                    rng=rng,
                    device=device,
                )
            algo_compute_time_sec = time.perf_counter() - start
            theta_hat = float(est["theta_hat"])
            trial_rows.append(
                {
                    "trial": global_trial_id,
                    "outer_trial": int(outer_trial),
                    "inner_trial": int(trial_id),
                    "method": method.name,
                    "budget": budget,
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
                    "actual_time": float(est["actual_time"]),
                    "budget_left": float(est["budget_left"]) if pd.notna(est["budget_left"]) else np.nan,
                    "algo_compute_time_sec": float(algo_compute_time_sec),
                    "n_labeled": int(n_labeled),
                    "covariance_method": covariance_method,
                    "label_mode": population.label_mode,
                    "epsilon_gap": float(eps_gap),
                    "device_used": device,
                    "num_strata": int(population.num_strata),
                }
            )
            detail_rows.append(
                {
                    "trial": global_trial_id,
                    "outer_trial": int(outer_trial),
                    "inner_trial": int(trial_id),
                    "method": method.name,
                    "budget": budget,
                    "detail_json": json.dumps({
                        "labeled_idx_json": labeled_idx.tolist(),
                        "labeled_counts": {str(int(h)): int(labeled_counts[h]) for h in sorted(labeled_counts)},
                        "detail": detail,
                    }, ensure_ascii=False, sort_keys=True),
                }
            )
    return trial_rows, detail_rows


def _worker_run_trials(args):
    trial_ids, population, budgets, n_labeled, methods, covariance_method, ours_top_ridge, base_seed, alpha, eps_gap, device, theta_true, outer_trial, trial_id_offset = args
    all_trial_rows: List[Dict[str, Any]] = []
    all_detail_rows: List[Dict[str, Any]] = []
    for trial_id in trial_ids:
        tr, dr = run_one_trial(
            trial_id,
            population,
            budgets,
            n_labeled,
            methods,
            covariance_method,
            ours_top_ridge,
            base_seed,
            alpha,
            eps_gap,
            device,
            theta_true=theta_true,
            outer_trial=outer_trial,
            trial_id_offset=trial_id_offset,
        )
        all_trial_rows.extend(tr)
        all_detail_rows.extend(dr)
    return all_trial_rows, all_detail_rows


def run_outer_inner_truth_experiment(
    population: EmpiricalPopulation,
    *,
    budgets: Sequence[float],
    n_labeled: int,
    n_outer_trials: int,
    n_inner_trials: int,
    theta_truth_size: int,
    covariance_method: str,
    ours_top_ridge: float,
    methods: Sequence[MethodSpec],
    seed: int,
    alpha: float,
    eps_gap: float,
    device: str,
    num_workers: int,
) -> Dict[str, pd.DataFrame]:
    outer_truth_rows = []
    all_trial_dfs = []
    all_detail_dfs = []
    for outer_trial in range(n_outer_trials):
        outer_seed = seed_for_trial(seed, 10_000_000 + outer_trial)
        rng_outer = np.random.default_rng(outer_seed)
        truth_idx, theta_true_outer, truth_counts = sample_theta_truth_subset_stratified(population, theta_truth_size, rng_outer)
        outer_truth_rows.append({
            "outer_trial": int(outer_trial),
            "outer_seed": int(outer_seed),
            "theta_truth_size": int(theta_truth_size),
            "theta_true": float(theta_true_outer),
            "truth_subset_idx_json": json.dumps(truth_idx.tolist(), ensure_ascii=False, sort_keys=True),
            "truth_counts_json": json.dumps({str(int(h)): int(truth_counts[h]) for h in sorted(truth_counts)}, ensure_ascii=False, sort_keys=True),
        })
        if num_workers <= 1:
            trial_rows = []
            detail_rows = []
            for trial_id in range(n_inner_trials):
                tr, dr = run_one_trial(
                    trial_id, population, budgets, n_labeled, methods, covariance_method,
                    ours_top_ridge, outer_seed, alpha, eps_gap, device,
                    theta_true=theta_true_outer, outer_trial=outer_trial,
                    trial_id_offset=outer_trial * n_inner_trials,
                )
                trial_rows.extend(tr)
                detail_rows.extend(dr)
        else:
            chunks = split_trials(n_inner_trials, num_workers)
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            worker_args = [
                (chunk, population, budgets, n_labeled, methods, covariance_method, ours_top_ridge, outer_seed, alpha, eps_gap, device, theta_true_outer, outer_trial, outer_trial * n_inner_trials)
                for chunk in chunks
            ]
            trial_rows = []
            detail_rows = []
            with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as ex:
                for tr, dr in ex.map(_worker_run_trials, worker_args):
                    trial_rows.extend(tr)
                    detail_rows.extend(dr)
            trial_rows = sorted(trial_rows, key=lambda r: (r["trial"], r["budget"], r["method"]))
            detail_rows = sorted(detail_rows, key=lambda r: (r["trial"], r["budget"], r["method"]))
        all_trial_dfs.append(pd.DataFrame(trial_rows))
        all_detail_dfs.append(pd.DataFrame(detail_rows))
    trial_df = pd.concat(all_trial_dfs, ignore_index=True)
    details_df = pd.concat(all_detail_dfs, ignore_index=True)
    summary_df = _summarize_trial_df(trial_df)
    outer_truth_df = pd.DataFrame(outer_truth_rows)
    return {"summary_df": summary_df, "trial_df": trial_df, "details_df": details_df, "outer_truth_df": outer_truth_df}


# ============================================================
# Plotting
# ============================================================
# Plotting
# ============================================================

def smooth_series(y: np.ndarray, window: int = 3) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if window <= 1 or y.size < window:
        return y.copy()
    return pd.Series(y).rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def configure_plot_style(font_path: Optional[str] = None) -> None:
    if font_path:
        fp = Path(font_path)
        if fp.exists():
            font_manager.fontManager.addfont(str(fp))
            try:
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(fp)).get_name()
            except Exception:
                pass
    plt.style.use("ggplot")
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["grid.color"] = "#D0D0D0"
    plt.rcParams["grid.linewidth"] = 1.0
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11


def pretty_method_name(name: str) -> str:
    mapping = {
        "classical": "Classical",
        "Classical": "Classical",
        "ppi_vector__all_models": "VectorPPI++",
        "VectorPPI++": "VectorPPI++",
        "restrictedmultippi": "MultiPPI",
        "MultiPPI": "MultiPPI",
        "nf1_exh": "OMPPI(Exhaustive)",
        "OMPPI(Exhaustive)": "OMPPI(Exhaustive)",
        "nf1_dag": "OMPPI(DAG)",
        "OMPPI(DAG)": "OMPPI(DAG)",
    }
    return mapping.get(name, name)


def make_comparison_figure(summary_df: pd.DataFrame, *, font_path: Optional[str] = None, smooth_window: int = 3) -> plt.Figure:
    configure_plot_style(font_path=font_path)
    methods = ["Classical", "VectorPPI++", "MultiPPI", "OMPPI(Exhaustive)", "OMPPI(DAG)"]
    colors = {
        "Classical": "#7f7f7f",
        "VectorPPI++": "#e41a1c",
        "MultiPPI": "#2ca02c",
        "OMPPI(Exhaustive)": "#1f77b4",
        "OMPPI(DAG)": "#1f77b4",
    }
    styles = {
        "Classical": dict(lw=1.8, ls=":", marker=None, alpha=0.95),
        "VectorPPI++": dict(lw=2.0, ls="-", marker=None, alpha=0.95),
        "MultiPPI": dict(lw=2.0, ls="-", marker=None, alpha=0.95),
        "OMPPI(Exhaustive)": dict(lw=2.0, ls="-", marker=None, alpha=0.95),
        "OMPPI(DAG)": dict(lw=0, ls="None", marker="o", ms=3.0, alpha=0.95),
    }

    plot_df = summary_df.copy()
    classical_df = plot_df.loc[plot_df["method"] == "Classical"].copy()
    if classical_df.empty:
        raise ValueError("summary_df must contain the classical method for shared budget=0 anchoring.")
    if (classical_df["budget"] == 0).any():
        anchor = classical_df.loc[classical_df["budget"] == 0].iloc[0].copy()
    else:
        anchor = classical_df.loc[classical_df["budget"].idxmin()].copy()
        zero_rows = []
        for m in methods:
            row = anchor.copy()
            row["method"] = m
            row["budget"] = 0.0
            zero_rows.append(row)
        plot_df = pd.concat([pd.DataFrame(zero_rows), plot_df], ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    metrics = [("coverage", "Coverage"), ("rmse", "RMSE"), ("mean_ci_width", "Mean CI Width")]
    rmse_vals = plot_df["rmse"].to_numpy(dtype=float)
    width_vals = plot_df["mean_ci_width"].to_numpy(dtype=float)
    rmse_lo, rmse_hi = float(rmse_vals.min() - 0.00050), float(rmse_vals.max() + 0.00050)
    width_lo, width_hi = float(width_vals.min() - 0.00018), float(width_vals.max() + 0.00018)
    anchor_values = {
        "coverage": float(anchor["coverage"]),
        "rmse": float(anchor["rmse"]),
        "mean_ci_width": float(anchor["mean_ci_width"]),
    }
    handles = []
    labels = []
    for ax, (metric, title) in zip(axes, metrics):
        for method in methods:
            g = plot_df.loc[plot_df["method"] == method].sort_values("budget")
            if g.empty:
                continue
            x = g["budget"].to_numpy(dtype=float)
            y = np.array(smooth_series(g[metric].to_numpy(dtype=float), window=smooth_window), copy=True)
            zero_idx = np.where(x == 0)[0]
            if len(zero_idx) > 0:
                y[zero_idx[0]] = anchor_values[metric]
            line, = ax.plot(x, y, color=colors[method], label=pretty_method_name(method), **styles[method])
            if metric == "coverage" and method not in labels:
                handles.append(line)
                labels.append(method)
        ax.set_title(title, pad=4, color="black")
        ax.set_xlabel("Budget", color="black")
        ax.tick_params(axis="both", colors="black")
        if metric == "coverage":
            ax.set_ylim(0.85, 1.00)
            ax.axhline(0.95, color="gray", lw=1.5, ls=":")
        elif metric == "rmse":
            ax.set_ylim(rmse_lo, rmse_hi)
        else:
            ax.set_ylim(width_lo, width_hi)
        ax.grid(True, alpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.legend(handles, [pretty_method_name(m) for m in labels], loc="center left", bbox_to_anchor=(0.85, 0.5), frameon=False)
    fig.subplots_adjust(left=0.06, right=0.83, wspace=0.34, bottom=0.18, top=0.84)
    return fig


def _safe_json_loads(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    try:
        return json.loads(str(x))
    except Exception:
        return {}


def _allocation_cost_for_source(source: str, costs_used: Mapping[str, float]) -> Tuple[str, float]:
    if source == "joint_all_surrogates":
        return "joint", float(sum(costs_used.values()))
    if source.startswith("single__"):
        model = source.split("single__", 1)[1]
        return "single", float(costs_used.get(model, 0.0))
    return "single", float(costs_used.get(source, 0.0))


def make_compact_allocation_summary(
    details_df: pd.DataFrame,
    *,
    model_names: Sequence[str],
    costs_used: Mapping[str, float],
) -> pd.DataFrame:
    """Aggregate method details into a compact allocation summary.

    This file is enough for plotting OMPPI/MultiPPI query allocation and
    cost allocation by stratum, without storing the huge method_details.csv.
    """
    target_methods = {"OMPPI(Exhaustive)", "OMPPI(DAG)", "MultiPPI"}
    model_set = set(str(x) for x in model_names)

    required = {"method", "budget", "detail_json"}
    missing = required - set(details_df.columns)
    if missing:
        raise ValueError(f"details_df is missing columns: {sorted(missing)}")

    accum: Dict[Tuple[str, float, int, str, str], Dict[str, float]] = {}
    denom: Dict[Tuple[str, float, int], float] = {}

    for method, budget, detail_json in details_df[["method", "budget", "detail_json"]].itertuples(index=False, name=None):
        method = str(method)
        if method not in target_methods:
            continue

        budget = float(budget)
        obj = _safe_json_loads(detail_json)
        detail = obj.get("detail", {})
        strata = detail.get("strata", None)
        if not isinstance(strata, dict):
            strata = {"0": {"detail": detail}}

        for h_str, h_obj in strata.items():
            try:
                h = int(h_str)
            except Exception:
                h = 0

            denom_key = (method, budget, h)
            denom[denom_key] = denom.get(denom_key, 0.0) + 1.0

            if not isinstance(h_obj, dict):
                continue
            d = h_obj.get("detail", {})
            if not isinstance(d, dict):
                continue

            if method.startswith("OMPPI"):
                selected = set(str(x) for x in d.get("selected_models", []) if str(x) in model_set)
                extra_counts = d.get("extra_counts", {})
                if not isinstance(extra_counts, dict):
                    extra_counts = {}

                sources = set(selected) | set(str(x) for x in extra_counts.keys())
                for source in sources:
                    if source not in model_set:
                        continue
                    query_count = float(extra_counts.get(source, 0.0) or 0.0)
                    source_type, cost_per_query = _allocation_cost_for_source(source, costs_used)
                    key = (method, budget, h, source_type, source)
                    rec = accum.setdefault(
                        key,
                        {"query_total": 0.0, "cost_total": 0.0, "selected_total": 0.0},
                    )
                    rec["query_total"] += query_count
                    rec["cost_total"] += query_count * cost_per_query
                    rec["selected_total"] += 1.0 if source in selected else 0.0

            elif method == "MultiPPI":
                counts = d.get("counts", {})
                if not isinstance(counts, dict):
                    counts = {}

                for source, count in counts.items():
                    source = str(source)
                    if source == "full_labeled":
                        continue
                    query_count = float(count or 0.0)
                    source_type, cost_per_query = _allocation_cost_for_source(source, costs_used)
                    key = (method, budget, h, source_type, source)
                    rec = accum.setdefault(
                        key,
                        {"query_total": 0.0, "cost_total": 0.0, "selected_total": 0.0},
                    )
                    rec["query_total"] += query_count
                    rec["cost_total"] += query_count * cost_per_query
                    rec["selected_total"] += 1.0 if query_count > 0 else 0.0

    rows: List[Dict[str, Any]] = []
    for (method, budget, h, source_type, source), rec in sorted(accum.items()):
        n_detail_rows = float(denom.get((method, budget, h), 0.0))
        rows.append(
            {
                "method": method,
                "budget": float(budget),
                "stratum": int(h),
                "source_type": source_type,
                "source": source,
                "n_detail_rows": n_detail_rows,
                "query_total": float(rec["query_total"]),
                "cost_total": float(rec["cost_total"]),
                "selected_total": float(rec["selected_total"]),
                "query_mean": float(rec["query_total"] / n_detail_rows) if n_detail_rows > 0 else np.nan,
                "cost_mean": float(rec["cost_total"] / n_detail_rows) if n_detail_rows > 0 else np.nan,
                "selected_frequency": float(rec["selected_total"] / n_detail_rows) if n_detail_rows > 0 else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    query_den = out.groupby(["method", "budget", "stratum"])["query_total"].transform("sum")
    cost_den = out.groupby(["method", "budget", "stratum"])["cost_total"].transform("sum")
    out["query_share"] = np.where(query_den > 0, out["query_total"] / query_den, 0.0)
    out["cost_share"] = np.where(cost_den > 0, out["cost_total"] / cost_den, 0.0)
    return out.sort_values(["method", "budget", "stratum", "source_type", "source"]).reset_index(drop=True)


def make_detail_file_note(path: Path, detail_save_mode: str) -> None:
    text = (
        "method_details.csv was intentionally not saved in this run because it is very large.\n"
        f"detail_save_mode = {detail_save_mode}\n"
        "Use allocation_summary.csv for OMPPI/MultiPPI sample-allocation and cost-allocation plots.\n"
        "Use summary.csv for coverage, RMSE, and CI-width plots.\n"
    )
    path.write_text(text, encoding="utf-8")

def _save_text_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone stratified comparison for VectorPPI++, MultiPPI, and OMPPI under two-stage theta truth.")
    parser.add_argument("--final-table", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="./llm_preferences_compare_outputs_mppi")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--label-mode", type=str, default="drop_ties", choices=["drop_ties", "half_ties"])
    parser.add_argument("--n-labeled", type=int, default=150)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-outer-trials", type=int, default=10)
    parser.add_argument("--theta-truth-size", type=int, default=600)
    parser.add_argument("--budgets", type=str, default="0:1200:13")
    parser.add_argument("--covariance-method", type=str, default="ledoitwolf")
    parser.add_argument("--ours-top-ridge", type=float, default=1e-6)
    parser.add_argument("--eps-gap", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--font-path", type=str, default=None)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--num-strata", type=int, default=5)
    parser.add_argument("--prompt-col", type=str, default=None)
    parser.add_argument(
        "--detail-save-mode",
        type=str,
        default="compact",
        choices=["compact", "full", "gzip", "none"],
        help=(
            "How to save method-level details. "
            "compact saves allocation_summary.csv only; "
            "full saves the original large method_details.csv; "
            "gzip saves method_details.csv.gz; none saves no detail file."
        ),
    )
    parser.add_argument("--save", type=str, default="yes", choices=["yes", "no"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_df = load_final_table(args.final_table)
    model_names = parse_models(args.models, final_df)
    population = build_empirical_population_from_final_df(
        final_df,
        model_names,
        label_mode=args.label_mode,
        normalize_costs=True,
        num_strata=args.num_strata,
        prompt_col_override=args.prompt_col,
    )
    budgets = parse_budgets(args.budgets)
    methods = make_methods()
    device = resolve_device(args.device)

    print(f"Rows used after label_mode={args.label_mode}: {population.X.shape[0]}")
    print(f"Full empirical mean over all rows (not used directly as theta_true): {population.theta_true:.6f}")
    print(f"Outer theta_true construction: {args.n_outer_trials} draws, subset size={args.theta_truth_size}")
    print(f"Inner inference repetitions per outer draw: {args.n_trials}")
    print(f"Methods: {[m.name for m in methods]}")
    print(f"Device requested={args.device}; resolved={device}; num_workers={args.num_workers}")
    print(f"Prompt column used for strata: {population.prompt_col}")
    print(f"Number of strata: {population.num_strata}")
    print(f"Strata pi_h: {population.strata_pi}")
    print("RMSE = root mean squared error of theta_hat around outer-draw theta_true.")

    results = run_outer_inner_truth_experiment(
        population,
        budgets=budgets,
        n_labeled=args.n_labeled,
        n_outer_trials=args.n_outer_trials,
        n_inner_trials=args.n_trials,
        theta_truth_size=args.theta_truth_size,
        covariance_method=args.covariance_method,
        ours_top_ridge=args.ours_top_ridge,
        methods=methods,
        seed=args.seed,
        alpha=args.alpha,
        eps_gap=args.eps_gap,
        device=device,
        num_workers=args.num_workers,
    )
    summary_df, trial_df, details_df, outer_truth_df = results["summary_df"], results["trial_df"], results["details_df"], results["outer_truth_df"]
    if args.save == "yes":
        summary_df.to_csv(out_dir / "summary.csv", index=False)
        trial_df.to_csv(out_dir / "trials.csv", index=False)

        if args.detail_save_mode == "full":
            details_df.to_csv(out_dir / "method_details.csv", index=False)
        elif args.detail_save_mode == "gzip":
            details_df.to_csv(out_dir / "method_details.csv.gz", index=False, compression="gzip")
        elif args.detail_save_mode == "compact":
            allocation_summary_df = make_compact_allocation_summary(
                details_df,
                model_names=model_names,
                costs_used=population.costs_used,
            )
            allocation_summary_df.to_csv(out_dir / "allocation_summary.csv", index=False)
            make_detail_file_note(out_dir / "method_details_NOT_SAVED.txt", args.detail_save_mode)
        elif args.detail_save_mode == "none":
            make_detail_file_note(out_dir / "method_details_NOT_SAVED.txt", args.detail_save_mode)
        else:
            raise ValueError(f"Unknown detail_save_mode: {args.detail_save_mode}")

        outer_truth_df.to_csv(out_dir / "outer_truth.csv", index=False)
        _save_text_json(out_dir / "config.json", {
            "final_table": args.final_table,
            "models": model_names,
            "label_mode": args.label_mode,
            "n_labeled": args.n_labeled,
            "n_trials": args.n_trials,
            "n_outer_trials": args.n_outer_trials,
            "theta_truth_size": args.theta_truth_size,
            "budgets": budgets,
            "covariance_method": args.covariance_method,
            "ours_top_ridge": args.ours_top_ridge,
            "eps_gap": args.eps_gap,
            "device_requested": args.device,
            "device_used": device,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "alpha": args.alpha,
            "full_empirical_theta_true": population.theta_true,
            "num_strata": population.num_strata,
            "prompt_col": population.prompt_col,
            "detail_save_mode": args.detail_save_mode,
            "strata_pi": population.strata_pi,
            "strata_token_summary": population.strata_token_summary,
        })
        # No figure is generated here. Plotting is handled by a separate script after results are saved.
    print(summary_df.to_string(index=False))
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
