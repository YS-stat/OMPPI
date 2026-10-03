"""Data and shared helpers for the Chatbot Arena experiment.

The data file holds Chatbot Arena comparisons of GPT-4-1106-Preview and Claude-2.1 with the human
preference and the votes of ten LLM judges. The outcome Y is a human preference for
GPT-4-1106-Preview (ties are dropped), a judge's prediction is its share of votes for
GPT-4-1106-Preview, and comparisons are split into five prompt-length strata (o200k_base tokens).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tiktoken
from sklearn.covariance import LedoitWolf

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "chatbot_arena" / "chatbot_arena_judges.pkl"
GPT4, CLAUDE = "gpt-4-1106-preview", "claude-2.1"


@dataclass
class Population:
    df: pd.DataFrame
    model_names: List[str]
    X: np.ndarray                       # columns: Y, then one vote share per judge
    costs_raw: Dict[str, float]         # mean API cost per comparison (USD)
    costs_used: Dict[str, float]        # costs normalized by the most expensive judge
    times: Dict[str, float]             # mean API time per comparison (seconds)
    strata_labels: np.ndarray
    strata_pi: Dict[int, float]
    strata_token_summary: Dict[int, Dict[str, float]]


# ============================================================
# Loading
# ============================================================
def judge_names(df: pd.DataFrame) -> List[str]:
    cand_a = {c[: -len("__judge_num_A")] for c in df.columns if c.endswith("__judge_num_A")}
    cand_b = {c[: -len("__judge_num_B")] for c in df.columns if c.endswith("__judge_num_B")}
    return sorted(cand_a & cand_b)


def count_prompt_tokens(texts: Sequence[str]) -> np.ndarray:
    enc = tiktoken.get_encoding("o200k_base")
    vals = ["" if x is None else str(x) for x in texts]
    return np.asarray([len(enc.encode(v)) for v in vals], dtype=int)


def make_quantile_strata(token_counts: np.ndarray, num_strata: int = 5) -> np.ndarray:
    ranks = pd.Series(np.asarray(token_counts, dtype=int)).rank(method="first")
    labels = np.asarray(pd.qcut(ranks, q=num_strata, labels=False, duplicates="drop"), dtype=int)
    remap = {old: new for new, old in enumerate(sorted(np.unique(labels).tolist()))}
    return np.asarray([remap[int(x)] for x in labels], dtype=int)


def summarize_strata(token_counts: np.ndarray, strata_labels: np.ndarray) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for h in sorted(np.unique(strata_labels).tolist()):
        vals = token_counts[strata_labels == h]
        out[int(h)] = {"count": int(vals.size), "min": int(np.min(vals)), "max": int(np.max(vals)),
                       "mean": float(np.mean(vals)), "median": float(np.median(vals))}
    return out


def _vote_share(gpt4_votes: np.ndarray, claude_votes: np.ndarray) -> np.ndarray:
    total = gpt4_votes + claude_votes
    out = np.full(total.shape, 0.5, dtype=float)
    mask = total > 0
    out[mask] = gpt4_votes[mask] / total[mask]
    return out


def build_population(final_df: pd.DataFrame, model_names: Sequence[str], num_strata: int = 5) -> Population:
    """Keep the GPT-4-1106-Preview vs Claude-2.1 comparisons, form prompt-length strata (on all of
    them, ties included), drop ties, and compute the judges' vote shares and mean costs."""
    df = final_df.copy()
    a_name, b_name = df["model_a"].astype(str), df["model_b"].astype(str)
    df = df.loc[((a_name == GPT4) & (b_name == CLAUDE)) | ((a_name == CLAUDE) & (b_name == GPT4))].copy().reset_index(drop=True)

    token_counts = count_prompt_tokens(df["prompt"].astype(str).fillna("").tolist())
    strata_labels = make_quantile_strata(token_counts, num_strata=num_strata)
    strata_pi = {int(h): float(np.mean(strata_labels == h)) for h in sorted(np.unique(strata_labels).tolist())}
    strata_token_summary = summarize_strata(token_counts, strata_labels)

    df["prompt_token_count"] = token_counts
    df["stratum"] = strata_labels
    df["gpt4_is_A"] = df["model_a"].astype(str) == GPT4
    df["gpt4_is_B"] = df["model_b"].astype(str) == GPT4

    def _row_y(row: pd.Series) -> float:
        if row.get("winner_tie", 0) == 1:
            return np.nan
        if row.get("winner_model_a", 0) == 1:
            return 1.0 if row["gpt4_is_A"] else 0.0
        if row.get("winner_model_b", 0) == 1:
            return 1.0 if row["gpt4_is_B"] else 0.0
        return np.nan

    df["Y"] = df.apply(_row_y, axis=1)
    df = df.loc[df["Y"].notna()].copy().reset_index(drop=True)

    gpt4_is_A = df["gpt4_is_A"].to_numpy(dtype=bool)
    scores, costs_raw, times = [], {}, {}
    for model in model_names:
        a_votes = df[f"{model}__judge_num_A"].fillna(0).astype(float).to_numpy()
        b_votes = df[f"{model}__judge_num_B"].fillna(0).astype(float).to_numpy()
        scores.append(_vote_share(np.where(gpt4_is_A, a_votes, b_votes), np.where(gpt4_is_A, b_votes, a_votes)))
        costs_raw[model] = float(df[f"{model}__judge_total_cost_usd"].fillna(0.0).mean())
        times[model] = float(df[f"{model}__judge_total_api_time_sec"].fillna(0.0).mean())

    max_cost = max(costs_raw.values())
    return Population(
        df=df, model_names=list(model_names),
        X=np.column_stack([df["Y"].astype(float).to_numpy()] + scores),
        costs_raw=costs_raw, costs_used={m: costs_raw[m] / max_cost for m in model_names}, times=times,
        strata_labels=df["stratum"].to_numpy(dtype=int), strata_pi=strata_pi, strata_token_summary=strata_token_summary,
    )


def load_population(path: Path = DATA) -> Tuple[Population, List[str]]:
    df = pd.read_pickle(path)
    names = judge_names(df)
    return build_population(df, names), names


# ============================================================
# Covariance and seeds
# ============================================================
def _regularize_covariance(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    return cov + eps * np.eye(cov.shape[0], dtype=float)


def estimate_covariance(X: np.ndarray, method: str = "ledoitwolf", eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if method == "ledoitwolf":
        lw = LedoitWolf(store_precision=False, assume_centered=False)
        lw.fit(X)
        return _regularize_covariance(lw.covariance_, eps=eps)
    return _regularize_covariance(np.cov(X, rowvar=False, ddof=1), eps=eps)


def seed_for_trial(base_seed: int, trial_id: int) -> int:
    ss = np.random.SeedSequence([int(base_seed), int(trial_id), 20260406])
    return int(ss.generate_state(1, dtype=np.uint64)[0] % np.uint64(2**63 - 1))


# ============================================================
# OMPPI route search
# ============================================================
# A route lists the active judges by decreasing explained variance tau_k^2 (stats["per_model"]).
# Its score is the leading variance (Var(Y) - tau_1^2) / n_L + (sum_j sqrt((tau_j^2 - tau_{j+1}^2) c_j))^2 / B
# for a labeled layer of n_L comparisons and judge budget B.

def route_score(route: Sequence[str], stats: Dict, budget: float, n_labeled: int, eps_gap: float) -> float:
    var_y = float(stats["var_y"])
    if len(route) == 0:
        return var_y / max(n_labeled, 1)
    tau = [float(stats["per_model"][m]["tau2"]) for m in route]
    if var_y - tau[0] <= eps_gap:
        return float("inf")
    for i in range(len(tau) - 1):
        if tau[i] - tau[i + 1] <= eps_gap:
            return float("inf")
    if tau[-1] <= eps_gap:
        return float("inf")
    fixed_term = (var_y - tau[0]) / max(n_labeled, 1)
    path_sum = 0.0
    for i, m in enumerate(route):
        tau_next = tau[i + 1] if i + 1 < len(tau) else 0.0
        path_sum += math.sqrt(max(tau[i] - tau_next, 0.0) * float(stats["per_model"][m]["cost"]))
    if budget <= 0:
        return float("inf")
    return fixed_term + (path_sum ** 2) / budget


def select_route_exhaustive(model_names: Sequence[str], stats: Dict, budget: float, n_labeled: int, eps_gap: float) -> Tuple[List[str], float]:
    best_route: List[str] = []
    best_score = route_score([], stats, budget, n_labeled, eps_gap)
    for r in range(1, len(model_names) + 1):
        for subset in itertools.combinations(model_names, r):
            ordered = sorted(subset, key=lambda m: (-float(stats["per_model"][m]["tau2"]), m))
            score = route_score(ordered, stats, budget, n_labeled, eps_gap)
            if score < best_score - 1e-15:
                best_score, best_route = score, list(ordered)
    return best_route, float(best_score)


def select_route_dag(model_names: Sequence[str], stats: Dict, budget: float, n_labeled: int, eps_gap: float) -> Tuple[List[str], float]:
    """Shortest path over the judges ordered by tau_k^2 (Appendix A); same result as exhaustive search."""
    var_y = float(stats["var_y"])
    tau_map = {m: float(stats["per_model"][m]["tau2"]) for m in model_names}
    ordered = sorted(model_names, key=lambda m: (-tau_map[m], m))
    k = len(ordered)
    dp = np.full(k, np.inf, dtype=float)
    nxt = [-1] * k
    for i in range(k - 1, -1, -1):
        m_i = ordered[i]
        tau_i = tau_map[m_i]
        best, best_next = float("inf"), -1
        if tau_i > eps_gap:
            best = math.sqrt(tau_i * float(stats["per_model"][m_i]["cost"]))
        for j in range(i + 1, k):
            gap = tau_i - tau_map[ordered[j]]
            if gap <= eps_gap:
                continue
            cand = math.sqrt(gap * float(stats["per_model"][m_i]["cost"])) + dp[j]
            if cand < best - 1e-15:
                best, best_next = cand, j
        dp[i], nxt[i] = best, best_next
    best_route: List[str] = []
    best_score = route_score([], stats, budget, n_labeled, eps_gap)
    for i, m in enumerate(ordered):
        top_gap = var_y - tau_map[m]
        if top_gap <= eps_gap or not np.isfinite(dp[i]):
            continue
        score = top_gap / max(n_labeled, 1) + (dp[i] ** 2) / max(budget, 1e-12)
        if score < best_score - 1e-15:
            route, cur = [m], i
            while nxt[cur] != -1:
                cur = nxt[cur]
                route.append(ordered[cur])
            best_route, best_score = route, float(score)
    return best_route, float(best_score)
