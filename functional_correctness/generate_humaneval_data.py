#!/usr/bin/env python3
"""Build the HumanEval+ completion table (data/humaneval_plus/humaneval_plus_completions.csv).

Steps: (1) download the HumanEval+ problems through EvalPlus; (2) generate 30 completions per task with
deepseek-ai/deepseek-coder-6.7b-base (temperature 0.2, top-p 0.95, at most 384 new tokens), or read
existing samples with --samples-jsonl; (3) evaluate every completion with the full HumanEval+ suite
(original tests plus up to 50 EvalPlus tests) and with the partial evaluators Plus50, Plus25, Plus10,
OriginalTests and StaticOK, recording pass fractions and test workloads (costs).

Generated code that fails or times out counts as a failure; output of generated code is suppressed,
input() is disabled, and rare internal evaluator errors are scored as failures.

Safety: this script executes generated Python code locally in subprocesses, which is not a secure
sandbox. Run it in a container or virtual machine.

Usage, from the package root (paper settings, one GPU):
    python functional_correctness/generate_humaneval_data.py
Re-evaluate the stored completions without a GPU:
    python functional_correctness/generate_humaneval_data.py --samples-jsonl data/humaneval_plus/samples_generated.jsonl
Output: functional_correctness/results/generated_data/ (humaneval_plus_completions.csv and data_config.json,
and samples_generated.jsonl when completions are generated).
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import io
import json
import math
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def _bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_DECODER = {v: k for k, v in _bytes_to_unicode().items()}


def clean_byte_level_text(text: str) -> str:
    """Convert visible byte-level BPE markers such as Ġ and Ċ back to text.

    Some code models expose byte-level tokens literally after tokenizer.decode,
    especially tokens like Ġ for space and Ċ for newline. Those markers make
    generated Python invalid, so we convert them back before evaluation.
    """
    text = str(text or "")
    if not any(ch in text for ch in ("Ġ", "Ċ", "ĉ", "č")):
        return text
    try:
        buf = bytearray()
        for ch in text:
            if ch in _BYTE_DECODER:
                buf.append(_BYTE_DECODER[ch])
            else:
                buf.extend(ch.encode("utf-8"))
        return bytes(buf).decode("utf-8", errors="replace")
    except Exception:
        return text.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t").replace("č", "\r")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate and evaluate HumanEval+ completions.")
    p.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent / "results" / "generated_data"))
    p.add_argument("--output-csv", type=str, default="humaneval_plus_completions.csv")

    # Input / generation options
    p.add_argument("--samples-jsonl", type=str, default=None,
                   help="Optional EvalPlus-style JSONL with task_id and solution or completion. If given, skip HF generation.")
    p.add_argument("--reuse-existing-samples", type=str, choices=["yes", "no"], default="no",
                   help="If yes and out_dir/samples_generated.jsonl exists, reuse it instead of regenerating when --samples-jsonl is not set.")
    p.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-6.7b-base",
                   help="HF model used when --samples-jsonl is not provided.")
    p.add_argument("--samples-per-task", type=int, default=30)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--do-sample", type=str, choices=["yes", "no"], default="yes")
    p.add_argument("--torch-dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--task-limit", type=int, default=0,
                   help="If positive, use a deterministic random subset of this many tasks. Useful for quick tests.")
    p.add_argument("--task-sample-seed", type=int, default=12345,
                   help="Seed for selecting a task subset when --task-limit is positive.")
    p.add_argument("--prompt-style", type=str, choices=["raw", "body_only"], default="raw",
                   help="Generation prompt style. raw uses the HumanEval prompt directly; body_only adds a short comment asking for only the function body.")

    # Evaluation options
    p.add_argument("--timeout", type=float, default=3.0, help="Per-completion timeout in seconds.")
    p.add_argument("--canonical-timeout", type=float, default=120.0,
                   help="Timeout in seconds for canonical solution when building expected outputs.")
    p.add_argument("--shrink-plus-on-canonical-timeout", type=str, choices=["yes", "no"], default="yes",
                   help="If canonical solution times out on the selected plus tests, halve the plus-test subset until it passes.")
    p.add_argument("--max-plus-tests", type=int, default=50,
                   help="Cap the number of EvalPlus extra tests per task to keep evaluation tractable. Use -1 for all tests.")
    p.add_argument("--subset-seed", type=int, default=20260406)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--unsafe-local-execution", type=str, choices=["yes", "no"], default="yes",
                   help="Must be yes to run generated code locally. Use Docker for stronger isolation in production.")

    # Column names of the output table
    p.add_argument("--prompt-col", type=str, default="prompt")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_humaneval_plus() -> Dict[str, Dict[str, Any]]:
    try:
        from evalplus.data import get_human_eval_plus
    except Exception as exc:
        raise RuntimeError(
            "Could not import evalplus. Install it first, e.g.\n"
            "  pip install --upgrade evalplus\n"
        ) from exc
    problems = get_human_eval_plus()
    if not isinstance(problems, dict):
        # Older versions may return a list.
        problems = {p["task_id"]: p for p in problems}
    return problems


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def truncate_completion(text: str) -> str:
    """Cleanup for model generations after the HumanEval prompt.

    The goal is to keep the function body produced after the prompt and remove
    common continuation artifacts such as examples, tests, markdown fences, new
    top-level functions/classes, or interactive code. This is intentionally light:
    it improves local HumanEval execution without trying to repair wrong code.
    """
    text = clean_byte_level_text(str(text or "")).replace("\r\n", "\n")

    # If the model returns a fenced code block, keep the part inside the first block.
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            block = parts[1]
            if block.lstrip().startswith("python"):
                block = block.lstrip()[len("python"):]
            text = block
        else:
            text = parts[0]

    lines = text.splitlines()
    kept: List[str] = []
    seen_code = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if seen_code:
                kept.append(line)
            continue

        # Stop at common top-level continuations. Indented helper functions or
        # comments inside the target function are still kept.
        indent = len(line) - len(line.lstrip(" \t"))
        top_level = indent == 0
        if seen_code and top_level:
            if re.match(r"^(def |class |if __name__|print\(|input\(|assert |#|import |from |```)", stripped):
                break
        if top_level and re.match(r"^(if __name__|print\(|input\(|assert |```)", stripped):
            break

        kept.append(line)
        seen_code = True

    out = "\n".join(kept).rstrip()
    if not out:
        return "\n"

    # If the first nonempty generated line is an unindented statement rather than
    # a top-level definition/import, indent the whole generated block. This helps
    # code models that output `return ...` without preserving the prompt indent.
    nonempty = [ln for ln in out.splitlines() if ln.strip()]
    if nonempty:
        first = nonempty[0]
        first_indent = len(first) - len(first.lstrip(" \t"))
        if first_indent == 0 and not re.match(r"^(def |class |import |from )", first.strip()):
            out = "\n".join(("    " + ln if ln.strip() else ln) for ln in out.splitlines())

    return out.rstrip() + "\n"


def make_generation_prompt(prompt: str, style: str = "raw") -> str:
    if style == "body_only":
        return prompt + "\n    # Complete the function body below. Do not write tests, examples, input(), or print().\n"
    return prompt


def make_solution(prompt: str, completion_or_solution: str, already_solution: bool) -> str:
    s = str(completion_or_solution or "")
    if already_solution:
        return s if s.endswith("\n") else s + "\n"
    return prompt + truncate_completion(s)


def generate_samples_hf(
    problems: Mapping[str, Mapping[str, Any]],
    *,
    model_name: str,
    samples_per_task: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    torch_dtype: str,
    device_map: str,
    seed: int,
    prompt_style: str = "raw",
) -> List[Dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Could not import transformers/torch. Install them first, e.g.\n"
            "  pip install transformers accelerate torch\n"
        ) from exc

    if torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float32":
        dtype = torch.float32
    else:
        dtype = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    rows: List[Dict[str, Any]] = []
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    for task_id, problem in tqdm(list(problems.items()), desc="Generating completions"):
        prompt = problem["prompt"]
        gen_prompt = make_generation_prompt(prompt, prompt_style)
        inputs = tokenizer(gen_prompt, return_tensors="pt")
        # Move inputs to first model device if not using accelerate device_map.
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        for sample_id in range(samples_per_task):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0, inputs["input_ids"].shape[1]:]
            completion = clean_byte_level_text(tokenizer.decode(new_tokens, skip_special_tokens=True))
            solution = make_solution(prompt, completion, already_solution=False)
            rows.append({
                "task_id": task_id,
                "sample_id": int(sample_id),
                "prompt": prompt,
                "completion": completion,
                "solution": solution,
                "model": model_name,
            })
    return rows


def expected_code(problem: Mapping[str, Any]) -> str:
    prompt = str(problem["prompt"])
    sol = str(problem.get("canonical_solution", ""))
    # EvalPlus canonical_solution is usually a function body appended to prompt.
    # Only treat it as a standalone solution if the def/import starts at column 0.
    # Do not use lstrip() here: some correct canonical bodies begin with an
    # indented helper function inside the entry-point function.
    if sol.startswith("def ") or sol.startswith("import ") or sol.startswith("from "):
        return sol if sol.endswith("\n") else sol + "\n"
    return prompt + (sol if sol.endswith("\n") else sol + "\n")


def _as_call_args(inp: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    if isinstance(inp, tuple):
        return copy.deepcopy(inp), {}
    if isinstance(inp, list):
        return tuple(copy.deepcopy(inp)), {}
    return (copy.deepcopy(inp),), {}


def _to_bool_scalar(x: Any) -> bool:
    """Convert equality results, including numpy arrays/scalars, to a Python bool."""
    try:
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        arr = np.asarray(x)
        if arr.shape == ():
            return bool(arr.item())
        return bool(np.all(arr))
    except Exception:
        try:
            return bool(x)
        except Exception:
            return False


def _safe_equal(a: Any, b: Any, *, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    """Robust equality for HumanEval outputs.

    Generated code can return Python scalars, lists, tuples, dictionaries, numpy
    arrays, or nested mixtures. Plain ``a == b`` can return an array rather than
    a scalar bool, which later breaks ``np.mean``. This helper always returns a
    Python bool.
    """
    try:
        # Numpy arrays and array-like numeric outputs.
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            aa = np.asarray(a)
            bb = np.asarray(b)
            if aa.shape != bb.shape:
                return False
            if np.issubdtype(aa.dtype, np.number) and np.issubdtype(bb.dtype, np.number):
                return bool(np.allclose(aa, bb, atol=atol, rtol=rtol, equal_nan=True))
            return bool(np.array_equal(aa, bb))

        # Numeric scalars, including numpy scalar types.
        if isinstance(a, (float, int, np.floating, np.integer)) or isinstance(b, (float, int, np.floating, np.integer)):
            try:
                return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)
            except Exception:
                return _to_bool_scalar(a == b)

        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(_safe_equal(x, y, atol=atol, rtol=rtol) for x, y in zip(a, b))

        if isinstance(a, dict) and isinstance(b, dict):
            return set(a.keys()) == set(b.keys()) and all(_safe_equal(a[k], b[k], atol=atol, rtol=rtol) for k in a.keys())

        return _to_bool_scalar(a == b)
    except Exception:
        return False


def _run_code_worker(code: str, entry_point: str, inputs: Sequence[Any], q: mp.Queue) -> None:
    """Execute one candidate in a child process with stdout/stderr suppressed.

    Some model-generated completions contain top-level print/input statements or
    functions that print intermediate values. These messages can otherwise flood
    the nohup log and make the evaluator look stuck. We shadow input/print and
    redirect stdout/stderr inside the child process; errors are still returned in
    the result dictionary.
    """
    def _blocked_input(*args: Any, **kwargs: Any) -> str:
        raise EOFError("input disabled during evaluation")

    def _quiet_print(*args: Any, **kwargs: Any) -> None:
        return None

    try:
        ns: Dict[str, Any] = {"input": _blocked_input, "print": _quiet_print}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(code, ns)
            fn = ns.get(entry_point, None)
            static_ok = callable(fn)
            outputs: List[Any] = []
            errors: List[str] = []
            if not static_ok:
                q.put({"static_ok": False, "outputs": [], "errors": ["entry point missing"]})
                return
            for inp in inputs:
                args, kwargs = _as_call_args(inp)
                try:
                    outputs.append(fn(*args, **kwargs))
                    errors.append("")
                except Exception as exc:
                    outputs.append(None)
                    errors.append(repr(exc))
        q.put({"static_ok": True, "outputs": outputs, "errors": errors})
    except Exception as exc:
        q.put({"static_ok": False, "outputs": [], "errors": [repr(exc)]})


def run_code_with_timeout(code: str, entry_point: str, inputs: Sequence[Any], timeout: float) -> Dict[str, Any]:
    # On Linux, fork avoids pickling issues when this function is called from a
    # thread-pool evaluator. Fall back to spawn on platforms without fork.
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_run_code_worker, args=(code, entry_point, list(inputs), q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        return {"static_ok": False, "outputs": [], "errors": ["timeout"]}
    try:
        return q.get_nowait()
    except Exception:
        return {"static_ok": False, "outputs": [], "errors": ["no result"]}


def deterministic_plus_indices(task_id: str, n_plus: int, frac: float, seed: int) -> List[int]:
    if n_plus <= 0:
        return []
    m = max(1, int(math.ceil(frac * n_plus)))
    rng = np.random.default_rng(abs(hash((task_id, seed, frac))) % (2**32 - 1))
    return sorted(rng.choice(np.arange(n_plus), size=min(m, n_plus), replace=False).astype(int).tolist())


def cap_plus_inputs(problem: Mapping[str, Any], max_plus_tests: int, seed: int, task_id: str) -> List[Any]:
    plus = list(problem.get("plus_input", []) or [])
    if max_plus_tests is not None and max_plus_tests >= 0 and len(plus) > max_plus_tests:
        rng = np.random.default_rng(abs(hash((task_id, seed, "cap"))) % (2**32 - 1))
        idx = sorted(rng.choice(np.arange(len(plus)), size=max_plus_tests, replace=False).astype(int).tolist())
        plus = [plus[i] for i in idx]
    return plus


@dataclass
class TaskCache:
    task_id: str
    prompt: str
    entry_point: str
    base_inputs: List[Any]
    plus_inputs: List[Any]
    full_inputs: List[Any]
    expected_outputs: List[Any]
    plus50_full_indices: List[int]
    plus25_full_indices: List[int]
    plus10_full_indices: List[int]
    base_full_indices: List[int]


def build_task_cache(problem: Mapping[str, Any], task_id: str, timeout: float, canonical_timeout: float, max_plus_tests: int, subset_seed: int, shrink_plus_on_timeout: bool = True) -> TaskCache:
    base_inputs = list(problem.get("base_input", []) or [])
    plus_inputs = cap_plus_inputs(problem, max_plus_tests=max_plus_tests, seed=subset_seed, task_id=task_id)
    entry = str(problem["entry_point"])
    canon = expected_code(problem)

    # EvalPlus can include very large stress tests. For a tractable evaluation, if the
    # canonical solution times out on the selected plus-test subset, shrink that
    # subset rather than failing the whole data build.
    while True:
        full_inputs = base_inputs + plus_inputs
        result = run_code_with_timeout(canon, entry, full_inputs, timeout=max(float(canonical_timeout), float(timeout)))
        ok = bool(result.get("static_ok", False)) and len(result.get("outputs", [])) == len(full_inputs)
        if ok:
            break
        errors = result.get("errors", [])
        is_timeout = any("timeout" in str(e).lower() for e in errors)
        if shrink_plus_on_timeout and is_timeout and len(plus_inputs) > 0:
            new_len = len(plus_inputs) // 2
            print(f"[warn] {task_id}: canonical timed out on {len(plus_inputs)} plus tests; retrying with {new_len}.", flush=True)
            plus_inputs = plus_inputs[:new_len]
            continue
        raise RuntimeError(f"Canonical solution failed for {task_id}: {errors}")

    n_base = len(base_inputs)
    n_plus = len(plus_inputs)
    plus50 = deterministic_plus_indices(task_id, n_plus, 0.50, subset_seed)
    plus25 = deterministic_plus_indices(task_id, n_plus, 0.25, subset_seed)
    plus10 = deterministic_plus_indices(task_id, n_plus, 0.10, subset_seed)
    return TaskCache(
        task_id=task_id,
        prompt=str(problem["prompt"]),
        entry_point=entry,
        base_inputs=base_inputs,
        plus_inputs=plus_inputs,
        full_inputs=full_inputs,
        expected_outputs=list(result["outputs"]),
        base_full_indices=list(range(n_base)),
        plus50_full_indices=list(range(n_base)) + [n_base + i for i in plus50],
        plus25_full_indices=list(range(n_base)) + [n_base + i for i in plus25],
        plus10_full_indices=list(range(n_base)) + [n_base + i for i in plus10],
    )


def _mean_pass(pass_flags: Sequence[bool], indices: Sequence[int]) -> float:
    if not indices:
        return 0.0
    return float(np.mean([1.0 if pass_flags[i] else 0.0 for i in indices]))


def evaluate_one_sample(row: Mapping[str, Any], cache: TaskCache, timeout: float) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = run_code_with_timeout(str(row["solution"]), cache.entry_point, cache.full_inputs, timeout=timeout)
    elapsed = time.perf_counter() - t0
    outputs = result.get("outputs", [])
    static_ok = bool(result.get("static_ok", False)) and len(outputs) == len(cache.full_inputs)
    if not static_ok:
        pass_flags = [False] * len(cache.full_inputs)
    else:
        pass_flags = [bool(_safe_equal(o, e)) for o, e in zip(outputs, cache.expected_outputs)]

    pass_vals = [1.0 if bool(x) else 0.0 for x in pass_flags]
    full_pass_fraction = float(sum(pass_vals) / len(pass_vals)) if pass_vals else 0.0
    y_full = 1.0 if pass_flags and all(bool(x) for x in pass_flags) else 0.0
    base_frac = _mean_pass(pass_flags, cache.base_full_indices)
    plus50_frac = _mean_pass(pass_flags, cache.plus50_full_indices)
    plus25_frac = _mean_pass(pass_flags, cache.plus25_full_indices)
    plus10_frac = _mean_pass(pass_flags, cache.plus10_full_indices)

    full_count = max(1, len(cache.full_inputs))
    return {
        "task_id": cache.task_id,
        "prompt": cache.prompt,
        "entry_point": cache.entry_point,
        "sample_id": row.get("sample_id", None),
        "model": row.get("model", None),
        "completion": row.get("completion", None),
        "solution": row.get("solution", None),
        "Y_full_plus": y_full,
        "full_pass_fraction": full_pass_fraction,
        "f_plus_50": plus50_frac,
        "f_plus_25": plus25_frac,
        "f_plus_10": plus10_frac,
        "f_original_tests": base_frac,
        "f_static_ok": 1.0 if static_ok else 0.0,
        "n_base_tests": len(cache.base_inputs),
        "n_plus_tests_used": len(cache.plus_inputs),
        "n_full_tests": len(cache.full_inputs),
        "n_plus50_tests": len(cache.plus50_full_indices),
        "n_plus25_tests": len(cache.plus25_full_indices),
        "n_plus10_tests": len(cache.plus10_full_indices),
        "eval_elapsed_sec": float(elapsed),
        # Costs (test workloads); humaneval_experiment.py normalizes them by the full-suite cost.
        "cost_full_plus": float(full_count),
        "cost_plus_50": float(max(1, len(cache.plus50_full_indices))),
        "cost_plus_25": float(max(1, len(cache.plus25_full_indices))),
        "cost_plus_10": float(max(1, len(cache.plus10_full_indices))),
        "cost_original_tests": float(max(1, len(cache.base_full_indices))),
        "cost_static_ok": 1.0,
    }


def _failed_eval_row(row: Mapping[str, Any], cache: TaskCache, err: Exception) -> Dict[str, Any]:
    """Return a well-formed failed row instead of crashing the whole run."""
    return {
        "task_id": cache.task_id,
        "prompt": cache.prompt,
        "entry_point": cache.entry_point,
        "sample_id": row.get("sample_id", None),
        "model": row.get("model", None),
        "completion": row.get("completion", None),
        "solution": row.get("solution", None),
        "Y_full_plus": 0.0,
        "full_pass_fraction": 0.0,
        "f_plus_50": 0.0,
        "f_plus_25": 0.0,
        "f_plus_10": 0.0,
        "f_original_tests": 0.0,
        "f_static_ok": 0.0,
        "n_base_tests": len(cache.base_inputs),
        "n_plus_tests_used": len(cache.plus_inputs),
        "n_full_tests": len(cache.full_inputs),
        "n_plus50_tests": len(cache.plus50_full_indices),
        "n_plus25_tests": len(cache.plus25_full_indices),
        "n_plus10_tests": len(cache.plus10_full_indices),
        "eval_elapsed_sec": 0.0,
        "eval_error": repr(err),
        "cost_full_plus": float(max(1, len(cache.full_inputs))),
        "cost_plus_50": float(max(1, len(cache.plus50_full_indices))),
        "cost_plus_25": float(max(1, len(cache.plus25_full_indices))),
        "cost_plus_10": float(max(1, len(cache.plus10_full_indices))),
        "cost_original_tests": float(max(1, len(cache.base_full_indices))),
        "cost_static_ok": 1.0,
    }


def _eval_worker(args: Tuple[Dict[str, Any], TaskCache, float]) -> Dict[str, Any]:
    row, cache, timeout = args
    try:
        out = evaluate_one_sample(row, cache, timeout=timeout)
        out.setdefault("eval_error", "")
        return out
    except Exception as exc:
        # One malformed completion should be scored as failure, not kill a 7-hour data build.
        return _failed_eval_row(row, cache, exc)


def normalize_samples(samples: List[Dict[str, Any]], problems: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    counts: Dict[str, int] = {}
    for s in samples:
        task_id = str(s["task_id"])
        if task_id not in problems:
            raise ValueError(f"Unknown task_id in samples: {task_id}")
        sample_id = counts.get(task_id, 0)
        counts[task_id] = sample_id + 1
        prompt = problems[task_id]["prompt"]
        if "completion" in s and s["completion"] is not None:
            completion = clean_byte_level_text(str(s["completion"]))
            solution = make_solution(prompt, completion, already_solution=False)
        elif "solution" in s and s["solution"] is not None:
            solution = clean_byte_level_text(str(s["solution"]))
            solution = make_solution(prompt, solution, already_solution=True)
            completion = ""
        else:
            raise ValueError("Each sample must have solution or completion")
        rows.append({
            "task_id": task_id,
            "sample_id": int(s.get("sample_id", sample_id)),
            "prompt": prompt,
            "completion": completion,
            "solution": solution,
            "model": s.get("model", "provided_samples"),
        })
    return rows


def maybe_subset_problems(problems: Dict[str, Dict[str, Any]], task_limit: int, seed: int) -> Dict[str, Dict[str, Any]]:
    if task_limit is None or task_limit <= 0 or task_limit >= len(problems):
        return problems
    keys = sorted(problems.keys(), key=lambda s: int(str(s).split("/")[-1]) if "/" in str(s) else str(s))
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(keys, size=int(task_limit), replace=False).tolist())
    return {k: problems[k] for k in keys if k in chosen}


def main() -> None:
    args = parse_args()
    if args.unsafe_local_execution != "yes":
        raise SystemExit("Set --unsafe-local-execution yes to acknowledge local code execution risk.")
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HumanEval+ problems through EvalPlus...")
    problems = load_humaneval_plus()
    print(f"Loaded {len(problems)} tasks.")
    problems = maybe_subset_problems(problems, args.task_limit, args.task_sample_seed)
    if args.task_limit and args.task_limit > 0:
        print(f"Using a deterministic subset of {len(problems)} tasks.")

    samples_path = out_dir / "samples_generated.jsonl"
    if args.samples_jsonl:
        print(f"Reading samples from {args.samples_jsonl}")
        samples_raw = read_jsonl(args.samples_jsonl)
        samples = normalize_samples(samples_raw, problems)
    elif args.reuse_existing_samples == "yes" and samples_path.exists():
        print(f"Reusing existing generated samples from {samples_path}")
        samples_raw = read_jsonl(samples_path)
        samples = normalize_samples(samples_raw, problems)
    else:
        print(f"Generating samples with HF model: {args.model}")
        samples = generate_samples_hf(
            problems,
            model_name=args.model,
            samples_per_task=args.samples_per_task,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=(args.do_sample == "yes"),
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            seed=args.seed,
            prompt_style=args.prompt_style,
        )
        write_jsonl(samples_path, samples)
        print(f"Saved generated samples to {samples_path}")

    print("Building task-level test caches once...")
    used_task_ids = sorted({str(s["task_id"]) for s in samples})
    caches: Dict[str, TaskCache] = {}
    for task_id in tqdm(used_task_ids, desc="Building caches"):
        caches[task_id] = build_task_cache(
            problems[task_id],
            task_id,
            timeout=args.timeout,
            canonical_timeout=args.canonical_timeout,
            max_plus_tests=args.max_plus_tests,
            subset_seed=args.subset_seed,
            shrink_plus_on_timeout=(args.shrink_plus_on_canonical_timeout == "yes"),
        )

    print(f"Evaluating {len(samples)} completions locally...")
    worker_args = [(s, caches[str(s["task_id"])], args.timeout) for s in samples]
    if args.num_workers <= 1:
        rows = [_eval_worker(x) for x in tqdm(worker_args, desc="Evaluating")]
    else:
        # Use a thread pool here because each evaluator call starts its own
        # non-daemonic child process for timeout control. A multiprocessing Pool
        # creates daemonic workers, and daemonic workers cannot spawn children.
        with ThreadPool(processes=args.num_workers) as pool:
            rows = list(tqdm(pool.imap_unordered(_eval_worker, worker_args), total=len(worker_args), desc="Evaluating"))

    df = pd.DataFrame(rows)
    # Prompt word count (the experiment forms strata from o200k_base token counts instead).
    df["prompt_word_count"] = df["prompt"].fillna("").astype(str).str.split().str.len().astype(int)
    out_csv = out_dir / args.output_csv
    df.to_csv(out_csv, index=False)

    meta = {
        "n_rows": int(df.shape[0]),
        "n_tasks": int(df["task_id"].nunique()),
        "model": args.model if not args.samples_jsonl else "provided_samples",
        "samples_per_task_requested": args.samples_per_task,
        "max_plus_tests": args.max_plus_tests,
        "timeout": args.timeout,
        "canonical_timeout": args.canonical_timeout,
        "shrink_plus_on_canonical_timeout": args.shrink_plus_on_canonical_timeout,
        "theta_full_population": float(df["Y_full_plus"].mean()) if df.shape[0] else None,
        "columns": list(df.columns),
    }
    (out_dir / "data_config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if "eval_error" in df.columns:
        n_eval_errors = int((df["eval_error"].fillna("").astype(str) != "").sum())
        if n_eval_errors:
            print(f"[warn] {n_eval_errors} completions had internal evaluator errors and were scored as failures.", flush=True)
    summary_cols = ["Y_full_plus", "f_plus_50", "f_plus_25", "f_plus_10", "f_original_tests", "f_static_ok"]
    print(df[summary_cols].describe().to_string())
    if df.shape[0] > 1:
        print("\nCorrelation with Y_full_plus:")
        for col in summary_cols[1:]:
            try:
                print(f"  {col}: {float(df['Y_full_plus'].corr(df[col])):.4f}")
            except Exception:
                print(f"  {col}: nan")
    cost_cols = ["cost_full_plus", "cost_plus_50", "cost_plus_25", "cost_plus_10", "cost_original_tests", "cost_static_ok"]
    try:
        cost_mean = df[cost_cols].mean()
        print("\nMean normalized costs relative to full label:")
        print((cost_mean / max(cost_mean["cost_full_plus"], 1e-12)).to_string())
    except Exception:
        pass
    try:
        used = df.groupby("task_id")["n_plus_tests_used"].first()
        print("\nn_plus_tests_used summary:")
        print(used.describe().to_string())
        print("Smallest n_plus_tests_used:")
        print(used.sort_values().head(15).to_string())
    except Exception:
        pass
    print(f"Saved completion table to {out_csv}")


if __name__ == "__main__":
    main()
