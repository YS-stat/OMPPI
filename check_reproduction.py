#!/usr/bin/env python3
"""Compare rerun repetitions with the corresponding rows of the paper's runs.

Usage: python check_reproduction.py reference.csv rerun_trials.csv
Rows are matched on the repetition, method and budget; every column except computing time must agree
(numerical columns to 1e-12).
"""
import sys

import numpy as np
import pandas as pd

KEYS = ["outer_trial", "inner_trial", "method", "budget_key"]


def load(path):
    df = pd.read_csv(path, float_precision="round_trip")
    df["budget_key"] = df["budget"].round(6)
    return df.drop(columns=["algo_compute_time_sec"], errors="ignore").sort_values(KEYS).reset_index(drop=True)


def main():
    ref, new = load(sys.argv[1]), load(sys.argv[2])
    if len(ref) != len(new) or not ref[KEYS].astype(str).equals(new[KEYS].astype(str)):
        sys.exit(f"FAIL: rows differ ({len(ref)} reference rows, {len(new)} rerun rows)")
    bad = []
    for c in ref.columns:
        if pd.api.types.is_numeric_dtype(ref[c]):
            if not np.allclose(ref[c].to_numpy(float), new[c].to_numpy(float), rtol=0.0, atol=1e-12, equal_nan=True):
                bad.append(c)
        elif not ref[c].astype(str).equals(new[c].astype(str)):
            bad.append(c)
    if bad:
        sys.exit(f"FAIL: {len(ref)} rows, columns differ: {bad}")
    print(f"OK: {len(ref)} rows identical to the paper's runs ({sys.argv[1]})")


if __name__ == "__main__":
    main()
