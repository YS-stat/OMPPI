# OMPPI

Code and processed input data for **Orthogonalized Multilevel Prediction-Powered Inference (OMPPI)**.

OMPPI performs cost-aware statistical inference with multiple correlated auxiliary predictions. This repository contains the two simulation studies and the human-preference and functional-correctness experiments used in the paper.

## Repository structure

```text
OMPPI/
├── simulation_omppi/
│   ├── omppi_simulation_full_alignment_demo.py
│   └── run.sh
├── simulation_comparisons/
│   ├── omppi_vectorppi_multippi_geometry_sim.py
│   └── run.sh
├── human_preference/
│   ├── data/final_aligned_with_vertex_online.pkl
│   ├── mppi_llm_preferences_compare_stratified_piq.py
│   ├── judge_alignment_bias_variance_demo.py
│   ├── plot_llm_preferences_stratified_reference_cost.py
│   └── run.sh
├── functional_correctness/
│   ├── data/humaneval_plus_generated_outputs_final.csv
│   ├── data/data_config.json
│   ├── mppi_humaneval_execution_compare_fullcost_stratified_outertruth.py
│   ├── plot_humaneval_results.py
│   └── run.sh
├── requirements.txt
├── environment.yml
├── run_all.sh
└── .gitignore
```

Generated results are written to an `outputs/` directory inside each experiment folder and are intentionally not included in the repository.

## Environment

The code was tested with Python 3.11.15. The main package versions are pinned in `requirements.txt`.

### Conda setup

```bash
conda create -n omppi python=3.11 -y
conda activate omppi
pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate omppi
```

The authors' GPU environment used PyTorch 2.10.0 with CUDA 13.0. The human-preference and functional-correctness scripts fall back to CPU when CUDA is unavailable. A platform-specific PyTorch build may be installed separately when needed.

## Data

The processed input data required to run the two LLM evaluation experiments are included:

- `human_preference/data/final_aligned_with_vertex_online.pkl`: processed Chatbot Arena human labels and LLM-judge outputs.
- `functional_correctness/data/humaneval_plus_generated_outputs_final.csv`: generated completions with precomputed full and partial HumanEval+ evaluation outcomes.
- `functional_correctness/data/data_config.json`: configuration metadata for the functional-correctness input table.

The released functional-correctness scripts use precomputed evaluator outputs and do not execute generated code. The implementation used to generate the raw input data is available from the authors upon request. Users should also comply with the terms of the original benchmarks and model outputs.

## Reproducing the experiments

All commands below can be launched from the repository root. The default `run.sh` files use the paper-scale settings and can be computationally intensive.

### 1. General OMPPI simulation

```bash
bash simulation_omppi/run.sh
```

This experiment studies finite-sample coverage, estimation error, confidence-set size, and sub-hierarchy selection in a multivariate regression setting.

### 2. Comparison with MultiPPI and VectorPPI++

```bash
bash simulation_comparisons/run.sh
```

This experiment studies covariance-geometry sensitivity under correlated auxiliary predictions.

### 3. Human-preference evaluation

```bash
bash human_preference/run.sh
```

This command runs the stratified Chatbot Arena experiment, generates performance and allocation plots, and runs the judge-alignment diagnostic.

### 4. Functional-correctness evaluation

```bash
bash functional_correctness/run.sh
```

This command runs the stratified HumanEval+ inference experiment and generates the corresponding performance and allocation plots.

### Run all experiments

```bash
bash run_all.sh
```

Running all four experiments with the paper settings requires substantial compute and time.

## Runtime configuration

The shell scripts support a few optional environment variables:

```bash
# Force CPU execution for scripts with optional GPU acceleration.
OMPPI_DEVICE=cpu bash human_preference/run.sh

# Change the number of multiprocessing workers.
OMPPI_NUM_WORKERS=4 bash functional_correctness/run.sh

# Change BLAS/OpenMP threads for the simulation scripts.
OMPPI_NUM_THREADS=16 bash simulation_omppi/run.sh
```

## Expected outputs

- `simulation_omppi/outputs/`: the main simulation figure.
- `simulation_comparisons/outputs/`: the comparison summary CSV and combined figure.
- `human_preference/outputs/main/`: summary tables and main human-preference plots.
- `human_preference/outputs/judge_alignment/`: judge-alignment diagnostic tables and figure.
- `functional_correctness/outputs/main/`: summary tables and functional-correctness plots.

## Reproducibility notes

- The input tables are included so that model generation and evaluator execution do not need to be repeated.
- Random seeds and paper-scale budgets are specified directly in the scripts and shell commands.
- Output folders, logs, notebook checkpoints, and font binaries are excluded from version control.
- If a custom font is unavailable, plotting scripts fall back to DejaVu Sans.
