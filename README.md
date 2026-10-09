# Code and data for "Statistical Inference for LLM Evaluations: From Multiple Auxiliary Predictors to Valid Uncertainty Quantification"

This package reproduces every figure and table of the paper and its appendices that is computed from
data: the HumanEval+ functional-correctness experiment, the Chatbot Arena human-preference experiment,
and the two simulation studies. Each experiment has its own folder with the scripts, a `run.sh`, and a
`results/` folder that holds the outputs used in the paper.

## Requirements

- Python 3.11 (tested with 3.11.15 on Linux). Only CPUs are needed.
- Packages: `pip install -r requirements.txt` (numpy, pandas, scipy, scikit-learn, matplotlib, tiktoken,
  pyarrow, datasets). For example:

  ```bash
  conda create -n omppi python=3.11 -y
  conda activate omppi
  pip install -r requirements.txt
  ```
- Internet access on the first run: tiktoken downloads the `o200k_base` tokenizer used to form
  prompt-length strata, and `win_rates/arena_pair_ci.py` downloads the Chatbot Arena comparisons from
  Hugging Face.
- Regenerating the HumanEval+ input data (optional, see [Data](#data)) also needs a GPU and the packages in
  `requirements-data.txt`.

## Quick start

All commands are run from the package root.

```bash
bash run_all.sh quick      # about 1 minute: reruns the first 8 repetitions of every Monte Carlo run and
                           # checks that they agree with the paper's runs row by row
bash run_all.sh results    # about 6 minutes: all figures and tables from the stored Monte Carlo runs
bash run_all.sh            # full reproduction from scratch (about 4 hours with 60 workers)
```

The Monte Carlo runs are parallelized over processes; set `OMPPI_WORKERS` (default 8) to the number of
available cores. Each repetition and method uses its own seeded random-number stream, so the results do
not depend on the number of workers.

## Figures and tables

Paths are relative to the package root. A script writes its outputs to the `results/` folder of its
experiment.

| Paper | Content | Script | Output |
|---|---|---|---|
| Figure 1 | Nested sampling design (schematic) | – | – |
| Figure 2 | HumanEval+: coverage, RMSE and interval width by pilot size | `functional_correctness/humaneval_figures.py` | `functional_correctness/results/figures/humaneval_combined_share.pdf` |
| Table 1 | False-positive diagnostics of the HumanEval+ partial evaluators | `functional_correctness/humaneval_tables.py` | `functional_correctness/results/table1_false_positive.csv` |
| Figure C.1 | Multivariate simulation study | `simulation/omppi_simulation.py` | `simulation/results/omppi_simulation_main.pdf` |
| Figure D.1 | Win-rate intervals of model pairs in Chatbot Arena | `win_rates/arena_pair_ci.py` | `win_rates/results/pair_ci.csv`; the plotted numbers are printed |
| Figure D.2 | Judge-bias alignment in Chatbot Arena | `human_preference/judge_alignment.py` | `human_preference/results/judge_alignment/judge_alignment.pdf` |
| Figure D.3 | Chatbot Arena: coverage, RMSE and interval width by pilot size | `human_preference/arena_figures.py` | `human_preference/results/figures/arena_combined_share.pdf` |
| Table D.1 | LLM judges: explained variance, cost and time | `human_preference/judge_alignment.py` | `human_preference/results/judge_alignment/judge_table.csv` (`tau2_hat`, `cost_normalized`, `time_mean_sec`) |
| Figure D.4 | Correlations of the human outcome and the ten judges | `human_preference/arena_figures.py` | `human_preference/results/figures/arena_correlation.pdf` |
| Figure D.5 | Oracle reductions against the budget | `human_preference/arena_oracle.py`, then `arena_figures.py` | `human_preference/results/figures/arena_oracle_reduction.pdf` |
| Table D.2 | Chatbot Arena results by pilot size | `human_preference/arena_metrics.py` | `human_preference/results/metrics.csv` (`share_rmse`, `share_width`, `coverage_pool`; oracle row: `oracle_share`) |
| Figure D.6 | Chatbot Arena allocations at B = 1500 | `human_preference/arena_allocation.py`, then `arena_figures.py` | `human_preference/results/figures/arena_combined_allocation.pdf` |
| Table E.1 | HumanEval+ partial evaluators | `functional_correctness/humaneval_tables.py` | `functional_correctness/results/tableE1_evaluators.csv` |
| Table E.2 | Stratum-level oracle constants for HumanEval+ | `functional_correctness/humaneval_oracle.py` | `functional_correctness/results/oracle_strata.csv` and `oracle.csv` |
| Table E.3 | HumanEval+ results by pilot size | `functional_correctness/humaneval_metrics.py` | `functional_correctness/results/metrics.csv` (`share_rmse`, `share_width`, `coverage_pool`; oracle row: `oracle_share`) |
| Figure E.1 | HumanEval+ allocations at the largest budget | `functional_correctness/humaneval_figures.py` | `functional_correctness/results/figures/humaneval_combined_allocation.pdf` |
| Tables F.1, F.2 | Comparison with related methods (no computation) | – | – |
| Figure F.1 | Controlled covariance-perturbation comparison | `simulation_comparison/covariance_perturbation.py` | `simulation_comparison/results/omppi_vectorppi_multippi_geometry_combined.pdf` |

Numbers quoted in the text: the oracle reductions Delta_oracle(B) of Chatbot Arena (4.2% at B = 100, 8.9% at
B = 1500, 10.9% as B → ∞) are in `human_preference/results/oracle.csv`; Delta_oracle = 15.5% of HumanEval+
and the ratio 0.5327 with free partial evaluators are in `functional_correctness/results/oracle.csv`; the
shrinkage comparison in Appendix E (ratios 1.20, 0.90 and 0.89 at pilot size 200) is in
`functional_correctness/results/shrinkage.csv` (`humaneval_shrinkage.py`).

Figure 2, Table E.3, Figure E.1, Figure D.3 and Table D.2 summarize the Monte Carlo runs stored in
`results/pilot<n>/` (summary files of the paper's runs); the full `run.sh` of an experiment regenerates
those runs (below). All other outputs are computed directly from the data.

## Running each experiment

Every `run.sh` can be called from any directory.

| Command | What it does | Time |
|---|---|---|
| `bash win_rates/run.sh` | Figure D.1 | under a minute after the download |
| `bash functional_correctness/run.sh` | three Monte Carlo runs (pilot sizes 800, 400, 200), then oracle, metrics, figures, tables and the shrinkage comparison | 2 hours with 60 workers |
| `bash functional_correctness/run.sh results` | everything after the Monte Carlo runs, from `results/pilot<n>/` | 3 minutes |
| `bash human_preference/run.sh` | four Monte Carlo runs (pilot sizes 50, 100, 200, 400), then oracle, allocation, metrics, figures and judge alignment | 1.5 hours with 60 workers |
| `bash human_preference/run.sh results` | everything after the Monte Carlo runs, from `results/pilot<n>/` | 3 minutes |
| `bash simulation/run.sh` | Figure C.1 | 30 minutes with 48 BLAS threads |
| `bash simulation_comparison/run.sh` | Figure F.1 | under a minute |
| `bash functional_correctness/run.sh quick`, `bash human_preference/run.sh quick` | reproduction check (outputs in `quick_check/`) | under a minute each |

A Monte Carlo run writes `trials.csv` (one row per repetition, budget and method), `summary.csv`,
`diagnostics_summary.csv` and `config.json` to `results/pilot<n>/`; the HumanEval+ runs also write
`allocation_summary.csv` and `outer_truth.csv`. The `trials.csv` files of the paper's runs (about 0.8 GB
for Chatbot Arena and 1.6 GB for HumanEval+) are not included; `results/pilot<n>/trials_outer0_first8.csv`
holds their rows for the first 8 repetitions, which the quick check compares against. Individual runs
can be started directly, for example `python human_preference/arena_experiment.py --n-pilot 100` or
`python functional_correctness/humaneval_experiment.py --n-pilot 800 --n-trials 500`; `--help` lists the
options.

## Package structure

```text
├── README.md
├── requirements.txt, requirements-data.txt
├── run_all.sh
├── check_reproduction.py           compares rerun repetitions with rows of the paper's runs
├── data/
│   ├── chatbot_arena/              Chatbot Arena comparisons with the votes of ten LLM judges
│   └── humaneval_plus/             HumanEval+ completions with full and partial evaluator outcomes
├── win_rates/                      Figure D.1
│   └── arena_pair_ci.py
├── human_preference/               Chatbot Arena experiment: Figures D.2-D.6, Tables D.1-D.2
│   ├── arena_data.py               data loading, strata, covariance estimation, OMPPI route search
│   ├── arena_experiment.py         Monte Carlo runs (Classical, VectorPPI++, MultiPPI, OMPPI)
│   ├── arena_oracle.py             designs under the population covariance
│   ├── arena_allocation.py         allocations at the largest budget
│   ├── arena_metrics.py            checks and Table D.2
│   ├── arena_figures.py            Figures D.3-D.6
│   └── judge_alignment.py          Figure D.2 and Table D.1
├── functional_correctness/         HumanEval+ experiment: Figure 2, Table 1, Tables E.1-E.3, Figure E.1
│   ├── humaneval_experiment.py     Monte Carlo runs (Classical, VectorPPI++, MultiPPI, OMPPI)
│   ├── humaneval_oracle.py         designs under the population covariance (Table E.2)
│   ├── humaneval_metrics.py        checks and Table E.3
│   ├── humaneval_figures.py        Figure 2 and Figure E.1
│   ├── humaneval_tables.py         Table 1 and Table E.1
│   ├── humaneval_shrinkage.py      effect of covariance shrinkage (Appendix E)
│   └── generate_humaneval_data.py  builds data/humaneval_plus (optional; GPU)
├── simulation/                     Figure C.1
│   └── omppi_simulation.py
└── simulation_comparison/          Figure F.1
    └── covariance_perturbation.py
```

## Data

- `data/chatbot_arena/chatbot_arena_judges.pkl` (pandas pickle): the 1073 Chatbot Arena comparisons of
  GPT-4-1106-Preview and Claude-2.1 in the Hugging Face dataset `lmarena-ai/arena-human-preference-55k`
  (prompt, both responses and the human preference), with the votes of ten LLM judges (Gemini 2.5 Pro,
  Flash and Flash-Lite, Gemini 3.1 Flash Preview and Flash-Lite Preview, Qwen3 Next 80B, Qwen3 235B A22B,
  Qwen3 Coder 480B A35B, GPT-OSS 120B and 20B; ten votes per comparison, five in each presentation order)
  and the API cost and time of each judge. Ties are dropped in the analysis, which leaves 823 comparisons.
  The judges were queried through the providers' APIs; that step is not part of the package.
- `data/humaneval_plus/humaneval_plus_completions.csv`: 4920 completions (164 HumanEval+ tasks, 30
  completions each) of `deepseek-ai/deepseek-coder-6.7b-base`, with the full-suite outcome `Y_full_plus`,
  the pass fractions of the partial evaluators (`f_plus_50`, `f_plus_25`, `f_plus_10`, `f_original_tests`,
  `f_static_ok`) and their test workloads (`cost_*`). `samples_generated.jsonl` holds the generated
  completions and `data_config.json` the generation settings.
  `python functional_correctness/generate_humaneval_data.py` rebuilds the table (one GPU; it executes
  generated code, so run it in a container); adding
  `--samples-jsonl data/humaneval_plus/samples_generated.jsonl` re-evaluates the stored completions without
  a GPU. Its output goes to `functional_correctness/results/generated_data/`.
- Figure D.1 uses `lmarena-ai/arena-human-preference-55k` directly.

Please follow the terms of use of Chatbot Arena, HumanEval+ (EvalPlus) and the models whose outputs the
data contain.

## Notes

- Strata are prompt-length quintiles of `o200k_base` token counts (tiktoken), formed once on the full data.
- With one BLAS thread per process, which the two experiment `run.sh` scripts set, reruns agree with the
  paper's runs to the last digit; other thread counts can change rounding in the last digits.
- The figures in the paper use Helvetica. Set `OMPPI_FONT=/path/to/Helvetica.ttf` to use it; otherwise a
  similar sans-serif font is used.
- In the code, Classical denotes the labeled-only estimator (LO in the paper), and OMPPI(DAG) is the
  OMPPI estimator reported in the paper; OMPPI(Exhaustive) uses exhaustive search instead of the DAG search
  and selects the same designs.
