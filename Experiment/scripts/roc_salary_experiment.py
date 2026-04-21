"""Salary-offer acceptance rate vs. offered salary, per model.

Direct analog of roc_loan_experiment.py, with the "difficulty" axis being
the salary number (from pairwise_salary_experiment.SALARIES) instead of
credit score. Reuses the 32 strongest-and-weakest ♂/♀ names per ethnicity
from roc_loan_experiment so name-level effects are directly comparable.

Trial grid:  32 names × N salaries × 5 models × TRIALS.
Current shape: 32 × 18 × 5 × 25 = 72,000 calls.

Output (under results/roc_salary/):
  - roc_extreme_names_salary_t{TRIALS}.csv      long-form: name, salary, model, rate
  - roc_extreme_names_salary_raw_t{TRIALS}.json raw per-trial answers
  - roc_extreme_names_salary_curves_t{TRIALS}.png main ROC figure
"""
import json
from pathlib import Path

import pandas as pd

from experiment_runner import Experiment, MODELS, aggregate_yes_rate, run
from pairwise_salary_experiment import (
    JOB_APP_PROMPT_PREFIX, JOB_APP_PROMPT_SUFFIX, SALARIES,
)
from roc_loan_experiment import NAMES  # reused: 32 strong+weak ♂/♀ per ethnicity


TRIALS = 25
MODEL_ORDER = ("grok", "gpt", "claude", "deepseek", "gemini")
SAMPLING = "extreme_names"

_OUT_DIR = Path("results/roc_salary")
CSV_PATH  = _OUT_DIR / f"roc_{SAMPLING}_salary_t{TRIALS}.csv"
PLOT_PATH = _OUT_DIR / f"roc_{SAMPLING}_salary_curves_t{TRIALS}.png"
RAW_PATH  = _OUT_DIR / f"roc_{SAMPLING}_salary_raw_t{TRIALS}.json"


def build_prompt(ax, model_cfg):
    """Prefix carries the stable job description + the applicant's name;
    suffix carries the variable salary. Same split pairwise_salary uses,
    so prompt caching hits across TRIALS × models for each (name, salary).
    """
    return (
        JOB_APP_PROMPT_PREFIX.format(full_name=ax["name"]),
        JOB_APP_PROMPT_SUFFIX.format(salary_number=ax["salary"]),
    )


def _build_dataframe(rates: dict) -> pd.DataFrame:
    """rates maps (name, salary, model) -> rate or -1. Drop -1."""
    rows = []
    for (name, salary, model), rate in rates.items():
        if rate < 0:
            continue
        rows.append({
            "name": name,
            "salary": salary,
            "model": model,
            "rate": rate,
        })
    return pd.DataFrame(rows)


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    exp = Experiment(
        axes={
            "name":   NAMES,
            "salary": SALARIES,
        },
        models={k: MODELS[k] for k in MODEL_ORDER},
        trials=TRIALS,
        prompt_builder=build_prompt,
    )

    results = run(exp)
    RAW_PATH.write_text(json.dumps(results))

    rates = aggregate_yes_rate(
        results,
        group_by=["name", "salary"],
        min_responses=1,
        fail_threshold=max(1, TRIALS // 5),
    )

    df = _build_dataframe(rates)
    df.to_csv(CSV_PATH, index=False)
    print("wrote results:", CSV_PATH)

    # Plot via the standalone module so the rendering can be iterated on
    # without re-querying.
    from plot_roc_salary_curves import plot as plot_roc
    plot_roc(df, PLOT_PATH)


if __name__ == "__main__":
    main()
