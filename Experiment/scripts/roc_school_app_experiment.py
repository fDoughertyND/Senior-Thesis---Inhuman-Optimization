"""ROC-style profile sweep for school admissions.

Direct analog of roc_loan_experiment.py, with the "difficulty" axis being
the student-profile tier (A strongest → G weakest) swept for each school
instead of credit score. Reuses the 32 strongest-and-weakest ♂/♀ names
per ethnicity from roc_loan_experiment so name-level effects can be
compared 1:1 across the loan and school domains.

Trial grid:  32 names × 3 schools × N profile tiers × 5 models × TRIALS.
With the current 8-tier PROFILES and TRIALS=25 that's ~96,000 calls.

Output (under results/roc_school/):
  - roc_extreme_names_school_t{TRIALS}.csv   long-form: name, school, profile, model, rate
  - roc_extreme_names_school_raw_t{TRIALS}.json   raw per-trial answers
  - roc_extreme_names_school_curves_{school}_t{TRIALS}.png   one per school

The plot module (plot_roc_school_curves.py) emits per-school figures and
can stitch them into a combined mega.
"""
import json
from pathlib import Path

import pandas as pd

from experiment_runner import Experiment, MODELS, aggregate_yes_rate, run
from pairwise_school_app_experiment import (
    PROFILES, SCHOOL_APP_PROMPT, SCHOOLS,
)
from roc_loan_experiment import NAMES  # reused: 32 strong+weak ♂/♀ per ethnicity


TRIALS = 25
MODEL_ORDER = ("grok", "gpt", "claude", "deepseek", "gemini")
# Ordered list of profile tags, strongest-first; the plotter uses the list
# position as the numeric x-axis so the curves go A→G left-to-right.
PROFILE_ORDER = list(PROFILES.keys())
SAMPLING = "extreme_names"

_OUT_DIR = Path("results/roc_school")
CSV_PATH  = _OUT_DIR / f"roc_{SAMPLING}_school_t{TRIALS}.csv"
RAW_PATH  = _OUT_DIR / f"roc_{SAMPLING}_school_raw_t{TRIALS}.json"


def build_prompt(ax, model_cfg):
    """All three variables go into the cacheable prefix; for a given
    (name, school, profile) the same prefix is reused across TRIALS × models.
    """
    stats = PROFILES[ax["profile"]]
    prefix = SCHOOL_APP_PROMPT.format(
        full_name=ax["name"],
        school_name=SCHOOLS[ax["school"]],
        **stats,
    )
    return prefix, ""


def _build_dataframe(rates: dict) -> pd.DataFrame:
    """rates maps (name, school, profile, model) -> rate or -1. Drop -1."""
    rows = []
    for (name, school, profile, model), rate in rates.items():
        if rate < 0:
            continue
        rows.append({
            "name": name,
            "school": school,
            "profile": profile,
            "model": model,
            "rate": rate,
        })
    return pd.DataFrame(rows)


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    exp = Experiment(
        axes={
            "name":    NAMES,
            "school":  list(SCHOOLS.keys()),
            "profile": PROFILE_ORDER,
        },
        models={k: MODELS[k] for k in MODEL_ORDER},
        trials=TRIALS,
        prompt_builder=build_prompt,
    )

    results = run(exp)
    RAW_PATH.write_text(json.dumps(results))

    rates = aggregate_yes_rate(
        results,
        group_by=["name", "school", "profile"],
        min_responses=1,
        fail_threshold=max(1, TRIALS // 5),
    )

    df = _build_dataframe(rates)
    df.to_csv(CSV_PATH, index=False)
    print("wrote results:", CSV_PATH)

    # Plotting lives in a separate module so it can be iterated on without
    # re-querying the APIs — same split as roc_loan_experiment + plot_roc_curves.
    from plot_roc_school_curves import plot_all
    plot_all(df, out_dir=_OUT_DIR, trials=TRIALS)


if __name__ == "__main__":
    main()
