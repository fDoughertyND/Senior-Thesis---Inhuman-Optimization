"""Salary calibration for the age-variant prompt.

Sibling of salary_calibration.py — same mechanics, but uses the prompt
defined in roc_salary_age_experiment.py (which differs from pairwise_salary
in a few places). The applicant's age is held fixed at CALIBRATION_AGE
for every call (representative mid-career value) so the result reflects
"what salary sits at 50% acceptance for this prompt at this age."

Sweep: CALIBRATION_NAMES × SALARIES × 5 models × TRIALS at the fixed age.
Default: 3 × 18 × 5 × 25 = 6,750 calls.

Output (under results/salary_age_calibration/):
  - salary_age_calibration_t{TRIALS}.csv   long-form: model, salary, rate
  - salary_age_calibration_raw_t{TRIALS}.json
  - salary_age_calibration_t{TRIALS}.png   per-model heatmap, 50% cell outlined

Usage:
    python3 salary_age_calibration.py                # run + plot
    python3 salary_age_calibration.py --replot       # plot-only from CSV

Then paste the printed MODEL_SALARIES_AGE dict into
roc_salary_age_experiment.py.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_runner import Experiment, MODELS, aggregate_yes_rate, run
from pairwise_salary_experiment import SALARIES
from roc_salary_age_experiment import build_prompt as _age_build_prompt


CALIBRATION_NAMES = [
    "Alex Taylor",
    "Sam Morgan",
    "Jamie Parker",
]

# Hold age fixed at a representative mid-career value so the calibrated
# salary reflects the prompt's overall generosity, not any age effect.
CALIBRATION_AGE = 35

TRIALS = 25
MODEL_ORDER = ("grok", "gpt", "claude", "deepseek", "gemini")

_OUT_DIR = Path("results/salary_age_calibration")
CSV_PATH  = _OUT_DIR / f"salary_age_calibration_t{TRIALS}.csv"
PLOT_PATH = _OUT_DIR / f"salary_age_calibration_t{TRIALS}.png"
RAW_PATH  = _OUT_DIR / f"salary_age_calibration_raw_t{TRIALS}.json"


def build_prompt(ax, model_cfg):
    """Reuse the age-prompt builder; inject the fixed CALIBRATION_AGE and
    sweep model_cfg['salary'] as the variable."""
    return _age_build_prompt(
        {"name": ax["name"], "age": CALIBRATION_AGE},
        {"salary": ax["salary"]},
    )


def run_calibration() -> pd.DataFrame:
    exp = Experiment(
        axes={
            "name":   CALIBRATION_NAMES,
            "salary": SALARIES,
        },
        models={k: MODELS[k] for k in MODEL_ORDER},
        trials=TRIALS,
        prompt_builder=build_prompt,
    )

    results = run(exp)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(json.dumps(results))

    # Cross-name mean per (salary, model) — neutral baseline regardless of
    # which calibration name the result came from.
    rates = aggregate_yes_rate(
        results, group_by=["salary"],
        min_responses=1, fail_threshold=max(1, TRIALS // 5),
    )
    rows = []
    for (salary, model), rate in rates.items():
        if rate < 0:
            continue
        rows.append({"model": model, "salary": salary, "rate": rate})
    return pd.DataFrame(rows)


def suggest_interesting(df: pd.DataFrame) -> dict:
    out = {}
    for model, sub in df.groupby("model"):
        sub = sub.assign(dist=(sub["rate"] - 0.5).abs())
        pick = sub.loc[sub["dist"].idxmin()]
        out[model] = (pick["salary"], float(pick["rate"]))
    return out


def plot_calibration(df: pd.DataFrame, out_path: Path):
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    salaries = [s for s in SALARIES if s in df["salary"].unique()]
    interesting = suggest_interesting(df)

    mat = np.full((len(salaries), len(models)), np.nan)
    for j, model in enumerate(models):
        for i, salary in enumerate(salaries):
            row = df[(df["model"] == model) & (df["salary"] == salary)]
            if not row.empty:
                mat[i, j] = float(row["rate"].iloc[0])

    fig, ax = plt.subplots(figsize=(2 + 1.2 * len(models), 5.5))
    cmap = plt.get_cmap("RdYlGn")
    ax.imshow(mat, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_yticks(range(len(salaries)))
    ax.set_yticklabels([f"${s}" for s in salaries], fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=10)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            color = "black" if 0.25 < v < 0.75 else "white"
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    color=color, fontsize=9)

    for j, model in enumerate(models):
        if model in interesting:
            sal, _ = interesting[model]
            i = salaries.index(sal)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="black", lw=2.0))

    n_names = len(CALIBRATION_NAMES)
    ax.set_title(
        f"Salary calibration for the AGE prompt (age fixed at "
        f"{CALIBRATION_AGE}): hire rate by model, averaged over "
        f"{n_names} neutral names × {TRIALS} trials. "
        f"Outlined cell = closest to 50%.",
        fontsize=10, pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("wrote plot:", out_path)


def print_interesting_snippet(df: pd.DataFrame):
    picks = suggest_interesting(df)
    print("\n# Suggested 'interesting' salary per model for the age prompt —")
    print("# paste into roc_salary_age_experiment.py.")
    print("MODEL_SALARIES_AGE = {")
    for model in MODEL_ORDER:
        if model in picks:
            sal, rate = picks[model]
            print(f'    "{model}": "{sal}",   # rate={rate:.0%} at age='
                  f'{CALIBRATION_AGE}')
    print("}")


def replot_from_csv():
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found — run calibration once first.")
    df = pd.read_csv(CSV_PATH)
    plot_calibration(df, PLOT_PATH)
    print_interesting_snippet(df)


def main():
    df = run_calibration()
    df.to_csv(CSV_PATH, index=False)
    print("wrote results:", CSV_PATH)
    plot_calibration(df, PLOT_PATH)
    print_interesting_snippet(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--replot", action="store_true",
                   help="Skip API calls; re-plot from the existing CSV.")
    args = p.parse_args()
    if args.replot:
        replot_from_csv()
    else:
        main()
