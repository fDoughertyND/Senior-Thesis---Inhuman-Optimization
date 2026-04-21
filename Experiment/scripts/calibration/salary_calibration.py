"""Salary-offer calibration run.

Analog of school_app_calibration.py but 1-D — the salary-offer prompt has a
single difficulty knob ({salary_number}) instead of (school, profile). For
every model we sweep every salary tier in pairwise_salary_experiment.SALARIES
using a short list of ethnically-neutral calibration names and report the
acceptance-rate surface. The "interesting" salary per model is whichever
value sits closest to 50% — that's the natural inflection point where
name-level effects are most visible in the subsequent pairwise run.

Usage:
    python3 salary_calibration.py                 # run + write CSV + plot
    python3 salary_calibration.py --replot        # plot-only from CSV

Paste the printed MODEL_SALARIES dict into pairwise_salary_experiment.py.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_runner import Experiment, MODELS, aggregate_yes_rate, run
from pairwise_salary_experiment import (
    JOB_APP_PROMPT_PREFIX, JOB_APP_PROMPT_SUFFIX, SALARIES,
)


# --- Calibration knobs -----------------------------------------------------
# Deliberately generic / gender-neutral Anglo names to minimize name-bias
# when locating each model's 50% inflection point.

CALIBRATION_NAMES = [
    "Alex Taylor",
    "Sam Morgan",
    "Jamie Parker",
]

TRIALS = 25
MODEL_ORDER = ("grok", "gpt", "claude", "deepseek", "gemini")

_OUT_DIR = Path("results/salary_calibration")
CSV_PATH  = _OUT_DIR / f"salary_calibration_t{TRIALS}.csv"
PLOT_PATH = _OUT_DIR / f"salary_calibration_t{TRIALS}.png"
RAW_PATH  = _OUT_DIR / f"salary_calibration_raw_t{TRIALS}.json"


def build_prompt(ax, model_cfg):
    return (
        JOB_APP_PROMPT_PREFIX.format(full_name=ax["name"]),
        JOB_APP_PROMPT_SUFFIX.format(salary_number=ax["salary"]),
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

    # Cross-name mean per (salary, model) — matches school_app_calibration's
    # "neutral baseline" treatment: we don't keep the name axis so each cell
    # is the averaged acceptance rate across CALIBRATION_NAMES.
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
    """For each model, return the salary closest to 0.50 acceptance."""
    out = {}
    for model, sub in df.groupby("model"):
        sub = sub.assign(dist=(sub["rate"] - 0.5).abs())
        pick = sub.loc[sub["dist"].idxmin()]
        out[model] = (pick["salary"], float(pick["rate"]))
    return out


def plot_calibration(df: pd.DataFrame, out_path: Path):
    """One heatmap per model: rows = salaries (ascending), one column.
    We also render a single combined heatmap with models as columns for
    at-a-glance comparison. The closest-to-50% cell gets a black outline.
    """
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    # Render salaries in the order they appear in SALARIES (numerical order).
    salaries = [s for s in SALARIES if s in df["salary"].unique()]

    interesting = suggest_interesting(df)

    mat = np.full((len(salaries), len(models)), np.nan)
    for j, model in enumerate(models):
        for i, salary in enumerate(salaries):
            row = df[(df["model"] == model) & (df["salary"] == salary)]
            if not row.empty:
                mat[i, j] = float(row["rate"].iloc[0])

    fig, ax = plt.subplots(figsize=(2 + 1.2 * len(models), 4.5))
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
        f"Salary calibration: hire-at-salary rate by model, "
        f"averaged over {n_names} neutral names × {TRIALS} trials. "
        f"Outlined cell = closest to 50%.",
        fontsize=11, pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("wrote plot:", out_path)


def print_interesting_snippet(df: pd.DataFrame):
    picks = suggest_interesting(df)
    print("\n# Suggested 'interesting' salary per model — paste into")
    print("# pairwise_salary_experiment.py.")
    print("MODEL_SALARIES = {")
    for model in MODEL_ORDER:
        if model in picks:
            sal, rate = picks[model]
            print(f'    "{model}": "{sal}",   # rate={rate:.0%}')
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
