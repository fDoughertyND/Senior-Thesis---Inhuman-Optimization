"""School-app calibration run.

Analog of the per-model "interesting credit score" calibration used for the
loan pairwise experiment, adapted to the 2-D (school × profile) landscape.

For every model we sweep every (school, profile) cell using a short set of
ethnically-neutral calibration names and report the admission-rate surface.
The "interesting" cell per model is simply whichever (school, profile) sits
closest to 50% — that's the natural inflection point where name-level
effects are most visible.

Usage:
    python3 school_app_calibration.py                 # run + write CSV + plot
    python3 school_app_calibration.py --replot        # plot-only from CSV

Pipe the printed MODEL_SCHOOL_PROFILES dict into the main pairwise
experiment when you're ready to do the real run.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_runner import Experiment, MODELS, aggregate_yes_rate, run
from pairwise_school_app_experiment import (
    PROFILES, SCHOOL_APP_PROMPT, SCHOOLS,
)


# --- Calibration knobs -----------------------------------------------------
# These names are deliberately generic / gender-neutral Anglo so they act as
# a neutral baseline for locating each model's 50% inflection point. Using
# more than one averages out any residual per-name bias.

CALIBRATION_NAMES = [
    "Alex Taylor",
    "Sam Morgan",
    "Jamie Parker",
]

TRIALS = 25
MODEL_ORDER = ("grok", "gpt", "claude", "deepseek", "gemini")

_OUT_DIR = Path("results/school_calibration")
CSV_PATH  = _OUT_DIR / f"school_app_calibration_t{TRIALS}.csv"
PLOT_PATH = _OUT_DIR / f"school_app_calibration_t{TRIALS}.png"
RAW_PATH  = _OUT_DIR / f"school_app_calibration_raw_t{TRIALS}.json"


def build_prompt(ax, model_cfg):
    stats = PROFILES[ax["profile"]]
    prefix = SCHOOL_APP_PROMPT.format(
        full_name=ax["name"],
        school_name=SCHOOLS[ax["school"]],
        **stats,
    )
    return prefix, ""


def run_calibration() -> pd.DataFrame:
    exp = Experiment(
        axes={
            "name":    CALIBRATION_NAMES,
            "school":  list(SCHOOLS.keys()),
            "profile": list(PROFILES.keys()),
        },
        models={k: MODELS[k] for k in MODEL_ORDER},
        trials=TRIALS,
        prompt_builder=build_prompt,
    )

    results = run(exp)
    RAW_PATH.write_text(json.dumps(results))

    rates = aggregate_yes_rate(
        results, group_by=["school", "profile"],
        min_responses=1, fail_threshold=max(1, TRIALS // 5),
    )
    # rates maps (school, profile, model) -> mean rate across CALIBRATION_NAMES,
    # since we didn't group by name (intentional: we want the neutral
    # cross-name mean).

    rows = []
    for (school, profile, model), rate in rates.items():
        if rate < 0:
            continue
        rows.append({
            "model": model,
            "school": school,
            "profile": profile,
            "rate": rate,
        })
    return pd.DataFrame(rows)


def suggest_interesting(df: pd.DataFrame) -> dict:
    """For each model, return the (school, profile) cell closest to 0.50."""
    out = {}
    for model, sub in df.groupby("model"):
        sub = sub.assign(dist=(sub["rate"] - 0.5).abs())
        pick = sub.loc[sub["dist"].idxmin()]
        out[model] = (pick["school"], pick["profile"], float(pick["rate"]))
    return out


def plot_calibration(df: pd.DataFrame, out_path: Path):
    """One heatmap per model: rows = profiles (A–G, top strongest), cols =
    schools (sorted by selectivity). Cells = admission rate. The closest-
    to-50% cell gets a black outline so the inflection point pops.
    """
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    profiles = list(PROFILES.keys())       # A..G (strong → weak)
    schools  = list(SCHOOLS.keys())        # insertion order: connecticut, tufts, mit

    interesting = suggest_interesting(df)

    ncols = len(models)
    fig, axes = plt.subplots(1, ncols,
                             figsize=(3.6 * ncols, 5.2),
                             squeeze=False)
    cmap = plt.get_cmap("RdYlGn")

    for ax, model in zip(axes[0], models):
        mat = np.full((len(profiles), len(schools)), np.nan)
        for i, prof in enumerate(profiles):
            for j, sch in enumerate(schools):
                row = df[(df["model"] == model)
                         & (df["profile"] == prof)
                         & (df["school"] == sch)]
                if not row.empty:
                    mat[i, j] = float(row["rate"].iloc[0])

        ax.imshow(mat, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_yticks(range(len(profiles)))
        ax.set_yticklabels(profiles, fontsize=9)
        ax.set_xticks(range(len(schools)))
        ax.set_xticklabels(schools, rotation=30, ha="right", fontsize=8)
        ax.set_title(model, fontsize=11)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                color = "black" if 0.25 < v < 0.75 else "white"
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        color=color, fontsize=8)

        # Outline the closest-to-50% cell.
        if model in interesting:
            sch, prof, _ = interesting[model]
            j = schools.index(sch)
            i = profiles.index(prof)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="black", lw=2.0))

    n_names = len(CALIBRATION_NAMES)
    fig.suptitle(
        f"School-app calibration: admission rate by (school × profile), "
        f"averaged over {n_names} neutral names × {TRIALS} trials. "
        f"Outlined cell = closest to 50%.",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("wrote plot:", out_path)


def print_interesting_snippet(df: pd.DataFrame):
    picks = suggest_interesting(df)
    print("\n# Suggested 'interesting' (school, profile) per model —")
    print("# paste into your pairwise script if you adopt approach 2.")
    print("MODEL_SCHOOL_PROFILES = {")
    for model in MODEL_ORDER:
        if model in picks:
            sch, prof, rate = picks[model]
            print(f'    "{model}": ("{sch}", "{prof}"),'
                  f'   # rate={rate:.0%}')
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
