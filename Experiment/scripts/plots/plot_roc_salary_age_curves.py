"""ROC-style age-sweep plot for salary acceptance.

Mirrors plot_roc_salary_curves.py but x-axis is applicant age. Per-panel
title shows each model's calibrated salary from MODEL_SALARIES (the
salary seen by every prompt in its panel). No x-axis vertical marker —
there is no calibrated "interesting" age.

Importable:
    from plot_roc_salary_age_curves import plot, plot_by_ethnicity
    plot(df, out_path)

Standalone — re-render from the existing CSV without re-querying:
    python3 plot_roc_salary_age_curves.py
    python3 plot_roc_salary_age_curves.py --csv foo.csv --out bar.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

from roc_loan_experiment import (
    ETHNICITY_NAMES, ETHNICITY_OF_NAME, GENDER_OF_NAME, STRENGTH_OF_NAME,
)
from roc_salary_age_experiment import (
    AGES, CSV_PATH as DEFAULT_CSV_PATH, MODEL_ORDER,
    PLOT_PATH as DEFAULT_PLOT_PATH, TRIALS, _effective_salaries,
)

# Whichever of MODEL_SALARIES_AGE (preferred) / MODEL_SALARIES (fallback)
# is populated — drives the per-panel title annotation.
MODEL_SALARIES = _effective_salaries()


STRENGTH_STYLE = {"strong": "-", "weak": "--"}
GENDER_COLORS = {
    "masculine": "#6FB4E3",
    "feminine":  "#F4A6C0",
}

_tab10 = plt.get_cmap("tab10")
ETHNICITY_COLORS = {
    e: _tab10(i % 10) for i, e in enumerate(ETHNICITY_NAMES)
}

AGE_MIN = min(AGES)
AGE_MAX = max(AGES)


def _normalized_auc(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2:
        return float("nan")
    area = np.trapezoid(ys, xs)
    width = xs[-1] - xs[0]
    return float(area / width) if width > 0 else float("nan")


def _draw_gender_lines(ax, spag, mean_src, gender, color, draw_mean):
    sub = spag[spag["gender"] == gender]
    spag_alpha = 0.55 if draw_mean else 0.85
    spag_lw = 1.0 if draw_mean else 1.9
    for name, g in sub.groupby("name"):
        g = g.sort_values("age")
        linestyle = STRENGTH_STYLE.get(STRENGTH_OF_NAME.get(name, "strong"), "-")
        ax.plot(g["age"], g["rate"],
                color=color, alpha=spag_alpha, lw=spag_lw, linestyle=linestyle)
    if not draw_mean or sub.empty:
        return
    mean_line = (sub.groupby("age")["rate"].mean()
                    .reset_index().sort_values("age"))
    ax.plot(mean_line["age"], mean_line["rate"],
            color=color, lw=2.6, label=gender)


def _panel_aucs(spag, gender):
    sub = spag[spag["gender"] == gender]
    per_name = {}
    for name, g in sub.groupby("name"):
        g = g.sort_values("age")
        per_name[name] = _normalized_auc(g["age"].values, g["rate"].values)

    def mean_or_nan(names):
        vals = [per_name[n] for n in names if not np.isnan(per_name[n])]
        return float(np.mean(vals)) if vals else float("nan")

    strong = [n for n in per_name if STRENGTH_OF_NAME.get(n) == "strong"]
    weak   = [n for n in per_name if STRENGTH_OF_NAME.get(n) == "weak"]
    return {
        "best":  mean_or_nan(strong),
        "worst": mean_or_nan(weak),
        "avg":   mean_or_nan(list(per_name)),
    }


def _render_panel(ax, spag, mean_src, title, show_xlabel, show_ylabel,
                  draw_mean=True):
    per_gender_aucs = {}
    for gender, color in GENDER_COLORS.items():
        _draw_gender_lines(ax, spag, mean_src, gender, color, draw_mean)
        per_gender_aucs[gender] = _panel_aucs(spag, gender)

    rows = []
    for gender, color in GENDER_COLORS.items():
        aucs = per_gender_aucs[gender]
        short = "♂" if gender == "masculine" else "♀"
        for kind, symbol in (("best", "—"), ("worst", "--"), ("avg", "•")):
            v = aucs.get(kind, float("nan"))
            val_str = f"{v:.2f}" if not np.isnan(v) else "—"
            label = f"{symbol} {short} {kind:<5} AUC = {val_str}"
            rows.append((v, label, color))
    rows.sort(key=lambda r: float("-inf") if np.isnan(r[0]) else r[0],
              reverse=True)
    auc_entries = [
        TextArea(label, textprops=dict(color=color, fontsize=7.0))
        for _, label, color in rows
    ]
    if auc_entries:
        packed = VPacker(children=auc_entries, align="left", pad=0, sep=1)
        anchored = AnchoredOffsetbox(
            loc="upper left", child=packed, pad=0.3, borderpad=0.3,
            frameon=True, bbox_to_anchor=(0.02, 0.98),
            bbox_transform=ax.transAxes,
        )
        anchored.patch.set(facecolor="white", edgecolor="gray", alpha=0.85)
        ax.add_artist(anchored)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(AGE_MIN, AGE_MAX)
    ax.set_xticks(AGES)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    if show_xlabel:
        ax.set_xlabel("Applicant age", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Hire rate", fontsize=9)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender"] = df["name"].map(GENDER_OF_NAME)
    df["ethnicity"] = df["name"].map(ETHNICITY_OF_NAME)
    df["age"] = df["age"].astype(int)
    df = df.dropna(subset=["gender"])
    return df


def plot(df: pd.DataFrame, out_path, ethnicity=None):
    df = _prepare(df)
    if ethnicity is not None:
        spaghetti_df = df[df["ethnicity"] == ethnicity]
        if spaghetti_df.empty:
            raise SystemExit(f"No names matched ethnicity={ethnicity!r}")
    else:
        spaghetti_df = df

    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    cols = list(models_present)
    if ethnicity is None:
        cols = cols + ["__avg__"]

    avg_full = (df.groupby(["name", "age", "gender"], as_index=False)
                  ["rate"].mean())
    avg_spag = (spaghetti_df.groupby(["name", "age", "gender"],
                                     as_index=False)["rate"].mean())

    ncols = len(cols)
    fig, axes = plt.subplots(1, ncols,
                             figsize=(4.6 * ncols, 4.6),
                             squeeze=False)

    for c, panel in enumerate(cols):
        ax = axes[0][c]
        show_ylabel = c == 0
        if panel == "__avg__":
            spag = avg_spag
            mean_src = avg_full
            title = f"avg across {len(models_present)} models"
        else:
            spag = spaghetti_df[spaghetti_df["model"] == panel]
            mean_src = df[df["model"] == panel]
            cal = MODEL_SALARIES.get(panel)
            title = f"{panel} @ ${cal}" if cal else panel

        if ethnicity is not None:
            _render_panel(ax, spag, spag, title,
                          show_xlabel=True, show_ylabel=show_ylabel,
                          draw_mean=False)
        else:
            _render_panel(ax, spag, mean_src, title,
                          show_xlabel=True, show_ylabel=show_ylabel,
                          draw_mean=True)

    legend_lw = 2.6 if ethnicity is None else 1.9
    gender_handles = [
        plt.Line2D([], [], color=GENDER_COLORS[g], lw=legend_lw, label=g)
        for g in GENDER_COLORS
    ]
    strength_handles = [
        plt.Line2D([], [], color="gray", lw=legend_lw,
                   linestyle=STRENGTH_STYLE[s], label=f"{s} name")
        for s in STRENGTH_STYLE
    ]
    fig.legend(handles=gender_handles + strength_handles,
               loc="lower center",
               ncol=len(gender_handles) + len(strength_handles),
               frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))

    n_names = spaghetti_df["name"].nunique()
    prefix = f"{ethnicity} — " if ethnicity is not None else ""
    trend_note = ("per-name lines only (no overall trendline)"
                  if ethnicity is not None
                  else "trendline = mean per gender across names")
    fig.suptitle(
        f"{prefix}Hire rate vs. applicant age, by model "
        f"(salary fixed at each model's calibrated inflection; "
        f"N={n_names} names × {TRIALS} trials per point; {trend_note})",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("wrote plot:", out_path)


# --- Ethnicity-averaged variant --------------------------------------------

def _render_panel_by_ethnicity(ax, df_panel, title, show_xlabel, show_ylabel):
    aucs = {}
    for eth, color in ETHNICITY_COLORS.items():
        sub = df_panel[df_panel["ethnicity"] == eth]
        if sub.empty:
            continue
        line = (sub.groupby("age")["rate"].mean()
                  .reset_index().sort_values("age"))
        ax.plot(line["age"], line["rate"], color=color, lw=2.0, label=eth)
        aucs[eth] = _normalized_auc(line["age"].values, line["rate"].values)

    if aucs:
        def _sort_key(eth):
            v = aucs[eth]
            return float("-inf") if np.isnan(v) else v
        ordered = sorted(aucs.keys(), key=_sort_key, reverse=True)
        entries = [
            TextArea(
                f"{eth:<16s} AUC = "
                f"{(f'{aucs[eth]:.2f}' if not np.isnan(aucs[eth]) else '—')}",
                textprops=dict(color=ETHNICITY_COLORS[eth], fontsize=7.0),
            )
            for eth in ordered
        ]
        packed = VPacker(children=entries, align="left", pad=0, sep=1)
        anchored = AnchoredOffsetbox(
            loc="upper left", child=packed, pad=0.3, borderpad=0.3,
            frameon=True, bbox_to_anchor=(0.02, 0.98),
            bbox_transform=ax.transAxes,
        )
        anchored.patch.set(facecolor="white", edgecolor="gray", alpha=0.85)
        ax.add_artist(anchored)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(AGE_MIN, AGE_MAX)
    ax.set_xticks(AGES)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    if show_xlabel:
        ax.set_xlabel("Applicant age", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Hire rate", fontsize=9)


def plot_by_ethnicity(df: pd.DataFrame, out_path):
    df = _prepare(df)
    if df.empty:
        raise SystemExit("No rows to plot.")

    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    cols = list(models_present) + ["__avg__"]
    avg_df = (df.groupby(["name", "age", "ethnicity"],
                         as_index=False)["rate"].mean())

    ncols = len(cols)
    fig, axes = plt.subplots(1, ncols,
                             figsize=(4.6 * ncols, 4.8),
                             squeeze=False)

    for c, panel in enumerate(cols):
        ax = axes[0][c]
        show_ylabel = c == 0
        if panel == "__avg__":
            sub = avg_df
            title = f"avg across {len(models_present)} models"
        else:
            sub = df[df["model"] == panel]
            cal = MODEL_SALARIES.get(panel)
            title = f"{panel} @ ${cal}" if cal else panel
        _render_panel_by_ethnicity(ax, sub, title,
                                   show_xlabel=True, show_ylabel=show_ylabel)

    eth_handles = [
        plt.Line2D([], [], color=ETHNICITY_COLORS[e], lw=2.2, label=e)
        for e in ETHNICITY_NAMES
    ]
    fig.legend(handles=eth_handles, loc="lower center",
               ncol=len(eth_handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"Hire rate by ethnicity (averaged over all names), by model — "
        f"age sweep; TRIALS={TRIALS}",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("wrote plot:", out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--csv", default=str(DEFAULT_CSV_PATH),
                   help=f"Input CSV (default: {DEFAULT_CSV_PATH}).")
    p.add_argument("--out", default=str(DEFAULT_PLOT_PATH),
                   help=f"Output PNG (default: {DEFAULT_PLOT_PATH}).")
    args = p.parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found — run the experiment first.")
    df = pd.read_csv(csv_path)
    plot(df, args.out)


if __name__ == "__main__":
    main()
