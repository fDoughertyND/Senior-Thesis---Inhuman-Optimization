"""ROC-style credit-score sweep plot.

Importable:

    from plot_roc_curves import plot
    plot(df, out_path)

Standalone — re-render from an existing roc_loan_results CSV without
re-querying the APIs:

    python3 plot_roc_curves.py                                 # uses the default CSV for TRIALS
    python3 plot_roc_curves.py --csv roc_loan_results_t25.csv
    python3 plot_roc_curves.py --csv foo.csv --out bar.png

The plot is a 6-column × 2-row grid: columns are each model + an
"avg across models" column; rows are masculine and feminine. Each panel
shows per-name spaghetti (thin lines per name) with a bolded mean line
and a dotted vertical marker at the model's calibrated "interesting"
credit score.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

from pairwise_name_experiments import MODEL_SCORES
from roc_loan_experiment import (
    CREDIT_SCORES, CSV_PATH as DEFAULT_CSV_PATH, ETHNICITY_NAMES,
    ETHNICITY_OF_NAME, GENDER_OF_NAME, MODEL_ORDER,
    PLOT_PATH as DEFAULT_PLOT_PATH, STRENGTH_OF_NAME, TRIALS,
)


# Solid line for the strong (high-approval) names, dashed for the weak ones.
STRENGTH_STYLE = {"strong": "-", "weak": "--"}


GENDER_COLORS = {
    "masculine": "#6FB4E3",  # light blue — matches plot_name_length
    "feminine":  "#F4A6C0",  # light pink  — matches plot_name_length
}

# One tab10 color per ethnicity, in the insertion order of ETHNICITY_NAMES.
_tab10 = plt.get_cmap("tab10")
ETHNICITY_COLORS = {
    e: _tab10(i % 10) for i, e in enumerate(ETHNICITY_NAMES)
}


def _normalized_auc(xs, ys):
    """Trapezoidal integral normalized to [0, 1] over the x range.

    Since credit-score range is fixed (580–700), this is just the mean
    approval rate along the trendline weighted by the x spacing — higher
    values mean the model approves more generously across the score range.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2:
        return float("nan")
    area = np.trapezoid(ys, xs)
    width = xs[-1] - xs[0]
    return float(area / width) if width > 0 else float("nan")


def _draw_gender_lines(ax, spag, mean_src, gender, color, draw_mean):
    """Draw spaghetti (solid for strong names, dashed for weak) and
    optionally a bolded trendline for one gender. No return value — AUCs
    are computed separately via _panel_aucs.
    """
    sub = spag[spag["gender"] == gender]
    spag_alpha = 0.55 if draw_mean else 0.85
    spag_lw = 1.0 if draw_mean else 1.9
    for name, g in sub.groupby("name"):
        g = g.sort_values("credit_score")
        linestyle = STRENGTH_STYLE.get(STRENGTH_OF_NAME.get(name, "strong"), "-")
        ax.plot(g["credit_score"], g["rate"],
                color=color, alpha=spag_alpha, lw=spag_lw,
                linestyle=linestyle)

    if not draw_mean:
        return
    mean_rows = mean_src[mean_src["gender"] == gender]
    if mean_rows.empty:
        return
    mean_line = (mean_rows.groupby("credit_score")["rate"].mean()
                          .reset_index().sort_values("credit_score"))
    ax.plot(mean_line["credit_score"], mean_line["rate"],
            color=color, lw=2.6, label=gender)


def _panel_aucs(spag, gender):
    """Per-name AUCs aggregated by strength for one gender.

    Returns {"best": mean AUC of strong names, "worst": mean AUC of weak
    names, "avg": mean AUC across all names}. Each value is the mean over
    whichever rows are present (handles multi-name aggregates cleanly).
    """
    sub = spag[spag["gender"] == gender]
    per_name = {}
    for name, g in sub.groupby("name"):
        g = g.sort_values("credit_score")
        per_name[name] = _normalized_auc(
            g["credit_score"].values, g["rate"].values,
        )

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
                  score_mark=None, draw_mean=True):
    """One panel: overlaid masculine (blue) and feminine (pink) lines.

    `spag` is the DataFrame used for spaghetti; `mean_src` is the DataFrame
    used for the bolded per-gender trendline. Pass the same DataFrame for
    both to get "spaghetti mean" behaviour; pass the full df as `mean_src`
    while filtering `spag` to an ethnicity to draw ethnicity spaghetti
    against the overall trend.
    """
    per_gender_aucs = {}
    for gender, color in GENDER_COLORS.items():
        _draw_gender_lines(ax, spag, mean_src, gender, color, draw_mean)
        per_gender_aucs[gender] = _panel_aucs(spag, gender)

    # Per-gender AUC annotation: best (strong-name mean), worst (weak-name
    # mean), and the avg. Sorted descending by AUC so the highest-approval
    # metric is at the top of the box. Lines are colored per gender, and
    # "best" is marked solid, "worst" dashed, matching the spaghetti.
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

    if score_mark is not None:
        ax.axvline(score_mark, color="gray", alpha=0.45, lw=1.0, linestyle=":")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(min(CREDIT_SCORES), max(CREDIT_SCORES))
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    if show_xlabel:
        ax.set_xlabel("Estimated credit score", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Approval rate", fontsize=9)


def plot(df: pd.DataFrame, out_path, ethnicity=None):
    """Render the per-model grid, with masculine (light blue) and feminine
    (light pink) lines overlaid in each panel.

    df is long-form (name, credit_score, model, rate). If `ethnicity` is
    provided, the spaghetti is filtered to that ethnicity's names while the
    bolded per-gender trendlines are still drawn from the FULL df — so each
    panel shows that ethnicity's names against the overall mean.
    """
    df = df.copy()
    df["gender"] = df["name"].map(GENDER_OF_NAME)
    df["ethnicity"] = df["name"].map(ETHNICITY_OF_NAME)
    df = df.dropna(subset=["gender"])

    if ethnicity is not None:
        spaghetti_df = df[df["ethnicity"] == ethnicity]
        if spaghetti_df.empty:
            raise SystemExit(f"No names matched ethnicity={ethnicity!r}")
    else:
        spaghetti_df = df

    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    # Only show the "avg across models" column on the aggregate (all-ethnicities)
    # figure; it isn't meaningful on a single-ethnicity panel set since the
    # trendline there is already the overall mean.
    cols = list(models_present)
    if ethnicity is None:
        cols = cols + ["__avg__"]

    # Cross-model mean per (name, credit_score, gender) so the avg column
    # has one line per name for spaghetti and one per gender for the mean.
    avg_full = (df.groupby(["name", "credit_score", "gender"], as_index=False)
                  ["rate"].mean())
    avg_spag = (spaghetti_df.groupby(["name", "credit_score", "gender"],
                                     as_index=False)["rate"].mean())

    ncols = len(cols)
    fig, axes = plt.subplots(1, ncols,
                             figsize=(4.6 * ncols, 4.6),
                             squeeze=False)

    for c, panel in enumerate(cols):
        ax = axes[0][c]
        show_ylabel = c == 0
        show_xlabel = True

        if panel == "__avg__":
            spag = avg_spag
            mean_src = avg_full
            title = f"avg across {len(models_present)} models"
            score_mark = None
        else:
            spag = spaghetti_df[spaghetti_df["model"] == panel]
            mean_src = df[df["model"] == panel]
            score = MODEL_SCORES.get(panel)
            title = f"{panel} @ {score}" if score is not None else panel
            score_mark = score

        # On per-ethnicity figures, skip the bold overall-mean trendline —
        # those lines dominate the handful of in-ethnicity spaghetti lines
        # and the ethnicity's own pattern is what we're trying to see.
        # AUC is computed from the ethnicity's own data (spag) instead.
        if ethnicity is not None:
            _render_panel(ax, spag, spag, title,
                          show_xlabel, show_ylabel,
                          score_mark=score_mark, draw_mean=False)
        else:
            _render_panel(ax, spag, mean_src, title,
                          show_xlabel, show_ylabel,
                          score_mark=score_mark, draw_mean=True)

    # Shared legend at the bottom: gender colors + strength line style.
    # Line width matches the spaghetti thickness actually used in the panels.
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
    # NAMES in roc_loan_experiment.py are the single strongest-approval
    # masculine + feminine name per ethnicity (picked from prior pairwise
    # runs), so make that explicit in the title.
    sample_note = "strongest ♂ & ♀ name per ethnicity"
    fig.suptitle(
        f"{prefix}Loan approval rate vs. credit score, by model "
        f"({sample_note}; N={n_names} names × {TRIALS} trials per point; "
        f"{trend_note})",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("wrote plot:", out_path)


# --- Ethnicity-averaged variant --------------------------------------------
# One line per ethnicity (averaged across its 4 names), no gender / strength
# split. Matches plot_roc_school_curves.plot_by_ethnicity.

def _render_panel_by_ethnicity(ax, df_panel, title, show_xlabel, show_ylabel,
                               score_mark=None):
    aucs = {}
    for eth, color in ETHNICITY_COLORS.items():
        sub = df_panel[df_panel["ethnicity"] == eth]
        if sub.empty:
            continue
        line = (sub.groupby("credit_score")["rate"].mean()
                  .reset_index().sort_values("credit_score"))
        ax.plot(line["credit_score"], line["rate"],
                color=color, lw=2.0, label=eth)
        aucs[eth] = _normalized_auc(line["credit_score"].values,
                                    line["rate"].values)

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

    if score_mark is not None:
        ax.axvline(score_mark, color="gray", alpha=0.45, lw=1.0, linestyle=":")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(min(CREDIT_SCORES), max(CREDIT_SCORES))
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    if show_xlabel:
        ax.set_xlabel("Estimated credit score", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Approval rate", fontsize=9)


def plot_by_ethnicity(df: pd.DataFrame, out_path):
    """Ethnicity-averaged variant of plot() — one line per ethnicity, no
    gender or strength split. One panel per model + an avg-across-models
    panel at the end.
    """
    df = df.copy()
    df["gender"] = df["name"].map(GENDER_OF_NAME)
    df["ethnicity"] = df["name"].map(ETHNICITY_OF_NAME)
    df = df.dropna(subset=["gender", "ethnicity"])
    if df.empty:
        raise SystemExit("No rows to plot.")

    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    cols = list(models_present) + ["__avg__"]
    # Cross-model mean per (name, credit_score, ethnicity) for the avg column.
    avg_df = (df.groupby(["name", "credit_score", "ethnicity"],
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
            score_mark = None
        else:
            sub = df[df["model"] == panel]
            title = (f"{panel} @ {MODEL_SCORES[panel]}"
                     if panel in MODEL_SCORES else panel)
            score_mark = MODEL_SCORES.get(panel)
        _render_panel_by_ethnicity(ax, sub, title,
                                   show_xlabel=True, show_ylabel=show_ylabel,
                                   score_mark=score_mark)

    eth_handles = [
        plt.Line2D([], [], color=ETHNICITY_COLORS[e], lw=2.2, label=e)
        for e in ETHNICITY_NAMES
    ]
    fig.legend(handles=eth_handles, loc="lower center",
               ncol=len(eth_handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"Approval rate by ethnicity (averaged over all names), by model — "
        f"TRIALS={TRIALS}",
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
