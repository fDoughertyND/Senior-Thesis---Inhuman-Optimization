"""Per-name approval-rate heatmaps across ethnic pairwise results.

For each pairwise CSV, render a heatmap where:
  - rows = every full name in the group, grouped male-first then female with
    a visual gap between the two blocks
  - cols = each model (ordered by calibrated credit score) + an
    "avg across models" column
  - cells = actual loan approval rate (0-1) on an RdYlGn colormap

"Male" vs "female" is determined by a configurable axis ("first" or "last"
in name_lists[group]) and a split index — tokens [:split] are male,
[split:] are female.

Usage:
    python3 plot_gender_gap.py                       # uses GROUPS below
    python3 plot_gender_gap.py --out gap.png
    python3 plot_gender_gap.py --csv path/to/pairwise_foo_...csv --csv ...

To add a new group: drop its CSV following the
`pairwise_{group}_interesting_score_loans_t{N}.csv` naming and append a
(group, trials, split, gender_axis) entry to GROUPS. `split` defaults to 3
(3 male / 3 female); `gender_axis` is "first" for western-order groups and
"last" for Chinese-order (surname-first) groups.
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pairwise_name_experiments import MODEL_SCORES, name_lists


# (group, trials, split, gender_axis). Extend this list as more groups are run.
# gender_axis is "first" (western-order: given-name carries gender) or
# "last"  (surname-first order, e.g. Chinese: second token carries gender).
GROUPS = [
    ("spanish",          10, 3, "first"),
    ("islamic",          10, 3, "first"),
    ("indian",           10, 3, "first"),
    ("caucasian",        10, 3, "first"),
    ("african",          10, 3, "first"),
    ("african_american", 25, 3, "first"),
    ("israeli",          25, 3, "first"),
    ("chinese",          10, 3, "last"),
]


# --- Domain presets --------------------------------------------------------
# Each entry describes where to find the pairwise CSVs, how to parse their
# filenames, what label to use in titles, and where to write the default
# heatmap. "loan" is the default for backward compatibility; other modules
# (plot_name_length, plot_avg_confusion_mega) still read loan data.

DOMAINS = {
    "loan": {
        "dir":        Path("results/pairwise_loan"),
        "infix":      "interesting_score_loans",
        "outcome":    "approval rate",
        "default_out": "results/analysis/gender_gap_heatmap.png",
    },
    "school": {
        "dir":        Path("results/pairwise_school"),
        "infix":      "interesting_school_app",
        "outcome":    "acceptance rate",
        "default_out": "results/analysis/gender_gap_heatmap_school.png",
    },
    "salary": {
        "dir":        Path("results/pairwise_salary"),
        "infix":      "interesting_salary",
        "outcome":    "hire rate",
        "default_out": "results/analysis/gender_gap_heatmap_salary.png",
    },
}

# Mutable so the CLI can override it at startup without threading a kwarg
# through every helper. Callers that import _csv_path / GROUPS without
# touching the CLI (e.g. plot_name_length_single) see the "loan" defaults.
DOMAIN = "loan"


def _csv_path(group, trials):
    cfg = DOMAINS[DOMAIN]
    return cfg["dir"] / f"pairwise_{group}_{cfg['infix']}_t{trials}.csv"


def _parse_csv_filename(path):
    # Group name can contain underscores (e.g. "african_american"), so capture
    # everything up to the domain-specific infix.
    infix = re.escape(DOMAINS[DOMAIN]["infix"])
    m = re.match(
        rf"pairwise_(.+?)_{infix}_t(\d+)\.csv$",
        Path(path).name,
    )
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _ordered_names(group, split, gender_axis):
    """Return (male_names, female_names) as ordered lists of full-name strings.

    Male block iterates male gender-tokens × all other-axis tokens;
    female block is the same for female gender-tokens.
    Full-name token order follows the CSV convention for the group
    (surname-first for Chinese, given-first otherwise).
    """
    firsts = name_lists[group]["first"]
    lasts = name_lists[group]["last"]

    if gender_axis == "first":
        male_tokens, female_tokens = firsts[:split], firsts[split:]
        other_tokens = lasts
        compose = lambda gtok, otok: f"{gtok} {otok}"
    elif gender_axis == "last":
        male_tokens, female_tokens = lasts[:split], lasts[split:]
        other_tokens = firsts
        compose = lambda gtok, otok: f"{otok} {gtok}"
    else:
        raise ValueError(f"gender_axis must be 'first' or 'last', got {gender_axis!r}")

    male_names = [compose(g, o) for g in male_tokens for o in other_tokens]
    female_names = [compose(g, o) for g in female_tokens for o in other_tokens]
    return male_names, female_names


def _rates_matrix(csv_path, names, models):
    """Build a len(names) × len(models) matrix of approval rates (NaN if missing)."""
    df = pd.read_csv(csv_path)
    by_name = {str(r["name"]).strip(): r for _, r in df.iterrows()}

    mat = np.full((len(names), len(models)), np.nan)
    for i, name in enumerate(names):
        row = by_name.get(name)
        if row is None:
            continue
        for j, model in enumerate(models):
            if model not in df.columns:
                continue
            try:
                mat[i, j] = float(row[model])
            except (TypeError, ValueError):
                pass
    return mat


def _resolve_entries(csv_args):
    """Return list of (group, trials, split, gender_axis, csv_path). CLI overrides GROUPS.

    For --csv, gender_axis and split default to whatever is configured for the
    group in GROUPS (falling back to split=3, axis="first" if unlisted).
    """
    cfg_by_group = {g: (s, ax) for g, _, s, ax in GROUPS}
    entries = []
    if csv_args:
        for c in csv_args:
            group, trials = _parse_csv_filename(c)
            if group is None:
                raise SystemExit(f"Could not parse group/trials from {c}")
            if group not in name_lists:
                raise SystemExit(f"Unknown group '{group}' (not in name_lists)")
            split, axis = cfg_by_group.get(group, (3, "first"))
            entries.append((group, trials, split, axis, Path(c)))
        return entries
    # Trial counts in GROUPS reflect the loan runs. For other domains the
    # same group may have been run at a different trial count, so if the
    # exact (group, trials) file isn't there, glob for any trial count for
    # that group and use the first match.
    domain_dir = DOMAINS[DOMAIN]["dir"]
    infix = DOMAINS[DOMAIN]["infix"]
    for group, trials, split, axis in GROUPS:
        path = _csv_path(group, trials)
        if not path.exists():
            candidates = sorted(domain_dir.glob(
                f"pairwise_{group}_{infix}_t*.csv"
            ))
            if candidates:
                path = candidates[-1]  # highest-trials match
                _, trials = _parse_csv_filename(path)
            else:
                print(f"warning: skipping missing {path}")
                continue
        entries.append((group, trials, split, axis, path))
    return entries


def _union_models(entries):
    """Models ordered by MODEL_SCORES (with any unknown models appended)."""
    all_models = set()
    for _, _, _, _, path in entries:
        cols = [c for c in pd.read_csv(path, nrows=0).columns if c != "name"]
        all_models.update(cols)
    ordered = [m for m in MODEL_SCORES if m in all_models]
    ordered += sorted(all_models - set(ordered))
    return ordered


def _annotate(ax, data, row_offset=0, fs=7):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i + row_offset, "—", ha="center", va="center",
                        color="black", fontsize=fs)
                continue
            color = "black" if 0.25 < v < 0.75 else "white"
            ax.text(j, i + row_offset, f"{v:.0%}", ha="center", va="center",
                    color=color, fontsize=fs)


def _render_panel(fig, spec, group, trials, male_names, female_names,
                  models, matrix):
    """Render one group's heatmap into the gridspec cell `spec`.

    The male block and female block are shown as two imshow axes with a small
    vertical gap between them; a shared col-axis of models runs underneath.
    """
    n_male = len(male_names)
    n_female = len(female_names)
    n_cols = len(models) + 1  # +1 for avg-across-models

    inner = spec.subgridspec(
        2, 1,
        height_ratios=[n_male, n_female],
        hspace=0.06,
    )
    ax_top = fig.add_subplot(inner[0, 0])
    ax_bot = fig.add_subplot(inner[1, 0])

    cmap = plt.get_cmap("RdYlGn")

    im = ax_top.imshow(matrix[:n_male], vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax_top.set_yticks(range(n_male))
    ax_top.set_yticklabels(male_names, fontsize=7)
    ax_top.set_xticks([])
    ax_top.set_title(f"{group} (t={trials})  masculine top / feminine bottom",
                     fontsize=11)
    # Visual separator before the avg-across-models column.
    ax_top.axvline(len(models) - 0.5, color="black", lw=1.0)
    _annotate(ax_top, matrix[:n_male], fs=6)

    ax_bot.imshow(matrix[n_male:], vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax_bot.set_yticks(range(n_female))
    ax_bot.set_yticklabels(female_names, fontsize=7)
    ax_bot.set_xticks(range(n_cols))
    col_labels = [
        f"{m}\n@{MODEL_SCORES[m]}" if m in MODEL_SCORES else m for m in models
    ] + ["avg across\nmodels"]
    ax_bot.set_xticklabels(col_labels, fontsize=8)
    ax_bot.axvline(len(models) - 0.5, color="black", lw=1.0)
    _annotate(ax_bot, matrix[n_male:], fs=6)

    return im


def _render_summary(fig, spec, panels, models):
    """Bottom heatmap: rows grouped male-block-then-female-block (one row per
    ethnicity in each block) with a gap between, cols = models, plus a
    detached "avg across models" column of boxes on the right.

    Cells are the mean approval rate across all names in that
    (group, gender) block, for that model.
    """
    groups = [p[0] for p in panels]
    n_groups = len(groups)

    male_rows = np.full((n_groups, len(models)), np.nan)
    female_rows = np.full((n_groups, len(models)), np.nan)
    for gi, (_, _, male_names, female_names, mat) in enumerate(panels):
        n_male = len(male_names)
        rate_mat = mat[:, :len(models)]  # drop the appended avg-across-models col
        with np.errstate(all="ignore"):
            male_rows[gi] = np.nanmean(rate_mat[:n_male], axis=0)
            female_rows[gi] = np.nanmean(rate_mat[n_male:], axis=0)

    with np.errstate(all="ignore"):
        male_avg = np.nanmean(male_rows, axis=1, keepdims=True)
        female_avg = np.nanmean(female_rows, axis=1, keepdims=True)

    # Each gender block has a per-model avg-across-ethnicities strip below it,
    # and the right avg-across-models column has a matching overall corner cell.
    # Layout: 5 rows × 3 cols.
    #   row 0: male block          row 1: male per-model avg strip
    #   row 2: spacer gap
    #   row 3: female block        row 4: female per-model avg strip
    # cols: main heatmap | spacer | avg-across-models column
    inner = spec.subgridspec(
        5, 3,
        height_ratios=[n_groups, 1, 0.55, n_groups, 1],
        width_ratios=[len(models), 0.3, 1.0],
        hspace=0.08, wspace=0.06,
    )
    ax_m        = fig.add_subplot(inner[0, 0])
    ax_m_strip  = fig.add_subplot(inner[1, 0])
    ax_m_avg    = fig.add_subplot(inner[0, 2])
    ax_m_corner = fig.add_subplot(inner[1, 2])
    ax_f        = fig.add_subplot(inner[3, 0])
    ax_f_strip  = fig.add_subplot(inner[4, 0])
    ax_f_avg    = fig.add_subplot(inner[3, 2])
    ax_f_corner = fig.add_subplot(inner[4, 2])

    cmap = plt.get_cmap("RdYlGn")

    def _draw_block(ax, block, label_suffix, deltas=None):
        """If deltas is provided (same shape as block), each cell is annotated
        with its value plus a signed delta (e.g. "42% (+5pp)").
        """
        ax.imshow(block, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_yticks(range(n_groups))
        ax.set_yticklabels(
            [f"{g} {label_suffix}" for g in groups], fontsize=9,
        )
        ax.set_xticks([])
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                v = block[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center",
                            color="black", fontsize=8)
                    continue
                color = "black" if 0.25 < v < 0.75 else "white"
                if deltas is not None and not np.isnan(deltas[i, j]):
                    ax.text(j, i, f"{v:.0%} ({deltas[i, j] * 100:+.0f})",
                            ha="center", va="center",
                            color=color, fontsize=8)
                else:
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                            color=color, fontsize=9)

    def _draw_strip(ax, row, ylabel, show_xticks, deltas=None):
        # A single-row strip: avg across ethnicities per model.
        ax.imshow(row[np.newaxis, :], vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_yticks([0])
        ax.set_yticklabels([ylabel], fontsize=9)
        if show_xticks:
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(
                [f"{m}\n@{MODEL_SCORES[m]}" if m in MODEL_SCORES else m
                 for m in models],
                fontsize=9,
            )
        else:
            ax.set_xticks([])
        for j in range(row.shape[0]):
            v = row[j]
            if np.isnan(v):
                ax.text(j, 0, "—", ha="center", va="center",
                        color="black", fontsize=9)
                continue
            color = "black" if 0.25 < v < 0.75 else "white"
            if deltas is not None and not np.isnan(deltas[j]):
                ax.text(j, 0, f"{v:.0%} ({deltas[j] * 100:+.0f})",
                        ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")
            else:
                ax.text(j, 0, f"{v:.0%}", ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")

    def _draw_avg_col(ax, col, deltas=None):
        # Per-ethnicity avg across models (one cell per ethnicity row).
        ax.imshow(col, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_yticks([])
        ax.set_xticks([])
        for i in range(col.shape[0]):
            v = col[i, 0]
            if np.isnan(v):
                ax.text(0, i, "—", ha="center", va="center",
                        color="black", fontsize=9)
                continue
            color = "black" if 0.25 < v < 0.75 else "white"
            if deltas is not None and not np.isnan(deltas[i, 0]):
                ax.text(0, i, f"{v:.0%} ({deltas[i, 0] * 100:+.0f})",
                        ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")
            else:
                ax.text(0, i, f"{v:.0%}", ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")

    def _draw_corner(ax, value, show_xticks, delta=None):
        ax.imshow(np.array([[value]]), vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_yticks([])
        if show_xticks:
            ax.set_xticks([0])
            ax.set_xticklabels(["avg across\nmodels"], fontsize=9)
        else:
            ax.set_xticks([])
        if np.isnan(value):
            ax.text(0, 0, "—", ha="center", va="center", fontsize=10)
            return
        color = "black" if 0.25 < value < 0.75 else "white"
        if delta is not None and not np.isnan(delta):
            ax.text(0, 0, f"{value:.0%} ({delta * 100:+.0f})",
                    ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")
        else:
            ax.text(0, 0, f"{value:.0%}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    with np.errstate(all="ignore"):
        male_strip = np.nanmean(male_rows, axis=0)       # per-model avg over ethnicities
        female_strip = np.nanmean(female_rows, axis=0)
        male_overall = float(np.nanmean(male_rows))
        female_overall = float(np.nanmean(female_rows))

    # Per-ethnicity delta (feminine − masculine), shown only in the feminine
    # cells so each row directly quotes its gap.
    female_minus_male = female_rows - male_rows
    female_minus_male_avg = female_avg - male_avg

    strip_delta = female_strip - male_strip
    overall_delta = female_overall - male_overall

    _draw_block(ax_m, male_rows, "masculine")
    _draw_block(ax_f, female_rows, "feminine", deltas=female_minus_male)
    _draw_strip(ax_m_strip, male_strip, "masculine\navg", show_xticks=False)
    _draw_strip(ax_f_strip, female_strip, "feminine\navg", show_xticks=True,
                deltas=strip_delta)
    _draw_avg_col(ax_m_avg, male_avg)
    _draw_avg_col(ax_f_avg, female_avg, deltas=female_minus_male_avg)
    _draw_corner(ax_m_corner, male_overall, show_xticks=False)
    _draw_corner(ax_f_corner, female_overall, show_xticks=True,
                 delta=overall_delta)

    outcome = DOMAINS[DOMAIN]["outcome"]
    ax_m.set_title(
        f"Mean {outcome} by (ethnicity, gender) × model  —  "
        f"masculine top / feminine bottom",
        fontsize=11,
    )


def _render_ethnicity_summary(fig, spec, panels, models):
    """Heatmap: rows = ethnicities (gender-agnostic), cols = models,
    plus an "avg across models" column on the right and a per-model
    avg-across-ethnicities strip below, meeting at a bolded overall corner.
    """
    groups = [p[0] for p in panels]
    n_groups = len(groups)

    block = np.full((n_groups, len(models)), np.nan)
    for gi, (_, _, _, _, mat) in enumerate(panels):
        rate_mat = mat[:, :len(models)]  # drop appended avg-across-models col
        with np.errstate(all="ignore"):
            block[gi] = np.nanmean(rate_mat, axis=0)  # all names in the group

    with np.errstate(all="ignore"):
        right_avg = np.nanmean(block, axis=1, keepdims=True)

    # 1 row × 3 cols: main block | narrow gap | avg-across-models column.
    inner = spec.subgridspec(
        1, 3,
        width_ratios=[len(models), 0.08, 1.0],
        wspace=0.02,
    )
    ax     = fig.add_subplot(inner[0, 0])
    ax_avg = fig.add_subplot(inner[0, 2])

    cmap = plt.get_cmap("RdYlGn")

    # aspect="equal" forces each cell to render as a true square. All four
    # axes share the same unit cell, so they line up into one coherent grid.
    ax.imshow(block, vmin=0, vmax=1, cmap=cmap, aspect="equal")
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(groups, fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(
        [f"{m}\n@{MODEL_SCORES[m]}" if m in MODEL_SCORES else m for m in models],
        fontsize=9,
    )
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            v = block[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        color="black", fontsize=8)
                continue
            color = "black" if 0.25 < v < 0.75 else "white"
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    color=color, fontsize=9)

    ax_avg.imshow(right_avg, vmin=0, vmax=1, cmap=cmap, aspect="equal")
    ax_avg.set_yticks([])
    ax_avg.set_xticks([0])
    ax_avg.set_xticklabels(["avg across\nmodels"], fontsize=9)
    ax_avg.xaxis.set_label_position("top")
    ax_avg.xaxis.tick_top()
    for i in range(right_avg.shape[0]):
        v = right_avg[i, 0]
        if np.isnan(v):
            ax_avg.text(0, i, "—", ha="center", va="center",
                        color="black", fontsize=9)
            continue
        color = "black" if 0.25 < v < 0.75 else "white"
        ax_avg.text(0, i, f"{v:.0%}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    # Title sits above the top-of-axis x-tick labels.
    ax.set_title(
        f"Mean {DOMAINS[DOMAIN]['outcome']} by ethnicity × model "
        f"(genders combined)",
        fontsize=11, pad=36,
    )


def plot_per_name_heatmap(entries, out_path):
    models = _union_models(entries)
    if not models:
        raise SystemExit("No model columns found in any CSV.")

    # Precompute panels so we know figure sizing.
    panels = []
    max_rows = 0
    for group, trials, split, axis, path in entries:
        male_names, female_names = _ordered_names(group, split, axis)
        names = male_names + female_names
        rate_mat = _rates_matrix(path, names, models)
        with np.errstate(all="ignore"):
            avg_col = np.nanmean(rate_mat, axis=1, keepdims=True)
        matrix = np.hstack([rate_mat, avg_col])
        panels.append((group, trials, male_names, female_names, matrix))
        max_rows = max(max_rows, len(names))

    # 2 columns × ceil(N/2) rows of per-name panels, plus a summary row below.
    inner_cols = min(len(panels), 2)
    inner_rows = (len(panels) + inner_cols - 1) // inner_cols

    panel_w = max(7.0, 1.1 * (len(models) + 1) + 3.5)
    panel_h = max(5.5, 0.22 * max_rows + 1.5)
    # Gender summary: 2 gender blocks (n_groups rows each) + 2 per-model avg strips.
    gender_h = max(3.5, 0.45 * (len(panels) * 2 + 2) + 1.8)
    # Ethnicity summary: 1 block (n_groups rows). Cell size ~0.75in → ~50%
    # larger than the earlier ~0.5in. aspect="equal" inside the panel clamps
    # actual cell size to whichever of width/height is tighter.
    ethnicity_h = max(3.5, 0.75 * len(panels) + 2.0)
    fig = plt.figure(
        figsize=(panel_w * inner_cols,
                 panel_h * inner_rows + gender_h + ethnicity_h + 1.2)
    )
    # Top: per-name panels; middle: gender summary; bottom: ethnicity summary.
    outer = fig.add_gridspec(
        3, 1,
        height_ratios=[panel_h * inner_rows, gender_h, ethnicity_h],
        hspace=0.22,
    )
    top_grid = outer[0].subgridspec(inner_rows, inner_cols, hspace=0.45, wspace=0.55)

    im = None
    for idx, (group, trials, male_names, female_names, matrix) in enumerate(panels):
        spec = top_grid[idx // inner_cols, idx % inner_cols]
        im = _render_panel(fig, spec, group, trials,
                           male_names, female_names, models, matrix) or im

    for idx in range(len(panels), inner_rows * inner_cols):
        blank = fig.add_subplot(top_grid[idx // inner_cols, idx % inner_cols])
        blank.axis("off")

    # Narrow the summary heatmaps to ~67% of the full figure width so the
    # individual cells read narrower, without shrinking the figure overall.
    def _pad(spec, content_frac=0.67):
        pad = (1.0 - content_frac) / 2
        sub = spec.subgridspec(
            1, 3,
            width_ratios=[pad, content_frac, pad],
            wspace=0.0,
        )
        return sub[0, 1]

    _render_summary(fig, _pad(outer[1]), panels, models)
    _render_ethnicity_summary(fig, _pad(outer[2]), panels, models)

    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.12, 0.010, 0.76])
        fig.colorbar(im, cax=cbar_ax, label="Approval rate")

    fig.suptitle(
        f"Per-name {DOMAINS[DOMAIN]['outcome']}s by model "
        f"(male block above, female below)",
        fontsize=14, y=0.995,
    )
    fig.subplots_adjust(left=0.06, right=0.91, top=0.95, bottom=0.05)

    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("wrote plot:", out_path)


def main():
    global DOMAIN
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--domain", choices=list(DOMAINS), default="loan",
                   help="Which pairwise results to read (default: loan).")
    p.add_argument("--csv", action="append", default=[],
                   help="Override: specific pairwise CSV to include (repeatable).")
    p.add_argument("--out", default=None,
                   help="Output PNG path. Defaults depend on --domain.")
    args = p.parse_args()

    DOMAIN = args.domain  # flips _csv_path / _parse_csv_filename to the right paths
    out = args.out or DOMAINS[args.domain]["default_out"]

    entries = _resolve_entries(args.csv)
    if not entries:
        raise SystemExit("No CSVs resolved; check GROUPS or --csv arguments.")
    plot_per_name_heatmap(entries, out)


if __name__ == "__main__":
    main()
