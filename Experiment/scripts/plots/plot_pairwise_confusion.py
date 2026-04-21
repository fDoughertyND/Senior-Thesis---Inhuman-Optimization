"""Pairwise first × last name confusion-matrix plot.

Importable:

    from plot_pairwise_confusion import plot
    plot(firsts, lasts, model_names, rates, model_scores, group, trials, out_path)

Standalone — regenerate a plot from an existing pairwise CSV:

    python3 plot_pairwise_confusion.py pairwise_spanish_interesting_score_loans_t25.csv
    python3 plot_pairwise_confusion.py <csv> --group spanish --trials 25 --out mine.png

Group and trials are parsed from the filename when possible; pass flags to
override. Firsts/lasts/model-score mappings are pulled from
pairwise_name_experiments (name_lists, MODEL_SCORES).
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Default male/female boundary in the first-name list (first 3 rows, last 3).
# Pass split=None (or an out-of-range index) to render a single unified block
# instead — useful for lists that aren't 3-male/3-female or have >6 entries.
DEFAULT_SPLIT = 3


def plot(firsts, lasts, model_names, rates, model_scores, group, trials,
         out_path, split=DEFAULT_SPLIT, outcome="approval rate",
         layout=None):
    """Render the confusion-matrix plot and save to out_path.

    firsts, lasts: axis labels (lists of strings).
    model_names: ordered list of models to render.
    rates: dict[(first, last, model)] -> float in [0, 1] or -1/None if missing.
    model_scores: dict[model] -> credit score shown in subplot title (may omit).
    group, trials: used only in the suptitle.
    split: row index at which to insert a visual gap (e.g. male/female split).
        Pass None — or any value <=0 or >=len(firsts) — to render the matrix
        as a single block with no gap. This is the right choice for name lists
        that aren't gendered 3/3 or that have more than 6 rows.
    """
    if split is None or split <= 0 or split >= len(firsts):
        split = None  # normalize to "no split"
    cmap = plt.get_cmap("RdYlGn")

    # Build one panel per model, plus a final "across-models average" panel
    # (useful for filling the otherwise-empty corner when len(models) % 3 != 0,
    # and a handy overall summary regardless of layout).
    panels = []  # list of (title, matrix)
    for model in model_names:
        matrix = np.full((len(firsts), len(lasts)), np.nan)
        for i, first in enumerate(firsts):
            for j, last in enumerate(lasts):
                r = rates.get((first, last, model))
                if r is not None and r >= 0:
                    matrix[i, j] = r
        title = f"{model} @ {model_scores[model]}" if model in model_scores else model
        panels.append((title, matrix))

    if panels:
        stacked = np.stack([m for _, m in panels], axis=0)
        with np.errstate(all="ignore"):
            avg_matrix = np.nanmean(stacked, axis=0)
        panels.append((f"avg across {len(model_names)} models", avg_matrix))

    # layout=(rows, cols) overrides the default 2-column auto layout — useful
    # when a wide name list reads better at e.g. 3×2 than 2×3. Falls back to
    # the auto layout if layout is None or doesn't have room for every panel.
    if layout is not None and layout[0] * layout[1] >= len(panels):
        outer_rows, outer_cols = layout
    else:
        outer_cols = min(len(panels), 2)
        outer_rows = (len(panels) + outer_cols - 1) // outer_cols

    # Scale figure size to the data so large name lists stay readable.
    panel_w = max(6.5, 0.85 * len(lasts) + 1.5)
    panel_h = max(5.5, 0.55 * len(firsts) + 1.0)
    fig = plt.figure(figsize=(panel_w * outer_cols, panel_h * outer_rows + 0.5))
    outer = fig.add_gridspec(outer_rows, outer_cols, hspace=0.55, wspace=0.45)

    # Scale annotation font down a bit as the grid grows.
    cell_fs = max(5, min(9, int(80 / max(len(firsts), len(lasts)))))

    im = None
    for idx, (title, matrix) in enumerate(panels):
        with np.errstate(all="ignore"):
            row_avg = np.nanmean(matrix, axis=1).reshape(-1, 1)
            col_avg = np.nanmean(matrix, axis=0).reshape(1, -1)
            grand = np.array([[np.nanmean(matrix)]])

        cell = outer[idx // outer_cols, idx % outer_cols]

        if split is None:
            # Simple 3x3 layout: main matrix + row-avg strip + col-avg strip
            # + grand-avg corner, with small gaps.
            inner = cell.subgridspec(
                3, 3,
                height_ratios=[len(firsts), 0.28, 1],
                width_ratios=[len(lasts), 0.25, 1],
                hspace=0.05, wspace=0.05,
            )
            ax_main = fig.add_subplot(inner[0, 0])
            ax_row = fig.add_subplot(inner[0, 2])
            ax_col = fig.add_subplot(inner[2, 0])
            ax_corner = fig.add_subplot(inner[2, 2])

            im = ax_main.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
            ax_main.set_yticks(range(len(firsts)))
            ax_main.set_yticklabels(firsts)
            ax_main.set_xticks([])
            ax_main.set_title(title)
            _annotate(ax_main, matrix, fs=cell_fs)

            ax_row.imshow(row_avg, vmin=0, vmax=1, cmap=cmap, aspect="auto")
            ax_row.set_xticks([])
            ax_row.set_yticks([])
            ax_row.set_title("avg", fontsize=9)
            _annotate(ax_row, row_avg, fs=cell_fs)
        else:
            # 5x3 layout with a male/female (or arbitrary) gap between rows.
            inner = cell.subgridspec(
                5, 3,
                height_ratios=[split, 0.18, len(firsts) - split, 0.28, 1],
                width_ratios=[len(lasts), 0.25, 1],
                hspace=0.05, wspace=0.05,
            )
            ax_top = fig.add_subplot(inner[0, 0])
            ax_bot = fig.add_subplot(inner[2, 0])
            ax_row_top = fig.add_subplot(inner[0, 2])
            ax_row_bot = fig.add_subplot(inner[2, 2])
            ax_col = fig.add_subplot(inner[4, 0])
            ax_corner = fig.add_subplot(inner[4, 2])

            im = ax_top.imshow(matrix[:split], vmin=0, vmax=1, cmap=cmap, aspect="auto")
            ax_top.set_yticks(range(split))
            ax_top.set_yticklabels(firsts[:split])
            ax_top.set_xticks([])
            ax_top.set_title(title)
            _annotate(ax_top, matrix[:split], fs=cell_fs)

            ax_bot.imshow(matrix[split:], vmin=0, vmax=1, cmap=cmap, aspect="auto")
            ax_bot.set_yticks(range(len(firsts) - split))
            ax_bot.set_yticklabels(firsts[split:])
            ax_bot.set_xticks([])
            _annotate(ax_bot, matrix[split:], fs=cell_fs)

            ax_row_top.imshow(row_avg[:split], vmin=0, vmax=1, cmap=cmap, aspect="auto")
            ax_row_top.set_xticks([])
            ax_row_top.set_yticks([])
            ax_row_top.set_title("avg", fontsize=9)
            _annotate(ax_row_top, row_avg[:split], fs=cell_fs)

            ax_row_bot.imshow(row_avg[split:], vmin=0, vmax=1, cmap=cmap, aspect="auto")
            ax_row_bot.set_xticks([])
            ax_row_bot.set_yticks([])
            _annotate(ax_row_bot, row_avg[split:], fs=cell_fs)

        ax_col.imshow(col_avg, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax_col.set_yticks([0])
        ax_col.set_yticklabels(["avg"])
        ax_col.set_xticks(range(len(lasts)))
        ax_col.set_xticklabels(lasts, rotation=45, ha="right")
        _annotate(ax_col, col_avg, fs=cell_fs)

        ax_corner.imshow(grand, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax_corner.set_xticks([])
        ax_corner.set_yticks([])
        _annotate(ax_corner, grand, fs=max(cell_fs, 8))

    for idx in range(len(panels), outer_rows * outer_cols):
        blank = fig.add_subplot(outer[idx // outer_cols, idx % outer_cols])
        blank.axis("off")

    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.12, 0.012, 0.76])
        fig.colorbar(im, cax=cbar_ax, label="Approval rate")

    fig.suptitle(
        f"Pairwise {group} {outcome} (N={trials} per cell)",
        fontsize=14, y=0.995,
    )
    fig.subplots_adjust(left=0.05, right=0.91, top=0.93, bottom=0.07)

    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _annotate(ax, data, fs=7):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            color = "black" if 0.25 < v < 0.75 else "white"
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    color=color, fontsize=fs)


# --- CSV-driven standalone entry point ------------------------------------

def _rates_from_csv(df, firsts, lasts, model_names):
    """Rebuild the (first, last, model) -> rate dict from the legacy CSV
    (one row per full_name, one column per model)."""
    rates = {}
    lookup = {f"{f} {l}": (f, l) for f in firsts for l in lasts}
    for _, row in df.iterrows():
        name = row.get("name")
        if name not in lookup:
            continue
        first, last = lookup[name]
        for model in model_names:
            if model not in df.columns:
                continue
            try:
                rates[(first, last, model)] = float(row[model])
            except (TypeError, ValueError):
                pass
    return rates


def _parse_csv_filename(path):
    """Pull (group, trials) from 'pairwise_{group}_..._t{trials}.csv'."""
    m = re.match(r"pairwise_([^_]+)_.*_t(\d+)\.csv$", Path(path).name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("csv", help="pairwise CSV produced by pairwise_name_experiments.py")
    p.add_argument("--group", help="override group (defaults to parsing the filename)")
    p.add_argument("--trials", type=int, help="override trial count for the title")
    p.add_argument("--out", help="output PNG path (defaults to <csv-stem>.png)")
    p.add_argument("--outcome", default="approval rate",
                   help="outcome label in the suptitle "
                        "(e.g. 'approval rate', 'acceptance rate', 'hire rate').")
    split_grp = p.add_mutually_exclusive_group()
    split_grp.add_argument(
        "--split", type=int, default=DEFAULT_SPLIT,
        help=(f"row index for the male/female gap (default: {DEFAULT_SPLIT}). "
              "Use --no-split for lists that aren't gendered 3/3."),
    )
    split_grp.add_argument(
        "--no-split", dest="split", action="store_const", const=None,
        help="render each matrix as a single unified block (no row gap).",
    )
    args = p.parse_args()

    inferred_group, inferred_trials = _parse_csv_filename(args.csv)
    group = args.group or inferred_group
    if group is None:
        raise SystemExit(
            "Could not infer group from filename; pass --group."
        )
    trials = args.trials if args.trials is not None else (inferred_trials or 0)

    # Pull axis labels + calibrated scores from the experiment module.
    from pairwise_name_experiments import MODEL_SCORES, name_lists

    if group not in name_lists:
        raise SystemExit(
            f"Unknown group '{group}'. Known groups: {sorted(name_lists)}"
        )

    firsts = name_lists[group]["first"]
    lasts = name_lists[group]["last"]

    df = pd.read_csv(args.csv)
    model_names = [c for c in df.columns if c != "name"]
    rates = _rates_from_csv(df, firsts, lasts, model_names)

    out = args.out or str(Path(args.csv).with_suffix(".png"))
    plot(firsts, lasts, model_names, rates, MODEL_SCORES, group, trials, out,
         split=args.split, outcome=args.outcome)
    print("wrote plot:", out)


if __name__ == "__main__":
    main()
