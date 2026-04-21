"""Mega-image of per-ethnicity average confusion matrices.

For every pairwise CSV in the chosen results subfolder (loan or school),
computes the per-cell (first × last) Yes-rate averaged across all models
present in that CSV, then renders one panel per group in a single figure.

Cell colormap matches the existing pairwise plots (RdYlGn). When a group is
in the gendered set (3 male / 3 female firsts), a faint horizontal divider
is drawn between the two name blocks.

Usage:
    python3 plot_avg_confusion_mega.py                       # loan mega
    python3 plot_avg_confusion_mega.py --domain school       # school mega
    python3 plot_avg_confusion_mega.py --domain school --out custom.png --cols 2
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pairwise_name_experiments import name_lists


# --- Domain presets --------------------------------------------------------
# Each entry names the results directory, the file-suffix that identifies a
# pairwise CSV, the outcome label used in the suptitle, and the default
# output path for the mega image.

DOMAINS = {
    "loan": {
        "dir":        Path("results/pairwise_loan"),
        "infix":      "interesting_score_loans",
        "outcome":    "approval rate",
        "default_out": "results/analysis/avg_confusion_mega.png",
    },
    "school": {
        "dir":        Path("results/pairwise_school"),
        "infix":      "interesting_school_app",
        "outcome":    "acceptance rate",
        "default_out": "results/analysis/avg_confusion_mega_school.png",
    },
    "salary": {
        "dir":        Path("results/pairwise_salary"),
        "infix":      "interesting_salary",
        "outcome":    "hire rate",
        "default_out": "results/analysis/avg_confusion_mega_salary.png",
    },
}


def _discover(domain_cfg):
    """Return list of (group, trials, csv_path) for every CSV in the domain
    directory that matches the expected filename pattern.
    """
    pattern = re.compile(
        rf"^pairwise_(.+?)_{re.escape(domain_cfg['infix'])}_t(\d+)\.csv$"
    )
    entries = []
    for path in sorted(domain_cfg["dir"].glob("pairwise_*.csv")):
        m = pattern.match(path.name)
        if not m:
            continue
        group = m.group(1)
        trials = int(m.group(2))
        if group not in name_lists:
            print(f"warning: no name_lists entry for {group!r}, skipping {path.name}")
            continue
        entries.append((group, trials, path))
    return entries


# Groups whose first-name list is ordered 3 male / 3 female (everything else
# renders as a single unified block with split=None).
_GENDERED = {"spanish", "islamic", "indian", "caucasian",
             "african", "african_american", "israeli"}


def _split_for(group):
    return 3 if group in _GENDERED else None


def _avg_matrix(csv_path, group):
    """Return (firsts, lasts, matrix) where matrix[i, j] = mean approval
    across all model columns for name firsts[i]+lasts[j]. NaN where missing.
    """
    firsts = name_lists[group]["first"]
    lasts = name_lists[group]["last"]
    df = pd.read_csv(csv_path)
    models = [c for c in df.columns if c != "name"]
    by_name = {str(r["name"]).strip(): r for _, r in df.iterrows()}

    matrix = np.full((len(firsts), len(lasts)), np.nan)
    for i, first in enumerate(firsts):
        for j, last in enumerate(lasts):
            # Two token orders occur across groups; try both.
            candidates = (f"{first} {last}", f"{last} {first}")
            row = next((by_name[c] for c in candidates if c in by_name), None)
            if row is None:
                continue
            vals = []
            for m in models:
                try:
                    vals.append(float(row[m]))
                except (TypeError, ValueError):
                    pass
            if vals:
                matrix[i, j] = float(np.nanmean(vals))
    return firsts, lasts, matrix, models


def _annotate(ax, data, fs=7):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            color = "black" if 0.25 < v < 0.75 else "white"
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    color=color, fontsize=fs)


def _render_panel(fig, spec, firsts, lasts, matrix, split, title):
    """Render one group's avg confusion matrix with row-avg, col-avg, and
    grand-avg strips mirroring plot_pairwise_confusion's panel layout.

    When `split` is a valid index (0 < split < len(firsts)), a small vertical
    gap is inserted between rows [:split] and [split:] (the male/female break)
    and a matching gap is placed in the row-avg strip.
    """
    cmap = plt.get_cmap("RdYlGn")
    cell_fs = 7

    with np.errstate(all="ignore"):
        row_avg = np.nanmean(matrix, axis=1).reshape(-1, 1)
        col_avg = np.nanmean(matrix, axis=0).reshape(1, -1)
        grand = np.array([[np.nanmean(matrix)]])

    use_split = split is not None and 0 < split < len(firsts)

    if use_split:
        inner = spec.subgridspec(
            5, 3,
            height_ratios=[split, 0.25, len(firsts) - split, 0.32, 1],
            width_ratios=[len(lasts), 0.3, 1],
            hspace=0.08, wspace=0.08,
        )
        ax_top = fig.add_subplot(inner[0, 0])
        ax_bot = fig.add_subplot(inner[2, 0])
        ax_row_top = fig.add_subplot(inner[0, 2])
        ax_row_bot = fig.add_subplot(inner[2, 2])
        ax_col = fig.add_subplot(inner[4, 0])
        ax_corner = fig.add_subplot(inner[4, 2])

        ax_top.imshow(matrix[:split], vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax_top.set_yticks(range(split))
        ax_top.set_yticklabels(firsts[:split], fontsize=8)
        ax_top.set_xticks([])
        ax_top.set_title(title, fontsize=10)
        _annotate(ax_top, matrix[:split], fs=cell_fs)

        ax_bot.imshow(matrix[split:], vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax_bot.set_yticks(range(len(firsts) - split))
        ax_bot.set_yticklabels(firsts[split:], fontsize=8)
        ax_bot.set_xticks([])
        _annotate(ax_bot, matrix[split:], fs=cell_fs)

        ax_row_top.imshow(row_avg[:split], vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax_row_top.set_xticks([])
        ax_row_top.set_yticks([])
        ax_row_top.set_title("avg", fontsize=8)
        _annotate(ax_row_top, row_avg[:split], fs=cell_fs)

        ax_row_bot.imshow(row_avg[split:], vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax_row_bot.set_xticks([])
        ax_row_bot.set_yticks([])
        _annotate(ax_row_bot, row_avg[split:], fs=cell_fs)
    else:
        inner = spec.subgridspec(
            3, 3,
            height_ratios=[len(firsts), 0.32, 1],
            width_ratios=[len(lasts), 0.3, 1],
            hspace=0.08, wspace=0.08,
        )
        ax_main = fig.add_subplot(inner[0, 0])
        ax_row = fig.add_subplot(inner[0, 2])
        ax_col = fig.add_subplot(inner[2, 0])
        ax_corner = fig.add_subplot(inner[2, 2])

        ax_main.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax_main.set_yticks(range(len(firsts)))
        ax_main.set_yticklabels(firsts, fontsize=8)
        ax_main.set_xticks([])
        ax_main.set_title(title, fontsize=10)
        _annotate(ax_main, matrix, fs=cell_fs)

        ax_row.imshow(row_avg, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax_row.set_xticks([])
        ax_row.set_yticks([])
        ax_row.set_title("avg", fontsize=8)
        _annotate(ax_row, row_avg, fs=cell_fs)

    ax_col.imshow(col_avg, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax_col.set_yticks([0])
    ax_col.set_yticklabels(["avg"], fontsize=8)
    ax_col.set_xticks(range(len(lasts)))
    ax_col.set_xticklabels(lasts, rotation=45, ha="right", fontsize=8)
    _annotate(ax_col, col_avg, fs=cell_fs)

    ax_corner.imshow(grand, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax_corner.set_xticks([])
    ax_corner.set_yticks([])
    _annotate(ax_corner, grand, fs=max(cell_fs, 8))


def plot_mega(out_path, ncols=2, domain="loan"):
    if domain not in DOMAINS:
        raise SystemExit(f"Unknown domain {domain!r}; choose from {list(DOMAINS)}.")
    cfg = DOMAINS[domain]

    panels = []
    for group, trials, path in _discover(cfg):
        firsts, lasts, matrix, models = _avg_matrix(path, group)
        title = f"{group} (t={trials}, avg across {len(models)} models)"
        panels.append((firsts, lasts, matrix, _split_for(group), title))

    if not panels:
        raise SystemExit(
            f"No {domain} CSVs found under {cfg['dir']}/; run the matching "
            f"pairwise experiment first."
        )

    n = len(panels)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols

    # Scale panel size so labels + cells fit without clipping.
    max_firsts = max(len(p[0]) for p in panels)
    max_lasts = max(len(p[1]) for p in panels)
    panel_w = max(5.5, 0.8 * max_lasts + 2.5)
    panel_h = max(5.5, 0.8 * max_firsts + 2.5)

    fig = plt.figure(figsize=(panel_w * ncols, panel_h * nrows))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.45, wspace=0.4)

    for idx, (firsts, lasts, matrix, split, title) in enumerate(panels):
        spec = gs[idx // ncols, idx % ncols]
        _render_panel(fig, spec, firsts, lasts, matrix, split, title)

    for idx in range(n, nrows * ncols):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        ax.axis("off")

    # Shared colorbar.
    cbar_ax = fig.add_axes([0.93, 0.12, 0.010, 0.76])
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("RdYlGn"),
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label=cfg["outcome"].capitalize())

    fig.suptitle(
        f"Per-ethnicity average confusion matrices "
        f"({cfg['outcome']}, averaged over all models)",
        fontsize=14, y=0.995,
    )
    fig.subplots_adjust(left=0.06, right=0.91, top=0.95, bottom=0.05)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("wrote mega-image:", out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--domain", choices=list(DOMAINS), default="loan",
                   help="Which pairwise results to build the mega from "
                        "(default: loan).")
    p.add_argument("--out", default=None,
                   help="Output PNG path. Defaults depend on --domain.")
    p.add_argument("--cols", type=int, default=2,
                   help="Panels per row (default: 2).")
    args = p.parse_args()
    out = args.out or DOMAINS[args.domain]["default_out"]
    plot_mega(out, ncols=args.cols, domain=args.domain)


if __name__ == "__main__":
    main()
