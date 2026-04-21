"""Single-ethnicity version of plot_roc_curves + mega-image combiner.

Renders the same 6-column × 2-row model × gender grid as plot_roc_curves,
but filters the spaghetti lines to ONE ethnic group while the bolded
trendline stays the all-ethnicity overall mean. With --all, generates one
PNG per ethnicity plus an "all ethnicities" aggregate, then stacks the
whole set into a mega-image (one row per figure, so tall + narrow).

Usage:
    python3 plot_roc_curves_single.py --group spanish
    python3 plot_roc_curves_single.py --group chinese --out custom.png
    python3 plot_roc_curves_single.py --all              # one PNG per group + mega
"""
import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

from roc_loan_experiment import (
    CSV_PATH as DEFAULT_CSV_PATH, ETHNICITY_NAMES,
    PLOT_PATH as DEFAULT_PLOT_PATH, SAMPLING, TRIALS,
)
from plot_roc_curves import plot as plot_roc, plot_by_ethnicity


def _load_df(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found — run the experiment first.")
    return pd.read_csv(csv_path)


def _build_mega(png_paths, out_path, ncols=1):
    """Stack PNG tiles into one mega-image in a grid of ncols columns."""
    imgs = [mpimg.imread(p) for p in png_paths]
    n = len(imgs)
    nrows = (n + ncols - 1) // ncols

    # Row heights scale with each tile's native aspect so nothing gets
    # squashed when rows have different shapes.
    row_heights = []
    for r in range(nrows):
        row_imgs = imgs[r * ncols:(r + 1) * ncols]
        row_heights.append(max(img.shape[0] / img.shape[1] for img in row_imgs))

    tile_w = 25.2  # matches plot_roc_curves' 4.2 * 6-column figure width
    fig_h = sum(row_heights) * tile_w
    fig = plt.figure(figsize=(tile_w * ncols, fig_h))
    gs = fig.add_gridspec(nrows, ncols, height_ratios=row_heights,
                          hspace=0.03, wspace=0.02)

    for idx, img in enumerate(imgs):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for idx in range(n, nrows * ncols):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        ax.axis("off")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("wrote mega-image:", out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--group",
                     help="Single ethnic group (must exist in ETHNICITY_NAMES).")
    sel.add_argument("--all", action="store_true",
                     help="Render one PNG per ethnicity + aggregate, then "
                          "composite a mega-image.")
    p.add_argument("--csv", default=str(DEFAULT_CSV_PATH),
                   help=f"Input CSV (default: {DEFAULT_CSV_PATH}).")
    p.add_argument("--out", default=None,
                   help="Output PNG path (ignored with --all). "
                        f"Defaults to results/roc_loan/roc_{SAMPLING}_loans_<group>.png.")
    p.add_argument("--mega-out",
                   default=f"results/roc_loan/roc_{SAMPLING}_loans_all_t{TRIALS}.png",
                   help="Filename for the combined mega-image (with --all).")
    p.add_argument("--mega-cols", type=int, default=1,
                   help="Columns in the mega-image tile grid (default: 1).")
    args = p.parse_args()

    df = _load_df(Path(args.csv))

    if args.all:
        # Mega layout matches the school plotter's alternation idiom: the
        # two aggregate views (gender-split + ethnicity-avg) sit together
        # at the top, then the per-ethnicity detail rows follow.
        aggregate_path = str(DEFAULT_PLOT_PATH)
        plot_roc(df, aggregate_path)
        eth_avg_path = (
            f"results/roc_loan/roc_{SAMPLING}_loans_ethnicity_avg_t{TRIALS}.png"
        )
        plot_by_ethnicity(df, eth_avg_path)

        written = [aggregate_path, eth_avg_path]
        for group in ETHNICITY_NAMES:
            out = f"results/roc_loan/roc_{SAMPLING}_loans_{group}_t{TRIALS}.png"
            plot_roc(df, out, ethnicity=group)
            written.append(out)
        _build_mega(written, args.mega_out, ncols=args.mega_cols)
    else:
        if args.group not in ETHNICITY_NAMES:
            raise SystemExit(
                f"Unknown group {args.group!r}. "
                f"Known: {list(ETHNICITY_NAMES)}"
            )
        out = args.out or f"results/roc_loan/roc_{SAMPLING}_loans_{args.group}_t{TRIALS}.png"
        plot_roc(df, out, ethnicity=args.group)


if __name__ == "__main__":
    main()
