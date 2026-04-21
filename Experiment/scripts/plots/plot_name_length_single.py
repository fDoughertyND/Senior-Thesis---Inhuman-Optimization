"""Single-ethnicity version of plot_name_length.

Produces the same 6-panel figure (one per model + an across-models panel) with
masculine / feminine / overall regression lines, but restricted to ONE ethnic
group so the scatter isn't cluttered.

Usage:
    python3 plot_name_length_single.py --group spanish
    python3 plot_name_length_single.py --group african_american --out foo.png
    python3 plot_name_length_single.py --all          # one PNG per group,
                                                       # plus a combined mega-image.
"""
import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import plot_gender_gap as _pgg
from plot_gender_gap import DOMAINS, GROUPS, _csv_path, _parse_csv_filename
from plot_name_length import plot_name_length


def _entries_for_group(group):
    """Look up (group, trials, split, gender_axis, path) for the current
    plot_gender_gap.DOMAIN. Falls back to globbing if the exact trial count
    in GROUPS isn't present in the chosen domain directory.
    """
    for g, trials, split, axis in GROUPS:
        if g != group:
            continue
        path = _csv_path(group, trials)
        if not path.exists():
            domain_dir = DOMAINS[_pgg.DOMAIN]["dir"]
            infix = DOMAINS[_pgg.DOMAIN]["infix"]
            candidates = sorted(domain_dir.glob(
                f"pairwise_{group}_{infix}_t*.csv"
            ))
            if not candidates:
                raise SystemExit(
                    f"No CSV for group {group!r} under {domain_dir}/"
                )
            path = candidates[-1]
            _, trials = _parse_csv_filename(path)
        return [(g, trials, split, axis, path)]
    raise SystemExit(
        f"Unknown group {group!r}. Known: {[g for g, *_ in GROUPS]}"
    )


def _build_mega(png_paths, out_path, ncols=1):
    """Stack PNG tiles into one mega-image. Tile heights may differ (the
    per-group tiles are 2×3 panels; the aggregate tile at the bottom is
    the same shape, so one column stacks cleanly)."""
    imgs = [mpimg.imread(p) for p in png_paths]
    n = len(imgs)
    nrows = (n + ncols - 1) // ncols

    # Row-heights proportional to each image's pixel height so nothing
    # gets squashed when the tile aspect differs between rows.
    row_heights = []
    for r in range(nrows):
        row_imgs = imgs[r * ncols:(r + 1) * ncols]
        row_heights.append(max(img.shape[0] / img.shape[1] for img in row_imgs))

    # Base tile width ~ each per-group figure's native inches (18.75in).
    tile_w = 18.75
    fig_h = sum(row_heights) * tile_w  # each row_height is height/width
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
                     help="Single ethnic group name (must exist in GROUPS).")
    sel.add_argument("--all", action="store_true",
                     help="Render one PNG per group configured in GROUPS, "
                          "then composite a mega-image.")
    p.add_argument("--domain", choices=list(DOMAINS), default="loan",
                   help="Which pairwise results to read (default: loan).")
    p.add_argument("--out", default=None,
                   help="Output PNG path (ignored with --all). "
                        "Defaults to results/analysis/"
                        "name_length_[<domain>_]<group>.png.")
    p.add_argument("--mega-out", default=None,
                   help="Combined mega-image filename (with --all). "
                        "Defaults to results/analysis/"
                        "name_length_[<domain>_]all.png.")
    p.add_argument("--mega-cols", type=int, default=1,
                   help="Columns in the mega-image tile grid (default: 1, "
                        "so each per-group figure becomes a full-width row).")
    args = p.parse_args()

    _pgg.DOMAIN = args.domain
    tag = "" if args.domain == "loan" else f"{args.domain}_"

    out_dir = "results/analysis"
    mega_out = args.mega_out or f"{out_dir}/name_length_{tag}all.png"

    if args.all:
        written = []
        for group, *_ in GROUPS:
            try:
                entries = _entries_for_group(group)
            except SystemExit as e:
                print(f"warning: {e}")
                continue
            out = f"{out_dir}/name_length_{tag}{group}.png"
            plot_name_length(entries, out, split_by="gender")
            written.append(out)

        # Aggregate figures at the bottom of the mega.
        aggregate_entries = []
        domain_dir = DOMAINS[args.domain]["dir"]
        infix = DOMAINS[args.domain]["infix"]
        for g, t, s, ax in GROUPS:
            p_ = _csv_path(g, t)
            if not p_.exists():
                candidates = sorted(domain_dir.glob(
                    f"pairwise_{g}_{infix}_t*.csv"
                ))
                if not candidates:
                    continue
                p_ = candidates[-1]
                _, t = _parse_csv_filename(p_)
            aggregate_entries.append((g, t, s, ax, p_))

        ethnicity_agg = f"{out_dir}/name_length_{tag}ethnicity.png"
        gender_agg = f"{out_dir}/name_length_{tag}gender.png"
        plot_name_length(aggregate_entries, ethnicity_agg, split_by="ethnicity")
        plot_name_length(aggregate_entries, gender_agg, split_by="gender")
        written.extend([ethnicity_agg, gender_agg])
        _build_mega(written, mega_out, ncols=args.mega_cols)
    else:
        entries = _entries_for_group(args.group)
        out = args.out or f"{out_dir}/name_length_{tag}{args.group}.png"
        plot_name_length(entries, out, split_by="gender")


if __name__ == "__main__":
    main()
