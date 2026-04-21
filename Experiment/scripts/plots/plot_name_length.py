"""Model approval rate vs. name length.

For every pairwise CSV configured in plot_gender_gap.GROUPS, builds a long-form
table of (group, name, length, model, rate) where `length` is the character
count of the full name with spaces stripped. Renders a figure with one scatter
subplot per model showing every name point colored by ethnic group, plus a
linear-regression line and Pearson r annotation. A final "avg across models"
subplot uses each name's mean rate across models.

Usage:
    python3 plot_name_length.py                  # uses GROUPS from plot_gender_gap
    python3 plot_name_length.py --out foo.png
    python3 plot_name_length.py --csv <path> --csv <path>
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

from pairwise_name_experiments import MODEL_SCORES
import plot_gender_gap as _pgg
from plot_gender_gap import (
    DOMAINS, GROUPS, _csv_path, _ordered_names, _parse_csv_filename,
    _union_models,
)


def _resolve_entries(csv_args):
    """Reuse the group/trial config from GROUPS; --csv overrides by path.

    Returns tuples (group, trials, split, gender_axis, path). Falls back to
    globbing for any trial count if the exact (group, trials) file isn't in
    the current domain's directory — same behaviour as plot_gender_gap.
    """
    cfg_by_group = {g: (s, ax) for g, _, s, ax in GROUPS}
    if csv_args:
        entries = []
        for c in csv_args:
            group, trials = _parse_csv_filename(c)
            if group is None:
                raise SystemExit(f"Could not parse group/trials from {c}")
            split, axis = cfg_by_group.get(group, (3, "first"))
            entries.append((group, trials, split, axis, Path(c)))
        return entries
    entries = []
    domain_dir = DOMAINS[_pgg.DOMAIN]["dir"]
    infix = DOMAINS[_pgg.DOMAIN]["infix"]
    for group, trials, split, axis in GROUPS:
        path = _csv_path(group, trials)
        if not path.exists():
            candidates = sorted(domain_dir.glob(
                f"pairwise_{group}_{infix}_t*.csv"
            ))
            if candidates:
                path = candidates[-1]
                _, trials = _parse_csv_filename(path)
            else:
                print(f"warning: skipping missing {path}")
                continue
        entries.append((group, trials, split, axis, path))
    return entries


def _build_long_table(entries, models):
    """Return a long-form DataFrame with one row per (group, name, model).

    Includes a "gender" column ("masculine" | "feminine" | None) derived from
    each group's split+gender_axis config.
    """
    rows = []
    for group, _trials, split, axis, path in entries:
        try:
            male_names, female_names = _ordered_names(group, split, axis)
        except (KeyError, ValueError):
            male_names, female_names = [], []
        gender_of = {n: "masculine" for n in male_names}
        gender_of.update({n: "feminine" for n in female_names})

        df = pd.read_csv(path)
        for _, r in df.iterrows():
            name = str(r["name"]).strip()
            if not name:
                continue
            length = len(name.replace(" ", ""))
            gender = gender_of.get(name)
            for model in models:
                if model not in df.columns:
                    continue
                try:
                    rate = float(r[model])
                except (TypeError, ValueError):
                    continue
                rows.append((group, name, length, gender, model, rate))
    return pd.DataFrame(
        rows, columns=["group", "name", "length", "gender", "model", "rate"],
    )


def _fit_line(x, y):
    """Return (slope, intercept, pearson_r) or (nan, nan, nan) if degenerate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.allclose(x[mask], x[mask][0]):
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    with np.errstate(invalid="ignore"):
        r = float(np.corrcoef(x[mask], y[mask])[0, 1])
    return float(slope), float(intercept), r


GENDER_STYLE = {
    # "glyph" is a unicode stand-in for the matplotlib scatter marker so the
    # stat box can hint at shape in addition to color.
    "masculine": {"marker": "o", "line": "-",  "color": "#6FB4E3", "glyph": "●"},
    "feminine":  {"marker": "^", "line": "--", "color": "#F4A6C0", "glyph": "▲"},
}
OVERALL_COLOR = "black"


def _render_panel(ax, df_sub, group_colors, title, split_by):
    """Scatter + per-subset regression lines.

    split_by:
        "gender"    -> one fit line per gender (light blue / light pink).
        "ethnicity" -> one fit line per ethnic group (tab10 colors).
    The scatter always encodes ethnicity as color and gender as marker shape.
    """
    rng = np.random.default_rng(42)
    for gender, style in GENDER_STYLE.items():
        sub_g = df_sub[df_sub["gender"] == gender]
        if sub_g.empty:
            continue
        for group, color in group_colors.items():
            sub = sub_g[sub_g["group"] == group]
            if sub.empty:
                continue
            jitter = rng.uniform(-0.18, 0.18, size=len(sub))
            ax.scatter(sub["length"] + jitter, sub["rate"],
                       s=18, alpha=0.55, color=color,
                       marker=style["marker"], edgecolors="none")

    if split_by == "gender":
        fit_specs = [
            (f"{GENDER_STYLE[gender]['glyph']} {gender}",
             (df_sub["gender"] == gender),
             GENDER_STYLE[gender]["color"],
             GENDER_STYLE[gender]["line"])
            for gender in GENDER_STYLE
        ]
    elif split_by == "ethnicity":
        fit_specs = [
            (f"● {group}",
             (df_sub["group"] == group), group_colors[group], "-")
            for group in sorted(df_sub["group"].unique())
        ]
    else:
        raise ValueError(f"unknown split_by: {split_by!r}")

    # Thinner lines when splitting by ethnicity since there are many more of them.
    fit_lw = 2.0 if split_by == "gender" else 1.3
    # Each entry = (label_text, color) to render inline as a colored line in
    # the stat box. This replaces the separate fit-lines legend.
    entries = []
    for label, mask, color, linestyle in fit_specs:
        sub_g = df_sub[mask]
        slope, intercept, r = _fit_line(sub_g["length"].values,
                                        sub_g["rate"].values)
        if not np.isfinite(slope):
            entries.append((f"{label}: n={len(sub_g)} (fit n/a)", color))
            continue
        xs = np.array([sub_g["length"].min(), sub_g["length"].max()])
        ax.plot(xs, slope * xs + intercept, color=color, lw=fit_lw,
                linestyle=linestyle)
        entries.append((
            f"{label}: slope {slope*100:+.2f}pp/char, r={r:+.2f}, n={len(sub_g)}",
            color,
        ))

    slope_o, intercept_o, r_o = _fit_line(df_sub["length"].values,
                                          df_sub["rate"].values)
    if np.isfinite(slope_o):
        xs = np.array([df_sub["length"].min(), df_sub["length"].max()])
        ax.plot(xs, slope_o * xs + intercept_o, color=OVERALL_COLOR,
                lw=1.6, linestyle=":")
        entries.append((
            f"overall: slope {slope_o*100:+.2f}pp/char, r={r_o:+.2f}, n={len(df_sub)}",
            OVERALL_COLOR,
        ))

    # In gender-split mode the fit lines don't name each ethnicity, so the
    # stat box needs its own scatter-color key. Ethnicity-split mode is
    # already keyed because every group has its own fit entry.
    scatter_key = []
    if split_by == "gender":
        present_groups = sorted(df_sub["group"].unique())
        for g in present_groups:
            scatter_key.append((f"● {g}", group_colors[g]))

    text_areas = [
        TextArea(text, textprops=dict(color=color, fontsize=7.5))
        for text, color in scatter_key + entries
    ]
    packed = VPacker(children=text_areas, align="left", pad=0, sep=1)
    anchored = AnchoredOffsetbox(
        loc="upper left", child=packed, pad=0.3, borderpad=0.3,
        frameon=True, bbox_to_anchor=(0.02, 0.98),
        bbox_transform=ax.transAxes,
    )
    anchored.patch.set(facecolor="white", edgecolor="gray", alpha=0.85)
    ax.add_artist(anchored)

    ax.set_title(title, fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("name length (chars, spaces stripped)", fontsize=8)
    ax.set_ylabel(DOMAINS[_pgg.DOMAIN]["outcome"], fontsize=8)
    ax.grid(True, alpha=0.25)


def plot_name_length(entries, out_path, split_by="gender"):
    models = _union_models(entries)
    if not models:
        raise SystemExit("No model columns found in any CSV.")
    df = _build_long_table(entries, models)
    if df.empty:
        raise SystemExit("No data rows parsed; check input CSVs.")
    df = df.dropna(subset=["gender"])  # only masculine/feminine-classified names
    if df.empty:
        raise SystemExit("No gender-classified names; check GROUPS config.")

    groups = sorted(df["group"].unique())
    cmap = plt.get_cmap("tab10")
    group_colors = {g: cmap(i % 10) for i, g in enumerate(groups)}

    # One subplot per model, plus one for per-name mean across models.
    panels = models + ["__avg__"]
    n = len(panels)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    # ~20% larger than the previous 5.2 × 4.2 per-panel size.
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.25 * ncols, 5.05 * nrows),
                             squeeze=False)

    # Per-name mean across models (gender is constant per name, keep it).
    mean_df = (df.groupby(["group", "name", "length", "gender"],
                          as_index=False)["rate"].mean())

    for idx, panel in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        if panel == "__avg__":
            _render_panel(ax, mean_df, group_colors,
                          title="per-name avg across models",
                          split_by=split_by)
        else:
            title = (f"{panel} @ {MODEL_SCORES[panel]}"
                     if panel in MODEL_SCORES else panel)
            _render_panel(ax, df[df["model"] == panel], group_colors, title,
                          split_by=split_by)

    # Hide any leftover empty axes.
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    # All legend info — scatter colors, gender markers, fit-line styles — is
    # now shown inline in each panel's stat box.

    # When the figure covers a single ethnic group, lead the title with that
    # group name so per-group PNGs identify themselves at a glance.
    prefix = f"{groups[0]} — " if len(groups) == 1 else ""
    outcome = DOMAINS[_pgg.DOMAIN]["outcome"]
    fig.suptitle(
        f"{prefix}{outcome.capitalize()} vs. full-name length, by model "
        f"(fits split by {split_by})",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print("wrote plot:", out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--domain", choices=list(DOMAINS), default="loan",
                   help="Which pairwise results to read (default: loan).")
    p.add_argument("--csv", action="append", default=[],
                   help="Override: specific pairwise CSV to include (repeatable).")
    p.add_argument("--out", default=None,
                   help="Output PNG path. Defaults to "
                        "results/analysis/name_length_[<domain>_]<split>.png.")
    p.add_argument("--split", choices=("gender", "ethnicity"), default="gender",
                   help="What the per-panel fit lines break out by.")
    args = p.parse_args()

    _pgg.DOMAIN = args.domain  # flips CSV paths in plot_gender_gap helpers

    entries = _resolve_entries(args.csv)
    if not entries:
        raise SystemExit("No CSVs resolved; check GROUPS or --csv arguments.")
    tag = "" if args.domain == "loan" else f"{args.domain}_"
    out = args.out or f"results/analysis/name_length_{tag}{args.split}.png"
    plot_name_length(entries, out, split_by=args.split)


if __name__ == "__main__":
    main()
