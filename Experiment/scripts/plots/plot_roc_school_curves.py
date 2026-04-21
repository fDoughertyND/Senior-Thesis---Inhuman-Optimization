"""Per-school ROC-style curves for school admissions.

One figure per school: 1 row × (N models + avg) cols, masculine (light
blue) and feminine (light pink) lines overlaid, per-name spaghetti with
the per-gender mean bolded on top. x-axis is the student-profile tier
(A strongest → G weakest, with CD between C and D); y-axis is admission
rate. Per-panel AUC box breaks down by best (strong-name mean), worst
(weak-name mean), and avg, same as plot_roc_curves.

Importable:

    from plot_roc_school_curves import plot, plot_all
    plot_all(df, out_dir, trials)          # one PNG per school + mega

Standalone — re-render from the existing CSV without re-querying:

    python3 plot_roc_school_curves.py                         # per-school + mega
    python3 plot_roc_school_curves.py --school mit            # just MIT
    python3 plot_roc_school_curves.py --csv path --out-dir d  # overrides
"""
import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

from pairwise_school_app_experiment import PROFILES, SCHOOLS
from roc_loan_experiment import (
    ETHNICITY_NAMES, ETHNICITY_OF_NAME, GENDER_OF_NAME, STRENGTH_OF_NAME,
)
from roc_school_app_experiment import (
    CSV_PATH as DEFAULT_CSV_PATH, MODEL_ORDER, PROFILE_ORDER,
    TRIALS, _OUT_DIR as DEFAULT_OUT_DIR,
)


STRENGTH_STYLE = {"strong": "-", "weak": "--"}
GENDER_COLORS = {
    "masculine": "#6FB4E3",
    "feminine":  "#F4A6C0",
}

# One tab10 color per ethnicity, in the insertion order of ETHNICITY_NAMES.
_tab10 = plt.get_cmap("tab10")
ETHNICITY_COLORS = {
    e: _tab10(i % 10) for i, e in enumerate(ETHNICITY_NAMES)
}

# Map profile tags to integer x-positions (A=0, B=1, ..., G=7 with CD=3).
PROFILE_X = {p: i for i, p in enumerate(PROFILE_ORDER)}


def _normalized_auc(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2:
        return float("nan")
    area = np.trapezoid(ys, xs)
    width = xs[-1] - xs[0]
    return float(area / width) if width > 0 else float("nan")


def _draw_gender_lines(ax, spag, gender, color, draw_mean=True):
    """Per-name spaghetti for one gender (solid=strong, dashed=weak).

    When draw_mean is False the spaghetti is drawn thicker + more opaque so
    it reads well without the bolded mean trendline on top (used for
    single-ethnicity panels where the ethnicity's 4 spaghetti lines would
    otherwise be swamped by a cross-ethnicity mean).
    """
    sub = spag[spag["gender"] == gender]
    spag_alpha = 0.55 if draw_mean else 0.85
    spag_lw = 1.0 if draw_mean else 1.9
    for name, g in sub.groupby("name"):
        g = g.sort_values("profile_x")
        linestyle = STRENGTH_STYLE.get(STRENGTH_OF_NAME.get(name, "strong"), "-")
        ax.plot(g["profile_x"], g["rate"],
                color=color, alpha=spag_alpha, lw=spag_lw, linestyle=linestyle)

    if not draw_mean or sub.empty:
        return
    mean_line = (sub.groupby("profile_x")["rate"].mean()
                    .reset_index().sort_values("profile_x"))
    ax.plot(mean_line["profile_x"], mean_line["rate"],
            color=color, lw=2.6, label=gender)


def _panel_aucs(spag, gender):
    sub = spag[spag["gender"] == gender]
    per_name = {}
    for name, g in sub.groupby("name"):
        g = g.sort_values("profile_x")
        per_name[name] = _normalized_auc(
            g["profile_x"].values, g["rate"].values,
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


def _render_panel(ax, spag, title, show_xlabel, show_ylabel, draw_mean=True):
    per_gender_aucs = {}
    for gender, color in GENDER_COLORS.items():
        _draw_gender_lines(ax, spag, gender, color, draw_mean=draw_mean)
        per_gender_aucs[gender] = _panel_aucs(spag, gender)

    # AUC annotation, colored per gender, sorted by AUC descending.
    rows = []
    for gender, color in GENDER_COLORS.items():
        aucs = per_gender_aucs[gender]
        short = "♂" if gender == "masculine" else "♀"
        for kind, symbol in (("best", "—"), ("worst", "--"), ("avg", "•")):
            v = aucs.get(kind, float("nan"))
            val_str = f"{v:.2f}" if not np.isnan(v) else "—"
            label = f"{symbol} {short} {kind:<5} AUC = {val_str}"
            rows.append((v, label, color))
    # NaNs fall to the bottom.
    rows.sort(key=lambda r: float("-inf") if np.isnan(r[0]) else r[0],
              reverse=True)
    auc_entries = [
        TextArea(label, textprops=dict(color=color, fontsize=7.0))
        for _, label, color in rows
    ]
    if auc_entries:
        packed = VPacker(children=auc_entries, align="left", pad=0, sep=1)
        anchored = AnchoredOffsetbox(
            loc="upper right", child=packed, pad=0.3, borderpad=0.3,
            frameon=True, bbox_to_anchor=(0.98, 0.98),
            bbox_transform=ax.transAxes,
        )
        anchored.patch.set(facecolor="white", edgecolor="gray", alpha=0.85)
        ax.add_artist(anchored)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.3, len(PROFILE_ORDER) - 0.7)
    ax.set_xticks(range(len(PROFILE_ORDER)))
    ax.set_xticklabels(PROFILE_ORDER, fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    if show_xlabel:
        ax.set_xlabel("Profile tier (A strongest → G weakest)", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Admission rate", fontsize=9)


def _render_panel_by_ethnicity(ax, df_panel, title, show_xlabel, show_ylabel):
    """One panel: one averaged line per ethnicity (no gender/strength split).

    df_panel should already be filtered to a single model (or averaged across
    models for the avg column). Each ethnicity's line = mean rate across its
    names at each profile tier. AUC per ethnicity is annotated in a compact
    colored box.
    """
    aucs = {}
    for eth, color in ETHNICITY_COLORS.items():
        sub = df_panel[df_panel["ethnicity"] == eth]
        if sub.empty:
            continue
        line = (sub.groupby("profile_x")["rate"].mean()
                  .reset_index().sort_values("profile_x"))
        ax.plot(line["profile_x"], line["rate"],
                color=color, lw=2.0, label=eth)
        aucs[eth] = _normalized_auc(line["profile_x"].values,
                                    line["rate"].values)

    if aucs:
        # Sort descending by AUC so the strongest-approving ethnicity is at
        # the top of the box — NaNs fall to the bottom.
        def _sort_key(eth):
            v = aucs[eth]
            return (float("-inf") if np.isnan(v) else v)
        ordered = sorted(aucs.keys(), key=_sort_key, reverse=True)
        entries = []
        for eth in ordered:
            v = aucs[eth]
            val = f"{v:.2f}" if not np.isnan(v) else "—"
            entries.append(TextArea(
                f"{eth:<16s} AUC = {val}",
                textprops=dict(color=ETHNICITY_COLORS[eth], fontsize=7.0),
            ))
        packed = VPacker(children=entries, align="left", pad=0, sep=1)
        anchored = AnchoredOffsetbox(
            loc="upper right", child=packed, pad=0.3, borderpad=0.3,
            frameon=True, bbox_to_anchor=(0.98, 0.98),
            bbox_transform=ax.transAxes,
        )
        anchored.patch.set(facecolor="white", edgecolor="gray", alpha=0.85)
        ax.add_artist(anchored)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.3, len(PROFILE_ORDER) - 0.7)
    ax.set_xticks(range(len(PROFILE_ORDER)))
    ax.set_xticklabels(PROFILE_ORDER, fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    if show_xlabel:
        ax.set_xlabel("Profile tier (A strongest → G weakest)", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Admission rate", fontsize=9)


def plot_by_ethnicity(df: pd.DataFrame, out_path, school=None):
    """Ethnicity-averaged variant of plot() — one line per ethnicity, no
    gender or strength split. If `school` is given, filters to that school;
    otherwise collapses across schools (cross-school ethnicity summary).
    """
    df = _prepare(df)
    if school is not None:
        df = df[df["school"] == school]
        school_label = SCHOOLS.get(school, school)
    else:
        # Collapse across schools: mean per (name, profile, model).
        df = (df
              .groupby(["name", "profile", "profile_x", "gender",
                        "ethnicity", "model"], as_index=False)
              ["rate"].mean())
        school_label = f"avg across {df['model'].nunique()} models × all schools"
    if df.empty:
        raise SystemExit("No rows to plot.")

    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    cols = list(models_present) + ["__avg__"]
    # Cross-model mean per (name, profile, ethnicity) for the avg column.
    avg_df = (df.groupby(["name", "profile", "profile_x", "ethnicity"],
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
            title = panel
        _render_panel_by_ethnicity(ax, sub, title,
                                   show_xlabel=True, show_ylabel=show_ylabel)

    # Shared legend of the 8 ethnicity colors.
    eth_handles = [
        plt.Line2D([], [], color=ETHNICITY_COLORS[e], lw=2.2, label=e)
        for e in ETHNICITY_NAMES
    ]
    fig.legend(handles=eth_handles, loc="lower center",
               ncol=len(eth_handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    prefix = "" if school is None else f"{school_label} — "
    title_school = "cross-school average" if school is None else school_label
    fig.suptitle(
        f"{prefix}admission rate by ethnicity (averaged over all names) — "
        f"{title_school}; TRIALS={TRIALS}",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("wrote plot:", out_path)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Attach gender, ethnicity, and numeric profile_x columns; drop rows
    whose name isn't in GENDER_OF_NAME (e.g. stray rows)."""
    df = df.copy()
    df["gender"] = df["name"].map(GENDER_OF_NAME)
    df["ethnicity"] = df["name"].map(ETHNICITY_OF_NAME)
    df["profile_x"] = df["profile"].map(PROFILE_X)
    df = df.dropna(subset=["gender", "profile_x"])
    return df


def plot(df: pd.DataFrame, out_path, school: str, ethnicity=None):
    """Render one figure for one school (N model panels + avg).

    If `ethnicity` is provided the per-panel spaghetti is filtered to that
    ethnicity's 4 names (strong ♂/♀ + weak ♂/♀ for that group) and the
    bold cross-ethnicity mean trendline is omitted — same convention the
    loan plotter uses so the ethnicity's own lines stay readable. Per-
    panel AUC is then computed from that ethnicity's own data.
    """
    df = _prepare(df)
    df = df[df["school"] == school]
    if df.empty:
        raise SystemExit(f"No rows for school={school!r}")

    if ethnicity is not None:
        spaghetti_df = df[df["ethnicity"] == ethnicity]
        if spaghetti_df.empty:
            raise SystemExit(
                f"No rows for ethnicity={ethnicity!r} in school={school!r}"
            )
    else:
        spaghetti_df = df

    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    # Drop the "avg across models" column on per-ethnicity plots (the bold
    # trendline is turned off there so avg is less meaningful).
    cols = list(models_present)
    if ethnicity is None:
        cols = cols + ["__avg__"]

    avg_df = (spaghetti_df
              .groupby(["name", "profile_x", "profile", "gender"],
                       as_index=False)["rate"].mean())

    ncols = len(cols)
    fig, axes = plt.subplots(1, ncols,
                             figsize=(4.6 * ncols, 4.8),
                             squeeze=False)

    for c, panel in enumerate(cols):
        ax = axes[0][c]
        show_ylabel = c == 0
        if panel == "__avg__":
            spag = avg_df
            title = f"avg across {len(models_present)} models"
        else:
            spag = spaghetti_df[spaghetti_df["model"] == panel]
            title = panel
        _render_panel(ax, spag, title,
                      show_xlabel=True, show_ylabel=show_ylabel,
                      draw_mean=(ethnicity is None))

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
    school_name = SCHOOLS.get(school, school)
    prefix = f"{ethnicity} — " if ethnicity is not None else ""
    trend_note = ("per-name lines only (no overall trendline)"
                  if ethnicity is not None
                  else "trendline = mean per gender across names")
    fig.suptitle(
        f"{prefix}{school_name} — admission rate vs. profile tier "
        f"(strongest + weakest ♂/♀ name per ethnicity; "
        f"N={n_names} × {TRIALS} trials per point; {trend_note})",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("wrote plot:", out_path)


def _build_mega(png_paths, out_path, ncols=1):
    """Stack per-school PNGs into one mega image (1 col × N rows by default)."""
    imgs = [mpimg.imread(p) for p in png_paths]
    n = len(imgs)
    nrows = (n + ncols - 1) // ncols

    row_heights = []
    for r in range(nrows):
        row_imgs = imgs[r * ncols:(r + 1) * ncols]
        row_heights.append(max(img.shape[0] / img.shape[1] for img in row_imgs))

    tile_w = 25.2
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


def plot_school_ethnicity_mega(df: pd.DataFrame, school: str,
                               out_dir: Path, trials: int) -> Path:
    """For one school, emit one PNG per ethnicity + the all-ethnicity
    aggregate for that school, then stitch them into a per-school mega.
    Returns the path of the mega image.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # Aggregate (all ethnicities in this school) at the top of the mega.
    agg_path = out_dir / (
        f"roc_extreme_names_school_{school}_curves_t{trials}.png"
    )
    plot(df, agg_path, school=school)
    written.append(agg_path)

    # One PNG per ethnicity — spaghetti filtered to that group.
    present_ethnicities = [
        e for e in ETHNICITY_NAMES if e in df["ethnicity"].unique()
    ]
    for eth in present_ethnicities:
        path = out_dir / (
            f"roc_extreme_names_school_{school}_{eth}_curves_t{trials}.png"
        )
        plot(df, path, school=school, ethnicity=eth)
        written.append(path)

    mega_path = out_dir / (
        f"roc_extreme_names_school_{school}_by_ethnicity_t{trials}.png"
    )
    _build_mega(written, mega_path, ncols=1)
    return mega_path


def plot_cross_school(df: pd.DataFrame, out_path: Path, trials: int):
    """One figure collapsing all schools: for each (name, profile, model) we
    take the mean rate across schools first, then render the standard
    N-model + avg panel grid. Useful as the bottom row of the cross-school
    mega to show the overall profile-tier curve per model.
    """
    df = _prepare(df)
    # Pretend there's one synthetic "all" school so we can reuse plot().
    collapsed = (df
                 .groupby(["name", "profile", "profile_x", "gender",
                           "ethnicity", "model"], as_index=False)
                 ["rate"].mean())
    collapsed["school"] = "__all__"
    # plot() looks up the school in SCHOOLS for a pretty title; patch that too.
    SCHOOLS["__all__"] = f"avg across {df['school'].nunique()} schools"
    try:
        plot(collapsed, out_path, school="__all__")
    finally:
        SCHOOLS.pop("__all__", None)


def plot_all(df: pd.DataFrame, out_dir: Path, trials: int):
    """Emit, for every school in df:
        - one aggregate PNG (all names)
        - one PNG per ethnicity (spaghetti filtered to that group)
        - one per-school mega stacking the above
       Plus a cross-school mega stacking:
        - each school's aggregate row,
        - a final "avg across all schools" row at the very bottom.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _prepare(df)

    schools_present = [s for s in SCHOOLS if s in df["school"].unique()]
    aggregate_paths = []
    for school in schools_present:
        plot_school_ethnicity_mega(df, school, out_dir, trials)
        # The per-school aggregate (gender-split) was written inside the
        # per-school mega; pair it with the per-school ethnicity-avg view.
        aggregate_paths.append(out_dir / (
            f"roc_extreme_names_school_{school}_curves_t{trials}.png"
        ))
        eth_avg_path = out_dir / (
            f"roc_extreme_names_school_{school}_ethnicity_avg_t{trials}.png"
        )
        plot_by_ethnicity(df, eth_avg_path, school=school)
        aggregate_paths.append(eth_avg_path)

    # Cross-school rows: first the gender-split avg, then the ethnicity-avg.
    cross_path = out_dir / (
        f"roc_extreme_names_school_cross_school_avg_t{trials}.png"
    )
    plot_cross_school(df, cross_path, trials)
    aggregate_paths.append(cross_path)

    cross_eth_path = out_dir / (
        f"roc_extreme_names_school_cross_school_ethnicity_avg_t{trials}.png"
    )
    plot_by_ethnicity(df, cross_eth_path, school=None)
    aggregate_paths.append(cross_eth_path)

    mega_path = out_dir / f"roc_extreme_names_school_all_t{trials}.png"
    _build_mega(aggregate_paths, mega_path, ncols=1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--csv", default=str(DEFAULT_CSV_PATH),
                   help=f"Input CSV (default: {DEFAULT_CSV_PATH}).")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                   help=f"Output directory (default: {DEFAULT_OUT_DIR}).")
    p.add_argument("--school", default=None,
                   help="Render only this school (default: every school + mega).")
    p.add_argument("--ethnicity", default=None,
                   help="With --school: filter spaghetti to this ethnicity.")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found — run the experiment first.")
    df = pd.read_csv(csv_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.school is not None:
        stem = f"roc_extreme_names_school_{args.school}"
        if args.ethnicity is not None:
            stem = f"{stem}_{args.ethnicity}"
        path = out_dir / f"{stem}_curves_t{TRIALS}.png"
        plot(df, path, school=args.school, ethnicity=args.ethnicity)
    else:
        plot_all(df, out_dir, trials=TRIALS)


if __name__ == "__main__":
    main()
