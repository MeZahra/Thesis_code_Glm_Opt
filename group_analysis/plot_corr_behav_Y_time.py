import argparse
import colorsys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECTION_PATH = REPO_ROOT / "projection_voxel_bold_thr90.npy"
DEFAULT_MANIFEST_PATH = Path("/Data/zahra/results_beta_preprocessed/group_concat/concat_manifest_group.tsv")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "behave_vs_bold" / "projection_voxel_bold_thr90_time"

REQUIRED_MANIFEST_COLUMNS = {
    "sub_tag",
    "ses",
    "run",
    "offset_start",
    "offset_end",
    "n_trials",
    "n_trials_source",
}

COND_ORDER = [(1, 1), (1, 2), (2, 1), (2, 2)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spaghetti plots for time-based projection signal with 9-TR trial grouping."
    )
    parser.add_argument("--projection-path", type=Path, default=DEFAULT_PROJECTION_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trial-len", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_manifest(path):
    manifest = pd.read_csv(path, sep="\t")
    missing_cols = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_cols)}")

    manifest = manifest.sort_values("offset_start").reset_index(drop=True)
    for col in ["ses", "run", "offset_start", "offset_end", "n_trials", "n_trials_source"]:
        manifest[col] = pd.to_numeric(manifest[col], errors="raise").astype(np.int64)
    return manifest


def infer_layout(total_timepoints, manifest_df, trial_len):
    kept_timepoints = int(manifest_df["n_trials"].sum()) * int(trial_len)
    source_timepoints = int(manifest_df["n_trials_source"].sum()) * int(trial_len)
    if total_timepoints == kept_timepoints:
        return "kept_only"
    if total_timepoints == source_timepoints:
        return "source_all"
    raise ValueError(
        "Projection length does not match manifest totals. "
        f"projection={total_timepoints}, kept={kept_timepoints}, source={source_timepoints}"
    )


def split_projection_to_runs(projection, manifest_df, trial_len, layout):
    run_segments = []
    cursor = 0
    for row in manifest_df.itertuples(index=False):
        n_trials = int(row.n_trials) if layout == "kept_only" else int(row.n_trials_source)
        span = int(n_trials) * int(trial_len)
        chunk = projection[cursor : cursor + span]
        if chunk.size != span:
            raise ValueError(
                f"Unexpected segment size for {row.sub_tag} ses-{row.ses} run-{row.run}: "
                f"expected {span}, got {chunk.size}."
            )
        run_segments.append(
            {
                "sub_tag": str(row.sub_tag),
                "ses": int(row.ses),
                "run": int(row.run),
                "n_trials": int(n_trials),
                "trial_matrix": chunk.reshape(n_trials, int(trial_len)),
            }
        )
        cursor += span
    if cursor != projection.size:
        raise ValueError(f"Did not consume full projection vector ({cursor} / {projection.size}).")
    return run_segments


def build_light_colormap(n_colors=256, saturation=0.28, value=0.95):
    hues = np.linspace(0.0, 1.0, int(n_colors), endpoint=False, dtype=np.float64)
    rgb = [colorsys.hsv_to_rgb(float(h), float(saturation), float(value)) for h in hues]
    return ListedColormap(rgb, name="light_hsv")


def plot_spaghetti(ax, trial_matrix, cmap, lw=1.0, alpha=0.92):
    trial_matrix = np.asarray(trial_matrix, dtype=np.float64)
    n_trials, trial_len = trial_matrix.shape
    tr_axis = np.arange(trial_len, dtype=np.int64)
    if n_trials <= 1:
        color_values = np.array([0.5], dtype=np.float64)
    else:
        color_values = np.linspace(0.0, 1.0, n_trials, dtype=np.float64)

    for i in range(n_trials):
        ax.plot(tr_axis, trial_matrix[i], color=cmap(float(color_values[i])), lw=lw, alpha=alpha)

    ax.set_xticks(tr_axis)
    ax.grid(alpha=0.2)


def select_four_diverse_runs(run_segments, rng):
    selected = []
    used_subjects = set()

    for ses, run in COND_ORDER:
        candidates = [seg for seg in run_segments if int(seg["ses"]) == ses and int(seg["run"]) == run]
        if not candidates:
            continue
        available = [seg for seg in candidates if seg["sub_tag"] not in used_subjects]
        pool = available if available else candidates
        choice = pool[int(rng.integers(0, len(pool)))]
        selected.append(choice)
        used_subjects.add(choice["sub_tag"])

    if len(selected) < 4:
        remaining = [
            seg for seg in run_segments if (seg["sub_tag"], seg["ses"], seg["run"]) not in
            {(s["sub_tag"], s["ses"], s["run"]) for s in selected}
        ]
        if remaining:
            extra_idx = rng.choice(len(remaining), size=min(4 - len(selected), len(remaining)), replace=False)
            for idx in np.sort(extra_idx):
                selected.append(remaining[int(idx)])

    if len(selected) < 4:
        raise ValueError(f"Need at least 4 runs to sample; found {len(selected)}.")
    return selected[:4]


def select_four_subjects_with_all_conditions(run_segments, rng):
    by_subject = {}
    for seg in run_segments:
        by_subject.setdefault(str(seg["sub_tag"]), set()).add((int(seg["ses"]), int(seg["run"])))

    eligible = [sub for sub, conds in by_subject.items() if set(COND_ORDER).issubset(conds)]
    eligible = np.array(sorted(eligible))
    if eligible.size < 4:
        raise ValueError(f"Need at least 4 eligible subjects with all session/run conditions; found {eligible.size}.")

    chosen = rng.choice(eligible, size=4, replace=False)
    return np.sort(chosen)


def plot_random_four_runs(run_segments, output_path, selection_csv_path, cmap, rng):
    selected = select_four_diverse_runs(run_segments, rng)
    selection_df = pd.DataFrame(
        [{"sub_tag": s["sub_tag"], "ses": s["ses"], "run": s["run"], "n_trials": s["n_trials"]} for s in selected]
    )
    selection_df.to_csv(selection_csv_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, seg in zip(axes, selected):
        plot_spaghetti(ax, seg["trial_matrix"], cmap=cmap)
        ax.set_title(f"{seg['sub_tag']} | ses-{seg['ses']} run-{seg['run']} | n={seg['n_trials']}")
        ax.set_ylabel("Projection signal")

    for ax in axes:
        ax.set_xlabel("TR within trial")

    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.88)
    cbar.set_label("Relative trial index (early -> late)")

    fig.suptitle("Random 4 runs: all trials over time (spaghetti)", fontsize=14)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return selection_df


def plot_random_four_subjects_4x4(run_segments, output_path, selection_csv_path, cmap, rng):
    selected_subjects = select_four_subjects_with_all_conditions(run_segments, rng)
    selection_df = pd.DataFrame({"sub_tag": selected_subjects})
    selection_df.to_csv(selection_csv_path, index=False)

    lookup = {(str(seg["sub_tag"]), int(seg["ses"]), int(seg["run"])): seg for seg in run_segments}

    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True, constrained_layout=True)
    axes = np.asarray(axes)

    for row_idx, sub_tag in enumerate(selected_subjects):
        for col_idx, (ses, run) in enumerate(COND_ORDER):
            ax = axes[row_idx, col_idx]
            seg = lookup.get((str(sub_tag), int(ses), int(run)))
            if seg is None:
                ax.axis("off")
                continue

            plot_spaghetti(ax, seg["trial_matrix"], cmap=cmap, lw=0.95, alpha=0.9)
            if row_idx == 0:
                ax.set_title(f"ses-{ses} run-{run}", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"{sub_tag}\nSignal", fontsize=10)
            else:
                ax.set_ylabel("Signal", fontsize=9)
            if row_idx == 3:
                ax.set_xlabel("TR within trial")
            ax.text(
                0.02,
                0.98,
                f"n={seg['n_trials']}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.7, "ec": "0.8"},
            )

    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("Relative trial index (early -> late)")

    fig.suptitle("Random 4 subjects: session/run spaghetti grid (4 x 4)", fontsize=14)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return selection_df


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projection = np.asarray(np.load(args.projection_path), dtype=np.float64).reshape(-1)
    manifest = load_manifest(args.manifest_path)
    layout = infer_layout(projection.size, manifest, trial_len=args.trial_len)
    run_segments = split_projection_to_runs(projection, manifest, trial_len=args.trial_len, layout=layout)

    rng = np.random.default_rng(args.seed)
    cmap = build_light_colormap()

    fig1_path = output_dir / "spaghetti_random4_runs.png"
    fig1_sel_path = output_dir / "spaghetti_random4_runs_selection.csv"
    fig2_path = output_dir / "spaghetti_random4subjects_4x4.png"
    fig2_sel_path = output_dir / "spaghetti_random4subjects_selection.csv"

    sel_runs_df = plot_random_four_runs(
        run_segments=run_segments,
        output_path=fig1_path,
        selection_csv_path=fig1_sel_path,
        cmap=cmap,
        rng=rng,
    )
    sel_subjects_df = plot_random_four_subjects_4x4(
        run_segments=run_segments,
        output_path=fig2_path,
        selection_csv_path=fig2_sel_path,
        cmap=cmap,
        rng=rng,
    )

    print(f"Projection path: {args.projection_path}")
    print(f"Manifest path: {args.manifest_path}")
    print(f"Output dir: {output_dir}")
    print(f"Trial length: {args.trial_len}")
    print(f"Inferred layout: {layout}")
    print(f"Runs available: {len(run_segments)}")
    print(f"Selected runs: {len(sel_runs_df)}")
    print(f"Selected subjects: {len(sel_subjects_df)}")
    print(f"Saved: {fig1_path}")
    print(f"Saved: {fig1_sel_path}")
    print(f"Saved: {fig2_path}")
    print(f"Saved: {fig2_sel_path}")


if __name__ == "__main__":
    main()
