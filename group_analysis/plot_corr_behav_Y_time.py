import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze trial-time projection (TR-level) with emphasis on amplitude changes and temporal shift "
            "within/between subjects and sessions."
        )
    )
    parser.add_argument("--projection-path", type=Path, default=DEFAULT_PROJECTION_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trial-len", type=int, default=9, help="Number of TRs per trial.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--detail-subjects",
        type=int,
        default=0,
        help="Number of subjects for detailed per-subject figures; 0 means all subjects.",
    )
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
    global_trial_cursor = 0

    for row in manifest_df.itertuples(index=False):
        n_trials = int(row.n_trials) if layout == "kept_only" else int(row.n_trials_source)
        n_timepoints = n_trials * int(trial_len)
        start = cursor
        stop = cursor + n_timepoints
        run_flat = projection[start:stop]
        if run_flat.size != n_timepoints:
            raise ValueError(
                f"Unexpected segment size for {row.sub_tag} ses-{row.ses} run-{row.run}: "
                f"expected {n_timepoints}, got {run_flat.size}."
            )

        run_trials = run_flat.reshape(n_trials, int(trial_len))
        run_segments.append(
            {
                "sub_tag": str(row.sub_tag),
                "ses": int(row.ses),
                "run": int(row.run),
                "n_trials": int(n_trials),
                "trial_start_global": int(global_trial_cursor),
                "trial_end_global": int(global_trial_cursor + n_trials),
                "time_start": int(start),
                "time_end": int(stop),
                "trial_matrix": run_trials,
            }
        )
        cursor = stop
        global_trial_cursor += n_trials

    if cursor != projection.size:
        raise ValueError(f"Did not consume full projection vector ({cursor} / {projection.size}).")
    return run_segments


def best_lag_corr(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"x and y must have the same size, got {x.size} vs {y.size}.")

    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        lags = np.arange(-(x.size - 1), x.size, dtype=np.int64)
        corr = np.full(lags.shape, np.nan, dtype=np.float64)
        return np.nan, np.nan, lags, corr

    x0 = x[finite] - np.mean(x[finite])
    y0 = y[finite] - np.mean(y[finite])
    denom = np.linalg.norm(x0) * np.linalg.norm(y0)
    lags = np.arange(-(x0.size - 1), x0.size, dtype=np.int64)
    if (not np.isfinite(denom)) or np.isclose(denom, 0.0):
        corr = np.full(lags.shape, np.nan, dtype=np.float64)
        return np.nan, np.nan, lags, corr

    corr = np.correlate(x0, y0, mode="full") / denom
    best_idx = int(np.nanargmax(np.abs(corr)))
    return int(lags[best_idx]), float(corr[best_idx]), lags, corr


def shift_with_nan(x, lag):
    x = np.asarray(x, dtype=np.float64).ravel()
    lag = int(lag)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if lag == 0:
        out[:] = x
        return out
    if lag > 0:
        out[lag:] = x[:-lag]
    else:
        out[:lag] = x[-lag:]
    return out


def _subplot_grid(n_panels, max_cols=4):
    n_panels = int(max(1, n_panels))
    n_cols = int(min(max_cols, n_panels))
    n_rows = int(np.ceil(n_panels / n_cols))
    return n_rows, n_cols


def _safe_nanmax_minus_nanmin(x):
    x = np.asarray(x, dtype=np.float64)
    if np.count_nonzero(np.isfinite(x)) == 0:
        return np.nan
    return float(np.nanmax(x) - np.nanmin(x))


def build_trial_metrics(run_segments, trial_len):
    rows = []
    signals = []
    tr_cols = [f"tr{i}" for i in range(int(trial_len))]

    for seg in run_segments:
        mat = np.asarray(seg["trial_matrix"], dtype=np.float64)
        for trial_idx in range(mat.shape[0]):
            signal = np.asarray(mat[trial_idx], dtype=np.float64)
            finite_signal = signal[np.isfinite(signal)]
            if finite_signal.size == 0:
                baseline = np.nan
                peak_val = np.nan
                trough_val = np.nan
                mean_val = np.nan
                auc_sum = np.nan
                peak_tr = np.nan
            else:
                baseline = float(np.nanmean(signal[:2])) if signal.size >= 2 else float(np.nanmean(signal))
                peak_val = float(np.nanmax(signal))
                trough_val = float(np.nanmin(signal))
                mean_val = float(np.nanmean(signal))
                auc_sum = float(np.nansum(signal))
                peak_tr = int(np.nanargmax(signal))

            row = {
                "sub_tag": seg["sub_tag"],
                "ses": int(seg["ses"]),
                "run": int(seg["run"]),
                "trial_index_in_run": int(trial_idx),
                "trial_index_global": int(seg["trial_start_global"] + trial_idx),
                "signal_index": int(len(signals)),
                "baseline_tr01": baseline,
                "peak_value": peak_val,
                "trough_value": trough_val,
                "signal_range": _safe_nanmax_minus_nanmin(signal),
                "amp_peak_baseline": float(peak_val - baseline) if np.isfinite(peak_val) and np.isfinite(baseline) else np.nan,
                "mean_value": mean_val,
                "auc_sum": auc_sum,
                "peak_tr": peak_tr,
            }
            for tr_idx, col in enumerate(tr_cols):
                row[col] = float(signal[tr_idx])
            rows.append(row)
            signals.append(signal)

    trial_df = pd.DataFrame(rows)
    if signals:
        signal_matrix = np.vstack(signals).astype(np.float64, copy=False)
    else:
        signal_matrix = np.empty((0, int(trial_len)), dtype=np.float64)
    return trial_df, signal_matrix


def build_templates(trial_df, signal_matrix, group_cols):
    templates = {}
    for key, grp in trial_df.groupby(group_cols, sort=True):
        if len(group_cols) == 1 and isinstance(key, tuple):
            norm_key = key[0]
        else:
            norm_key = key
        signal_idx = grp["signal_index"].to_numpy(dtype=np.int64)
        template = np.nanmean(signal_matrix[signal_idx], axis=0)
        templates[norm_key] = np.asarray(template, dtype=np.float64)
    return templates


def add_alignment_metrics(trial_df, signal_matrix, templates_sub_ses, templates_sub):
    lag_ses = []
    corr_ses = []
    lag_sub = []
    corr_sub = []

    for row in trial_df.itertuples(index=False):
        signal = signal_matrix[int(row.signal_index)]

        key_sub_ses = (str(row.sub_tag), int(row.ses))
        template_sub_ses = templates_sub_ses[key_sub_ses]
        best_lag_ses, best_corr_ses, _, _ = best_lag_corr(signal, template_sub_ses)
        lag_ses.append(best_lag_ses)
        corr_ses.append(best_corr_ses)

        key_sub = str(row.sub_tag)
        template_sub = templates_sub[key_sub]
        best_lag_sub, best_corr_sub, _, _ = best_lag_corr(signal, template_sub)
        lag_sub.append(best_lag_sub)
        corr_sub.append(best_corr_sub)

    out = trial_df.copy()
    out["lag_to_session_template"] = np.asarray(lag_ses, dtype=np.float64)
    out["corr_to_session_template"] = np.asarray(corr_ses, dtype=np.float64)
    out["lag_to_subject_template"] = np.asarray(lag_sub, dtype=np.float64)
    out["corr_to_subject_template"] = np.asarray(corr_sub, dtype=np.float64)
    return out


def build_run_summary(run_segments):
    rows = []
    for seg in run_segments:
        vals = np.asarray(seg["trial_matrix"], dtype=np.float64).ravel()
        finite = vals[np.isfinite(vals)]
        rows.append(
            {
                "sub_tag": seg["sub_tag"],
                "ses": seg["ses"],
                "run": seg["run"],
                "n_trials": seg["n_trials"],
                "n_timepoints": int(vals.size),
                "mean_signal": float(np.mean(finite)) if finite.size else np.nan,
                "std_signal": float(np.std(finite)) if finite.size else np.nan,
                "min_signal": float(np.min(finite)) if finite.size else np.nan,
                "max_signal": float(np.max(finite)) if finite.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_subject_session_summary(trial_df):
    summary = (
        trial_df.groupby(["sub_tag", "ses"], as_index=False)
        .agg(
            n_trials=("signal_index", "size"),
            n_runs=("run", "nunique"),
            amp_mean=("amp_peak_baseline", "mean"),
            amp_median=("amp_peak_baseline", "median"),
            amp_std=("amp_peak_baseline", "std"),
            lag_session_mean=("lag_to_session_template", "mean"),
            lag_session_median=("lag_to_session_template", "median"),
            lag_subject_mean=("lag_to_subject_template", "mean"),
            lag_subject_median=("lag_to_subject_template", "median"),
            corr_session_mean=("corr_to_session_template", "mean"),
            corr_subject_mean=("corr_to_subject_template", "mean"),
            peak_tr_median=("peak_tr", "median"),
        )
        .sort_values(["sub_tag", "ses"])
        .reset_index(drop=True)
    )
    return summary


def build_subject_session_shift_summary(templates_sub_ses):
    subjects = sorted({key[0] for key in templates_sub_ses.keys()})
    rows = []

    for sub_tag in subjects:
        key1 = (sub_tag, 1)
        key2 = (sub_tag, 2)
        if key1 not in templates_sub_ses or key2 not in templates_sub_ses:
            continue

        ses1 = np.asarray(templates_sub_ses[key1], dtype=np.float64)
        ses2 = np.asarray(templates_sub_ses[key2], dtype=np.float64)
        best_lag, best_corr, lags, corr_curve = best_lag_corr(ses1, ses2)
        zero_idx = np.where(lags == 0)[0]
        corr_zero = float(corr_curve[zero_idx[0]]) if zero_idx.size else np.nan
        rows.append(
            {
                "sub_tag": sub_tag,
                "best_lag_ses1_vs_ses2": best_lag,
                "best_corr_ses1_vs_ses2": best_corr,
                "abs_best_corr_ses1_vs_ses2": float(np.abs(best_corr)) if np.isfinite(best_corr) else np.nan,
                "corr_zero_lag_ses1_vs_ses2": corr_zero,
                "range_ses1": _safe_nanmax_minus_nanmin(ses1),
                "range_ses2": _safe_nanmax_minus_nanmin(ses2),
                "peak_ses1": float(np.nanmax(ses1)),
                "peak_ses2": float(np.nanmax(ses2)),
                "peak_tr_ses1": int(np.nanargmax(ses1)),
                "peak_tr_ses2": int(np.nanargmax(ses2)),
            }
        )

    return pd.DataFrame(rows).sort_values("sub_tag").reset_index(drop=True)


def _plot_paired_by_subject(ax, values_s1, values_s2, subjects, y_label, title):
    x_pos = [1, 2]
    for sub_tag in subjects:
        v1 = values_s1[sub_tag]
        v2 = values_s2[sub_tag]
        ax.plot(x_pos, [v1, v2], color="0.75", lw=1.0, zorder=1)
        ax.scatter(x_pos, [v1, v2], color=["tab:blue", "tab:orange"], s=22, zorder=2)

    mean_s1 = float(np.nanmean([values_s1[s] for s in subjects])) if subjects else np.nan
    mean_s2 = float(np.nanmean([values_s2[s] for s in subjects])) if subjects else np.nan
    ax.plot(x_pos, [mean_s1, mean_s2], color="k", lw=2.5, marker="o", markersize=5, zorder=3, label="Group mean")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Session 1", "Session 2"])
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=False)


def plot_subject_session_alignment_grid(templates_sub_ses, shift_df, output_path):
    subjects = sorted({key[0] for key in templates_sub_ses.keys()})
    if not subjects:
        return

    shift_map = {}
    if not shift_df.empty:
        shift_map = shift_df.set_index("sub_tag").to_dict(orient="index")

    trial_len = int(next(iter(templates_sub_ses.values())).size)
    tr_axis = np.arange(trial_len, dtype=np.int64)
    n_rows, n_cols = _subplot_grid(len(subjects), max_cols=4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, sub_tag in zip(axes, subjects):
        key1 = (sub_tag, 1)
        key2 = (sub_tag, 2)
        if key1 in templates_sub_ses:
            ax.plot(tr_axis, templates_sub_ses[key1], lw=2.0, color="tab:blue", label="ses-1 mean")
        if key2 in templates_sub_ses:
            ax.plot(tr_axis, templates_sub_ses[key2], lw=2.0, color="tab:orange", label="ses-2 mean")

        if key1 in templates_sub_ses and key2 in templates_sub_ses and sub_tag in shift_map:
            lag = int(shift_map[sub_tag]["best_lag_ses1_vs_ses2"])
            corr = float(shift_map[sub_tag]["best_corr_ses1_vs_ses2"])
            ses2_shifted = shift_with_nan(templates_sub_ses[key2], lag)
            ax.plot(tr_axis, ses2_shifted, lw=1.6, ls="--", color="tab:green", label="ses-2 shifted")
            ax.text(
                0.02,
                0.98,
                f"lag={lag} TR\n|r|={abs(corr):.2f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.75, "ec": "0.8"},
            )

        ax.set_title(str(sub_tag), fontsize=9)
        ax.set_xticks(tr_axis)
        ax.grid(alpha=0.2)

    for ax in axes[len(subjects) :]:
        ax.axis("off")

    axes[0].legend(loc="best", fontsize=8, frameon=False)
    fig.supxlabel("TR within trial")
    fig.supylabel("Mean projection")
    fig.suptitle("Per-subject session templates and aligned session-2 curve", fontsize=12)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_subject_session_paired_metrics(subject_session_df, shift_df, output_path):
    if subject_session_df.empty:
        return

    pivot_amp = subject_session_df.pivot(index="sub_tag", columns="ses", values="amp_median")
    pivot_lag = subject_session_df.pivot(index="sub_tag", columns="ses", values="lag_subject_median")
    shared_subjects = sorted(set(pivot_amp.index) & set(pivot_lag.index))
    shared_subjects = [sub for sub in shared_subjects if 1 in pivot_amp.columns and 2 in pivot_amp.columns and 1 in pivot_lag.columns and 2 in pivot_lag.columns]
    if not shared_subjects:
        return

    values_amp_s1 = {sub: float(pivot_amp.loc[sub, 1]) for sub in shared_subjects}
    values_amp_s2 = {sub: float(pivot_amp.loc[sub, 2]) for sub in shared_subjects}
    values_lag_s1 = {sub: float(pivot_lag.loc[sub, 1]) for sub in shared_subjects}
    values_lag_s2 = {sub: float(pivot_lag.loc[sub, 2]) for sub in shared_subjects}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    _plot_paired_by_subject(
        axes[0],
        values_s1=values_amp_s1,
        values_s2=values_amp_s2,
        subjects=shared_subjects,
        y_label="Trial amplitude (peak - baseline TR0/1)",
        title="Session paired amplitude (within subject)",
    )

    _plot_paired_by_subject(
        axes[1],
        values_s1=values_lag_s1,
        values_s2=values_lag_s2,
        subjects=shared_subjects,
        y_label="Median lag to subject template (TR)",
        title="Session paired lag (within subject)",
    )

    if shift_df is not None and not shift_df.empty:
        s = shift_df.set_index("sub_tag").reindex(shared_subjects)
        x = np.arange(len(shared_subjects), dtype=np.int64)
        y = s["best_lag_ses1_vs_ses2"].to_numpy(dtype=np.float64)
        c = s["abs_best_corr_ses1_vs_ses2"].to_numpy(dtype=np.float64)
        sc = axes[2].scatter(x, y, c=c, cmap="viridis", s=55, edgecolor="k", linewidth=0.25)
        axes[2].axhline(0.0, color="0.3", lw=1.0, ls="--")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(shared_subjects, rotation=90, fontsize=7)
        axes[2].set_ylabel("Best lag ses1 vs ses2 (TR)")
        axes[2].set_title("Between-session shift by subject")
        axes[2].grid(alpha=0.2)
        cbar = fig.colorbar(sc, ax=axes[2], shrink=0.92)
        cbar.set_label("|best corr|")
    else:
        axes[2].axis("off")

    fig.suptitle("Amplitude and shift comparisons across sessions", fontsize=12)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def build_between_subject_similarity(templates_sub_ses):
    sessions = sorted({int(key[1]) for key in templates_sub_ses.keys()})
    by_session = {}
    pair_rows = []

    for ses in sessions:
        subjects = sorted([key[0] for key in templates_sub_ses.keys() if int(key[1]) == int(ses)])
        if not subjects:
            continue

        n_sub = len(subjects)
        corr_mat = np.full((n_sub, n_sub), np.nan, dtype=np.float64)
        lag_mat = np.full((n_sub, n_sub), np.nan, dtype=np.float64)

        for i, sub_a in enumerate(subjects):
            for j, sub_b in enumerate(subjects):
                lag, corr, _, _ = best_lag_corr(templates_sub_ses[(sub_a, ses)], templates_sub_ses[(sub_b, ses)])
                corr_mat[i, j] = float(np.abs(corr)) if np.isfinite(corr) else np.nan
                lag_mat[i, j] = float(lag) if np.isfinite(lag) else np.nan
                if i < j:
                    pair_rows.append(
                        {
                            "ses": int(ses),
                            "sub_a": sub_a,
                            "sub_b": sub_b,
                            "best_lag": lag,
                            "best_corr": corr,
                            "abs_best_corr": float(np.abs(corr)) if np.isfinite(corr) else np.nan,
                        }
                    )

        by_session[int(ses)] = {"subjects": subjects, "abs_corr": corr_mat, "lag": lag_mat}

    pair_df = pd.DataFrame(pair_rows)
    return by_session, pair_df


def plot_between_subject_similarity_heatmaps(by_session, output_path):
    if not by_session:
        return

    sessions = sorted(by_session.keys())
    n_cols = len(sessions)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.2 * n_cols, 9.0), constrained_layout=True)
    axes = np.atleast_2d(axes)
    if axes.shape[1] != n_cols:
        axes = axes.reshape(2, n_cols)

    for col_idx, ses in enumerate(sessions):
        payload = by_session[ses]
        subjects = payload["subjects"]
        corr_mat = np.asarray(payload["abs_corr"], dtype=np.float64)
        lag_mat = np.asarray(payload["lag"], dtype=np.float64)

        ax_corr = axes[0, col_idx]
        im_corr = ax_corr.imshow(corr_mat, vmin=0.0, vmax=1.0, cmap="viridis")
        ax_corr.set_title(f"Session {ses}: between-subject |corr|")
        ax_corr.set_xticks(np.arange(len(subjects)))
        ax_corr.set_yticks(np.arange(len(subjects)))
        ax_corr.set_xticklabels(subjects, rotation=90, fontsize=7)
        ax_corr.set_yticklabels(subjects, fontsize=7)
        fig.colorbar(im_corr, ax=ax_corr, shrink=0.88)

        max_abs_lag = max(1.0, float(np.nanmax(np.abs(lag_mat))))
        ax_lag = axes[1, col_idx]
        im_lag = ax_lag.imshow(lag_mat, vmin=-max_abs_lag, vmax=max_abs_lag, cmap="coolwarm")
        ax_lag.set_title(f"Session {ses}: between-subject lag (TR)")
        ax_lag.set_xticks(np.arange(len(subjects)))
        ax_lag.set_yticks(np.arange(len(subjects)))
        ax_lag.set_xticklabels(subjects, rotation=90, fontsize=7)
        ax_lag.set_yticklabels(subjects, fontsize=7)
        fig.colorbar(im_lag, ax=ax_lag, shrink=0.88)

    fig.suptitle("Between-subject similarity and shift within each session", fontsize=12)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _select_subjects_for_detail(all_subjects, detail_subjects, seed):
    all_subjects = np.asarray(sorted(all_subjects))
    if all_subjects.size == 0:
        return np.array([], dtype=object)
    if int(detail_subjects) <= 0 or int(detail_subjects) >= all_subjects.size:
        return all_subjects
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(all_subjects, size=int(detail_subjects), replace=False))
    return chosen


def plot_subject_detail_panels(trial_df, templates_sub_ses, output_dir, detail_subjects, seed):
    all_subjects = sorted(trial_df["sub_tag"].astype(str).unique())
    selected_subjects = _select_subjects_for_detail(all_subjects, detail_subjects, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sub_tag in selected_subjects:
        sub_df = trial_df[trial_df["sub_tag"].astype(str) == str(sub_tag)].copy()
        if sub_df.empty:
            continue
        sub_df = sub_df.sort_values(["ses", "run", "trial_index_in_run"]).reset_index(drop=True)
        sub_df["trial_seq_in_session"] = sub_df.groupby("ses").cumcount() + 1

        key1 = (str(sub_tag), 1)
        key2 = (str(sub_tag), 2)
        ses1_template = templates_sub_ses.get(key1)
        ses2_template = templates_sub_ses.get(key2)
        trial_len = int(len(ses1_template if ses1_template is not None else ses2_template))
        tr_axis = np.arange(trial_len, dtype=np.int64)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        ax0, ax1, ax2, ax3 = axes.reshape(-1)

        if ses1_template is not None:
            ax0.plot(tr_axis, ses1_template, lw=2.2, color="tab:blue", label="ses-1 mean")
        if ses2_template is not None:
            ax0.plot(tr_axis, ses2_template, lw=2.2, color="tab:orange", label="ses-2 mean")

        if ses1_template is not None and ses2_template is not None:
            lag, corr, lags, corr_curve = best_lag_corr(ses1_template, ses2_template)
            ses2_shifted = shift_with_nan(ses2_template, lag)
            ax0.plot(tr_axis, ses2_shifted, lw=1.8, ls="--", color="tab:green", label=f"ses-2 shifted (lag={lag})")
            ax0.text(
                0.02,
                0.98,
                f"best lag={lag} TR\nbest corr={corr:.2f}",
                transform=ax0.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.75, "ec": "0.8"},
            )

            ax1.plot(lags, corr_curve, color="tab:purple", lw=1.8)
            ax1.axvline(lag, color="tab:red", lw=1.2, ls="--")
            ax1.axhline(0.0, color="0.3", lw=1.0)
            ax1.set_title("Cross-correlation: ses-1 vs ses-2 templates")
            ax1.set_xlabel("Lag (TR)")
            ax1.set_ylabel("Normalized correlation")
            ax1.grid(alpha=0.25)
        else:
            ax1.axis("off")

        for ses_val, color in [(1, "tab:blue"), (2, "tab:orange")]:
            ses_df = sub_df[sub_df["ses"].astype(int) == int(ses_val)].sort_values(["run", "trial_index_in_run"])
            if ses_df.empty:
                continue
            x = ses_df["trial_seq_in_session"].to_numpy(dtype=np.int64)
            y_amp = ses_df["amp_peak_baseline"].to_numpy(dtype=np.float64)
            y_lag = ses_df["lag_to_session_template"].to_numpy(dtype=np.float64)
            y_amp_smooth = pd.Series(y_amp).rolling(window=7, center=True, min_periods=1).median().to_numpy()
            y_lag_smooth = pd.Series(y_lag).rolling(window=7, center=True, min_periods=1).median().to_numpy()

            ax2.scatter(x, y_amp, s=16, color=color, alpha=0.45)
            ax2.plot(x, y_amp_smooth, lw=2.0, color=color, label=f"ses-{ses_val}")

            ax3.scatter(x, y_lag, s=16, color=color, alpha=0.45)
            ax3.plot(x, y_lag_smooth, lw=2.0, color=color, label=f"ses-{ses_val}")

        ax0.set_title("Session mean curves (and shifted alignment)")
        ax0.set_xlabel("TR within trial")
        ax0.set_ylabel("Projection")
        ax0.set_xticks(tr_axis)
        ax0.grid(alpha=0.25)
        ax0.legend(loc="best", fontsize=8, frameon=False)

        ax2.set_title("Trial amplitude progression in session")
        ax2.set_xlabel("Trial index within session")
        ax2.set_ylabel("Amplitude (peak - baseline TR0/1)")
        ax2.grid(alpha=0.25)
        ax2.legend(loc="best", fontsize=8, frameon=False)

        ax3.set_title("Trial lag progression to session template")
        ax3.set_xlabel("Trial index within session")
        ax3.set_ylabel("Lag (TR)")
        ax3.axhline(0.0, color="0.3", lw=1.0, ls="--")
        ax3.grid(alpha=0.25)
        ax3.legend(loc="best", fontsize=8, frameon=False)

        fig.suptitle(f"{sub_tag}: amplitude change and temporal shift", fontsize=12)
        out_path = output_dir / f"{sub_tag}_trial_amplitude_shift_detail.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    return selected_subjects


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projection = np.asarray(np.load(args.projection_path), dtype=np.float64).reshape(-1)
    manifest = load_manifest(args.manifest_path)
    layout = infer_layout(projection.size, manifest, trial_len=args.trial_len)
    run_segments = split_projection_to_runs(projection, manifest, trial_len=args.trial_len, layout=layout)

    trial_df, signal_matrix = build_trial_metrics(run_segments, trial_len=args.trial_len)
    templates_sub_ses = build_templates(trial_df, signal_matrix, ["sub_tag", "ses"])
    templates_sub = build_templates(trial_df, signal_matrix, ["sub_tag"])
    trial_df = add_alignment_metrics(trial_df, signal_matrix, templates_sub_ses, templates_sub)

    run_summary_df = build_run_summary(run_segments)
    subject_session_df = build_subject_session_summary(trial_df)
    session_shift_df = build_subject_session_shift_summary(templates_sub_ses)
    by_session_similarity, pairwise_df = build_between_subject_similarity(templates_sub_ses)

    run_summary_path = output_dir / "run_time_summary.csv"
    trial_metrics_path = output_dir / "trial_time_metrics.csv"
    subject_session_path = output_dir / "subject_session_trial_summary.csv"
    subject_shift_path = output_dir / "subject_session_mean_shift.csv"
    pairwise_path = output_dir / "between_subject_pairwise_shift_similarity.csv"

    run_summary_df.to_csv(run_summary_path, index=False)
    trial_df.to_csv(trial_metrics_path, index=False)
    subject_session_df.to_csv(subject_session_path, index=False)
    session_shift_df.to_csv(subject_shift_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)

    alignment_grid_path = output_dir / "subject_session_mean_alignment_grid.png"
    paired_metrics_path = output_dir / "subject_session_paired_metrics.png"
    between_subject_heatmap_path = output_dir / "between_subject_similarity_heatmaps.png"
    subject_detail_dir = output_dir / "subject_details"

    plot_subject_session_alignment_grid(
        templates_sub_ses=templates_sub_ses,
        shift_df=session_shift_df,
        output_path=alignment_grid_path,
    )
    plot_subject_session_paired_metrics(
        subject_session_df=subject_session_df,
        shift_df=session_shift_df,
        output_path=paired_metrics_path,
    )
    plot_between_subject_similarity_heatmaps(
        by_session=by_session_similarity,
        output_path=between_subject_heatmap_path,
    )
    selected_subjects = plot_subject_detail_panels(
        trial_df=trial_df,
        templates_sub_ses=templates_sub_ses,
        output_dir=subject_detail_dir,
        detail_subjects=args.detail_subjects,
        seed=args.seed,
    )

    print(f"Projection path: {args.projection_path}")
    print(f"Manifest path: {args.manifest_path}")
    print(f"Output dir: {output_dir}")
    print(f"Trial length: {args.trial_len}")
    print(f"Inferred layout: {layout}")
    print(f"Runs segmented: {len(run_segments)}")
    print(f"Trials analyzed: {len(trial_df)}")
    print(f"Subjects analyzed: {trial_df['sub_tag'].nunique() if not trial_df.empty else 0}")
    print(f"Detailed-subject figures: {len(selected_subjects)}")
    print(f"Saved: {run_summary_path}")
    print(f"Saved: {trial_metrics_path}")
    print(f"Saved: {subject_session_path}")
    print(f"Saved: {subject_shift_path}")
    print(f"Saved: {pairwise_path}")
    print(f"Saved: {alignment_grid_path}")
    print(f"Saved: {paired_metrics_path}")
    print(f"Saved: {between_subject_heatmap_path}")
    print(f"Saved details in: {subject_detail_dir}")


if __name__ == "__main__":
    main()
