import colorsys
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags


PROJECTION_PATH = Path(
    "/home/zkavian/Thesis_code_Glm_Opt/results/behave_vs_bold/"
    "projection_voxel_foldavg_sub9_ses1_task0.8_bold0.8_beta0.5_smooth0.2_gamma1_bold_thr90.npy")
MANIFEST_PATH = Path("/Data/zahra/results_beta_preprocessed/group_concat/concat_manifest_group.tsv")
BEHAVIOR_ROOT = Path("/Data/zahra/behaviour")

BEHAVIOR_COLUMN = 0
N_SUBJECTS = 5
RNG_SEED = 42

projection = np.asarray(np.load(PROJECTION_PATH), dtype=np.float64).reshape(-1)
manifest = pd.read_csv(MANIFEST_PATH, sep="	").sort_values("offset_start").reset_index(drop=True)

required_cols = {
    "sub_tag",
    "ses",
    "run",
    "offset_start",
    "offset_end",
    "source_offset_start",
    "source_offset_end",
    "n_trials",
    "n_trials_source",
    "trial_keep_path",
}
missing_cols = required_cols - set(manifest.columns)

def sample_subject_rows(manifest_df, n_subjects=5, seed=42):
    rng = np.random.default_rng(seed)

    subjects = np.array(sorted(manifest_df["sub_tag"].unique()))
    if n_subjects > subjects.size:
        raise ValueError(f"Requested {n_subjects} subjects, but only {subjects.size} are available.")

    selected_subjects = rng.choice(subjects, size=n_subjects, replace=False)

    combos = (
        manifest_df[["ses", "run"]]
        .drop_duplicates()
        .sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        .to_records(index=False)
        .tolist()
    )

    picked_rows = []
    for idx, sub in enumerate(selected_subjects):
        sub_rows = manifest_df[manifest_df["sub_tag"] == sub]
        target_ses, target_run = combos[idx % len(combos)]
        preferred = sub_rows[(sub_rows["ses"] == target_ses) & (sub_rows["run"] == target_run)]
        pool = preferred if not preferred.empty else sub_rows
        picked_rows.append(pool.iloc[int(rng.integers(0, len(pool)))])

    sampled = pd.DataFrame(picked_rows).reset_index(drop=True)
    sampled = sampled.sort_values(["sub_tag", "ses", "run"]).reset_index(drop=True)
    sampled["kept_trial_indices_manifest"] = sampled.apply(lambda row: np.arange(int(row.offset_start), int(row.offset_end), dtype=int), axis=1,)
    return sampled


sampled_runs = sample_subject_rows(manifest, n_subjects=N_SUBJECTS, seed=RNG_SEED)
sampled_runs[[
    "sub_tag",
    "ses",
    "run",
    "offset_start",
    "offset_end",
    "source_offset_start",
    "source_offset_end",
    "n_trials",
    "n_trials_source",
]]

def _subject_digits(sub_tag):
    match = re.search(r"(\d+)$", str(sub_tag))
    if match is None:
        raise ValueError(f"Could not parse subject digits from '{sub_tag}'.")
    return match.group(1)


def resolve_behavior_path(sub_tag, ses, run, behavior_root):
    subject_digits = _subject_digits(sub_tag)
    return behavior_root / f"PSPD{subject_digits}_ses_{int(ses)}_run_{int(run)}.npy"


def load_behavior_column(path, behavior_column):
    behavior = np.asarray(np.load(path), dtype=np.float64)
    if behavior.ndim == 1:
        if behavior_column != 0:
            raise IndexError(f"Behavior is 1D in {path}; requested column {behavior_column}.")
        return behavior
    if behavior.ndim != 2:
        raise ValueError(f"Behavior must be 1D or 2D in {path}, got shape {behavior.shape}.")
    if not (0 <= int(behavior_column) < behavior.shape[1]):
        raise IndexError(
            f"Behavior column {behavior_column} is out of bounds for {path} with shape {behavior.shape}."
        )
    return behavior[:, int(behavior_column)]


def zscore_1d(x):
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    std = np.std(x)
    if (not np.isfinite(std)) or np.isclose(std, 0.0):
        return np.zeros_like(x)
    return (x - mean) / std


def crosscorr_with_best_lag(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size:
        raise ValueError(f"x and y must have the same length, got {x.size} vs {y.size}.")
    if x.size < 2:
        raise ValueError("Need at least 2 aligned samples for cross-correlation.")

    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    denom = np.linalg.norm(x0) * np.linalg.norm(y0)
    if np.isclose(denom, 0.0):
        lags = correlation_lags(x0.size, y0.size, mode="full")
        corr = np.full(lags.shape, np.nan, dtype=np.float64)
        return corr, lags, np.nan, np.nan

    corr = correlate(x0, y0, mode="full", method="auto") / denom
    lags = correlation_lags(x0.size, y0.size, mode="full")
    peak_idx = int(np.nanargmax(np.abs(corr)))
    return corr, lags, int(lags[peak_idx]), float(corr[peak_idx])


def top_k_ranked_lag_stats(corr, lags, top_k=3):
    corr_vals = np.asarray(corr, dtype=np.float64)
    lag_vals = np.asarray(lags, dtype=np.int64)
    finite_mask = np.isfinite(corr_vals)
    if not np.any(finite_mask):
        return {
            "top_lags": np.array([], dtype=np.int64),
            "top_corr": np.array([], dtype=np.float64),
            "first_rank_lag": np.nan,
            "first_rank_abs_corr": np.nan,
            "min_topk_lag": np.nan,
            "min_topk_abs_corr": np.nan,
        }

    finite_corr = corr_vals[finite_mask]
    finite_lags = lag_vals[finite_mask]
    k = min(int(top_k), finite_corr.size)
    top_idx = np.argsort(np.abs(finite_corr))[::-1][:k]
    top_lags = finite_lags[top_idx]
    top_corr = finite_corr[top_idx]

    min_lag_idx = int(np.argmin(top_lags))
    return {
        "top_lags": top_lags.astype(np.int64, copy=False),
        "top_corr": top_corr.astype(np.float64, copy=False),
        "first_rank_lag": int(top_lags[0]),
        "first_rank_abs_corr": float(np.abs(top_corr[0])),
        "min_topk_lag": int(top_lags[min_lag_idx]),
        "min_topk_abs_corr": float(np.abs(top_corr[min_lag_idx])),
    }


def integer_hist_bins(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([-0.5, 0.5], dtype=np.float64)
    vmin = int(np.floor(np.min(values)))
    vmax = int(np.ceil(np.max(values)))
    if vmin == vmax:
        return np.array([vmin - 0.5, vmax + 0.5], dtype=np.float64)
    return np.arange(vmin - 0.5, vmax + 1.5, 1.0, dtype=np.float64)


def light_subject_colors(n_subjects, saturation=0.25, value=0.95):
    if int(n_subjects) <= 0:
        return []
    hues = np.linspace(0.0, 1.0, int(n_subjects), endpoint=False, dtype=np.float64)
    return [colorsys.hsv_to_rgb(float(h), float(saturation), float(value)) for h in hues]


def plot_subject_projection_over_trials(manifest_df, projection_values, output_path):
    subject_tags = sorted(manifest_df["sub_tag"].astype(str).unique())
    colors = light_subject_colors(len(subject_tags))
    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)

    for color, sub_tag in zip(colors, subject_tags):
        sub_rows = manifest_df[manifest_df["sub_tag"] == sub_tag].sort_values("offset_start")
        for _, row in sub_rows.iterrows():
            start = int(row.offset_start)
            end = int(row.offset_end)
            trial_idx = np.arange(start, end, dtype=int)
            proj_segment = np.asarray(projection_values[start:end], dtype=np.float64)
            finite_mask = np.isfinite(proj_segment)
            if np.any(finite_mask):
                ax.plot(trial_idx[finite_mask], proj_segment[finite_mask], color=color, lw=1.0, alpha=0.8)

    ax.set_title(f"Projected signal over all kept trials ({len(subject_tags)} subjects)")
    ax.set_xlabel("Global kept-trial index")
    ax.set_ylabel("Projected signal")
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=300)


analysis_records = []
for _, row in sampled_runs.iterrows():
    sub_tag = str(row.sub_tag)
    ses = int(row.ses)
    run = int(row.run)

    start = int(row.offset_start)
    end = int(row.offset_end)
    source_start = int(row.source_offset_start)
    source_end = int(row.source_offset_end)

    proj_segment = projection[start:end]

    trial_keep_path = Path(str(row.trial_keep_path))
    keep_mask = np.asarray(np.load(trial_keep_path), dtype=bool)
    n_trials_source = int(row.n_trials_source)
    if keep_mask.size != n_trials_source:
        raise ValueError(
            f"trial_keep length mismatch for {sub_tag} ses-{ses} run-{run}: "
            f"{keep_mask.size} vs {n_trials_source}."
        )

    behavior_path = resolve_behavior_path(sub_tag, ses, run, BEHAVIOR_ROOT)

    behavior = load_behavior_column(behavior_path, BEHAVIOR_COLUMN)
    behavior = behavior[:n_trials_source]

    behavior_kept = behavior[keep_mask]
    finite_mask = np.isfinite(behavior_kept) & np.isfinite(proj_segment)
    x = proj_segment[finite_mask]
    y = behavior_kept[finite_mask]

    corr, lags, best_lag, best_corr = crosscorr_with_best_lag(x, y)

    kept_trial_indices_manifest = np.arange(start, end, dtype=int)
    kept_trial_indices_source_local = np.flatnonzero(keep_mask)

    analysis_records.append(
        {
            "sub_tag": sub_tag,
            "ses": ses,
            "run": run,
            "behavior_path": str(behavior_path),
            "trial_keep_path": str(trial_keep_path),
            "offset_start": start,
            "offset_end": end,
            "source_offset_start": source_start,
            "source_offset_end": source_end,
            "kept_trial_indices_manifest": kept_trial_indices_manifest,
            "kept_trial_indices_source_local": kept_trial_indices_source_local,
            "finite_aligned_indices_manifest": kept_trial_indices_manifest[finite_mask],
            "finite_aligned_indices_source_local": kept_trial_indices_source_local[finite_mask],
            "projection": x,
            "behavior": y,
            "corr": corr,
            "lags": lags,
            "best_lag": best_lag,
            "best_corr": best_corr,
            "n_trials_kept_manifest": int(end - start),
            "n_trials_aligned_finite": int(x.size),
        }
    )

summary_df = pd.DataFrame(
    [
        {
            "sub_tag": rec["sub_tag"],
            "ses": rec["ses"],
            "run": rec["run"],
            "offset_start": rec["offset_start"],
            "offset_end": rec["offset_end"],
            "source_offset_start": rec["source_offset_start"],
            "source_offset_end": rec["source_offset_end"],
            "n_trials_kept_manifest": rec["n_trials_kept_manifest"],
            "n_trials_aligned_finite": rec["n_trials_aligned_finite"],
            "best_lag": rec["best_lag"],
            "best_corr": rec["best_corr"],
            "abs_best_corr": np.abs(rec["best_corr"]),
        }
        for rec in analysis_records
    ]
)

print(summary_df)

n_rows = len(analysis_records)
fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.5 * n_rows), constrained_layout=True)
if n_rows == 1:
    axes = np.array([axes])

for i, rec in enumerate(analysis_records):
    x = rec["projection"]
    y = rec["behavior"]

    ax_ts = axes[i, 0]
    ax_ts.plot(zscore_1d(x), label="projection (z)", lw=1.6)
    ax_ts.plot(zscore_1d(y), label="behavior (z)", lw=1.6, alpha=0.9)
    ax_ts.set_title(
        f"{rec['sub_tag']} ses-{rec['ses']} run-{rec['run']} "
        f"| best lag={rec['best_lag']} | best corr={rec['best_corr']:.3f}"
    )
    ax_ts.set_xlabel("Aligned kept-trial index")
    ax_ts.set_ylabel("z-score")
    ax_ts.grid(alpha=0.3)
    if i == 0:
        ax_ts.legend(loc="upper right")

    corr_vals = np.asarray(rec["corr"], dtype=np.float64)
    lag_vals = np.asarray(rec["lags"], dtype=np.int64)
    ranked = top_k_ranked_lag_stats(corr_vals, lag_vals, top_k=3)
    if ranked["top_lags"].size:
        top_idx = np.arange(ranked["top_lags"].size, dtype=np.int64)
        top_pairs = [
            f"lag {int(ranked['top_lags'][j])} (|corr|={abs(ranked['top_corr'][j]):.3f})"
            for j in top_idx
        ]
        corr_title = " | ".join(top_pairs)
    else:
        corr_title = "no finite corr"

    ax_corr = axes[i, 1]
    ax_corr.plot(rec["lags"], rec["corr"], color="tab:green", lw=1.5)
    ax_corr.axvline(rec["best_lag"], color="tab:red", ls="--", lw=1.2)
    ax_corr.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    ax_corr.set_title(corr_title)
    ax_corr.set_xlabel("Lag (trials)")
    ax_corr.set_ylabel("Normalized cross-correlation")
    ax_corr.grid(alpha=0.3)

fig.savefig("projection_behavior_crosscorr.png", dpi=300)

# --- All-runs / all-subjects extension ---
all_records = []
for _, row in manifest.iterrows():
    sub_tag = str(row.sub_tag)
    ses = int(row.ses)
    run = int(row.run)

    start = int(row.offset_start)
    end = int(row.offset_end)
    proj_segment = projection[start:end]

    trial_keep_path = Path(str(row.trial_keep_path))
    keep_mask = np.asarray(np.load(trial_keep_path), dtype=bool)
    n_trials_source = int(row.n_trials_source)
    if keep_mask.size != n_trials_source:
        raise ValueError(
            f"trial_keep length mismatch for {sub_tag} ses-{ses} run-{run}: "
            f"{keep_mask.size} vs {n_trials_source}."
        )

    behavior_path = resolve_behavior_path(sub_tag, ses, run, BEHAVIOR_ROOT)
    behavior = load_behavior_column(behavior_path, BEHAVIOR_COLUMN)
    behavior = behavior[:n_trials_source]

    behavior_kept = behavior[keep_mask]
    finite_mask = np.isfinite(behavior_kept) & np.isfinite(proj_segment)
    x = proj_segment[finite_mask]
    y = behavior_kept[finite_mask]
    if x.size < 2:
        continue

    corr, lags, best_lag, best_corr = crosscorr_with_best_lag(x, y)
    ranked = top_k_ranked_lag_stats(corr, lags, top_k=3)

    all_records.append(
        {
            "sub_tag": sub_tag,
            "ses": ses,
            "run": run,
            "n_trials_aligned_finite": int(x.size),
            "best_lag": best_lag,
            "best_corr": best_corr,
            "abs_best_corr": float(np.abs(best_corr)),
            "first_rank_lag": ranked["first_rank_lag"],
            "min_top3_lag": ranked["min_topk_lag"],
            "first_rank_abs_corr": ranked["first_rank_abs_corr"],
            "min_top3_abs_corr": ranked["min_topk_abs_corr"],
        }
    )

all_summary_df = pd.DataFrame(all_records)
print(
    f"All-runs summary: {len(all_summary_df)} runs, "
    f"{all_summary_df['sub_tag'].nunique() if not all_summary_df.empty else 0} subjects"
)
print(all_summary_df.head())
all_summary_df.to_csv("projection_behavior_crosscorr_all_subjects_summary.csv", index=False)

if not all_summary_df.empty:
    lag_bins = integer_hist_bins(
        np.concatenate(
            [
                all_summary_df["first_rank_lag"].to_numpy(dtype=np.float64),
                all_summary_df["min_top3_lag"].to_numpy(dtype=np.float64),
            ]
        )
    )

    fig_lag, ax_lag = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    first_rank_lag_vals = all_summary_df["first_rank_lag"].to_numpy(dtype=np.float64)
    min_top3_lag_vals = all_summary_df["min_top3_lag"].to_numpy(dtype=np.float64)

    ax_lag[0].hist(first_rank_lag_vals[np.isfinite(first_rank_lag_vals)], bins=lag_bins, color="tab:blue", alpha=0.8)
    ax_lag[0].set_title("First-rank lag (max |corr|)")
    ax_lag[0].set_xlabel("Lag (trials)")
    ax_lag[0].set_ylabel("Run count")
    ax_lag[0].grid(alpha=0.3)

    ax_lag[1].hist(min_top3_lag_vals[np.isfinite(min_top3_lag_vals)], bins=lag_bins, color="tab:orange", alpha=0.8)
    ax_lag[1].set_title("Minimum lag among top-3 |corr|")
    ax_lag[1].set_xlabel("Lag (trials)")
    ax_lag[1].set_ylabel("Run count")
    ax_lag[1].grid(alpha=0.3)

    fig_lag.suptitle(
        f"Lag distribution across all runs ({len(all_summary_df)} runs, "
        f"{all_summary_df['sub_tag'].nunique()} subjects)"
    )
    fig_lag.savefig("projection_behavior_crosscorr_all_subjects_lag_hist.png", dpi=300)

    abs_first_vals = all_summary_df["first_rank_abs_corr"].to_numpy(dtype=np.float64)
    abs_min3_vals = all_summary_df["min_top3_abs_corr"].to_numpy(dtype=np.float64)
    combined_abs = np.concatenate([abs_first_vals, abs_min3_vals])
    combined_abs = combined_abs[np.isfinite(combined_abs)]
    if combined_abs.size == 0:
        corr_bins = 20
    else:
        corr_bins = np.linspace(0.0, float(np.max(combined_abs)), 31)
        if np.isclose(corr_bins[0], corr_bins[-1]):
            corr_bins = 20

    fig_abs, ax_abs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax_abs[0].hist(abs_first_vals[np.isfinite(abs_first_vals)], bins=corr_bins, color="tab:green", alpha=0.8)
    ax_abs[0].set_title("|corr| at first-rank lag")
    ax_abs[0].set_xlabel("|corr|")
    ax_abs[0].set_ylabel("Run count")
    ax_abs[0].grid(alpha=0.3)

    ax_abs[1].hist(abs_min3_vals[np.isfinite(abs_min3_vals)], bins=corr_bins, color="tab:red", alpha=0.8)
    ax_abs[1].set_title("|corr| at min(top-3 lag)")
    ax_abs[1].set_xlabel("|corr|")
    ax_abs[1].set_ylabel("Run count")
    ax_abs[1].grid(alpha=0.3)

    fig_abs.suptitle("Absolute-correlation distributions for selected lags")
    fig_abs.savefig("projection_behavior_crosscorr_all_subjects_abs_corr_hist.png", dpi=300)

plot_subject_projection_over_trials(
    manifest_df=manifest,
    projection_values=projection,
    output_path="projection_all_trials_all_subjects.png",
)

plt.show()
