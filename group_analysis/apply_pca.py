#!/usr/bin/env python3
"""
Precompute EMPCA models for all subject/session datasets in /Data/zahra.

The script:
1) discovers sub/ses/run entries from /Data/zahra/bold_data,
2) loads run-wise preprocessed active BOLD arrays,
3) applies trial_keep_run*.npy masks from /Data/zahra/results_glm (drops False trials),
4) concatenates runs on shared voxels,
5) fits EMPCA and saves models where group_analysis/obj_param.py can load them.
"""

import argparse
import gc
import os
import re
import sys
from collections import defaultdict
from glob import glob
from os.path import join

import numpy as np

try:
    from empca.empca import empca
except ModuleNotFoundError:
    repo_root = os.path.abspath(join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from empca.empca import empca


DEFAULT_DATA_ROOT = "/Data/zahra"
DEFAULT_BOLD_ROOT = join(DEFAULT_DATA_ROOT, "bold_data")
DEFAULT_PREPROC_ROOT = join(DEFAULT_DATA_ROOT, "results_beta_preprocessed")
DEFAULT_GLM_ROOT = join(DEFAULT_DATA_ROOT, "results_glm")
DEFAULT_CACHE_DIR = join(DEFAULT_PREPROC_ROOT, "group_concat")

BOLD_RE = re.compile(r"^(sub-pd\d+)_ses-(\d+)_run-(\d+)_task-mv_bold_corrected_smoothed_mnireg-2mm\.nii\.gz$")
MNI_SHAPE = (91, 109, 91)


def _normalize_subject(subject):
    if subject is None:
        return None
    subject = str(subject).strip()
    if re.fullmatch(r"\d+", subject):
        return f"sub-pd{int(subject):03d}"
    match = re.fullmatch(r"sub-pd(\d+)", subject)
    if match:
        return f"sub-pd{int(match.group(1)):03d}"
    raise ValueError(f"Invalid --subject value: {subject}")


def _discover_subject_sessions(bold_root, subject=None, session=None):
    pattern = join(bold_root, "sub-pd*_ses-*_run-*_task-mv_bold_corrected_smoothed_mnireg-2mm.nii.gz")
    discovered = defaultdict(set)
    for path in sorted(glob(pattern)):
        match = BOLD_RE.match(os.path.basename(path))
        if not match:
            continue
        sub_tag = match.group(1)
        ses = int(match.group(2))
        run = int(match.group(3))
        if subject and sub_tag != subject:
            continue
        if session is not None and ses != session:
            continue
        discovered[(sub_tag, ses)].add(run)
    return {key: sorted(runs) for key, runs in discovered.items()}


def _resolve_existing_path(*candidates):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("None of these files exist:\n" + "\n".join(f"- {c}" for c in candidates if c))


def _run_tag(sub_tag, ses, run):
    return f"{sub_tag}_ses-{ses}_run-{run}"


def _normalize_active_coords(active_coords):
    coords = np.asarray(active_coords)
    if coords.dtype == object and coords.size == 3:
        axes = [np.asarray(axis, dtype=np.int64).ravel() for axis in coords]
    elif coords.ndim == 2 and coords.shape[0] == 3:
        axes = [np.asarray(coords[idx], dtype=np.int64).ravel() for idx in range(3)]
    elif coords.ndim == 2 and coords.shape[1] == 3:
        axes = [np.asarray(coords[:, idx], dtype=np.int64).ravel() for idx in range(3)]
    else:
        raise ValueError(f"Unexpected active_coords shape: {coords.shape}")
    if not (axes[0].size == axes[1].size == axes[2].size):
        raise ValueError(f"active_coords axes must have matching lengths, got {[axis.size for axis in axes]}")
    return axes


def _flat_from_active_coords(coords):
    axes = _normalize_active_coords(coords)
    return np.ravel_multi_index(tuple(axes), MNI_SHAPE)


def _load_run_arrays(preproc_root, sub_tag, ses, run):
    tag = _run_tag(sub_tag, ses, run)
    sub_dir = join(preproc_root, sub_tag)
    bold_path = _resolve_existing_path(join(sub_dir, f"active_bold_{tag}.npy"),
        join(sub_dir, f"active_bold_{tag}.npy.npy"))
    flat_path = join(sub_dir, f"active_flat_indices__{tag}.npy")
    coords_path = join(sub_dir, f"active_coords_{tag}.npy")

    bold = np.asarray(np.load(bold_path, mmap_mode="r"), dtype=np.float32)
    n_vox = int(bold.shape[0])

    flat_from_file = None
    if os.path.exists(flat_path):
        flat_from_file = np.asarray(np.load(flat_path, mmap_mode="r"), dtype=np.int64).ravel()

    flat_from_coords = None
    if os.path.exists(coords_path):
        flat_from_coords = _flat_from_active_coords(np.load(coords_path, allow_pickle=True))

    if bold.ndim != 3:
        raise ValueError(f"active_bold must be 3D, got {bold.shape} ({bold_path})")

    flat = None
    if flat_from_coords is not None and flat_from_coords.size == n_vox:
        flat = flat_from_coords
        if flat_from_file is not None and flat_from_file.size != n_vox:
            print(f"  {tag}: active_flat_indices length {flat_from_file.size} mismatch; using active_coords-derived flat indices ({n_vox}).", flush=True)
    elif flat_from_file is not None and flat_from_file.size == n_vox:
        flat = flat_from_file
    elif flat_from_file is not None and flat_from_file.size > n_vox:
        flat = flat_from_file[:n_vox]
        print(f"  {tag}: trimming active_flat_indices from {flat_from_file.size} to {n_vox} to match active_bold.", flush=True )
    elif flat_from_coords is not None and flat_from_coords.size > n_vox:
        flat = flat_from_coords[:n_vox]
        print(f"  {tag}: trimming active_coords-derived flat indices from {flat_from_coords.size} to {n_vox} to match active_bold.", flush=True)
    else:
        raise ValueError(f"Voxel count mismatch for {tag}: active_bold has {n_vox}, "
            f"active_flat_indices has {None if flat_from_file is None else flat_from_file.size}, "
            f"active_coords has {None if flat_from_coords is None else flat_from_coords.size}.")

    return {"tag": tag, "bold": bold, "flat": flat, "n_trials": int(bold.shape[1]), "trial_len": int(bold.shape[2])}


def _load_trial_keep_mask(glm_root, sub_tag, ses, run):
    path = join(glm_root, sub_tag, f"ses-{ses}", "GLMOutputs-mni-std", f"trial_keep_run{run}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing trial_keep mask: {path}")
    keep = np.asarray(np.load(path), dtype=bool).ravel()
    if keep.ndim != 1:
        raise ValueError(f"trial_keep mask must be 1D, got {keep.shape} ({path})")
    return keep, path


def _apply_trial_keep(bold, keep_mask, tag):
    if bold.shape[1] == keep_mask.size:
        return bold[:, keep_mask, :], int(np.count_nonzero(~keep_mask))
    kept = int(np.count_nonzero(keep_mask))
    if bold.shape[1] == kept:
        return bold, 0
    raise ValueError(f"trial_keep length mismatch for {tag}: active_bold has {bold.shape[1]} trials, "
        f"trial_keep has {keep_mask.size} entries ({kept} True).")


def _combine_runs_on_shared_voxels(run_payloads):
    if not run_payloads:
        raise ValueError("No runs provided.")
    trial_len = run_payloads[0]["trial_len"]
    for payload in run_payloads[1:]:
        if payload["trial_len"] != trial_len:
            raise ValueError(f"Trial length mismatch: {run_payloads[0]['tag']}={trial_len}, {payload['tag']}={payload['trial_len']}")

    shared_flat = np.asarray(run_payloads[0]["flat"], dtype=np.int64)
    for payload in run_payloads[1:]:
        shared_flat = np.intersect1d(shared_flat, payload["flat"], assume_unique=False)
    if shared_flat.size == 0:
        tags = ", ".join(payload["tag"] for payload in run_payloads)
        raise RuntimeError(f"No shared voxels across runs: {tags}")

    aligned = []
    for payload in run_payloads:
        flat = np.asarray(payload["flat"], dtype=np.int64)
        sorter = np.argsort(flat)
        sorted_flat = flat[sorter]
        pos = np.searchsorted(sorted_flat, shared_flat)
        if np.any(pos >= sorted_flat.size) or not np.all(sorted_flat[pos] == shared_flat):
            raise RuntimeError(f"Failed to align shared voxels for {payload['tag']}.")
        idx = sorter[pos]
        aligned.append(np.asarray(payload["bold"][idx], dtype=np.float32))

    combined_bold = np.concatenate(aligned, axis=1)
    return combined_bold


def _remove_empty_trials(bold):
    keep_trials = np.any(np.isfinite(bold), axis=(0, 2))
    removed_trials = int(np.count_nonzero(~keep_trials))
    if removed_trials > 0:
        bold = bold[:, keep_trials, :]
    return bold, removed_trials


def _fit_empca_from_bold(bold, nvec, niter):
    x = np.asarray(bold, dtype=np.float32).reshape(bold.shape[0], -1)
    keep_cols = np.any(np.isfinite(x), axis=0)
    removed_cols = int(np.count_nonzero(~keep_cols))
    if removed_cols > 0:
        x = x[:, keep_cols]
    if x.shape[1] == 0:
        raise RuntimeError("All columns are empty after filtering.")

    data = x.T
    w = np.isfinite(data)
    y = np.where(w, data, np.float32(0.0)).astype(np.float32, copy=False)
    row_weight = w.sum(axis=0, keepdims=True).astype(np.float32)
    mean = np.divide((y * w).sum(axis=0, keepdims=True), row_weight, out=np.zeros(row_weight.shape, dtype=np.float32), where=row_weight > 0)
    y -= mean
    var = np.divide((w * y ** 2).sum(axis=0, keepdims=True), row_weight, out=np.zeros(row_weight.shape, dtype=np.float32), where=row_weight > 0)
    scale = np.sqrt(var)
    np.divide(y, np.maximum(scale, np.float32(1e-6)), out=y, where=row_weight > 0)

    y = np.ascontiguousarray(y.T)
    w = np.ascontiguousarray(w.T)
    effective_nvec = max(1, min(int(nvec), y.shape[0], y.shape[1]))
    print(f"begin empca (nvec={effective_nvec}, niter={int(niter)}, voxels={y.shape[0]}, features={y.shape[1]})...", flush=True)
    model = empca(y, w, nvec=effective_nvec, niter=int(niter))
    del x, keep_cols, data, w, y, row_weight, mean, var, scale
    gc.collect()
    return model, removed_cols


def _process_subject_session(sub_tag, ses, runs, args):
    label = f"{sub_tag}_ses-{ses}"
    out_path = join(args.cache_dir, f"empca_model_{label}.npy")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] {label}: model exists ({out_path})", flush=True)
        return

    run_payloads = []
    total_removed_by_trial_keep = 0
    for run in runs:
        try:
            payload = _load_run_arrays(args.preproc_root, sub_tag, ses, run)
        except FileNotFoundError as exc:
            print(f"  run-{run}: missing preprocessed inputs ({exc}); skipping run.", flush=True)
            continue
        try:
            keep_mask, keep_path = _load_trial_keep_mask(args.glm_root, sub_tag, ses, run)
        except FileNotFoundError as exc:
            print(f"  run-{run}: missing trial_keep mask ({exc}); skipping run.", flush=True)
            continue
        filtered_bold, removed_count = _apply_trial_keep(payload["bold"], keep_mask, payload["tag"])
        print(f"  run-{run}: trials {payload['n_trials']} -> {filtered_bold.shape[1]} (removed {removed_count} using {keep_path})", flush=True)
        if filtered_bold.shape[1] == 0:
            print(f"  run-{run}: no trials left after trial_keep; skipping run.", flush=True)
            continue
        payload["bold"] = filtered_bold
        payload["n_trials"] = int(filtered_bold.shape[1])
        total_removed_by_trial_keep += removed_count
        run_payloads.append(payload)

    if not run_payloads:
        raise RuntimeError("No usable runs after applying trial_keep.")

    combined_bold = _combine_runs_on_shared_voxels(run_payloads)
    combined_bold, removed_empty_trials = _remove_empty_trials(combined_bold)
    if combined_bold.shape[1] == 0:
        raise RuntimeError("No trials left after removing empty trials.")

    if args.dry_run:
        print(f"[dry-run] {label}: combined_bold={combined_bold.shape}, trial_keep_removed={total_removed_by_trial_keep}, empty_trials_removed={removed_empty_trials}", flush=True)
        return

    model, removed_cols = _fit_empca_from_bold(combined_bold, nvec=args.nvec, niter=args.niter)
    np.save(out_path, model)
    print(f"[ok] {label}: saved {out_path} | combined_bold={combined_bold.shape} | "
        f"trial_keep_removed={total_removed_by_trial_keep} | empty_trials_removed={removed_empty_trials} | "
        f"empty_columns_removed={removed_cols}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fit and save EMPCA models for all subject/sessions in /Data/zahra.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--bold-root", default=None)
    parser.add_argument("--preproc-root", default=None)
    parser.add_argument("--glm-root", default=None)
    parser.add_argument("--cache-dir", default=os.environ.get("FMRI_EMPCA_CACHE_DIR", DEFAULT_CACHE_DIR))
    parser.add_argument("--subject", default=None, help="Subject filter, e.g. sub-pd009 or 9")
    parser.add_argument("--session", type=int, default=None, help="Session filter, e.g. 1")
    parser.add_argument("--nvec", type=int, default=int(os.environ.get("FMRI_EMPCA_NVEC", "100")))
    parser.add_argument("--niter", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict-missing", action="store_true",help="Fail when required run inputs are missing instead of skipping those runs/sessions.",)
    args = parser.parse_args()

    args.subject = _normalize_subject(args.subject)
    data_root = os.path.expanduser(args.data_root)
    args.bold_root = os.path.expanduser(args.bold_root or join(data_root, "bold_data"))
    args.preproc_root = os.path.expanduser(args.preproc_root or join(data_root, "results_beta_preprocessed"))
    args.glm_root = os.path.expanduser(args.glm_root or join(data_root, "results_glm"))
    args.cache_dir = os.path.expanduser(args.cache_dir)
    os.makedirs(args.cache_dir, exist_ok=True)

    discovered = _discover_subject_sessions(args.bold_root, subject=args.subject, session=args.session)
    print(f"bold_root: {args.bold_root}", flush=True)
    print(f"preproc_root: {args.preproc_root}", flush=True)
    print(f"glm_root: {args.glm_root}", flush=True)
    print(f"cache_dir (obj_param-compatible): {args.cache_dir}", flush=True)
    print("obj_param label convention expected by this script: FMRI_GROUP_LABEL=sub-pdXXX_ses-Y", flush=True)

    failures = []
    for (sub_tag, ses), runs in sorted(discovered.items()):
        print(f"\n=== Processing {sub_tag} ses-{ses} (runs={runs}) ===", flush=True)
        try:
            _process_subject_session(sub_tag, ses, runs, args)
        except RuntimeError as exc:
            if not args.strict_missing and str(exc) == "No usable runs after applying trial_keep.":
                print(f"[skip] {sub_tag} ses-{ses}: {exc}", flush=True)
            else:
                failures.append((sub_tag, ses, str(exc)))
                print(f"[failed] {sub_tag} ses-{ses}: {exc}", flush=True)
        except FileNotFoundError as exc:
            if args.strict_missing:
                failures.append((sub_tag, ses, str(exc)))
                print(f"[failed] {sub_tag} ses-{ses}: {exc}", flush=True)
            else:
                print(f"[skip] {sub_tag} ses-{ses}: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append((sub_tag, ses, str(exc)))
            print(f"[failed] {sub_tag} ses-{ses}: {exc}", flush=True)

    if failures:
        print("\nFailures:", flush=True)
        for sub_tag, ses, message in failures:
            print(f"  - {sub_tag} ses-{ses}: {message}", flush=True)
        raise SystemExit(1)

    print("\nAll subject/session EMPCA models finished successfully.", flush=True)


if __name__ == "__main__":
    main()
