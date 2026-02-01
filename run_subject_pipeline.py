#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = REPO_ROOT / "mian_second_obj_25.py"
BOLD_SCRIPT = REPO_ROOT / "bold_task_viz.py"
DICE_SCRIPT = REPO_ROOT / "Dice.py"


def _parse_id(raw, label):
    value = str(raw).strip().lower()
    value = re.sub(rf"^{label}[-_]*", "", value)
    if not value.isdigit():
        raise argparse.ArgumentTypeError(f"{label} must be numeric (got {raw!r}).")
    return int(value)


def _resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _first_existing(paths=(), glob_patterns=()):
    for path in paths:
        if path and path.exists():
            return path
    for root, pattern in glob_patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _collect_missing(label, missing, paths=(), glob_patterns=()):
    found = _first_existing(paths, glob_patterns)
    if found is None:
        display = [str(p) for p in paths if p]
        for root, pattern in glob_patterns:
            display.append(str(root / pattern))
        missing.append((label, display))
    return found


def _print_missing(title, missing, stream):
    if not missing:
        return
    print(title, file=stream)
    for label, paths in missing:
        print(f"- {label}", file=stream)
        for path in paths:
            print(f"  - {path}", file=stream)


def _ensure_data_link(results_dir, data_root):
    link = results_dir / "data"
    if link.exists():
        return
    try:
        link.symlink_to(data_root, target_is_directory=True)
    except OSError:
        print(
            f"Warning: could not create data symlink at {link}; "
            "relative 'data/...' paths may not resolve.",
            file=sys.stderr,
        )


def _default_output_prefix(results_dir, sub_label, ses_label, task, bold, beta, smooth, gamma):
    return results_dir / (
        f"voxel_weights_mean_foldavg_sub{sub_label}_ses{ses_label}"
        f"_task{task:g}_bold{bold:g}_beta{beta:g}_smooth{smooth:g}_gamma{gamma:g}"
    )


def _resolve_output_prefix(prefix, results_dir):
    if prefix is not None:
        return _resolve_path(prefix)
    matches = sorted(results_dir.glob("voxel_weights_mean_foldavg_*_motor_voxel_indicies.npz"))
    if len(matches) == 1:
        suffix = "_motor_voxel_indicies.npz"
        return Path(str(matches[0])[: -len(suffix)])
    return None


def _ensure_indices(prefix, results_dir):
    motor = Path(f"{prefix}_motor_voxel_indicies.npz")
    selected = Path(f"{prefix}_selected_voxel_indicies.npz")
    if motor.exists() and selected.exists():
        return prefix
    auto_prefix = _resolve_output_prefix(None, results_dir)
    if auto_prefix and auto_prefix != prefix:
        motor = Path(f"{auto_prefix}_motor_voxel_indicies.npz")
        selected = Path(f"{auto_prefix}_selected_voxel_indicies.npz")
        if motor.exists() and selected.exists():
            return auto_prefix
    missing = []
    if not motor.exists():
        missing.append(str(motor))
    if not selected.exists():
        missing.append(str(selected))
    raise FileNotFoundError("Missing indices files:\n" + "\n".join(f"- {path}" for path in missing))


def _run_step(label, cmd, cwd=None, env=None):
    print(f"\n==> {label}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run mian_second_obj_25.py, bold_task_viz.py, and Dice.py in sequence "
            "with data and results organized by subject/session."
        )
    )
    parser.add_argument("--sub", required=True, type=lambda v: _parse_id(v, "sub"))
    parser.add_argument("--ses", required=True, type=lambda v: _parse_id(v, "ses"))
    parser.add_argument("--data-dir", default="data", help="Base data directory (default: data).")
    parser.add_argument("--results-dir", default="results", help="Base results directory (default: results).")
    parser.add_argument("--runs", default="1,2", help="Comma-separated run numbers for bold_task_viz.py.")
    parser.add_argument("--trial-keep-run", type=int, default=None, help="Pass through to bold_task_viz.py.")
    parser.add_argument("--trial-len", type=int, default=9, help="Pass through to bold_task_viz.py.")
    parser.add_argument("--task-alpha", type=float, default=0.8, help="Used to build output prefix.")
    parser.add_argument("--bold-alpha", type=float, default=1.0, help="Used to build output prefix.")
    parser.add_argument("--beta-alpha", type=float, default=0.5, help="Used to build output prefix.")
    parser.add_argument("--smooth-alpha", type=float, default=1.2, help="Used to build output prefix.")
    parser.add_argument("--gamma", type=float, default=1.5, help="Used to build output prefix.")
    parser.add_argument("--output-prefix", default=None, help="Override voxel weight prefix for bold_task_viz.py.")
    args = parser.parse_args()

    sub_int = args.sub
    ses_int = args.ses
    sub_str = str(sub_int)
    ses_str = str(ses_int)
    sub0_label = f"0{sub_str}"
    subj_dir_name = f"sub{sub_int:02d}_ses{ses_int:02d}"

    data_base = _resolve_path(args.data_dir)
    data_root = data_base / subj_dir_name
    alt_data_root = data_base / f"sub{sub_str}_ses{ses_str}"
    if not data_root.is_dir() and not alt_data_root.is_dir():
        if data_base.is_dir() and data_base.name in (subj_dir_name, f"sub{sub_str}_ses{ses_str}"):
            data_root = data_base
            data_base = data_base.parent
        else:
            raise SystemExit(
                f"Subject data folder not found. Tried:\n- {data_root}\n- {alt_data_root}"
            )
    if not data_root.is_dir():
        data_root = alt_data_root

    results_base = _resolve_path(args.results_dir)
    results_dir = results_base / subj_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    _ensure_data_link(results_dir, data_base)

    for script in (MAIN_SCRIPT, BOLD_SCRIPT, DICE_SCRIPT):
        if not script.exists():
            raise SystemExit(f"Missing script: {script}")

    missing = []
    optional = []

    _collect_missing(
        "T1w brain",
        missing,
        paths=[data_root / f"sub-pd00{sub_str}_ses-{ses_str}_T1w_brain.nii.gz"],
    )
    _collect_missing(
        "Brain mask",
        missing,
        paths=[data_root / f"sub-pd00{sub_str}_ses-{ses_str}_T1w_brain_mask.nii.gz"],
    )
    _collect_missing(
        "CSF mask",
        missing,
        paths=[data_root / f"sub-pd00{sub_str}_ses-{ses_str}_T1w_brain_pve_0.nii.gz"],
    )
    _collect_missing(
        "Gray mask",
        missing,
        paths=[data_root / f"sub-pd00{sub_str}_ses-{ses_str}_T1w_brain_pve_1.nii.gz"],
    )

    _collect_missing(
        "nan_mask run2",
        missing,
        paths=[
            data_root / f"nan_mask_flat_sub{sub0_label}_ses{ses_str}_run2.npy",
            _resolve_path(f"data/nan_mask_flat_sub{sub_str}_ses{ses_str}_run2.npy"),
        ],
    )
    _collect_missing(
        "active_coords run2",
        missing,
        paths=[
            data_root / f"active_coords_sub{sub0_label}_ses{ses_str}_run2.npy",
            _resolve_path(f"data/sub{sub0_label}_ses0{ses_str}/active_coords_sub{sub0_label}_ses{ses_str}_run2.npy"),
        ],
    )
    _collect_missing(
        "beta_volume_filter run2",
        missing,
        paths=[
            data_root / f"beta_volume_filter_sub{sub0_label}_ses{ses_str}_run2.npy",
            _resolve_path(f"data/sub{sub0_label}_ses0{ses_str}/beta_volume_filter_sub{sub0_label}_ses{ses_str}_run2.npy"),
        ],
    )
    _collect_missing(
        "active_bold run2",
        missing,
        paths=[
            data_root / f"active_bold_sub{sub0_label}_ses{ses_str}_run2.npy",
            _resolve_path(f"data/sub{sub0_label}_ses0{ses_str}/active_bold_sub{sub0_label}_ses{ses_str}_run2.npy"),
        ],
    )
    _collect_missing(
        "BOLD run2",
        missing,
        paths=[
            data_root / f"fmri_sub{sub_str}_ses{ses_str}_run2.nii.gz",
            data_root / f"fmri_sub{sub_str}_ses{ses_str}_run2.npy",
        ],
        glob_patterns=[
            (data_root, f"sub-pd00{sub_str}_ses-{ses_str}_run-2*_bold*_reg.nii.gz"),
        ],
    )

    if ses_int == 1:
        behav_name = f"PSPD00{sub_str}_OFF_behav_metrics.mat"
    else:
        behav_name = f"PSPD00{sub_str}_ON_behav_metrics.mat"
    _collect_missing("Behavior metrics", missing, paths=[data_root / behav_name])

    _collect_missing("GLMsingle model", missing, paths=[data_root / "TYPED_FITHRF_GLMDENOISE_RR.npy"])
    _collect_missing("GLMsingle design", missing, paths=[data_root / "DESIGNINFO.npy"])

    run1_core = [
        data_root / f"nan_mask_flat_sub{sub0_label}_ses{ses_str}_run1.npy",
        data_root / f"active_coords_sub{sub0_label}_ses{ses_str}_run1.npy",
        data_root / f"beta_volume_filter_sub{sub0_label}_ses{ses_str}_run1.npy",
    ]
    run1_complete = all(path.exists() for path in run1_core)
    if not run1_complete:
        optional.append(
            (
                "Run1 core inputs (script will fall back to run2-only mode)",
                [str(path) for path in run1_core],
            )
        )
    else:
        _collect_missing(
            "active_bold run1",
            missing,
            paths=[data_root / f"active_bold_sub{sub0_label}_ses{ses_str}_run1.npy"],
        )
        _collect_missing(
            "BOLD run1",
            missing,
            paths=[
                data_root / f"fmri_sub{sub_str}_ses{ses_str}_run1.nii.gz",
                data_root / f"fmri_sub{sub_str}_ses{ses_str}_run1.npy",
            ],
            glob_patterns=[
                (data_root, f"sub-pd00{sub_str}_ses-{ses_str}_run-1*_bold*_reg.nii.gz"),
            ],
        )

    for run in (1, 2):
        path = data_root / f"trial_keep_run{run}.npy"
        if not path.exists():
            optional.append((f"trial_keep_run{run}.npy (optional)", [str(path)]))

    if missing:
        _print_missing("Missing required inputs:", missing, sys.stderr)
        raise SystemExit(1)
    if optional:
        _print_missing("Optional inputs not found:", optional, sys.stderr)

    env = os.environ.copy()
    env["SUB"] = sub_str
    env["SES"] = ses_str
    env["FMRI_OPT_DATA_DIR"] = str(data_base)

    _run_step(
        "Running mian_second_obj_25.py",
        [sys.executable, str(MAIN_SCRIPT)],
        cwd=results_dir,
        env=env,
    )

    output_prefix = _resolve_output_prefix(args.output_prefix, results_dir)
    if output_prefix is None:
        output_prefix = _default_output_prefix(
            results_dir,
            sub_str,
            ses_str,
            args.task_alpha,
            args.bold_alpha,
            args.beta_alpha,
            args.smooth_alpha,
            args.gamma,
        )
    output_prefix = _ensure_indices(output_prefix, results_dir)

    bold_cmd = [
        sys.executable,
        str(BOLD_SCRIPT),
        "--runs",
        args.runs,
        "--model-path",
        str(data_root / "TYPED_FITHRF_GLMDENOISE_RR.npy"),
        "--design-path",
        str(data_root / "DESIGNINFO.npy"),
        "--out-dir",
        str(results_dir),
        "--output-prefix",
        str(output_prefix),
        "--trial-len",
        str(args.trial_len),
        "--trial-keep-dir",
        str(data_root),
    ]
    if args.trial_keep_run is not None:
        bold_cmd.extend(["--trial-keep-run", str(args.trial_keep_run)])

    _run_step("Running bold_task_viz.py", bold_cmd)

    nii_files = sorted(results_dir.glob("*.nii.gz"))
    if not nii_files:
        raise SystemExit(f"No .nii.gz files found in {results_dir} for Dice.py.")

    _run_step(
        "Running Dice.py",
        [sys.executable, str(DICE_SCRIPT), "--input-dir", str(results_dir)],
    )


if __name__ == "__main__":
    main()
