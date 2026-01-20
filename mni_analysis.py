#!/usr/bin/env python3
"""
Native-space analysis wrapper:
1) Resolve native-space inputs.
2) Stage inputs into an isolated analysis workspace (mni_space).
3) Run Beta_preprocessing.py writing outputs into mni_space.
4) Run mian_second_obj_25.py against the same mni_space.

This keeps native inputs separate from derived outputs and avoids reusing
existing EMPCA/derived files from native directories.
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run(cmd, env=None, cwd=None):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def resolve_native_dir(data_dir, sub_tag, ses, sub_zero):
    data_dir = Path(data_dir).expanduser().resolve()
    direct = data_dir / f"{sub_tag}_ses-{ses}_T1w_brain.nii.gz"
    if direct.exists():
        return data_dir
    candidate = data_dir / f"sub{sub_zero}_ses{int(ses):02d}"
    candidate_anat = candidate / f"{sub_tag}_ses-{ses}_T1w_brain.nii.gz"
    if candidate_anat.exists():
        return candidate
    raise FileNotFoundError(
        "\n".join(
            [
                "Could not locate native-space anatomy.",
                f"Checked: {direct}",
                f"Checked: {candidate_anat}",
                "Hint: set --data-dir to the folder containing native T1/BOLD/mask files.",
            ]
        )
    )


def ensure_native_inputs(native_dir, sub_tag, ses, runs):
    required = [
        native_dir / f"{sub_tag}_ses-{ses}_T1w_brain.nii.gz",
        native_dir / f"{sub_tag}_ses-{ses}_T1w_brain_mask.nii.gz",
        native_dir / f"{sub_tag}_ses-{ses}_T1w_brain_pve_0.nii.gz",
        native_dir / f"{sub_tag}_ses-{ses}_T1w_brain_pve_1.nii.gz",
    ]
    for run in runs:
        required.append(
            native_dir / f"{sub_tag}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg.nii.gz"
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        msg = ["Missing native-space inputs:"]
        msg.extend([f"  - {path}" for path in missing])
        msg.append("Hint: set --data-dir to the folder containing native inputs.")
        raise FileNotFoundError("\n".join(msg))


def _safe_symlink(src, dest):
    src = Path(src).resolve()
    dest = Path(dest)
    if dest.is_symlink():
        if dest.resolve() == src:
            return
        dest.unlink()
    elif dest.exists():
        raise RuntimeError(f"{dest} exists and is not a symlink; remove it to continue.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(src), str(dest))


def _resolve_behavior_file(native_dir, sub, sub_zero, ses_int):
    state = "OFF" if ses_int == 1 else "ON"
    expected_name = f"PSPD00{sub}_{state}_behav_metrics.mat"
    candidates = [
        native_dir / expected_name,
        native_dir / f"PSPD0{sub_zero}_{state}_behav_metrics.mat",
        native_dir / f"PSPD{sub:03d}_{state}_behav_metrics.mat",
        native_dir / f"PSPD{sub:02d}_{state}_behav_metrics.mat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, expected_name
    for candidate in native_dir.glob(f"PSPD*{sub}*_{state}_behav_metrics.mat"):
        return candidate, expected_name
    return None, expected_name


def _resolve_glmsingle_file(explicit_path, native_dir, sub, sub_zero, ses):
    if explicit_path:
        explicit = Path(explicit_path).expanduser().resolve()
        if not explicit.exists():
            raise FileNotFoundError(f"GLMsingle file not found: {explicit}")
        return explicit
    candidates = [
        native_dir / "TYPED_FITHRF_GLMDENOISE_RR.npy",
        native_dir / "GLMsingle" / f"GLMOutputs-sub{sub}-ses{ses}-std" / "TYPED_FITHRF_GLMDENOISE_RR.npy",
        native_dir / "GLMsingle" / f"GLMOutputs-sub{sub_zero}-ses{ses}-std" / "TYPED_FITHRF_GLMDENOISE_RR.npy",
    ]
    parent = native_dir.parent
    if parent and parent != native_dir:
        candidates.extend(
            [
                parent / "TYPED_FITHRF_GLMDENOISE_RR.npy",
                parent / "GLMsingle" / f"GLMOutputs-sub{sub}-ses{ses}-std" / "TYPED_FITHRF_GLMDENOISE_RR.npy",
                parent / "GLMsingle" / f"GLMOutputs-sub{sub_zero}-ses{ses}-std" / "TYPED_FITHRF_GLMDENOISE_RR.npy",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _stage_inputs(native_dir, analysis_data_dir, sub_tag, sub_tag_main, ses, runs, sub):
    inputs = [
        f"{sub_tag}_ses-{ses}_T1w_brain.nii.gz",
        f"{sub_tag}_ses-{ses}_T1w_brain_mask.nii.gz",
        f"{sub_tag}_ses-{ses}_T1w_brain_pve_0.nii.gz",
        f"{sub_tag}_ses-{ses}_T1w_brain_pve_1.nii.gz",
    ]
    for name in inputs:
        src = native_dir / name
        _safe_symlink(src, analysis_data_dir / name)
        if sub_tag_main != sub_tag:
            alias_name = name.replace(sub_tag, sub_tag_main, 1)
            _safe_symlink(src, analysis_data_dir / alias_name)

    for run in runs:
        bold_name = f"{sub_tag}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg.nii.gz"
        src = native_dir / bold_name
        _safe_symlink(src, analysis_data_dir / bold_name)
        if sub_tag_main != sub_tag:
            alias_name = bold_name.replace(sub_tag, sub_tag_main, 1)
            _safe_symlink(src, analysis_data_dir / alias_name)
        fmri_alias = analysis_data_dir / f"fmri_sub{sub}_ses{ses}_run{run}.nii.gz"
        _safe_symlink(src, fmri_alias)


def _stage_trial_keep(native_dir, analysis_data_dir, runs):
    for run in runs:
        src = native_dir / f"trial_keep_run{run}.npy"
        if src.exists():
            _safe_symlink(src, analysis_data_dir / src.name)


def _ensure_output_aliases(data_dir, beta_prefix, main_prefix, ses, runs):
    if beta_prefix == main_prefix:
        return
    patterns = [
        "nan_mask_flat_{prefix}_ses{ses}_run{run}.npy",
        "beta_volume_filter_{prefix}_ses{ses}_run{run}.npy",
        "active_coords_{prefix}_ses{ses}_run{run}.npy",
        "active_bold_{prefix}_ses{ses}_run{run}.npy",
    ]
    for run in runs:
        for pattern in patterns:
            src = data_dir / pattern.format(prefix=beta_prefix, ses=ses, run=run)
            if not src.exists():
                continue
            dest = data_dir / pattern.format(prefix=main_prefix, ses=ses, run=run)
            _safe_symlink(src, dest)


def _map_runs_for_main(runs):
    unique = sorted(set(runs))
    if len(unique) == 1:
        return {2: unique[0]}
    if len(unique) == 2:
        return {1: unique[0], 2: unique[1]}
    raise ValueError(
        "mian_second_obj_25.py supports at most 2 runs; "
        f"requested {len(unique)} runs: {unique}."
    )


def _alias_run_outputs(data_dir, prefix, ses, run_map):
    patterns = [
        "nan_mask_flat_{prefix}_ses{ses}_run{run}.npy",
        "beta_volume_filter_{prefix}_ses{ses}_run{run}.npy",
        "active_coords_{prefix}_ses{ses}_run{run}.npy",
        "active_bold_{prefix}_ses{ses}_run{run}.npy",
    ]
    for target_run, source_run in run_map.items():
        if target_run == source_run:
            continue
        for pattern in patterns:
            src = data_dir / pattern.format(prefix=prefix, ses=ses, run=source_run)
            if not src.exists():
                raise FileNotFoundError(
                    f"Missing preprocessing output for run {source_run}: {src}"
                )
            dest = data_dir / pattern.format(prefix=prefix, ses=ses, run=target_run)
            _safe_symlink(src, dest)


def _alias_bold_runs(data_dir, sub, ses, run_map):
    for target_run, source_run in run_map.items():
        if target_run == source_run:
            continue
        src = data_dir / f"fmri_sub{sub}_ses{ses}_run{source_run}.nii.gz"
        if not src.exists():
            raise FileNotFoundError(f"Missing BOLD source for run {source_run}: {src}")
        dest = data_dir / f"fmri_sub{sub}_ses{ses}_run{target_run}.nii.gz"
        _safe_symlink(src, dest)


def _alias_trial_keep(data_dir, run_map):
    for target_run, source_run in run_map.items():
        if target_run == source_run:
            continue
        src = data_dir / f"trial_keep_run{source_run}.npy"
        if not src.exists():
            continue
        dest = data_dir / f"trial_keep_run{target_run}.npy"
        _safe_symlink(src, dest)


def _verify_main_second_outputs(data_dir, main_prefix, ses, target_runs):
    required = []
    for run in target_runs:
        required.extend(
            [
                data_dir / f"nan_mask_flat_{main_prefix}_ses{ses}_run{run}.npy",
                data_dir / f"beta_volume_filter_{main_prefix}_ses{ses}_run{run}.npy",
                data_dir / f"active_coords_{main_prefix}_ses{ses}_run{run}.npy",
                data_dir / f"active_bold_{main_prefix}_ses{ses}_run{run}.npy",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        msg = ["Missing required preprocessing outputs:"]
        msg.extend([f"  - {path}" for path in missing])
        raise FileNotFoundError("\n".join(msg))


def _default_analysis_dir(native_dir, sub_zero, ses):
    stamp = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    return (native_dir / "mni_analysis" / f"sub{sub_zero}_ses{ses}" / stamp).resolve()


def _parse_runs(runs_str):
    runs = [int(r.strip()) for r in str(runs_str).split(",") if r.strip()]
    if not runs:
        raise ValueError("No runs provided; use --runs like '1,2'.")
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", default=os.environ.get("SUB", "9"))
    parser.add_argument("--ses", default=os.environ.get("SES", "1"))
    parser.add_argument("--data-dir", default=os.environ.get("FMRI_OPT_DATA_DIR", "data"))
    parser.add_argument("--analysis-dir", default=None, help="Workspace for staged inputs and outputs.")
    parser.add_argument("--mni-dir", default=None, help="Ignored; MNI conversion disabled for analysis.")
    parser.add_argument("--mni-template", default=None, help="Ignored; MNI conversion disabled for analysis.")
    parser.add_argument("--runs", default="1,2")
    parser.add_argument("--glmsingle-file", default=None)
    args = parser.parse_args()

    sub = int(args.sub)
    ses = str(args.ses)
    ses_int = int(ses)
    sub_zero = f"{sub:02d}"
    sub_tag = f"sub-pd{sub:03d}"
    sub_tag_main = f"sub-pd00{sub}"

    if args.mni_dir or args.mni_template:
        print("Note: MNI conversion is disabled; ignoring --mni-dir/--mni-template.", flush=True)

    runs = _parse_runs(args.runs)

    native_dir = resolve_native_dir(args.data_dir, sub_tag, ses, sub_zero)
    ensure_native_inputs(native_dir, sub_tag, ses, runs)

    analysis_root = (
        Path(args.analysis_dir).expanduser().resolve()
        if args.analysis_dir
        else _default_analysis_dir(native_dir, sub_zero, ses)
    )
    if analysis_root.name == "mni_space":
        mni_space_dir = analysis_root
        analysis_dir = analysis_root.parent
    else:
        analysis_dir = analysis_root
        mni_space_dir = analysis_dir / "mni_space"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    mni_space_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analysis workspace: {analysis_dir}", flush=True)
    print(f"mni_space: {mni_space_dir}", flush=True)

    _stage_inputs(native_dir, mni_space_dir, sub_tag, sub_tag_main, ses, runs, sub)
    _stage_trial_keep(native_dir, mni_space_dir, runs)

    behavior_src, behavior_name = _resolve_behavior_file(native_dir, sub, sub_zero, ses_int)
    if behavior_src is None:
        raise FileNotFoundError(
            "\n".join(
                [
                    "Could not locate behavior metrics file.",
                    f"Expected name in workspace: {analysis_data_dir / behavior_name}",
                    f"Searched in: {native_dir}",
                ]
            )
        )
    _safe_symlink(behavior_src, mni_space_dir / behavior_name)

    glmsingle_src = None
    if args.glmsingle_file:
        glmsingle_src = _resolve_glmsingle_file(args.glmsingle_file, native_dir, sub, sub_zero, ses)
    else:
        existing_glm = mni_space_dir / "TYPED_FITHRF_GLMDENOISE_RR.npy"
        if existing_glm.exists():
            glmsingle_src = existing_glm
        else:
            glmsingle_src = _resolve_glmsingle_file(None, native_dir, sub, sub_zero, ses)

    glmsingle_staged = None
    if glmsingle_src is not None:
        if mni_space_dir in glmsingle_src.parents:
            glmsingle_staged = glmsingle_src
        else:
            glmsingle_staged = mni_space_dir / glmsingle_src.name
            _safe_symlink(glmsingle_src, glmsingle_staged)
    if glmsingle_staged is None:
        raise FileNotFoundError(
            "\n".join(
                [
                    "GLMsingle output not found.",
                    "Provide --glmsingle-file or place TYPED_FITHRF_GLMDENOISE_RR.npy in the input directory.",
                    f"Searched in: {native_dir}",
                ]
            )
        )

    project_dir = Path(__file__).resolve().parent
    beta_script = project_dir / "Beta_preprocessing.py"
    main_script = project_dir / "mian_second_obj_25.py"

    env = os.environ.copy()
    env["SUB"] = sub_zero
    env["SES"] = ses
    env["FMRI_OPT_DATA_DIR"] = str(mni_space_dir)
    beta_cmd = [
        sys.executable,
        str(beta_script),
        "--output-dir",
        str(mni_space_dir),
        "--runs",
        args.runs,
    ]
    if glmsingle_staged is not None:
        beta_cmd.extend(["--glmsingle-file", str(glmsingle_staged)])
    run(beta_cmd, cwd=project_dir, env=env)

    beta_prefix = f"sub{sub_zero}"
    main_prefix = f"sub0{sub}"
    _ensure_output_aliases(mni_space_dir, beta_prefix, main_prefix, ses, runs)
    run_map = _map_runs_for_main(runs)
    _alias_run_outputs(mni_space_dir, main_prefix, ses, run_map)
    _alias_bold_runs(mni_space_dir, sub, ses, run_map)
    _alias_trial_keep(mni_space_dir, run_map)
    target_runs = sorted(run_map.keys())
    _verify_main_second_outputs(mni_space_dir, main_prefix, ses, target_runs)

    empca_dir = mni_space_dir / "data" / f"sub0{sub}_ses0{ses}"
    empca_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SUB"] = str(sub)
    env["SES"] = ses
    env["FMRI_OPT_DATA_DIR"] = str(mni_space_dir)
    run([sys.executable, str(main_script)], cwd=mni_space_dir, env=env)


if __name__ == "__main__":
    main()
