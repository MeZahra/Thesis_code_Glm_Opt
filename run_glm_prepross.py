#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
GLM_SCRIPT = REPO_ROOT / "GLMsingle" / "main_glm.py"
BETA_SCRIPT = REPO_ROOT / "Beta_preprocessing.py"

# TRIAL_CONFIGS = [{"trial_metric": "dvars", "trial_z": 3.5, "trial_fallback": 95, "trial_max_drop": 0.2},
#     {"trial_metric": "std", "trial_z": 3.0, "trial_fallback": 95, "trial_max_drop": 0.15},
#     {"trial_metric": "mean_abs", "trial_z": 3.0, "trial_fallback": 95, "trial_max_drop": 0.1}]
TRIAL_CONFIGS = [{"trial_metric": "dvars", "trial_z": 0, "trial_fallback": 0, "trial_max_drop": 0.2}]

SUBJECT_ID = "09"
SESSION_ID = "1"
TRIAL_ONSETS = "blocks" #'go_times'

# Environment variables for GLM configuration:
# - GLM_TRIAL_METRIC: Trial quality metric (dvars, std, mean_abs)
# - GLM_TRIAL_Z: Z-score threshold for trial outliers
# - GLM_TRIAL_FALLBACK: Fallback percentile for trial filtering
# - GLM_TRIAL_MAX_DROP: Maximum fraction of trials to drop
# - GLM_TRIAL_ONSETS: Trial onset source ('blocks' or 'go_times')
# - GLM_ROI_STAT: ROI ranking statistic (percentile_95, percentile_90, mean, peak)


def _run(cmd, env=None):
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=REPO_ROOT)


def main():
    base_env = os.environ.copy()
    for cfg in TRIAL_CONFIGS:
        env = base_env.copy()
        env["GLM_TRIAL_METRIC"] = cfg["trial_metric"]
        env["GLM_TRIAL_Z"] = str(cfg["trial_z"])
        env["GLM_TRIAL_FALLBACK"] = str(cfg["trial_fallback"])
        env["GLM_TRIAL_MAX_DROP"] = str(cfg["trial_max_drop"])
        env["GLM_TRIAL_ONSETS"] = TRIAL_ONSETS
        _run([sys.executable, str(GLM_SCRIPT)], env=env)

        output_dir = REPO_ROOT / "GLMsingle" / f"GLMOutputs-sub{SUBJECT_ID}-ses{SESSION_ID}-{cfg['trial_metric']}"
        glmsingle_file = output_dir / "TYPED_FITHRF_GLMDENOISE_RR.npy"
        if not glmsingle_file.exists():
            raise FileNotFoundError(f"Missing GLMsingle output: {glmsingle_file}")

        mask_indices = output_dir / "mask_indices.npy"
        roi_stat = os.environ.get("GLM_ROI_STAT", "percentile_95")
        for skip_ttest, tag in ((False, "ttest"), (True, "skip-ttest")):
            args = [sys.executable, str(BETA_SCRIPT), "--gray-threshold", "0", "--skip-hampel", "--output-dir", str(output_dir),
                    "--glmsingle-file", str(glmsingle_file), "--output-tag", tag, "--roi-stat", roi_stat]
            if skip_ttest:
                args.append("--skip-ttest")
            if mask_indices.exists():
                args.extend(["--mask-indices", str(mask_indices)])
            _run(args)


if __name__ == "__main__":
    main()
