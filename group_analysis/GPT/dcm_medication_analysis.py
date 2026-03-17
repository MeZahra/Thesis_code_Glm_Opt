from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_io import (
    CIRCUIT_BASE_ROIS,
    TMP_DCM_ROOT,
    aggregate_matrix_by_base_roi,
    bh_fdr,
    ensure_dir,
    list_paired_dcm_subjects,
    load_dcm_labels,
    load_subject_dcm_matrix,
    load_subject_dcm_vector,
    paired_delta_stats,
    read_json,
    to_serializable,
    write_json,
)


def _select_analysis_matrix(
    matrix: np.ndarray,
    labels: list[str],
    analysis_rois: list[str],
    aggregate_to_base_rois: bool,
) -> tuple[np.ndarray, list[str]]:
    if aggregate_to_base_rois:
        reduced, reduced_labels, _ = aggregate_matrix_by_base_roi(
            matrix,
            labels,
            include_base_rois=analysis_rois,
        )
        return reduced, reduced_labels

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    missing = [label for label in analysis_rois if label not in label_to_idx]
    if missing:
        raise ValueError(f"Requested ROI labels were not found in DCM labels: {missing}")
    keep_idx = [label_to_idx[label] for label in analysis_rois]
    selected = np.asarray(matrix, dtype=float)[np.ix_(keep_idx, keep_idx)]
    return selected, list(analysis_rois)


def _plot_circuit_matrices(
    off_mean: np.ndarray,
    on_mean: np.ndarray,
    delta_mean: np.ndarray,
    q_matrix: np.ndarray,
    labels: list[str],
    out_path: Path,
) -> None:
    n_labels = len(labels)
    annotate_cells = n_labels <= 12
    axis_fontsize = 9 if n_labels <= 12 else 6
    panel_width = max(4.8, 0.42 * n_labels + 2.5)
    fig_height = max(4.8, 0.32 * n_labels + 1.5)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(panel_width * 3, fig_height),
        constrained_layout=True,
    )
    shared_lim = float(
        np.nanmax(np.abs(np.concatenate([off_mean.ravel(), on_mean.ravel()])))
    )
    delta_lim = float(np.nanmax(np.abs(delta_mean))) or 1.0
    panels = [
        (off_mean, "Mean OFF", shared_lim),
        (on_mean, "Mean ON", shared_lim),
        (delta_mean, "Mean ON - OFF", delta_lim),
    ]

    for ax, (matrix, title, vmax) in zip(axes, panels):
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=axis_fontsize)
        ax.set_yticklabels(labels, fontsize=axis_fontsize)
        if annotate_cells:
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    mark = ""
                    if title == "Mean ON - OFF" and np.isfinite(q_matrix[row, col]):
                        if q_matrix[row, col] < 0.05:
                            mark = "*"
                    ax.text(
                        col,
                        row,
                        f"{matrix[row, col]:.2f}{mark}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_dcm_medication_analysis(
    out_dir: Path,
    circuit_rois: list[str] | None = None,
    aggregate_to_base_rois: bool = True,
) -> dict:
    out_dir = ensure_dir(Path(out_dir))
    dcm_labels = load_dcm_labels()
    analysis_rois = (
        list(circuit_rois)
        if circuit_rois is not None
        else (list(CIRCUIT_BASE_ROIS) if aggregate_to_base_rois else list(dcm_labels))
    )
    group_summary = read_json(TMP_DCM_ROOT / "group_level" / "group_summary.json")
    subjects = list_paired_dcm_subjects()

    subject_rows = []
    comparison_rows = []
    off_mats = []
    on_mats = []
    delta_mats = []

    for subject in subjects:
        off_full = load_subject_dcm_matrix(subject, "dcm_session1_off.npy")
        on_full = load_subject_dcm_matrix(subject, "dcm_session2_on.npy")
        delta_full = load_subject_dcm_matrix(subject, "dcm_session2_minus_session1.npy")
        modulatory_full = load_subject_dcm_matrix(
            subject, "dcm_modulatory_session2_on_minus_session1_off.npy"
        )
        intrinsic_full = load_subject_dcm_matrix(subject, "dcm_intrinsic_session1_off.npy")
        constant_drive_full = load_subject_dcm_vector(subject, "dcm_constant_drive.npy")

        off, labels = _select_analysis_matrix(
            off_full,
            dcm_labels,
            analysis_rois=analysis_rois,
            aggregate_to_base_rois=aggregate_to_base_rois,
        )
        on, _ = _select_analysis_matrix(
            on_full,
            dcm_labels,
            analysis_rois=analysis_rois,
            aggregate_to_base_rois=aggregate_to_base_rois,
        )
        delta, _ = _select_analysis_matrix(
            delta_full,
            dcm_labels,
            analysis_rois=analysis_rois,
            aggregate_to_base_rois=aggregate_to_base_rois,
        )
        modulatory, _ = _select_analysis_matrix(
            modulatory_full,
            dcm_labels,
            analysis_rois=analysis_rois,
            aggregate_to_base_rois=aggregate_to_base_rois,
        )
        intrinsic, _ = _select_analysis_matrix(
            intrinsic_full,
            dcm_labels,
            analysis_rois=analysis_rois,
            aggregate_to_base_rois=aggregate_to_base_rois,
        )
        off_mats.append(off)
        on_mats.append(on)
        delta_mats.append(delta)

        for target_idx, target_roi in enumerate(labels):
            for source_idx, source_roi in enumerate(labels):
                subject_rows.append(
                    {
                        "subject": subject,
                        "parameter_type": "coupling",
                        "target_roi": target_roi,
                        "source_roi": source_roi,
                        "off_parameter": float(off[target_idx, source_idx]),
                        "on_parameter": float(on[target_idx, source_idx]),
                        "delta_on_minus_off": float(delta[target_idx, source_idx]),
                        "modulatory_delta": float(modulatory[target_idx, source_idx]),
                        "intrinsic_off_parameter": float(intrinsic[target_idx, source_idx]),
                    }
                )

        constant_drive_matrix, constant_drive_labels = _select_analysis_matrix(
            np.diag(constant_drive_full),
            dcm_labels,
            analysis_rois=analysis_rois,
            aggregate_to_base_rois=aggregate_to_base_rois,
        )
        for idx, roi in enumerate(constant_drive_labels):
            subject_rows.append(
                {
                    "subject": subject,
                    "parameter_type": "constant_drive",
                    "target_roi": roi,
                    "source_roi": "constant",
                    "off_parameter": float("nan"),
                    "on_parameter": float("nan"),
                    "delta_on_minus_off": float("nan"),
                    "modulatory_delta": float("nan"),
                    "intrinsic_off_parameter": float(constant_drive_matrix[idx, idx]),
                }
            )

    subject_df = pd.DataFrame(subject_rows)
    coupling_df = subject_df[subject_df["parameter_type"] == "coupling"].copy()

    for target_roi in analysis_rois:
        for source_roi in analysis_rois:
            edge_df = coupling_df[
                (coupling_df["target_roi"] == target_roi)
                & (coupling_df["source_roi"] == source_roi)
            ].sort_values("subject")
            delta = edge_df["delta_on_minus_off"].to_numpy()
            stats_dict = paired_delta_stats(delta)
            comparison_rows.append(
                {
                    "target_roi": target_roi,
                    "source_roi": source_roi,
                    "edge": f"{source_roi} -> {target_roi}",
                    "is_self_connection": bool(target_roi == source_roi),
                    "mean_off": float(edge_df["off_parameter"].mean()),
                    "mean_on": float(edge_df["on_parameter"].mean()),
                    "mean_modulatory_delta": float(edge_df["modulatory_delta"].mean()),
                    **stats_dict,
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df["q_fdr"] = bh_fdr(comparison_df["p_signflip"])
    comparison_df = comparison_df.sort_values(
        ["q_fdr", "p_signflip", "is_self_connection", "mean_delta"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    subject_path = out_dir / "dcm_subject_parameters.csv"
    group_path = out_dir / "dcm_group_comparison.csv"
    group_fdr_path = out_dir / "dcm_group_comparison_fdr.csv"
    summary_path = out_dir / "dcm_circuit_summary.json"
    fig_path = out_dir / "dcm_circuit_off_on.png"

    subject_df.to_csv(subject_path, index=False)
    comparison_df.to_csv(group_path, index=False)
    comparison_df.sort_values(["q_fdr", "p_signflip", "mean_delta"], ascending=[True, True, False]).to_csv(
        group_fdr_path, index=False
    )

    off_mean = np.mean(np.stack(off_mats, axis=0), axis=0)
    on_mean = np.mean(np.stack(on_mats, axis=0), axis=0)
    delta_mean = np.mean(np.stack(delta_mats, axis=0), axis=0)
    q_matrix = np.full_like(delta_mean, np.nan, dtype=float)
    for _, row in comparison_df.iterrows():
        tgt = analysis_rois.index(row["target_roi"])
        src = analysis_rois.index(row["source_roi"])
        q_matrix[tgt, src] = row["q_fdr"]
    _plot_circuit_matrices(off_mean, on_mean, delta_mean, q_matrix, analysis_rois, fig_path)

    summary = {
        "analysis_note": (
            "This module re-summarizes the precomputed bilinear neuronal-state model "
            "saved under results/connectivity/tmp/dynamic_causal_modeling; it does not "
            "refit a classical hemodynamic DCM."
        ),
        "aggregation_mode": "base_roi_average" if aggregate_to_base_rois else "full_roi",
        "input_root": TMP_DCM_ROOT,
        "n_subjects": len(subjects),
        "analysis_rois": analysis_rois,
        "n_rois": len(analysis_rois),
        "circuit_rois": analysis_rois if aggregate_to_base_rois else [],
        "session_definition": group_summary.get("session_definition", {}),
        "matrix_convention": group_summary.get("matrix_convention"),
        "group_mean_abs_offdiag_off": group_summary.get("global_mean_abs_offdiag_off"),
        "group_mean_abs_offdiag_on": group_summary.get("global_mean_abs_offdiag_on"),
        "group_mean_abs_offdiag_delta_on_minus_off": group_summary.get(
            "global_mean_abs_offdiag_delta_on_minus_off"
        ),
        "n_edges_tested": int(comparison_df.shape[0]),
        "n_edges_fdr_below_0_05": int((comparison_df["q_fdr"] < 0.05).sum()),
        "top_edges_by_abs_mean_delta": to_serializable(
            comparison_df.assign(abs_mean_delta=comparison_df["mean_delta"].abs())
            .sort_values("abs_mean_delta", ascending=False)
            .head(10)[["edge", "mean_delta", "p_signflip", "q_fdr"]]
            .to_dict(orient="records")
        ),
        "output_files": {
            "dcm_subject_parameters": subject_path,
            "dcm_group_comparison": group_path,
            "dcm_group_comparison_fdr": group_fdr_path,
            "dcm_circuit_off_on": fig_path,
        },
    }
    write_json(summary_path, to_serializable(summary))

    return {
        "subjects": subjects,
        "analysis_rois": analysis_rois,
        "circuit_rois": analysis_rois,
        "subject_parameters_path": subject_path,
        "group_comparison_path": group_path,
        "group_comparison_fdr_path": group_fdr_path,
        "summary_path": summary_path,
        "figure_path": fig_path,
    }
