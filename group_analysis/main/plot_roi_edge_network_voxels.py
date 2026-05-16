#!/usr/bin/env python3
"""Visualize selected ROI-network voxels used by a saved connectivity figure."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.patches import Rectangle
from nilearn import plotting


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]


DEFAULT_TARGET_FIGURE = (
    _REPO_ROOT
    / "results"
    / "connectivity"
    / "roi_edge_network"
    / "mutual_information_ksg"
    / "cross_subject_only_laplacian_spectral_distance_signed_distribution.png"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the selected voxels and hemisphere-split ROI nodes used by an "
            "roi_edge_network result, then save color-coded projection plots and a label image."
        )
    )
    parser.add_argument("--target-figure", type=Path, default=DEFAULT_TARGET_FIGURE)
    parser.add_argument(
        "--weight-img",
        type=Path,
        default=None,
        help=(
            "Optional voxel-weight NIfTI. When provided, nonzero voxels from this image are "
            "grouped by ROI atlas label instead of reading roi_nodes.csv."
        ),
    )
    parser.add_argument(
        "--roi-nodes",
        type=Path,
        default=None,
        help="roi_nodes.csv for the network. Defaults to the target figure's grandparent directory.",
    )
    parser.add_argument(
        "--roi-img",
        type=Path,
        default=None,
        help="Fitted ROI atlas NIfTI. If omitted, known connectivity atlas locations are tried.",
    )
    parser.add_argument(
        "--roi-summary",
        type=Path,
        default=None,
        help="ROI summary JSON with all_roi_names_in_order. Used for --weight-img labels.",
    )
    parser.add_argument(
        "--voxel-indices-path",
        type=Path,
        default=None,
        help="selected_voxel_indices.npz. If omitted, candidates are tried and matched to roi_nodes.csv.",
    )
    parser.add_argument(
        "--anat-img",
        type=Path,
        default=None,
        help="Anatomical NIfTI used as HTML background. If omitted, known MNI anatomy locations are tried.",
    )
    parser.add_argument("--midline-band-mm", type=float, default=1.0)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="Output path without extension. Defaults beside the target figure.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]


def _load_selected_ijk(path: Path, volume_shape: tuple[int, int, int]) -> np.ndarray:
    pack = np.load(path, allow_pickle=True)
    if "selected_ijk" in pack.files:
        ijk = np.asarray(pack["selected_ijk"], dtype=np.int32)
    elif "selected_flat_indices" in pack.files:
        flat = np.asarray(pack["selected_flat_indices"], dtype=np.int64)
        ijk = np.column_stack(np.unravel_index(flat, volume_shape)).astype(np.int32, copy=False)
    else:
        raise KeyError(f"{path} must contain selected_ijk or selected_flat_indices.")
    if ijk.ndim != 2 or ijk.shape[1] != 3:
        raise ValueError(f"Expected selected_ijk shape (N, 3); got {ijk.shape} from {path}")
    return ijk


def _candidate_roi_imgs(network_dir: Path) -> list[Path]:
    connectivity_root = network_dir.parent
    return [
        connectivity_root / "created_rois_fitted.nii.gz",
        connectivity_root / "atlas figure" / "created_rois_fitted.nii.gz",
        _REPO_ROOT / "results" / "connectivity" / "created_rois_fitted.nii.gz",
        _REPO_ROOT / "results" / "connectivity" / "atlas figure" / "created_rois_fitted.nii.gz",
    ]


def _candidate_roi_summaries(network_dir: Path) -> list[Path]:
    connectivity_root = network_dir.parent
    return [
        connectivity_root / "created_roi_summary.json",
        connectivity_root / "atlas figure" / "created_roi_summary.json",
        _REPO_ROOT / "results" / "connectivity" / "created_roi_summary.json",
        _REPO_ROOT / "results" / "connectivity" / "atlas figure" / "created_roi_summary.json",
    ]


def _candidate_voxel_indices(network_dir: Path) -> list[Path]:
    connectivity_root = network_dir.parent
    candidates: list[Path] = []
    for summary_path in [
        network_dir / "roi_edge_network_summary.json",
        network_dir / "1tmp" / "roi_edge_network_summary.json",
        connectivity_root / "tmp" / "roi_edge_network" / "roi_edge_network_summary.json",
    ]:
        if summary_path.exists():
            try:
                payload = _read_json(summary_path)
            except json.JSONDecodeError:
                continue
            raw_path = payload.get("voxel_indices_path")
            if raw_path:
                candidates.append(Path(raw_path))

    candidates.extend(
        [
            connectivity_root / "tmp" / "data" / "selected_voxel_indices.npz",
            connectivity_root / "data" / "selected_voxel_indices.npz",
            _REPO_ROOT / "results" / "connectivity" / "tmp" / "data" / "selected_voxel_indices.npz",
            _REPO_ROOT / "results" / "connectivity" / "data" / "selected_voxel_indices.npz",
        ]
    )

    out: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.expanduser()
        if not resolved.is_absolute():
            resolved = (_REPO_ROOT / resolved).resolve()
        else:
            resolved = resolved.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _roi_name_lookup(summary_path: Path | None) -> dict[int, str]:
    if summary_path is None or not summary_path.exists():
        return {}
    payload = _read_json(summary_path)
    names: dict[int, str] = {}
    rows = payload.get("roi_rows")
    if isinstance(rows, list):
        for row in rows:
            roi_id = row.get("roi_id")
            roi_name = row.get("roi_name")
            if isinstance(roi_id, int) and isinstance(roi_name, str) and roi_name:
                names[int(roi_id)] = roi_name
    all_names = payload.get("all_roi_names_in_order")
    if isinstance(all_names, list):
        for idx, name in enumerate(all_names, start=1):
            if idx not in names and isinstance(name, str) and name:
                names[idx] = name
    return names


def _candidate_anat_imgs(network_dir: Path) -> list[Path]:
    connectivity_root = network_dir.parent
    candidates: list[Path] = []
    for summary_path in [
        connectivity_root / "atlas figure" / "created_roi_summary.json",
        connectivity_root / "created_roi_summary.json",
    ]:
        if summary_path.exists():
            try:
                payload = _read_json(summary_path)
            except json.JSONDecodeError:
                continue
            raw_path = payload.get("anat_path")
            if raw_path:
                candidates.append(Path(raw_path))

    candidates.extend(
        [
            connectivity_root / "tmp" / "data" / "MNI152_T1_2mm_brain.nii.gz",
            connectivity_root / "data" / "MNI152_T1_2mm_brain.nii.gz",
            _REPO_ROOT / "results" / "connectivity" / "tmp" / "data" / "MNI152_T1_2mm_brain.nii.gz",
            _REPO_ROOT / "results" / "connectivity" / "data" / "MNI152_T1_2mm_brain.nii.gz",
        ]
    )

    out: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.expanduser()
        if not resolved.is_absolute():
            resolved = (_REPO_ROOT / resolved).resolve()
        else:
            resolved = resolved.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _assign_nodes(
    roi_nodes: pd.DataFrame,
    roi_data: np.ndarray,
    selected_ijk: np.ndarray,
    affine: np.ndarray,
    midline_band_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = selected_ijk.T
    base_roi_at_voxel = roi_data[x, y, z].astype(np.int32, copy=False)
    selected_coords_mm = nib.affines.apply_affine(affine, selected_ijk)
    selected_x_mm = selected_coords_mm[:, 0]

    node_assignment = np.zeros(selected_ijk.shape[0], dtype=np.int16)
    midline_band = float(max(0.0, midline_band_mm))

    node_lookup = {
        (int(row.base_roi_id), str(row.hemisphere)): int(row.node_id)
        for row in roi_nodes.itertuples(index=False)
    }
    base_ids = sorted(int(v) for v in roi_nodes["base_roi_id"].unique())

    for base_roi_id in base_ids:
        members = np.flatnonzero(base_roi_at_voxel == base_roi_id).astype(np.int64, copy=False)
        if members.size == 0:
            continue

        if (base_roi_id, "B") in node_lookup:
            node_assignment[members] = node_lookup[(base_roi_id, "B")]
            continue
        if (base_roi_id, "M") in node_lookup:
            node_assignment[members] = node_lookup[(base_roi_id, "M")]
            continue

        roi_x = selected_x_mm[members]
        left = members[roi_x < -midline_band]
        right = members[roi_x > midline_band]
        mid = members[np.abs(roi_x) <= midline_band]

        if mid.size > 0:
            if left.size >= right.size and left.size > 0:
                left = np.concatenate([left, mid])
            elif right.size > 0:
                right = np.concatenate([right, mid])

        if (base_roi_id, "L") in node_lookup:
            node_assignment[left] = node_lookup[(base_roi_id, "L")]
        if (base_roi_id, "R") in node_lookup:
            node_assignment[right] = node_lookup[(base_roi_id, "R")]

    return node_assignment, selected_coords_mm


def _counts_match(roi_nodes: pd.DataFrame, assignment: np.ndarray) -> bool:
    observed = np.bincount(assignment.astype(int), minlength=int(roi_nodes["node_id"].max()) + 1)
    expected = roi_nodes.set_index("node_id")["n_selected_voxels"].astype(int)
    for node_id, count in expected.items():
        if int(observed[int(node_id)]) != int(count):
            return False
    return int(np.count_nonzero(assignment > 0)) == int(expected.sum())


def _choose_voxel_indices(
    explicit_path: Path | None,
    candidates: list[Path],
    roi_nodes: pd.DataFrame,
    roi_data: np.ndarray,
    affine: np.ndarray,
    midline_band_mm: float,
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    if explicit_path is not None:
        candidates = [explicit_path]

    tried: list[str] = []
    for path in _existing(candidates):
        ijk = _load_selected_ijk(path, roi_data.shape)
        assignment, coords_mm = _assign_nodes(roi_nodes, roi_data, ijk, affine, midline_band_mm)
        tried.append(f"{path} ({ijk.shape[0]} voxels)")
        if explicit_path is not None or _counts_match(roi_nodes, assignment):
            if not _counts_match(roi_nodes, assignment):
                raise ValueError(
                    f"{path} does not reproduce roi_nodes.csv voxel counts. "
                    "Use the selected_voxel_indices.npz from the same network run."
                )
            return path, ijk, assignment, coords_mm

    raise FileNotFoundError(
        "Could not find a selected_voxel_indices.npz that matches roi_nodes.csv. Tried: "
        + "; ".join(tried or [str(p) for p in candidates])
    )


def _roi_nodes_from_weight_img(
    weight_img_path: Path,
    roi_data: np.ndarray,
    roi_affine: np.ndarray,
    roi_names: dict[int, str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    weight_img = nib.load(str(weight_img_path))
    if weight_img.shape != roi_data.shape:
        raise ValueError(
            f"Weight image shape {weight_img.shape} does not match ROI atlas shape {roi_data.shape}: "
            f"{weight_img_path}"
        )
    if not np.allclose(weight_img.affine, roi_affine):
        raise ValueError(f"Weight image affine does not match ROI atlas affine: {weight_img_path}")

    weight_data = np.nan_to_num(np.asarray(weight_img.get_fdata(), dtype=np.float64), nan=0.0)
    selected_ijk = np.column_stack(np.where(weight_data != 0.0)).astype(np.int32, copy=False)
    if selected_ijk.size == 0:
        raise ValueError(f"No nonzero voxels found in {weight_img_path}")

    x, y, z = selected_ijk.T
    roi_labels = roi_data[x, y, z].astype(np.int32, copy=False)
    coords_mm = nib.affines.apply_affine(roi_affine, selected_ijk)

    rows = []
    assignment = np.zeros(selected_ijk.shape[0], dtype=np.int16)
    unique_labels = sorted(int(v) for v in np.unique(roi_labels))
    for node_id, base_roi_id in enumerate(unique_labels, start=1):
        members = np.flatnonzero(roi_labels == base_roi_id)
        if base_roi_id > 0:
            node_name = roi_names.get(base_roi_id, f"ROI_{base_roi_id}")
        else:
            node_name = "Outside ROI atlas"
        centroid = np.mean(coords_mm[members], axis=0)
        assignment[members] = int(node_id)
        rows.append(
            {
                "node_id": int(node_id),
                "base_roi_id": int(base_roi_id),
                "hemisphere": "B",
                "node_name": node_name,
                "n_selected_voxels": int(members.size),
                "x_mm": float(centroid[0]),
                "y_mm": float(centroid[1]),
                "z_mm": float(centroid[2]),
            }
        )

    return pd.DataFrame(rows), selected_ijk, assignment, coords_mm


def _palette(n_colors: int) -> list[tuple[float, float, float, float]]:
    colors: list[tuple[float, float, float, float]] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.colormaps[cmap_name]
        colors.extend([cmap(i) for i in range(cmap.N)])
    return colors[:n_colors]


def _save_label_image(
    out_path: Path,
    roi_img: nib.Nifti1Image,
    selected_ijk: np.ndarray,
    assignment: np.ndarray,
) -> nib.Nifti1Image:
    label_data = np.zeros(roi_img.shape, dtype=np.int16)
    valid = assignment > 0
    x, y, z = selected_ijk[valid].T
    label_data[x, y, z] = assignment[valid].astype(np.int16, copy=False)
    label_img = nib.Nifti1Image(label_data, roi_img.affine, roi_img.header)
    label_img.set_data_dtype(np.int16)
    nib.save(label_img, str(out_path))
    return label_img


def _save_html_overlay(
    out_html: Path,
    label_img: nib.Nifti1Image,
    anat_img_path: Path,
    colors: list[tuple[float, float, float, float]],
    roi_nodes: pd.DataFrame,
    target_figure: Path,
) -> None:
    cmap = ListedColormap(colors)
    view = plotting.view_img(
        label_img,
        bg_img=str(anat_img_path),
        threshold=0.5,
        cmap=cmap,
        symmetric_cmap=False,
        vmin=1.0,
        vmax=float(len(colors)),
        opacity=0.72,
        colorbar=True,
        title=f"Selected ROI-network voxels for {target_figure.parent.name}/{target_figure.name}",
        resampling_interpolation="nearest",
    )
    view.save_as_html(str(out_html))
    _inject_html_legend(out_html, roi_nodes, colors)


def _inject_html_legend(
    html_path: Path,
    roi_nodes: pd.DataFrame,
    colors: list[tuple[float, float, float, float]],
) -> None:
    rows = []
    for row in roi_nodes.itertuples(index=False):
        node_id = int(row.node_id)
        swatch = to_hex(colors[node_id - 1])
        label = html.escape(f"{node_id}. {row.node_name} ({int(row.n_selected_voxels):,})")
        rows.append(
            "<div class=\"roiLegendRow\">"
            f"<span class=\"roiSwatch\" style=\"background:{swatch}\"></span>"
            f"<span>{label}</span>"
            "</div>"
        )

    style = """
    <style>
      #roiLegend {
        position: fixed;
        top: 12px;
        right: 12px;
        z-index: 9999;
        max-height: calc(100vh - 24px);
        width: 340px;
        overflow-y: auto;
        box-sizing: border-box;
        padding: 10px 12px;
        border: 1px solid #bdbdbd;
        background: rgba(255, 255, 255, 0.94);
        color: #111;
        font-family: Arial, sans-serif;
        font-size: 12px;
        line-height: 1.25;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.18);
      }
      #roiLegend strong {
        display: block;
        margin-bottom: 6px;
        font-size: 13px;
      }
      .roiLegendRow {
        display: flex;
        align-items: center;
        gap: 6px;
        margin: 3px 0;
      }
      .roiSwatch {
        display: inline-block;
        flex: 0 0 12px;
        width: 12px;
        height: 12px;
        border: 1px solid rgba(0, 0, 0, 0.35);
      }
    </style>
    """
    legend = (
        "<div id=\"roiLegend\">"
        "<strong>ROI node colors</strong>"
        "<div>Each color marks selected voxels from one hemisphere-split ROI.</div>"
        + "".join(rows)
        + "</div>"
    )

    text = html_path.read_text(encoding="utf-8")
    if "</head>" in text:
        text = text.replace("</head>", f"{style}\n  </head>", 1)
    else:
        text = style + text
    if "<body>" in text:
        text = text.replace("<body>", f"<body>\n    {legend}", 1)
    else:
        text = legend + text
    html_path.write_text(text, encoding="utf-8")


def _plot_projection_panel(
    ax: plt.Axes,
    coords_mm: np.ndarray,
    assignment: np.ndarray,
    roi_nodes: pd.DataFrame,
    colors: list[tuple[float, float, float, float]],
    dims: tuple[int, int],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    for row in roi_nodes.itertuples(index=False):
        node_id = int(row.node_id)
        idx = assignment == node_id
        if not np.any(idx):
            continue
        ax.scatter(
            coords_mm[idx, dims[0]],
            coords_mm[idx, dims[1]],
            s=1.0,
            c=[colors[node_id - 1]],
            alpha=0.56,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            [float(getattr(row, ["x_mm", "y_mm", "z_mm"][dims[0]]))],
            [float(getattr(row, ["x_mm", "y_mm", "z_mm"][dims[1]]))],
            s=26,
            c="black",
            marker="x",
            linewidths=0.8,
            alpha=0.85,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#d9d9d9", linewidth=0.4, alpha=0.75)
    ax.tick_params(labelsize=8)


def _add_legend_panel(
    ax: plt.Axes,
    roi_nodes: pd.DataFrame,
    colors: list[tuple[float, float, float, float]],
) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.985, "ROI node colors", fontsize=12, fontweight="bold", va="top")
    ax.text(0.0, 0.955, "Numbers match roi_nodes.csv node_id", fontsize=8, va="top")

    n = len(roi_nodes)
    n_cols = 2
    rows_per_col = int(np.ceil(n / n_cols))
    y_step = 0.89 / rows_per_col
    for idx, row in enumerate(roi_nodes.itertuples(index=False)):
        col = idx // rows_per_col
        row_in_col = idx % rows_per_col
        x0 = 0.02 + col * 0.49
        y = 0.91 - row_in_col * y_step
        node_id = int(row.node_id)
        ax.add_patch(
            Rectangle(
                (x0, y - 0.012),
                0.022,
                0.022,
                color=colors[node_id - 1],
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        label = f"{node_id}. {row.node_name} ({int(row.n_selected_voxels):,})"
        ax.text(x0 + 0.03, y, label, fontsize=6.4, va="center", transform=ax.transAxes)


def _save_projection_figure(
    out_png: Path,
    coords_mm: np.ndarray,
    assignment: np.ndarray,
    roi_nodes: pd.DataFrame,
    colors: list[tuple[float, float, float, float]],
    target_figure: Path,
) -> None:
    fig = plt.figure(figsize=(17.0, 10.0))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 1.18],
        height_ratios=[1.0, 1.0],
        left=0.05,
        right=0.985,
        top=0.90,
        bottom=0.07,
        wspace=0.22,
        hspace=0.28,
    )

    ax_axial = fig.add_subplot(gs[0, 0])
    ax_coronal = fig.add_subplot(gs[0, 1])
    ax_sagittal = fig.add_subplot(gs[1, 0])
    ax_counts = fig.add_subplot(gs[1, 1])
    ax_legend = fig.add_subplot(gs[:, 2])

    _plot_projection_panel(
        ax_axial,
        coords_mm,
        assignment,
        roi_nodes,
        colors,
        dims=(0, 1),
        xlabel="MNI x (mm)",
        ylabel="MNI y (mm)",
        title="Axial projection",
    )
    _plot_projection_panel(
        ax_coronal,
        coords_mm,
        assignment,
        roi_nodes,
        colors,
        dims=(0, 2),
        xlabel="MNI x (mm)",
        ylabel="MNI z (mm)",
        title="Coronal projection",
    )
    _plot_projection_panel(
        ax_sagittal,
        coords_mm,
        assignment,
        roi_nodes,
        colors,
        dims=(1, 2),
        xlabel="MNI y (mm)",
        ylabel="MNI z (mm)",
        title="Sagittal projection",
    )

    ordered = roi_nodes.sort_values("n_selected_voxels", ascending=True)
    bar_colors = [colors[int(node_id) - 1] for node_id in ordered["node_id"]]
    y = np.arange(len(ordered))
    ax_counts.barh(y, ordered["n_selected_voxels"], color=bar_colors)
    ax_counts.set_yticks(y)
    ax_counts.set_yticklabels([str(v) for v in ordered["node_id"]], fontsize=7)
    ax_counts.set_xlabel("Selected voxels")
    ax_counts.set_ylabel("Node ID")
    ax_counts.set_title("ROI node voxel counts", fontsize=11)
    ax_counts.grid(axis="x", color="#d9d9d9", linewidth=0.4, alpha=0.75)
    ax_counts.tick_params(axis="x", labelsize=8)

    _add_legend_panel(ax_legend, roi_nodes, colors)

    n_voxels = int(np.count_nonzero(assignment > 0))
    fig.suptitle(
        "Selected voxels and ROI nodes used by "
        f"{target_figure.parent.name}/{target_figure.name}\n"
        f"{len(roi_nodes)} ROI nodes, {n_voxels:,} selected voxels. "
        "Each color is one hemisphere-split ROI node; black x marks the node centroid.",
        fontsize=13,
        y=0.975,
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _save_color_lookup(
    out_path: Path,
    roi_nodes: pd.DataFrame,
    colors: list[tuple[float, float, float, float]],
) -> None:
    out = roi_nodes.copy()
    out["color_hex"] = [to_hex(colors[int(node_id) - 1]) for node_id in out["node_id"]]
    out.to_csv(out_path, index=False)


def _output_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def _prefix_path(prefix: Path, extension: str) -> Path:
    return prefix.parent / f"{prefix.name}{extension}"


def main() -> None:
    args = _parse_args()
    target_figure = (
        args.weight_img.expanduser().resolve()
        if args.weight_img is not None
        else args.target_figure.expanduser().resolve()
    )
    network_dir = target_figure.parent if args.weight_img is not None else target_figure.parent.parent

    roi_img_candidates = [args.roi_img.expanduser().resolve()] if args.roi_img else _candidate_roi_imgs(network_dir)
    roi_img_paths = _existing(roi_img_candidates)
    if not roi_img_paths:
        raise FileNotFoundError(
            "Could not find fitted ROI atlas. Tried: " + "; ".join(str(p) for p in roi_img_candidates)
        )
    roi_img_path = roi_img_paths[0]
    roi_img = nib.load(str(roi_img_path))
    roi_data = roi_img.get_fdata().astype(np.int32)

    roi_nodes_path: Path | None = None
    voxel_path: Path
    if args.weight_img is not None:
        roi_summary_paths = (
            [args.roi_summary.expanduser().resolve()] if args.roi_summary else _existing(_candidate_roi_summaries(network_dir))
        )
        roi_summary_path = roi_summary_paths[0] if roi_summary_paths else None
        roi_nodes, selected_ijk, assignment, coords_mm = _roi_nodes_from_weight_img(
            weight_img_path=target_figure,
            roi_data=roi_data,
            roi_affine=roi_img.affine,
            roi_names=_roi_name_lookup(roi_summary_path),
        )
        voxel_path = target_figure
    else:
        roi_nodes_path = (
            args.roi_nodes.expanduser().resolve()
            if args.roi_nodes is not None
            else (network_dir / "roi_nodes.csv").resolve()
        )
        if not roi_nodes_path.exists():
            raise FileNotFoundError(f"Could not find roi_nodes.csv: {roi_nodes_path}")

        roi_nodes = pd.read_csv(roi_nodes_path)
        required_cols = {
            "node_id",
            "base_roi_id",
            "hemisphere",
            "node_name",
            "n_selected_voxels",
            "x_mm",
            "y_mm",
            "z_mm",
        }
        missing = sorted(required_cols.difference(roi_nodes.columns))
        if missing:
            raise ValueError(f"{roi_nodes_path} is missing columns: {missing}")
        roi_nodes = roi_nodes.sort_values("node_id").reset_index(drop=True)

        voxel_path, selected_ijk, assignment, coords_mm = _choose_voxel_indices(
            explicit_path=args.voxel_indices_path.expanduser().resolve() if args.voxel_indices_path else None,
            candidates=_candidate_voxel_indices(network_dir),
            roi_nodes=roi_nodes,
            roi_data=roi_data,
            affine=roi_img.affine,
            midline_band_mm=float(args.midline_band_mm),
        )

    anat_candidates = [args.anat_img.expanduser().resolve()] if args.anat_img else _candidate_anat_imgs(network_dir)
    anat_paths = _existing(anat_candidates)
    if not anat_paths:
        raise FileNotFoundError(
            "Could not find an anatomy background image. Tried: " + "; ".join(str(p) for p in anat_candidates)
        )
    anat_img_path = anat_paths[0]

    out_prefix = (
        args.out_prefix.expanduser().resolve()
        if args.out_prefix is not None
        else target_figure.with_name(f"{_output_stem(target_figure)}_roi_voxels")
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    colors = _palette(len(roi_nodes))
    out_png = _prefix_path(out_prefix, ".png")
    out_html = _prefix_path(out_prefix, ".html")
    _save_projection_figure(
        out_png=out_png,
        coords_mm=coords_mm,
        assignment=assignment,
        roi_nodes=roi_nodes,
        colors=colors,
        target_figure=target_figure,
    )
    label_img = _save_label_image(
        out_path=out_prefix.with_name(f"{out_prefix.name}_node_labels.nii.gz"),
        roi_img=roi_img,
        selected_ijk=selected_ijk,
        assignment=assignment,
    )
    _save_html_overlay(
        out_html=out_html,
        label_img=label_img,
        anat_img_path=anat_img_path,
        colors=colors,
        roi_nodes=roi_nodes,
        target_figure=target_figure,
    )
    _save_color_lookup(out_prefix.with_name(f"{out_prefix.name}_color_lookup.csv"), roi_nodes, colors)

    print(f"Target figure: {target_figure}")
    print(f"ROI nodes: {roi_nodes_path if roi_nodes_path is not None else 'derived from nonzero weight voxels'}")
    print(f"ROI atlas: {roi_img_path}")
    print(f"Selected voxels: {voxel_path}")
    print(f"Anatomy background: {anat_img_path}")
    print(f"Saved projection figure: {out_png}")
    print(f"Saved projection PDF: {out_png.with_suffix('.pdf')}")
    print(f"Saved anatomy HTML overlay: {out_html}")
    print(f"Saved node-label NIfTI: {out_prefix.with_name(f'{out_prefix.name}_node_labels.nii.gz')}")
    print(f"Saved color lookup: {out_prefix.with_name(f'{out_prefix.name}_color_lookup.csv')}")


if __name__ == "__main__":
    main()
