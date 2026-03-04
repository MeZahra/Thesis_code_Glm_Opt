from __future__ import annotations

from . import (
    instantaneous_phase_sync,
    linear_correlation_network,
    linear_granger,
    mutual_information,
    mutual_information_ksg,
    nonlinear_granger,
    partial_correlation,
    wavelet_transform_coherence,
)

METRIC_ORDER = [
    wavelet_transform_coherence.METRIC_NAME,
    partial_correlation.METRIC_NAME,
    mutual_information.METRIC_NAME,
    mutual_information_ksg.METRIC_NAME,
    linear_correlation_network.METRIC_NAME,
    linear_granger.METRIC_NAME,
    nonlinear_granger.METRIC_NAME,
    instantaneous_phase_sync.METRIC_NAME,
]

METRIC_REGISTRY = {
    wavelet_transform_coherence.METRIC_NAME: wavelet_transform_coherence.compute_metric,
    partial_correlation.METRIC_NAME: partial_correlation.compute_metric,
    mutual_information.METRIC_NAME: mutual_information.compute_metric,
    mutual_information_ksg.METRIC_NAME: mutual_information_ksg.compute_metric,
    linear_correlation_network.METRIC_NAME: linear_correlation_network.compute_metric,
    "graph_correlation_network": linear_correlation_network.compute_metric,
    linear_granger.METRIC_NAME: linear_granger.compute_metric,
    nonlinear_granger.METRIC_NAME: nonlinear_granger.compute_metric,
    instantaneous_phase_sync.METRIC_NAME: instantaneous_phase_sync.compute_metric,
    "instantaneous_phase_sync_plv": instantaneous_phase_sync.compute_metric,
}


def normalize_metric_list(metric_arg: str | list[str] | None) -> list[str]:
    if metric_arg is None:
        return list(METRIC_ORDER)

    if isinstance(metric_arg, (list, tuple)):
        requested = [str(v).strip() for v in metric_arg if str(v).strip()]
    else:
        text = str(metric_arg).strip()
        if not text or text.lower() == "all":
            return list(METRIC_ORDER)
        requested = [item.strip() for item in text.split(",") if item.strip()]

    unknown = [name for name in requested if name not in METRIC_REGISTRY]
    if unknown:
        valid = ", ".join(METRIC_ORDER)
        raise ValueError(f"Unknown metrics: {unknown}. Valid metrics: {valid}")

    deduped = []
    seen = set()
    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped
