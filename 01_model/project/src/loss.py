"""Training losses shared by rolling models."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch


LossFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
_EPSILON = 1e-8


def _group_reduction_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    groups: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate vectors and return compact group codes plus group counts."""

    if prediction.ndim != 1 or target.ndim != 1 or groups.ndim != 1:
        raise ValueError("prediction, target, and groups must be one-dimensional")
    if prediction.numel() != target.numel() or prediction.numel() != groups.numel():
        raise ValueError("prediction, target, and groups must have equal lengths")
    if prediction.numel() == 0:
        raise ValueError("loss inputs cannot be empty")
    if prediction.device != target.device or prediction.device != groups.device:
        raise ValueError("prediction, target, and groups must use the same device")
    _, inverse, integer_counts = torch.unique(
        groups, sorted=False, return_inverse=True, return_counts=True
    )
    counts = integer_counts.to(dtype=prediction.dtype)
    return prediction, target, inverse, counts


def _group_sums(
    values: torch.Tensor,
    inverse: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    result = torch.zeros_like(counts, dtype=values.dtype)
    result.index_add_(0, inverse, values)
    return result


def _period_equal_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    groups: torch.Tensor,
) -> torch.Tensor:
    prediction, target, inverse, counts = _group_reduction_inputs(
        prediction, target, groups
    )
    squared_error = (prediction - target).square()
    return (_group_sums(squared_error, inverse, counts) / counts).mean()


def _grouped_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    inverse: torch.Tensor,
    counts: torch.Tensor,
    epsilon: float = _EPSILON,
) -> torch.Tensor:
    prediction_sum = _group_sums(prediction, inverse, counts)
    target_sum = _group_sums(target, inverse, counts)
    prediction_centered = prediction - prediction_sum[inverse] / counts[inverse]
    target_centered = target - target_sum[inverse] / counts[inverse]
    covariance = _group_sums(prediction_centered * target_centered, inverse, counts)
    prediction_ss = _group_sums(prediction_centered.square(), inverse, counts)
    target_ss = _group_sums(target_centered.square(), inverse, counts)
    denominator = (prediction_ss * target_ss).clamp_min(epsilon * epsilon).sqrt()
    correlations = covariance / denominator
    valid_groups = counts >= 2
    valid_count = valid_groups.to(dtype=prediction.dtype).sum()
    correlation_sum = (correlations * valid_groups).sum()
    return correlation_sum / valid_count.clamp_min(1.0)


def ComputeGroupedCorrelation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    groups: torch.Tensor,
    epsilon: float = _EPSILON,
) -> torch.Tensor:
    """Return the equally weighted mean Pearson correlation across valid groups."""

    prediction, target, inverse, counts = _group_reduction_inputs(
        prediction, target, groups
    )
    return _grouped_correlation(prediction, target, inverse, counts, epsilon)


def _cross_section_ic(
    prediction: torch.Tensor,
    target: torch.Tensor,
    groups: torch.Tensor,
) -> torch.Tensor:
    prediction, target, inverse, counts = _group_reduction_inputs(
        prediction, target, groups
    )
    ic_loss = 1.0 - _grouped_correlation(prediction, target, inverse, counts)
    fallback = (_group_sums((prediction - target).square(), inverse, counts) / counts).mean()
    return torch.where(
        torch.any(counts >= 2),
        ic_loss,
        fallback,
    )


_LOSS_REGISTRY: dict[str, LossFunction] = {
    "mse": _period_equal_mse,
    "cross_section_ic": _cross_section_ic,
}


def LossKinds() -> tuple[str, ...]:
    """Return all supported loss identifiers."""

    return tuple(sorted(_LOSS_REGISTRY))


def ValidateLossConfig(config: Mapping[str, object] | None) -> str:
    """Validate a training.loss mapping and return its kind."""

    if not isinstance(config, Mapping):
        raise ValueError("training.loss must be a mapping")
    kind = str(config.get("kind", ""))
    if kind not in _LOSS_REGISTRY:
        available = ", ".join(LossKinds())
        raise ValueError(f"Unsupported training.loss.kind: {kind}. Available: {available}")
    unknown = set(config).difference({"kind"})
    if unknown:
        raise ValueError(f"Unsupported training.loss fields: {sorted(unknown)}")
    return kind


def ComputeLoss(
    kind: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    groups: torch.Tensor,
) -> torch.Tensor:
    """Compute a registered period-equal training loss."""

    try:
        function = _LOSS_REGISTRY[kind]
    except KeyError as error:
        available = ", ".join(LossKinds())
        raise ValueError(f"Unsupported loss: {kind}. Available: {available}") from error
    return function(prediction, target, groups)


__all__ = ["ComputeGroupedCorrelation", "ComputeLoss", "LossKinds", "ValidateLossConfig"]
