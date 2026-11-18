"""Transform 注册表和 pipeline Factory。"""

from __future__ import annotations

from collections.abc import Mapping

from registry import TypedRegistry
from .pipeline import (
    CrossSectionRankTransform,
    ImputeTransform,
    SanitizeTransform,
    StandardizeTransform,
    Transform,
    TransformPipeline,
)


TRANSFORM_REGISTRY = TypedRegistry[Transform]("transform")
TRANSFORM_REGISTRY.Register("sanitize", SanitizeTransform)
TRANSFORM_REGISTRY.Register("cross_section_rank", CrossSectionRankTransform)
TRANSFORM_REGISTRY.Register("impute", ImputeTransform)
TRANSFORM_REGISTRY.Register("standardize", StandardizeTransform)

_DATASET_TRANSFORMS = {"sanitize", "cross_section_rank"}
_WINDOW_TRANSFORMS = {"impute", "standardize"}


def BuildDatasetPipeline(config: Mapping[str, object] | None) -> TransformPipeline:
    """创建一次作用于根 Dataset 的 pipeline。"""

    config = config or {}
    return _build_pipeline(list(config.get("dataset", []) or []), _DATASET_TRANSFORMS)


def BuildWindowPipeline(config: Mapping[str, object] | None) -> TransformPipeline:
    """创建每个窗口独立拟合的 pipeline。"""

    config = config or {}
    return _build_pipeline(list(config.get("window", []) or []), _WINDOW_TRANSFORMS)


def _build_pipeline(items: list[object], allowed: set[str]) -> TransformPipeline:
    """从一组类型化配置创建 pipeline。"""

    transforms: list[Transform] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("Transform configuration must be a mapping")
        kind = str(item.get("kind", ""))
        params = dict(item.get("params", {}) or {})
        if kind not in allowed:
            raise ValueError(f"Transform is not valid for this pipeline: {kind}")
        transforms.append(TRANSFORM_REGISTRY.Create(kind, **params))
    return TransformPipeline(transforms)


__all__ = ["TRANSFORM_REGISTRY", "BuildDatasetPipeline", "BuildWindowPipeline"]
