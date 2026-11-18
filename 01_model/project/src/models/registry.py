"""模型注册表。"""

from __future__ import annotations

from collections.abc import Mapping

from registry import TypedRegistry
from .mlp_ls import MlpLsModel
from .mlp import MlpModel


MODEL_REGISTRY = TypedRegistry[object]("model")
MODEL_REGISTRY.Register("mlp", MlpModel)
MODEL_REGISTRY.Register("mlp-ls", MlpLsModel)


def MakeModel(kind: str, params: Mapping[str, object] | None = None) -> object:
    """按名称创建模型。"""

    return MODEL_REGISTRY.Create(kind, params or {})


__all__ = ["MODEL_REGISTRY", "MakeModel"]
