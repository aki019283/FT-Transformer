"""训练策略注册表。"""

from __future__ import annotations

from collections.abc import Mapping

from registry import TypedRegistry
from .policies import TrainingPolicy


TRAINING_POLICY_REGISTRY = TypedRegistry[TrainingPolicy]("training policy")
for _kind in ("cold_start", "warm_start", "periodic_refit", "fixed"):
    TRAINING_POLICY_REGISTRY.Register(
        _kind,
        lambda config, kind=_kind: TrainingPolicy({**dict(config), "policy": kind}),
    )


def MakeTrainingPolicy(config: Mapping[str, object]) -> TrainingPolicy:
    """从配置创建训练策略。"""

    return TRAINING_POLICY_REGISTRY.Create(str(config.get("policy", "periodic_refit")), config)


__all__ = ["TRAINING_POLICY_REGISTRY", "MakeTrainingPolicy"]
