"""窗口间模型继承和训练预算策略。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class TrainingDecision:
    """一个窗口的初始化来源和训练预算。"""

    parent_bundle: Path | None
    epochs: int
    train: bool
    full_refit: bool


class TrainingPolicy:
    """根据窗口位置生成 TrainingDecision。"""

    def __init__(self, config: Mapping[str, object]) -> None:
        """读取训练预算和继承策略。"""

        self.kind = str(config.get("policy", "periodic_refit"))
        if self.kind not in {"cold_start", "warm_start", "periodic_refit", "fixed"}:
            raise ValueError(f"Unknown training policy: {self.kind}")
        self.full_epochs = int(config.get("full_epochs", 20))
        self.update_epochs = int(config.get("update_epochs", 5))
        self.update_learning_rate_scale = float(
            config.get("update_learning_rate_scale", 1.0)
        )
        self.refit_every_periods = int(config.get("refit_every_periods", 12))
        if self.full_epochs <= 0 or self.update_epochs <= 0:
            raise ValueError("Training epoch budgets must be positive")
        if self.update_learning_rate_scale <= 0:
            raise ValueError("update_learning_rate_scale must be positive")
        if self.refit_every_periods <= 0:
            raise ValueError("refit_every_periods must be positive")

    def Decide(self, window_index: int, previous_bundle: Path | None) -> TrainingDecision:
        """返回当前窗口的训练决策。"""

        if self.kind == "cold_start":
            return TrainingDecision(None, self.full_epochs, True, True)
        if self.kind == "fixed":
            if previous_bundle is None:
                raise ValueError("fixed policy requires an existing parent bundle")
            return TrainingDecision(previous_bundle, 0, False, False)
        if previous_bundle is None:
            return TrainingDecision(None, self.full_epochs, True, True)
        if self.kind == "periodic_refit" and window_index % self.refit_every_periods == 0:
            return TrainingDecision(None, self.full_epochs, True, True)
        return TrainingDecision(previous_bundle, self.update_epochs, True, False)


__all__ = ["TrainingDecision", "TrainingPolicy"]
