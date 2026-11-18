"""run 内 tuning 触发时间表。"""

from __future__ import annotations

from typing import Mapping

import pandas as pd


class TuningSchedule:
    """判断窗口是否创建新的 TuningVersion。"""

    def __init__(self, config: Mapping[str, object] | None) -> None:
        """读取 tuning 模式和触发周期。"""

        config = dict(config or {})
        self.mode = str(config.get("mode", "none"))
        if self.mode not in {"none", "once", "periodic", "explicit"}:
            raise ValueError(f"Unknown tuning mode: {self.mode}")
        self.every_periods = int(config.get("every_periods", 1))
        self.offset_periods = int(config.get("offset_periods", 0))
        self.explicit_periods = {
            str(value) for value in list(config.get("explicit_periods", []) or [])
        }
        if self.mode == "periodic" and self.every_periods <= 0:
            raise ValueError("every_periods must be positive")

    def ShouldTune(self, window_index: int, fit_time: pd.Timestamp) -> bool:
        """返回当前窗口是否触发 tuning。"""

        if self.mode == "none":
            return False
        if self.mode == "once":
            return window_index == 0
        if self.mode == "periodic":
            if window_index < self.offset_periods:
                return False
            return (window_index - self.offset_periods) % self.every_periods == 0
        return fit_time.isoformat() in self.explicit_periods or fit_time.strftime("%Y-%m-%d") in self.explicit_periods


__all__ = ["TuningSchedule"]
