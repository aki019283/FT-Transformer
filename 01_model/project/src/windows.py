"""滚动窗口边界和统一 Window 对象。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import pandas as pd

from data import PanelDataset


@dataclass
class Window:
    """一个可训练、可提交的滚动窗口。"""

    window_id: str
    fit_time: pd.Timestamp
    train: PanelDataset
    valid: PanelDataset
    test: PanelDataset
    train_periods: tuple[pd.Timestamp, ...]
    valid_periods: tuple[pd.Timestamp, ...]
    test_periods: tuple[pd.Timestamp, ...]
    tuning_version: str | None = None

    def WithDatasets(
        self,
        train: PanelDataset,
        valid: PanelDataset,
        test: PanelDataset,
    ) -> "Window":
        """返回替换特征表示后的窗口。"""

        return replace(self, train=train, valid=valid, test=test)

    def ToRecord(self) -> dict:
        """返回可写入窗口 manifest 的基础记录。"""

        return {
            "window_id": self.window_id,
            "fit_time": self.fit_time.isoformat(),
            "train_start": self.train_periods[0].isoformat(),
            "train_end": self.train_periods[-1].isoformat(),
            "valid_start": self.valid_periods[0].isoformat(),
            "valid_end": self.valid_periods[-1].isoformat(),
            "test_start": self.test_periods[0].isoformat(),
            "test_end": self.test_periods[-1].isoformat(),
            "train_rows": self.train.RowCount(),
            "valid_rows": self.valid.RowCount(),
            "test_rows": self.test.RowCount(),
            "tuning_version": self.tuning_version,
        }


class WindowBuilder:
    """按照观测周期生成严格时间顺序的滚动窗口。"""

    def __init__(self, config: Mapping[str, object]) -> None:
        """读取并校验窗口边界配置。"""

        self.kind = str(config.get("kind", "sliding"))
        if self.kind not in {"sliding", "expanding"}:
            raise ValueError(f"Unsupported window kind: {self.kind}")
        self.train_periods = int(config.get("train_periods", 60))
        self.valid_periods = int(config.get("valid_periods", 12))
        self.test_periods = int(config.get("test_periods", 1))
        self.step_periods = int(config.get("step_periods", 1))
        self.purge_periods = int(config.get("purge_periods", 0))
        self.embargo_periods = int(config.get("embargo_periods", 0))
        self.test_start = config.get("test_start")
        self.test_end = config.get("test_end")
        values = {
            "train_periods": self.train_periods,
            "valid_periods": self.valid_periods,
            "test_periods": self.test_periods,
            "step_periods": self.step_periods,
            "purge_periods": self.purge_periods,
            "embargo_periods": self.embargo_periods,
        }
        if any(value < 0 for value in values.values()):
            raise ValueError("Window lengths and gaps must be non-negative")
        if self.train_periods == 0 or self.valid_periods == 0 or self.test_periods == 0:
            raise ValueError("train_periods, valid_periods and test_periods must be positive")
        if self.step_periods == 0:
            raise ValueError("step_periods must be positive")

    def Build(self, dataset: PanelDataset) -> list[Window]:
        """根据 Dataset 周期生成窗口列表。"""

        periods = list(dataset.Periods())
        windows: list[Window] = []
        used_test_periods: set[pd.Timestamp] = set()
        start_index = 0
        window_number = 0
        while True:
            train_begin = 0 if self.kind == "expanding" else start_index
            train_end = start_index + self.train_periods
            valid_begin = train_end + self.purge_periods
            valid_end = valid_begin + self.valid_periods
            test_begin = valid_end + self.purge_periods + self.embargo_periods
            test_end = test_begin + self.test_periods
            if test_end > len(periods):
                break
            train_period_values = tuple(periods[train_begin:train_end])
            valid_period_values = tuple(periods[valid_begin:valid_end])
            test_period_values = tuple(periods[test_begin:test_end])
            if self._outside_requested_range(test_period_values):
                start_index += self.step_periods
                continue
            overlap = used_test_periods.intersection(test_period_values)
            if overlap:
                raise ValueError("Rolling test windows overlap")
            train = dataset.SlicePeriods(train_period_values[0], train_period_values[-1])
            valid = dataset.SlicePeriods(valid_period_values[0], valid_period_values[-1])
            test = dataset.SlicePeriods(test_period_values[0], test_period_values[-1])
            if train.IsEmpty() or valid.IsEmpty() or test.IsEmpty():
                start_index += self.step_periods
                continue
            windows.append(
                Window(
                    window_id=f"w{window_number:06d}",
                    fit_time=periods[test_begin - 1],
                    train=train,
                    valid=valid,
                    test=test,
                    train_periods=train_period_values,
                    valid_periods=valid_period_values,
                    test_periods=test_period_values,
                )
            )
            used_test_periods.update(test_period_values)
            window_number += 1
            start_index += self.step_periods
        return windows

    def _outside_requested_range(self, periods: tuple[pd.Timestamp, ...]) -> bool:
        """检查 test 周期是否完整位于请求范围。"""

        if not periods:
            return True
        if self.test_start is not None and periods[0] < pd.Timestamp(self.test_start):
            return True
        if self.test_end is not None and periods[-1] > pd.Timestamp(self.test_end):
            return True
        return False


__all__ = ["Window", "WindowBuilder"]
