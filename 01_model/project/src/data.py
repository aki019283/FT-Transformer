"""滚动训练使用的统一面板数据门面。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _PanelStorage:
    """保存一份已经排序并校验过的根数据。"""

    frame: pd.DataFrame


class PanelDataset:
    """提供不可变语义的面板数据访问。

    窗口切片共享根表，学习型变换只替换当前对象的特征块。所有公开方法都返回
    新对象或副本，因此训练组件不会通过 Dataset 修改其他窗口的数据。
    """

    def __init__(
        self,
        storage: _PanelStorage,
        time_col: str,
        asset_col: str,
        target_col: str,
        feature_cols: Sequence[str],
        rows: np.ndarray | None = None,
        feature_override: np.ndarray | None = None,
        target_override: np.ndarray | None = None,
    ) -> None:
        """绑定根 Storage、行选择和可选特征覆盖。"""

        self._storage = storage
        self.time_col = time_col
        self.asset_col = asset_col
        self.target_col = target_col
        self._feature_cols = tuple(feature_cols)
        self._rows = (
            np.arange(len(storage.frame), dtype=np.int64)
            if rows is None
            else np.asarray(rows, dtype=np.int64)
        )
        if feature_override is not None:
            values = np.asarray(feature_override, dtype=np.float32)
            if values.ndim != 2 or values.shape != (len(self._rows), len(self._feature_cols)):
                raise ValueError("feature_override shape does not match dataset")
            self._feature_override = values.copy()
        else:
            self._feature_override = None
        if target_override is not None:
            values = np.asarray(target_override, dtype=np.float32)
            if values.shape != (len(self._rows),):
                raise ValueError("target_override shape does not match dataset")
            self._target_override = values.copy()
        else:
            self._target_override = None

    @classmethod
    def Load(
        cls,
        path: str | Path,
        time_col: str,
        asset_col: str,
        target_col: str,
        feature_cols: Sequence[str] | None = None,
    ) -> "PanelDataset":
        """从 Parquet 加载并校验面板数据。"""

        frame = pd.read_parquet(path)
        return cls.FromFrame(frame, time_col, asset_col, target_col, feature_cols)

    @classmethod
    def FromFrame(
        cls,
        frame: pd.DataFrame,
        time_col: str,
        asset_col: str,
        target_col: str,
        feature_cols: Sequence[str] | None = None,
    ) -> "PanelDataset":
        """从 DataFrame 创建根 Dataset。"""

        required = {time_col, asset_col, target_col}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if feature_cols is None:
            feature_cols = [
                column for column in frame.columns
                if column not in required
            ]
        feature_cols = list(feature_cols)
        missing_features = sorted(set(feature_cols).difference(frame.columns))
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        if not feature_cols:
            raise ValueError("At least one feature column is required")

        normalized = frame[[time_col, asset_col, target_col, *feature_cols]].copy()
        normalized[time_col] = pd.to_datetime(normalized[time_col], errors="raise")
        normalized[asset_col] = normalized[asset_col].astype(str)
        normalized = normalized.sort_values([time_col, asset_col], kind="mergesort")
        normalized = normalized.reset_index(drop=True)
        if normalized.duplicated([time_col, asset_col]).any():
            raise ValueError("Dataset contains duplicate (time, asset) keys")
        for column in feature_cols:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized[target_col] = pd.to_numeric(normalized[target_col], errors="coerce")
        storage = _PanelStorage(normalized)
        return cls(storage, time_col, asset_col, target_col, feature_cols)

    @property
    def feature_cols(self) -> tuple[str, ...]:
        """返回特征列名称。"""

        return self._feature_cols

    @property
    def schema(self) -> tuple[str, ...]:
        """返回当前 Dataset 的列 schema。"""

        return (self.time_col, self.asset_col, self.target_col, *self._feature_cols)

    def __len__(self) -> int:
        """返回行数。"""

        return len(self._rows)

    def RowCount(self) -> int:
        """返回行数。"""

        return len(self._rows)

    def PeriodCount(self) -> int:
        """返回当前对象包含的周期数。"""

        return len(self.Periods())

    def Periods(self) -> pd.DatetimeIndex:
        """返回排序后的唯一周期。"""

        values = self._storage.frame.iloc[self._rows][self.time_col]
        return pd.DatetimeIndex(values.drop_duplicates().sort_values())

    def PeriodValues(self) -> pd.DatetimeIndex:
        """返回当前对象的周期，保留与 Periods 相同的语义。"""

        return self.Periods()

    def SlicePeriods(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> "PanelDataset":
        """按包含边界的周期范围创建零拷贝切片。"""

        times = self._storage.frame.iloc[self._rows][self.time_col]
        mask = np.ones(len(self._rows), dtype=bool)
        if start is not None:
            mask &= times.to_numpy() >= pd.Timestamp(start).to_datetime64()
        if end is not None:
            mask &= times.to_numpy() <= pd.Timestamp(end).to_datetime64()
        return self._select(mask)

    def TailPeriods(self, count: int) -> "PanelDataset":
        """返回最后 count 个周期。"""

        if count < 0:
            raise ValueError("count must be non-negative")
        periods = self.Periods()
        selected = set(periods[-count:]) if count else set()
        times = self._storage.frame.iloc[self._rows][self.time_col]
        return self._select(times.isin(selected).to_numpy())

    def WithFeatures(
        self,
        features: pd.DataFrame | np.ndarray,
        feature_cols: Sequence[str] | None = None,
    ) -> "PanelDataset":
        """创建只替换特征块的新 Dataset。"""

        if isinstance(features, pd.DataFrame):
            names = list(features.columns) if feature_cols is None else list(feature_cols)
            values = features.to_numpy(dtype=np.float32, copy=True)
        else:
            values = np.asarray(features, dtype=np.float32)
            names = list(self._feature_cols) if feature_cols is None else list(feature_cols)
        if values.ndim != 2:
            raise ValueError("feature values must be two-dimensional")
        if len(names) != values.shape[1]:
            raise ValueError("feature column count does not match feature values")
        if values.shape[0] != len(self._rows):
            raise ValueError("feature values must have one row per dataset row")
        return PanelDataset(
            self._storage,
            self.time_col,
            self.asset_col,
            self.target_col,
            names,
            self._rows,
            values,
            self._target_override,
        )

    def WithTarget(self, target: pd.Series | np.ndarray) -> "PanelDataset":
        """创建只替换目标列的新 Dataset。"""

        values = np.asarray(target, dtype=np.float32)
        if values.shape != (len(self._rows),):
            raise ValueError("target values must have one value per dataset row")
        return PanelDataset(
            self._storage,
            self.time_col,
            self.asset_col,
            self.target_col,
            self._feature_cols,
            self._rows,
            self._feature_override,
            values,
        )

    def Materialize(self) -> "PanelDataset":
        """把当前范围复制成独立根 Storage。"""

        return PanelDataset.FromFrame(
            self.ToFrame(),
            self.time_col,
            self.asset_col,
            self.target_col,
            self._feature_cols,
        )

    def ToFrame(self, copy: bool = True) -> pd.DataFrame:
        """返回当前 Dataset 的表格副本。"""

        frame = self._storage.frame.iloc[self._rows].copy()
        if self._feature_override is not None:
            frame.loc[:, list(self._feature_cols)] = self._feature_override
        if self._target_override is not None:
            frame.loc[:, self.target_col] = self._target_override
        return frame.copy() if copy else frame

    def FeatureArray(self) -> np.ndarray:
        """返回 float32 特征数组副本。"""

        if self._feature_override is not None:
            return self._feature_override.copy()
        return self.ToFrame()[list(self._feature_cols)].to_numpy(dtype=np.float32, copy=True)

    def TargetArray(self) -> np.ndarray:
        """返回目标数组副本。"""

        if self._target_override is not None:
            return self._target_override.copy()
        return self.ToFrame()[self.target_col].to_numpy(dtype=np.float32, copy=True)

    def TimeArray(self) -> np.ndarray:
        """返回时间数组副本。"""

        return self.ToFrame()[self.time_col].to_numpy(copy=True)

    def AssetArray(self) -> np.ndarray:
        """返回资产数组副本。"""

        return self.ToFrame()[self.asset_col].to_numpy(copy=True)

    def PeriodCodes(self) -> np.ndarray:
        """返回按当前周期排序编码的每行周期编号。"""

        times = pd.Series(self.TimeArray())
        return pd.Categorical(times, categories=self.Periods(), ordered=True).codes

    def IsEmpty(self) -> bool:
        """返回 Dataset 是否为空。"""

        return len(self._rows) == 0

    def _select(self, mask: np.ndarray) -> "PanelDataset":
        """使用相对当前对象的布尔掩码创建切片。"""

        selected = np.asarray(mask, dtype=bool)
        if selected.shape != (len(self._rows),):
            raise ValueError("selection mask has invalid shape")
        positions = self._rows[selected]
        override = self._feature_override[selected] if self._feature_override is not None else None
        target_override = self._target_override[selected] if self._target_override is not None else None
        return PanelDataset(
            self._storage,
            self.time_col,
            self.asset_col,
            self.target_col,
            self._feature_cols,
            positions,
            override,
            target_override,
        )


__all__ = ["PanelDataset"]
