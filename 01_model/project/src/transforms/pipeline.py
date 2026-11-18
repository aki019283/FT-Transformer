"""窗口特征变换及其 pipeline。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from data import PanelDataset
from windows import Window


class Transform:
    """定义 Dataset 和 Window transform 的最小接口。"""

    KIND = "transform"

    def Fit(self, train_data: PanelDataset) -> None:
        """从训练数据学习可选状态。"""

    def Apply(self, data: PanelDataset) -> PanelDataset:
        """应用变换并返回新 Dataset。"""

        return data

    def State(self) -> dict:
        """返回可序列化状态。"""

        return {"kind": self.KIND}


class SanitizeTransform(Transform):
    """把无穷和缺失特征转换为有限值。"""

    KIND = "sanitize"

    def Apply(self, data: PanelDataset) -> PanelDataset:
        """把无穷值转换为缺失值。"""

        values = data.FeatureArray()
        values[np.isinf(values)] = np.nan
        return data.WithFeatures(values, data.feature_cols)


class CrossSectionRankTransform(Transform):
    """在每个周期内把特征和可选目标转换为百分位排名。"""

    KIND = "cross_section_rank"

    def __init__(
        self,
        output_range: Sequence[float] = (0.0, 1.0),
        rank_target: bool = False,
    ) -> None:
        """设置输出百分位范围。"""

        if len(output_range) != 2 or output_range[1] <= output_range[0]:
            raise ValueError("output_range must contain two increasing values")
        self.output_range = (float(output_range[0]), float(output_range[1]))
        self.rank_target = bool(rank_target)

    def Apply(self, data: PanelDataset) -> PanelDataset:
        """按周期计算截面百分位。"""

        frame = data.ToFrame()
        ranks = frame.groupby(data.time_col, sort=False)[list(data.feature_cols)].rank(
            pct=True,
            method="average",
        )
        low, high = self.output_range
        values = ranks.to_numpy(dtype=np.float32)
        values = low + values * (high - low)
        transformed = data.WithFeatures(values, data.feature_cols)
        if self.rank_target:
            target = frame.groupby(data.time_col, sort=False)[data.target_col].rank(
                pct=True,
                method="average",
            )
            target_values = low + target.to_numpy(dtype=np.float32) * (high - low)
            transformed = transformed.WithTarget(target_values)
        return transformed

    def State(self) -> dict:
        """返回输出范围配置。"""

        return {
            "kind": self.KIND,
            "output_range": list(self.output_range),
            "rank_target": self.rank_target,
        }


class ImputeTransform(Transform):
    """使用训练集统计量填充缺失特征。"""

    KIND = "impute"

    def __init__(self, strategy: str = "median", value: float = 0.0) -> None:
        """设置缺失填充策略。"""

        if strategy not in {"median", "mean", "constant"}:
            raise ValueError(f"Unsupported impute strategy: {strategy}")
        self.strategy = strategy
        self.value = float(value)
        self.statistics: np.ndarray | None = None

    def Fit(self, train_data: PanelDataset) -> None:
        """从训练集计算各特征填充值。"""

        values = train_data.FeatureArray().astype(np.float64)
        values[~np.isfinite(values)] = np.nan
        if self.strategy == "median":
            statistics = np.nanmedian(values, axis=0)
        elif self.strategy == "mean":
            statistics = np.nanmean(values, axis=0)
        else:
            statistics = np.full(values.shape[1], self.value)
        self.statistics = np.nan_to_num(statistics, nan=self.value, posinf=self.value, neginf=self.value)

    def Apply(self, data: PanelDataset) -> PanelDataset:
        """使用训练集统计量填充缺失值。"""

        if self.statistics is None:
            raise RuntimeError("Impute transform must be fitted before Apply")
        values = data.FeatureArray().astype(np.float32)
        invalid = ~np.isfinite(values)
        values[invalid] = np.take(self.statistics, np.nonzero(invalid)[1])
        return data.WithFeatures(values, data.feature_cols)

    def State(self) -> dict:
        """返回填充配置和统计量。"""

        return {
            "kind": self.KIND,
            "strategy": self.strategy,
            "value": self.value,
            "statistics": None if self.statistics is None else self.statistics.tolist(),
        }


class StandardizeTransform(Transform):
    """使用训练集均值和标准差标准化特征。"""

    KIND = "standardize"

    def __init__(self, epsilon: float = 1e-6) -> None:
        """设置标准差下限。"""

        self.epsilon = float(epsilon)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def Fit(self, train_data: PanelDataset) -> None:
        """从训练集计算均值和标准差。"""

        values = train_data.FeatureArray().astype(np.float64)
        values[~np.isfinite(values)] = np.nan
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        self.mean = np.nan_to_num(mean, nan=0.0)
        self.scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
        self.scale[self.scale < self.epsilon] = 1.0

    def Apply(self, data: PanelDataset) -> PanelDataset:
        """应用训练集标准化参数。"""

        if self.mean is None or self.scale is None:
            raise RuntimeError("Standardize transform must be fitted before Apply")
        values = data.FeatureArray().astype(np.float32)
        values = (values - self.mean.astype(np.float32)) / self.scale.astype(np.float32)
        values[~np.isfinite(values)] = 0.0
        return data.WithFeatures(values, data.feature_cols)

    def State(self) -> dict:
        """返回标准化配置和统计量。"""

        return {
            "kind": self.KIND,
            "epsilon": self.epsilon,
            "mean": None if self.mean is None else self.mean.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
        }


class TransformPipeline:
    """按配置顺序拟合并应用一组变换。"""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        """保存有序 Transform 列表。"""

        self.transforms = list(transforms)

    def Fit(self, train_data: PanelDataset) -> None:
        """只使用当前窗口 train Dataset 拟合 Window transforms。"""

        current = train_data
        for transform in self.transforms:
            transform.Fit(current)
            current = transform.Apply(current)

    def Apply(self, window: Window) -> Window:
        """对窗口的 train/valid/test 统一应用已拟合状态。"""

        train, valid, test = window.train, window.valid, window.test
        for transform in self.transforms:
            train = transform.Apply(train)
            valid = transform.Apply(valid)
            test = transform.Apply(test)
        return window.WithDatasets(train, valid, test)

    def ApplyDataset(self, data: PanelDataset) -> PanelDataset:
        """对单个 Dataset 顺序应用全部变换。"""

        current = data
        for transform in self.transforms:
            current = transform.Apply(current)
        return current

    def Save(self, directory: str | Path) -> None:
        """保存变换配置和拟合状态。"""

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "transforms.json").write_text(
            json.dumps([transform.State() for transform in self.transforms], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def Load(cls, directory: str | Path) -> "TransformPipeline":
        """从 bundle 恢复变换顺序和 Window 状态。"""

        states = json.loads((Path(directory) / "transforms.json").read_text(encoding="utf-8"))
        transforms: list[Transform] = []
        for state in states:
            kind = state["kind"]
            if kind == "sanitize":
                transform = SanitizeTransform()
            elif kind == "cross_section_rank":
                transform = CrossSectionRankTransform(
                    state["output_range"],
                    bool(state.get("rank_target", False)),
                )
            elif kind == "impute":
                transform = ImputeTransform(state["strategy"], state["value"])
                transform.statistics = np.asarray(state["statistics"], dtype=np.float64)
            elif kind == "standardize":
                transform = StandardizeTransform(state["epsilon"])
                transform.mean = np.asarray(state["mean"], dtype=np.float64)
                transform.scale = np.asarray(state["scale"], dtype=np.float64)
            else:
                raise ValueError(f"Unknown transform in bundle: {kind}")
            transforms.append(transform)
        return cls(transforms)


__all__ = [
    "Transform",
    "SanitizeTransform",
    "CrossSectionRankTransform",
    "ImputeTransform",
    "StandardizeTransform",
    "TransformPipeline",
]
