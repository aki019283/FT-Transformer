"""模型接口定义。"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from data import PanelDataset


class Model(Protocol):
    """训练、预测和持久化模型的最小协议。"""

    def Fit(self, train_data: PanelDataset, valid_data: PanelDataset, options: dict) -> dict:
        """训练模型并返回训练摘要。"""

    def Predict(self, data: PanelDataset) -> pd.DataFrame:
        """对 Dataset 生成预测。"""

    def Save(self, directory: str) -> None:
        """保存模型。"""

    def Load(self, directory: str) -> None:
        """加载模型。"""


__all__ = ["Model"]
