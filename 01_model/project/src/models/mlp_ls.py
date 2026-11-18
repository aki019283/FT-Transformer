"""A minimal equal-weight Long/Short window ensemble."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from data import PanelDataset
from loss import ComputeGroupedCorrelation, ComputeLoss, ValidateLossConfig
from .mlp import MlpModel


class MlpLsModel:
    """Blend two identical MLPs trained on long and short history windows."""

    KIND = "mlp-ls"

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        self.params = dict(params or {})
        self.model_params = dict(self.params.get("model_params", {}) or {})
        self.short_train_periods = int(self.params.get("short_train_periods", 18))
        if self.short_train_periods <= 0:
            raise ValueError("short_train_periods must be positive")
        self.long_model = MlpModel(self.model_params)
        self.short_model = MlpModel(self.model_params)
        self.training_summary: dict = {}

    def Fit(self, train_data: PanelDataset, valid_data: PanelDataset, options: dict) -> dict:
        """Fit equal MLPs on full history and the latest short history."""

        if train_data.PeriodCount() <= self.short_train_periods:
            raise ValueError("short_train_periods must be smaller than the training window")
        if bool(options.get("reset_models", False)):
            self.long_model = MlpModel(self.model_params)
            self.short_model = MlpModel(self.model_params)
        short_train = train_data.TailPeriods(self.short_train_periods)
        long_summary = self.long_model.Fit(train_data, valid_data, options)
        short_summary = self.short_model.Fit(short_train, valid_data, options)
        ensemble_validation = self._validation(valid_data, options)
        self.training_summary = {
            "epochs": max(
                int(long_summary.get("epochs", 0)),
                int(short_summary.get("epochs", 0)),
            ),
            "short_train_periods": self.short_train_periods,
            "blend": "equal",
            "long": long_summary,
            "short": short_summary,
            **ensemble_validation,
        }
        return self.training_summary

    def SetModelParams(self, model_params: Mapping[str, object]) -> None:
        """Set the shared parameters used by the next full refit."""

        self.model_params = dict(model_params)
        self.params["model_params"] = self.model_params

    def Predict(self, data: PanelDataset) -> pd.DataFrame:
        """Return the arithmetic mean of Long and Short predictions."""

        long_prediction = self.long_model.Predict(data)
        short_prediction = self.short_model.Predict(data)
        result = long_prediction.copy()
        result["prediction"] = (
            0.5 * long_prediction["prediction"].to_numpy()
            + 0.5 * short_prediction["prediction"].to_numpy()
        ).astype(np.float32)
        return result

    def Save(self, directory: str | Path) -> None:
        """Save both MLP bundles and the minimal ensemble state."""

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.long_model.Save(path / "long")
        self.short_model.Save(path / "short")
        (path / "ensemble.json").write_text(
            json.dumps(
                {
                    "kind": self.KIND,
                    "params": self.params,
                    "training_summary": self.training_summary,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def Load(self, directory: str | Path) -> None:
        """Restore both MLP bundles."""

        path = Path(directory)
        state = json.loads((path / "ensemble.json").read_text(encoding="utf-8"))
        self.__init__(state["params"])
        self.long_model.Load(path / "long")
        self.short_model.Load(path / "short")
        self.training_summary = dict(state.get("training_summary", {}))

    def _validation(
        self,
        valid_data: PanelDataset,
        options: Mapping[str, object],
    ) -> dict[str, float]:
        prediction = self.Predict(valid_data)["prediction"].to_numpy(dtype=np.float32)
        target = valid_data.TargetArray().astype(np.float32)
        groups = valid_data.PeriodCodes().astype(np.int64)
        finite = np.isfinite(prediction) & np.isfinite(target)
        prediction_tensor = torch.from_numpy(prediction[finite])
        target_tensor = torch.from_numpy(target[finite])
        group_tensor = torch.from_numpy(groups[finite])
        loss_kind = ValidateLossConfig(options.get("loss"))
        return {
            "best_valid_loss": float(
                ComputeLoss(
                    loss_kind,
                    prediction_tensor,
                    target_tensor,
                    group_tensor,
                )
            ),
            "best_valid_mse": float(
                ComputeLoss("mse", prediction_tensor, target_tensor, group_tensor)
            ),
            "best_valid_target_ic": float(
                ComputeGroupedCorrelation(
                    prediction_tensor,
                    target_tensor,
                    group_tensor,
                )
            ),
        }


__all__ = ["MlpLsModel"]
