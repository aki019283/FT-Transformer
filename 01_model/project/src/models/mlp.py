"""Torch MLP model implementations for factor training."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from data import PanelDataset
from loss import ComputeGroupedCorrelation, ComputeLoss, ValidateLossConfig
from training.batching import BuildPeriodBatches


class _PlainMlpNetwork(nn.Module):
    """Legacy configurable MLP retained for the existing MLP-LS model."""

    def __init__(self, input_dim: int, hidden_dims: Iterable[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current, hidden_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = hidden_dim
        layers.append(nn.Linear(current, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


class _ResidualBlock(nn.Module):
    """Pre-norm residual MLP block used by the fixed baseline."""

    def __init__(self, width: int, expansion: int, dropout: float) -> None:
        super().__init__()
        expanded_width = width * expansion
        self.norm = nn.RMSNorm(width)
        self.layers = nn.Sequential(
            nn.Linear(width, expanded_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_width, width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.layers(self.norm(values))


class _ResidualMlpNetwork(nn.Module):
    """Fixed 128-wide, two-block residual MLP baseline."""

    WIDTH = 128
    BLOCKS = 2
    EXPANSION = 2

    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, self.WIDTH)
        self.blocks = nn.Sequential(
            *[
                _ResidualBlock(self.WIDTH, self.EXPANSION, dropout)
                for _ in range(self.BLOCKS)
            ]
        )
        self.output_norm = nn.RMSNorm(self.WIDTH)
        self.output = nn.Linear(self.WIDTH, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(self.input(values))
        return self.output(self.output_norm(hidden)).squeeze(-1)


class MlpModel:
    """Single-output Torch MLP for cross-sectional rolling prediction."""

    KIND = "mlp"

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        params = dict(params or {})
        self.params = params
        self.architecture = "residual"
        self.hidden_dims = (128, 64)
        if not self.hidden_dims or any(value <= 0 for value in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        self.dropout = float(params.get("dropout", 0.0))
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.seed = int(params.get("seed", 42))
        self.device_name = str(params.get("device", "auto"))
        self.device = self._resolve_device(self.device_name)
        self.network: nn.Module | None = None
        self.feature_cols: tuple[str, ...] | None = None
        self.training_summary: dict = {}

    def Fit(self, train_data: PanelDataset, valid_data: PanelDataset, options: dict) -> dict:
        """Train the network and restore the best validation checkpoint."""

        self._set_seed(int(options.get("seed", self.seed)))
        self.device = self._resolve_device(str(options.get("device", self.device_name)))
        self.feature_cols = train_data.feature_cols
        train_x, train_y, train_groups = self._arrays(train_data)
        valid_x, valid_y, valid_groups = self._arrays(valid_data)
        if len(train_x) == 0:
            raise ValueError("Training dataset is empty")
        if self.network is None:
            self.network = self._build_network(len(self.feature_cols)).to(self.device)
        optimizer_config = dict(options.get("optimizer", {}) or {})
        optimizer_name = str(optimizer_config.get("kind", "adam"))
        learning_rate = float(optimizer_config.get("learning_rate", 1e-3))
        learning_rate *= float(options.get("learning_rate_scale", 1.0))
        weight_decay = float(optimizer_config.get("weight_decay", 1e-4))
        if optimizer_name not in {"adam", "adamw"}:
            raise ValueError("training.optimizer.kind must be adam or adamw")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("training.optimizer parameters are invalid")
        optimizer_class = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
        optimizer = optimizer_class(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        epochs = int(options.get("epochs", 20))
        patience = int(options.get("early_stopping_patience", 5))
        periods_per_batch = int(options.get("periods_per_batch", 0))
        loss_kind = ValidateLossConfig(options.get("loss"))
        min_delta = float(options.get("early_stopping_min_delta", 0.0))
        if epochs <= 0 or patience <= 0:
            raise ValueError("epochs and early_stopping_patience must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate and learning_rate_scale must be positive")
        if periods_per_batch < 0:
            raise ValueError("periods_per_batch must be non-negative")
        use_amp = str(options.get("precision", "fp32")) == "amp" and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        scheduler = self._build_scheduler(optimizer, options)
        train_tensor = self._tensor(train_x)
        target_tensor = self._tensor(train_y)
        group_tensor = torch.as_tensor(train_groups, dtype=torch.long, device=self.device)
        valid_tensor = self._tensor(valid_x)
        best_loss = float("inf")
        best_validation: dict[str, float | int] | None = None
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        history: list[dict] = []
        for epoch in range(epochs):
            self.network.train()
            batches = BuildPeriodBatches(train_groups, periods_per_batch)
            epoch_losses: list[float] = []
            for indices in batches:
                index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                    prediction = self.network(train_tensor[index_tensor])
                    loss = ComputeLoss(
                        loss_kind,
                        prediction,
                        target_tensor[index_tensor],
                        group_tensor[index_tensor],
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(float(loss.detach().cpu()))
            validation = self._validation(
                valid_tensor,
                valid_y,
                valid_groups,
                loss_kind,
            )
            valid_loss = float(validation["valid_loss"])
            improved = np.isfinite(valid_loss) and valid_loss < best_loss - min_delta
            if improved:
                best_loss = valid_loss
                best_validation = dict(validation)
                best_state = copy.deepcopy(self.network.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if scheduler is not None and np.isfinite(valid_loss):
                scheduler.step(valid_loss)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(epoch_losses)),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    **validation,
                }
            )
            if stale_epochs >= patience:
                break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()
        best_validation = best_validation or {
            "valid_loss": float("nan"),
            "valid_mse": float("nan"),
            "valid_target_ic": float("nan"),
            "valid_rows": 0,
        }
        self.training_summary = {
            "epochs": len(history),
            "loss_kind": loss_kind,
            "best_valid_loss": float(best_validation["valid_loss"]),
            "best_valid_mse": float(best_validation["valid_mse"]),
            "best_valid_target_ic": float(best_validation["valid_target_ic"]),
            "history": history,
        }
        return self.training_summary

    def Predict(self, data: PanelDataset) -> pd.DataFrame:
        """Generate predictions for a Dataset."""

        if self.network is None or self.feature_cols is None:
            raise RuntimeError("Model is not fitted")
        if data.feature_cols != self.feature_cols:
            raise ValueError("Prediction feature schema does not match the fitted model")
        values = np.nan_to_num(data.FeatureArray(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        with torch.no_grad():
            predictions = self.network(self._tensor(values)).cpu().numpy()
        return pd.DataFrame(
            {
                data.time_col: data.TimeArray(),
                data.asset_col: data.AssetArray(),
                "prediction": predictions.astype(np.float32),
                data.target_col: data.TargetArray(),
            }
        )

    def Save(self, directory: str | Path) -> None:
        """Save model weights and reproducible configuration."""

        if self.network is None or self.feature_cols is None:
            raise RuntimeError("Cannot save an unfitted model")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "params": self.params,
                "feature_cols": list(self.feature_cols),
                "state_dict": self.network.state_dict(),
                "training_summary": self.training_summary,
            },
            path / "model.pt",
        )
        (path / "model.json").write_text(
            json.dumps(
                {
                    "kind": self.KIND,
                    "architecture": self.architecture,
                    "feature_cols": list(self.feature_cols),
                    "hidden_dims": list(self.hidden_dims) if self.architecture == "plain" else None,
                    "training_summary": self.training_summary,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def Load(self, directory: str | Path) -> None:
        """Restore a saved model bundle."""

        payload = torch.load(Path(directory) / "model.pt", map_location=self.device, weights_only=False)
        self.__init__(dict(payload["params"]))
        self.feature_cols = tuple(payload["feature_cols"])
        self.network = self._build_network(len(self.feature_cols)).to(self.device)
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval()
        self.training_summary = dict(payload.get("training_summary", {}))

    def _build_network(self, input_dim: int) -> nn.Module:
        if self.architecture == "residual":
            return _ResidualMlpNetwork(input_dim, self.dropout)
        return _PlainMlpNetwork(input_dim, self.hidden_dims, self.dropout)

    def _arrays(self, data: PanelDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = np.nan_to_num(data.FeatureArray(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        target = data.TargetArray().astype(np.float32)
        groups = data.PeriodCodes().astype(np.int64)
        valid = np.isfinite(target)
        return features[valid], target[valid], groups[valid]

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _validation(
        self,
        values: torch.Tensor,
        target: np.ndarray,
        groups: np.ndarray,
        loss_kind: str,
    ) -> dict[str, float | int]:
        if len(values) == 0:
            return {
                "valid_loss": float("nan"),
                "valid_mse": float("nan"),
                "valid_target_ic": float("nan"),
                "valid_rows": 0,
            }
        self.network.eval()
        with torch.no_grad():
            prediction_tensor = self.network(values)
            target_tensor = self._tensor(target)
            group_tensor = torch.as_tensor(groups, dtype=torch.long, device=self.device)
            loss = float(ComputeLoss(loss_kind, prediction_tensor, target_tensor, group_tensor).cpu())
            mse = float(ComputeLoss("mse", prediction_tensor, target_tensor, group_tensor).cpu())
            target_ic = float(
                ComputeGroupedCorrelation(prediction_tensor, target_tensor, group_tensor).cpu()
            )
        return {
            "valid_loss": loss,
            "valid_mse": mse,
            "valid_target_ic": target_ic,
            "valid_rows": int(len(values)),
        }

    @staticmethod
    def _build_scheduler(
        optimizer: torch.optim.Optimizer,
        options: Mapping[str, object],
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
        if str(options.get("lr_scheduler", "none")) != "plateau":
            return None
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(options.get("lr_scheduler_factor", 0.5)),
            patience=int(options.get("lr_scheduler_patience", 4)),
            min_lr=float(options.get("min_learning_rate", 1e-5)),
        )

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)

    @staticmethod
    def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


__all__ = ["MlpModel"]
