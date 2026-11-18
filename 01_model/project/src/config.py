from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

APP_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MlpFactorConfig:
    path: Path
    values: dict[str, Any]
    config_hash: str

    @classmethod
    def Load(cls, path: str | Path) -> "MlpFactorConfig":
        config_path = Path(path).expanduser()
        if not config_path.is_absolute():
            config_path = (APP_ROOT / config_path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        if not isinstance(raw, dict):
            raise ValueError("Config root must be a mapping")
        cls._validate(raw)
        semantic = {
            key: raw.get(key)
            for key in ("experiment", "data", "window", "transforms", "model", "training", "tuning")
        }
        semantic["data"] = {key: value for key, value in dict(raw.get("data", {})).items() if key != "name"}
        digest = json.dumps(semantic, sort_keys=True, separators=(",", ":"), default=str)
        return cls(config_path, raw, sha256(digest.encode("utf-8")).hexdigest())

    def Section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"Config section must be a mapping: {name}")
        return dict(value)

    def DataPath(self) -> Path:
        return self.Resolve(self.Section("data")["path"])

    def MergeDailyInfoPath(self) -> Path:
        return self.Resolve(self.Section("data")["merge_daily_info_path"])

    @staticmethod
    def Resolve(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (APP_ROOT / path).resolve()

    @classmethod
    def _validate(cls, values: Mapping[str, Any]) -> None:
        if "metadata" in values:
            raise ValueError("metadata is not supported")
        required = {"experiment", "data", "window", "model", "training", "artifacts"}
        missing = sorted(required.difference(values))
        if missing:
            raise ValueError(f"Missing config sections: {missing}")
        for name in required:
            if not isinstance(values[name], Mapping):
                raise ValueError(f"Config section must be a mapping: {name}")

        experiment = values["experiment"]
        data = values["data"]
        model = values["model"]
        artifacts = values["artifacts"]
        if not experiment.get("name") or not experiment.get("factor_name"):
            raise ValueError("experiment.name and experiment.factor_name are required")
        for field in ("name", "path", "merge_daily_info_path", "time_col", "asset_col", "target_col"):
            if not data.get(field):
                raise ValueError(f"data.{field} is required")
        if model.get("kind") not in {"mlp", "mlp-ls"}:
            raise ValueError("model.kind must be mlp or mlp-ls")
        forbidden = {"frequency", "label_horizon_periods"}.intersection(data)
        if forbidden:
            raise ValueError(f"Unsupported data fields: {sorted(forbidden)}")
        window = values["window"]
        if "kind" in window:
            raise ValueError("window.kind is fixed and must not be configured")
        for key in ("train_periods", "valid_periods", "test_periods"):
            if int(window.get(key, 0)) <= 0:
                raise ValueError(f"window.{key} must be positive")
        for key in ("step_periods", "purge_periods", "embargo_periods"):
            if int(window.get(key, 0)) < 0:
                raise ValueError(f"window.{key} must be non-negative")
        model_params = model.get("params", {}) or {}
        if not isinstance(model_params, Mapping):
            raise ValueError("model.params must be a mapping")
        unknown_model_params = set(model_params).difference({"dropout"})
        if unknown_model_params:
            raise ValueError(f"Unsupported model.params fields: {sorted(unknown_model_params)}")
        if "architecture" in model_params:
            raise ValueError("model.params.architecture is fixed and must not be configured")
        dropout = float(model_params.get("dropout", 0.0))
        if not 0.0 <= dropout < 1.0:
            raise ValueError("model.params.dropout must be in [0, 1)")
        if model["kind"] == "mlp-ls":
            ls = model.get("mlp-ls")
            if not isinstance(ls, Mapping) or int(ls.get("short_train_periods", 0)) <= 0:
                raise ValueError("model.mlp-ls.short_train_periods must be positive")
            if int(ls["short_train_periods"]) >= int(window["train_periods"]):
                raise ValueError("short_train_periods must be smaller than train_periods")
        from loss import ValidateLossConfig

        ValidateLossConfig(dict(values["training"].get("loss", {})))
        tuning = values.get("tuning", {}) or {}
        if not isinstance(tuning, Mapping):
            raise ValueError("tuning must be a mapping")
        legacy_tuning = {"metric", "metric_weights"}.intersection(tuning)
        if legacy_tuning:
            raise ValueError(f"Unsupported tuning fields: {sorted(legacy_tuning)}")
        optimizer = values["training"].get("optimizer")
        if not isinstance(optimizer, Mapping) or str(optimizer.get("kind")) not in {"adam", "adamw"}:
            raise ValueError("training.optimizer.kind must be adam or adamw")
        if float(optimizer.get("learning_rate", 0)) <= 0 or float(optimizer.get("weight_decay", -1)) < 0:
            raise ValueError("training.optimizer values are invalid")
        if not artifacts.get("run_root") or not artifacts.get("factor_path"):
            raise ValueError("artifacts.run_root and artifacts.factor_path are required")
        if Path(str(artifacts["factor_path"])).suffix.lower() != ".parquet":
            raise ValueError("artifacts.factor_path must use .parquet")
        if "publish_on_success" in artifacts:
            raise ValueError("artifacts.publish_on_success is fixed and must not be configured")


FactorConfig = MlpFactorConfig

__all__ = ["APP_ROOT", "FactorConfig", "MlpFactorConfig"]
