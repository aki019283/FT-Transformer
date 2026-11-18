"""Optuna tuning 和 run-local TuningVersion。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import optuna

from models import MakeModel
from windows import Window


@dataclass(frozen=True)
class TuningVersion:
    """一个 run 内不可变的分支参数版本。"""

    version: str
    parent_version: str | None
    trigger_window: str
    fit_time: str
    params: dict[str, dict[str, Any]]
    training_params: dict[str, dict[str, Any]]
    trial_counts: dict[str, int]
    best_values: dict[str, float | None]

    def ToDict(self) -> dict[str, Any]:
        """返回 manifest 记录。"""

        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "trigger_window": self.trigger_window,
            "fit_time": self.fit_time,
            "params": self.params,
            "training_params": self.training_params,
            "trial_counts": self.trial_counts,
            "best_values": self.best_values,
            "trial_count": sum(self.trial_counts.values()),
        }


class TuningEngine:
    """在当前窗口上为每个模型分支执行独立 Optuna 搜索。"""

    def __init__(
        self,
        config: Mapping[str, object] | None,
        seed: int = 42,
        training_options: Mapping[str, object] | None = None,
    ) -> None:
        """读取 trial 数量、预算和搜索空间。"""

        self.config = dict(config or {})
        self.seed = int(seed)
        self.trials = int(self.config.get("initial_trials", 10))
        self.refresh_trials = int(self.config.get("refresh_trials", 10))
        self.budgets = [int(value) for value in list(self.config.get("budgets", [10]) or [10])]
        self.search_space = dict(self.config.get("search_space", {}) or {})
        self.periods_per_batch = int(self.config.get("periods_per_batch", 8))
        self.fixed_trial_seed = bool(self.config.get("fixed_trial_seed", False))
        self.training_options = dict(training_options or {})
        if self.trials < 0 or self.refresh_trials < 0:
            raise ValueError("Tuning trial counts must be non-negative")
        if any(value <= 0 for value in self.budgets):
            raise ValueError("Tuning budgets must be positive")
        if self.periods_per_batch < 0:
            raise ValueError("periods_per_batch must be non-negative")

    def Tune(
        self,
        branches: Mapping[str, Window],
        base_params: Mapping[str, Mapping[str, object]],
        base_training_params: Mapping[str, Mapping[str, object]],
        run_dir: str | Path,
        version_index: int,
        parent_version: str | None,
    ) -> TuningVersion:
        """为所有分支创建一个统一的 TuningVersion。"""

        params: dict[str, dict[str, Any]] = {}
        training_params: dict[str, dict[str, Any]] = {}
        trial_counts: dict[str, int] = {}
        best_values: dict[str, float | None] = {}
        for branch_index, (branch, window) in enumerate(branches.items()):
            result = self._tune_branch(
                branch,
                window,
                base_params[branch],
                base_training_params.get(branch, {}),
                Path(run_dir) / "tuning" / f"v{version_index:03d}" / branch,
                version_index,
                branch_index,
            )
            params[branch] = result[0]
            training_params[branch] = result[1]
            trial_counts[branch] = result[2]
            best_values[branch] = result[3]
        version = TuningVersion(
            version=f"v{version_index:03d}",
            parent_version=parent_version,
            trigger_window=next(iter(branches.values())).window_id,
            fit_time=next(iter(branches.values())).fit_time.isoformat(),
            params=params,
            training_params=training_params,
            trial_counts=trial_counts,
            best_values=best_values,
        )
        version_dir = Path(run_dir) / "tuning" / version.version
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "tuning.json").write_text(
            json.dumps(version.ToDict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return version

    def _tune_branch(
        self,
        branch: str,
        window: Window,
        base_params: Mapping[str, object],
        base_training_params: Mapping[str, object],
        study_path: Path,
        version_index: int,
        branch_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any], int, float | None]:
        """执行一个分支的 study 并返回最佳参数。"""

        params = dict(base_params)
        training_params = copy.deepcopy(dict(base_training_params))
        study_path.mkdir(parents=True, exist_ok=True)
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{(study_path / 'study.db').as_posix()}"
        )
        try:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(
                    seed=self.seed + version_index * 101 + branch_index
                ),
                storage=storage,
                study_name=f"v{version_index:03d}_{branch}",
                load_if_exists=True,
            )
            trial_count = self.trials if version_index == 0 else self.refresh_trials
            if self.search_space and trial_count > 0:
                if not study.trials:
                    initial = self._initial_trial(params, training_params)
                    if initial:
                        study.enqueue_trial(initial)
                study.optimize(
                    lambda trial: self._objective(
                        trial, window, params, training_params
                    ),
                    n_trials=trial_count,
                )
            complete_trials = [
                trial for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
            ]
            if complete_trials:
                resolved = study.best_trial.user_attrs.get("resolved_params")
                if isinstance(resolved, Mapping):
                    params = dict(resolved)
                resolved_training = study.best_trial.user_attrs.get(
                    "resolved_training_params"
                )
                if isinstance(resolved_training, Mapping):
                    training_params = copy.deepcopy(dict(resolved_training))
            best_value = float(study.best_value) if complete_trials else None
            total_trials = len(study.trials)
        finally:
            storage.remove_session()
            storage.engine.dispose()
        return params, training_params, total_trials, best_value

    def _objective(
        self,
        trial: optuna.Trial,
        window: Window,
        base_params: Mapping[str, object],
        base_training_params: Mapping[str, object],
    ) -> float:
        """训练一个候选 MLP 并返回配置指定的 validation objective。"""

        trial_params = copy.deepcopy(dict(base_params))
        trial_training_params = copy.deepcopy(dict(base_training_params))
        for name, specification in self.search_space.items():
            value = self._suggest(trial, name, specification)
            root, path = name.split(".", 1)
            target = trial_params if root == "model" else trial_training_params
            self._set_nested(target, path, value)
        trial.set_user_attr("resolved_params", trial_params)
        trial.set_user_attr("resolved_training_params", trial_training_params)
        model = MakeModel("mlp", trial_params)
        trial_seed = self.seed if self.fixed_trial_seed else self.seed + trial.number
        options = self._merge_nested(self.training_options, trial_training_params)
        options.update({
            "epochs": self.budgets[-1],
            "early_stopping_patience": self.budgets[-1],
            "periods_per_batch": self.periods_per_batch,
            "seed": trial_seed,
        })
        options["early_stopping_patience"] = int(
            self.training_options.get("early_stopping_patience", self.budgets[-1])
        )
        summary = model.Fit(window.train, window.valid, options)
        value = summary.get("best_valid_loss")
        if value is None:
            raise RuntimeError("MLP training did not report best_valid_loss")
        return float(value)

    def _initial_trial(
        self,
        params: Mapping[str, object],
        training_params: Mapping[str, object],
    ) -> dict[str, object]:
        """Build an enqueue-compatible trial from the active parameter version."""

        initial: dict[str, object] = {}
        for name, specification in self.search_space.items():
            if not isinstance(specification, Mapping):
                continue
            root, path = name.split(".", 1)
            source = params if root == "model" else self._merge_nested(
                self.training_options, training_params
            )
            value = self._get_nested(source, path)
            if value is None:
                continue
            if str(specification.get("type", "float")) == "categorical":
                choices = list(specification.get("choices", []))
                if choices and isinstance(choices[0], (list, tuple)):
                    continue
            initial[name] = value
        return initial

    @staticmethod
    def _set_nested(target: dict[str, Any], path: str, value: object) -> None:
        current = target
        parts = path.split(".")
        for part in parts[:-1]:
            child = current.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Search path collides with a scalar: {path}")
            current = child
        current[parts[-1]] = value

    @staticmethod
    def _get_nested(source: Mapping[str, object], path: str) -> object | None:
        current: object = source
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        return current

    @classmethod
    def _merge_nested(
        cls,
        base: Mapping[str, object],
        overrides: Mapping[str, object],
    ) -> dict[str, Any]:
        result = copy.deepcopy(dict(base))
        for name, value in overrides.items():
            if isinstance(value, Mapping) and isinstance(result.get(name), Mapping):
                result[name] = cls._merge_nested(result[name], value)
            else:
                result[name] = copy.deepcopy(value)
        return result

    def _suggest(self, trial: optuna.Trial, name: str, specification: object) -> object:
        """根据类型化搜索空间创建 Optuna 参数。"""

        if not isinstance(specification, Mapping):
            raise ValueError(f"Search space entry must be a mapping: {name}")
        kind = str(specification.get("type", "float"))
        if kind == "int":
            return trial.suggest_int(
                name,
                int(specification["low"]),
                int(specification["high"]),
                log=bool(specification.get("log", False)),
            )
        if kind == "float":
            return trial.suggest_float(
                name,
                float(specification["low"]),
                float(specification["high"]),
                log=bool(specification.get("log", False)),
            )
        if kind == "categorical":
            choices = list(specification.get("choices", []))
            if choices and isinstance(choices[0], (list, tuple)):
                index = trial.suggest_int(f"{name}__index", 0, len(choices) - 1)
                return choices[index]
            return trial.suggest_categorical(name, choices)
        raise ValueError(f"Unsupported search space type: {kind}")


__all__ = ["TuningEngine", "TuningVersion"]
