"""MLP 因子训练与发布工作流。"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter

import pandas as pd

from artifacts import ArtifactManager, MakeRunId, RunStatus
from config import MlpFactorConfig
from data import PanelDataset
from metrics import ComputeMetrics
from training import MakeTrainingPolicy, TrainingDecision
from transforms import BuildDatasetPipeline, BuildWindowPipeline
from tuning.registry import MakeTuningSchedule
from windows import WindowBuilder


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanResult:
    """只读 plan 的摘要。"""

    run_id: str
    rows: int
    periods: int
    windows: int
    tuning_events: int

    def ToDict(self) -> dict[str, object]:
        """返回 CLI 可展示的摘要。"""

        return {
            "run_id": self.run_id,
            "rows": self.rows,
            "periods": self.periods,
            "windows": self.windows,
            "tuning_events": self.tuning_events,
        }


class MlpFactorWorkflow:
    """加载数据、训练窗口并提交最终因子的主入口。"""

    def __init__(self, config: MlpFactorConfig) -> None:
        """拆分 resolved config 供各业务模块使用。"""

        self.config = config
        self.data_config = config.Section("data")
        self.dataset_name = str(self.data_config["name"])
        self.window_config = config.Section("window")
        self.model_config = config.Section("model")
        self.model_kind = str(self.model_config.get("kind", "mlp"))
        self.training_config = config.Section("training")
        self.tuning_config = config.Section("tuning")
        self.artifact_config = config.Section("artifacts")

    def Plan(self) -> PlanResult:
        """只读构建数据和窗口计划。"""

        _LOGGER.info("Loading panel dataset")
        dataset = self._load_dataset()
        windows = WindowBuilder(self.window_config).Build(dataset)
        BuildWindowPipeline(self.config.values.get("transforms"))
        MakeTrainingPolicy(self.training_config)
        schedule = MakeTuningSchedule(self.tuning_config)
        events = sum(schedule.ShouldTune(index, window.fit_time) for index, window in enumerate(windows))
        return PlanResult(
            run_id=MakeRunId(),
            rows=dataset.RowCount(),
            periods=dataset.PeriodCount(),
            windows=len(windows),
            tuning_events=events,
        )

    def Run(
        self,
        resume: str | None = None,
        rebuild_from: str | None = None,
    ) -> dict[str, object]:
        """执行滚动训练并返回 run manifest。"""

        if rebuild_from is not None and resume is None:
            raise ValueError("--rebuild-from requires --resume")
        rebuild_boundary = pd.Timestamp(rebuild_from) if rebuild_from is not None else None
        _LOGGER.info("Loading panel dataset")
        dataset = self._load_dataset()
        windows = WindowBuilder(self.window_config).Build(dataset)
        manager = self._manager()
        if resume:
            manager.Resume(resume, self.config.config_hash)
            manager.UpdatePlan(
                self._data_summary(dataset),
                len(windows),
                allow_same_period_revision=rebuild_boundary is not None,
            )
        else:
            manager.Initialize(
                MakeRunId(),
                self.config.values,
                self.config.config_hash,
                self._data_summary(dataset),
                windows,
            )
        run_id = manager.run_id
        if run_id is None:
            raise RuntimeError("Run was not initialized")
        _LOGGER.info("Run %s contains %d planned windows", run_id, len(windows))
        with manager.Lock():
            if rebuild_boundary is not None:
                successful = manager.SuccessfulWindowIds()
                invalid = [
                    window.window_id for window in windows
                    if window.test_periods[-1] >= rebuild_boundary
                    and window.window_id in successful
                ]
                manager.InvalidateFrom(invalid)
                manager.InvalidateTuningFrom(rebuild_boundary)
            try:
                self._run_windows(manager, windows, dataset)
                manager.AssembleFactor(self.config.MergeDailyInfoPath())
                manager.Publish(self._factor_path())
                manager.Finish(RunStatus.SUCCEEDED)
            except KeyboardInterrupt:
                manager.Finish(RunStatus.INTERRUPTED, "Interrupted by user")
                raise
            except Exception as error:
                manager.Finish(RunStatus.FAILED, str(error))
                raise
        return manager.manifest

    def Status(self, run_id: str) -> dict[str, object]:
        """读取 run manifest。"""

        manager = self._manager()
        manager.Open(run_id, self.config.config_hash)
        return manager.manifest

    def Publish(self, run_id: str) -> Path:
        """发布指定 run 已经生成的因子。"""

        manager = self._manager()
        manager.Open(run_id, self.config.config_hash)
        with manager.Lock():
            return manager.Publish(
                self._factor_path()
            )

    def _run_windows(
        self,
        manager: ArtifactManager,
        windows: list,
        dataset: PanelDataset,
    ) -> None:
        """顺序执行未完成窗口并提交结果。"""

        from models import MakeModel
        from tuning.optuna import TuningEngine

        policy = MakeTrainingPolicy(self.training_config)
        schedule = MakeTuningSchedule(self.tuning_config)
        tuner = TuningEngine(
            self.tuning_config,
            int(self.config.values["experiment"].get("seed", 42)),
            self.training_config,
        )
        successful = manager.SuccessfulWindowIds()
        base_params = self._base_branch_params()
        base_training_params = {branch: {} for branch in base_params}
        versions = list(manager.manifest.get("tuning_versions", []))
        for index, window in enumerate(windows):
            if window.window_id in successful:
                _LOGGER.info("[%d/%d] Skip committed window %s", index + 1, len(windows), window.window_id)
                continue
            started_at = perf_counter()
            _LOGGER.info("[%d/%d] Train window %s", index + 1, len(windows), window.window_id)
            pipeline = BuildWindowPipeline(self.config.values.get("transforms"))
            pipeline.Fit(window.train)
            transformed_window = pipeline.Apply(window)
            parent_bundle = manager.PreviousBundle(window.window_id)
            decision = policy.Decide(index, parent_bundle)
            eligible_versions = [
                record for record in versions
                if str(record.get("trigger_window", "")) <= window.window_id
            ]
            active_record = max(
                eligible_versions,
                key=lambda record: (str(record.get("trigger_window", "")), str(record.get("version", ""))),
                default=None,
            )
            active_params = self._copy_branch_params(
                active_record.get("params", base_params) if active_record is not None else base_params
            )
            active_training_params = self._copy_branch_params(
                active_record.get("training_params", base_training_params)
                if active_record is not None else base_training_params
            )
            active_version = (
                str(active_record["version"])
                if active_record is not None else None
            )
            trigger_record = next(
                (
                    record for record in reversed(versions)
                    if record.get("trigger_window") == window.window_id
                ),
                None,
            )
            if trigger_record is not None:
                active_record = trigger_record
                active_params = self._copy_branch_params(trigger_record.get("params", base_params))
                active_training_params = self._copy_branch_params(
                    trigger_record.get("training_params", base_training_params)
                )
                active_version = str(trigger_record["version"])
                decision = TrainingDecision(None, policy.full_epochs, True, True)
            elif schedule.ShouldTune(index, window.fit_time):
                _LOGGER.info("Create tuning version for window %s", window.window_id)
                version = tuner.Tune(
                    self._tuning_branches(transformed_window),
                    active_params,
                    active_training_params,
                    manager.run_dir,
                    len(versions),
                    active_version,
                )
                versions.append(version.ToDict())
                manager.AddTuningVersion(version.ToDict())
                active_params = self._copy_branch_params(version.params)
                active_training_params = self._copy_branch_params(version.training_params)
                active_version = version.version
                decision = TrainingDecision(None, policy.full_epochs, True, True)
            transformed_window.tuning_version = active_version
            model = MakeModel(self.model_kind, self._model_params(active_params))
            if decision.parent_bundle is not None:
                model.Load(decision.parent_bundle)
            if decision.train:
                options = {
                    **self._merge_training_options(
                        self.training_config,
                        active_training_params.get("main", {}),
                    ),
                    "epochs": decision.epochs,
                    "seed": int(self.config.values["experiment"].get("seed", 42)) + index,
                    "reset_models": decision.full_refit,
                    "learning_rate_scale": (
                        1.0 if decision.full_refit else policy.update_learning_rate_scale
                    ),
                }
                summary = model.Fit(transformed_window.train, transformed_window.valid, options)
            else:
                summary = model.training_summary
            predictions = model.Predict(transformed_window.test)
            valid_predictions = model.Predict(transformed_window.valid)
            metrics = ComputeMetrics(valid_predictions, dataset.time_col, dataset.target_col)
            metrics["test_mean_ic"] = ComputeMetrics(predictions, dataset.time_col, dataset.target_col)["mean_ic"]
            metrics["training_epochs"] = int(summary.get("epochs", 0))
            metrics["best_valid_target_ic"] = summary.get("best_valid_target_ic")
            metrics["best_valid_mse"] = summary.get("best_valid_mse")
            metrics["best_valid_loss"] = summary.get("best_valid_loss")
            metrics["elapsed_seconds"] = perf_counter() - started_at
            manager.CommitWindow(
                transformed_window,
                predictions,
                model,
                pipeline,
                metrics,
                decision.parent_bundle,
                decision.full_refit,
            )
            _LOGGER.info(
                "[%d/%d] Committed window %s in %.2fs",
                index + 1,
                len(windows),
                window.window_id,
                metrics["elapsed_seconds"],
            )

    def _load_dataset(self) -> PanelDataset:
        """按配置加载统一 PanelDataset。"""

        dataset = PanelDataset.Load(
            self.config.DataPath(),
            str(self.data_config["time_col"]),
            str(self.data_config["asset_col"]),
            str(self.data_config["target_col"]),
            self.data_config.get("feature_cols"),
        )
        pipeline = BuildDatasetPipeline(self.config.values.get("transforms"))
        return pipeline.ApplyDataset(dataset)

    def _base_branch_params(self) -> dict[str, dict[str, object]]:
        """构建初始模型分支参数。"""

        params = dict(self.model_config.get("params", {}) or {})
        return {"main": params}

    def _copy_branch_params(self, value: object) -> dict[str, dict[str, object]]:
        """规范化 manifest 中保存的分支参数。"""

        if not isinstance(value, Mapping):
            raise ValueError("Tuning params must be a branch mapping")
        return {str(branch): dict(params) for branch, params in value.items()}

    @classmethod
    def _merge_training_options(
        cls,
        base: Mapping[str, object],
        overrides: Mapping[str, object],
    ) -> dict[str, object]:
        """Recursively merge tuned training overrides into the run config."""

        result = copy.deepcopy(dict(base))
        for name, value in overrides.items():
            if isinstance(value, Mapping) and isinstance(result.get(name), Mapping):
                result[name] = cls._merge_training_options(result[name], value)
            else:
                result[name] = copy.deepcopy(value)
        return result

    def _model_params(self, branch_params: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
        """把分支参数和融合配置组装成 Model 参数。"""

        if self.model_kind == "mlp":
            return dict(branch_params["main"])
        mlp_ls_config = dict(self.model_config.get("mlp-ls", {}) or {})
        return {
            "model_params": dict(branch_params["main"]),
            "short_train_periods": int(mlp_ls_config.get("short_train_periods", 18)),
        }

    def _tuning_branches(self, window):
        """为 tuning 构建 Long/Short 分支窗口。"""

        return {"main": window}

    def _manager(self) -> ArtifactManager:
        """创建指向当前 experiment 的产物管理器。"""

        values = self.config.values["experiment"]
        return ArtifactManager(
            self.config.Resolve(self.artifact_config["run_root"]),
            self.model_kind,
            self.dataset_name,
            str(values["name"]),
            str(values["factor_name"]),
        )

    def _factor_path(self) -> Path:
        """返回配置显式指定的稳定因子发布路径。"""

        return self.config.Resolve(self.artifact_config["factor_path"])

    def _data_summary(self, dataset: PanelDataset) -> dict[str, object]:
        """生成不包含内容指纹的数据摘要。"""

        periods = dataset.Periods()
        return {
            "path": str(self.config.DataPath()),
            "schema": list(dataset.schema),
            "rows": dataset.RowCount(),
            "periods": dataset.PeriodCount(),
            "start": periods[0].isoformat() if len(periods) else None,
            "end": periods[-1].isoformat() if len(periods) else None,
        }

__all__ = ["MlpFactorWorkflow", "PlanResult"]
