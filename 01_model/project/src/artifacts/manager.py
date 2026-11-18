"""Run 目录、原子窗口提交、恢复和因子发布。"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from time import sleep
from typing import TYPE_CHECKING, Iterator, Mapping
from uuid import uuid4

import pandas as pd

from alignment import AlignMonthlyFactor
from transforms import TransformPipeline
from windows import Window
from .manifests import RunStatus, WindowStatus

if TYPE_CHECKING:
    from models.base import Model


class ArtifactManager:
    """维护一个 experiment 下的 run 和窗口事务。"""

    def __init__(
        self,
        run_root: str | Path,
        model_name: str,
        dataset_name: str,
        experiment_name: str,
        factor_name: str,
    ) -> None:
        """Bind an experiment to its explicitly configured run root."""

        self.root = Path(run_root).expanduser().resolve()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.factor_name = factor_name
        self.run_id: str | None = None
        self.run_dir: Path | None = None
        self.manifest: dict = {}

    def Initialize(
        self,
        run_id: str,
        config: Mapping[str, object],
        config_hash: str,
        data_summary: Mapping[str, object],
        windows: list[Window],
    ) -> None:
        """创建一个新的 run。"""

        self._validate_run_id(run_id)
        self.run_id = run_id
        self.run_dir = self.root / "runs" / run_id
        if self.run_dir.exists():
            raise FileExistsError(f"Run already exists: {run_id}")
        (self.run_dir / "windows").mkdir(parents=True)
        (self.run_dir / "tuning").mkdir()
        self.manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "experiment": self.experiment_name,
            "model": self.model_name,
            "dataset": self.dataset_name,
            "factor_name": self.factor_name,
            "status": RunStatus.RUNNING,
            "created_at": self._now(),
            "updated_at": self._now(),
            "config_hash": config_hash,
            "resolved_config": dict(config),
            "data": dict(data_summary),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
            },
            "window_count": len(windows),
            "windows": [],
            "tuning_versions": [],
            "factor": None,
        }
        self._write_run_manifest()
        self._update_experiment_index()
        self.AppendEvent("RUN_STARTED", {"window_count": len(windows)})

    def Resume(self, run_id: str, config_hash: str) -> None:
        """加载已有 run 并校验训练配置。"""

        self._validate_run_id(run_id)
        self.run_id = run_id
        self.run_dir = self.root / "runs" / run_id
        path = self.run_dir / "run.json"
        if not path.is_file():
            raise FileNotFoundError(f"Run manifest not found: {path}")
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if self.manifest.get("config_hash") != config_hash:
            raise ValueError("Resume config does not match the existing run")
        self.manifest["status"] = RunStatus.RUNNING
        self._write_run_manifest()
        self.AppendEvent("RUN_RESUMED", {})

    def Open(self, run_id: str, config_hash: str) -> None:
        """只读加载已有 run 并校验配置。"""

        self._validate_run_id(run_id)
        self.run_id = run_id
        self.run_dir = self.root / "runs" / run_id
        path = self.run_dir / "run.json"
        if not path.is_file():
            raise FileNotFoundError(f"Run manifest not found: {path}")
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if self.manifest.get("config_hash") != config_hash:
            raise ValueError("Config does not match the existing run")

    def UpdatePlan(
        self,
        data_summary: Mapping[str, object],
        window_count: int,
        allow_same_period_revision: bool = False,
    ) -> None:
        """校验 append-only 数据约定并更新 run 的当前规模。"""

        previous = dict(self.manifest.get("data", {}))
        current = dict(data_summary)
        if previous.get("schema") != current.get("schema"):
            raise ValueError("Dataset schema changed during resume")
        if previous.get("start") != current.get("start"):
            raise ValueError("Dataset start period changed during resume")
        if int(current.get("rows", 0)) < int(previous.get("rows", 0)):
            raise ValueError("Dataset row count decreased during resume")
        if int(current.get("periods", 0)) < int(previous.get("periods", 0)):
            raise ValueError("Dataset period count decreased during resume")
        if previous.get("end") and current.get("end"):
            if pd.Timestamp(str(current["end"])) < pd.Timestamp(str(previous["end"])):
                raise ValueError("Dataset end period moved backwards during resume")
        if (
            int(current.get("periods", 0)) == int(previous.get("periods", 0))
            and int(current.get("rows", 0)) != int(previous.get("rows", 0))
            and not allow_same_period_revision
        ):
            raise ValueError("Rows changed without appending a new period")
        self.manifest["data"] = current
        self.manifest["window_count"] = int(window_count)
        self._write_run_manifest()

    @contextmanager
    def Lock(self) -> Iterator[None]:
        """使用独占文件限制同一 run 的并发写入。"""

        run_dir = self._require_run_dir()
        lock_path = run_dir / ".lock"
        descriptor: int | None = None
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            yield
        except FileExistsError as error:
            raise RuntimeError(f"Run is locked: {self.run_id}") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
                lock_path.unlink(missing_ok=True)

    def SuccessfulWindowIds(self) -> set[str]:
        """返回已经成功提交的窗口 ID。"""

        successful: set[str] = set()
        for record in self.manifest.get("windows", []):
            if record.get("status") != WindowStatus.SUCCEEDED:
                continue
            if not self._validate_window(record):
                raise RuntimeError(f"Committed window failed validation: {record.get('window_id')}")
            successful.add(str(record["window_id"]))
        return successful

    def PreviousBundle(self, before_window_id: str | None = None) -> Path | None:
        """返回指定窗口之前最近的成功 bundle。"""

        records = [
            record for record in self.manifest.get("windows", [])
            if record.get("status") == WindowStatus.SUCCEEDED
            and (before_window_id is None or str(record["window_id"]) < before_window_id)
        ]
        if not records:
            return None
        latest = max(records, key=lambda record: str(record["window_id"]))
        return self._require_run_dir() / "windows" / str(latest["window_id"]) / "bundle"

    def InvalidateFrom(self, window_ids: list[str]) -> None:
        """保留归档并使指定窗口失效。"""

        if not window_ids:
            return
        run_dir = self._require_run_dir()
        archive = (
            run_dir / "windows" / ".invalidated"
            / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid4().hex}"
        )
        archive.mkdir(parents=True, exist_ok=True)
        for window_id in window_ids:
            source = run_dir / "windows" / window_id
            if source.exists():
                self._replace(source, archive / window_id)
        invalid = set(window_ids)
        self.manifest["windows"] = [
            record for record in self.manifest.get("windows", [])
            if record.get("window_id") not in invalid
        ]
        self.manifest["factor"] = None
        self._write_run_manifest()
        self.AppendEvent("WINDOWS_INVALIDATED", {"window_ids": window_ids})

    def InvalidateTuningFrom(self, fit_time: pd.Timestamp) -> None:
        """归档数据截止时间受修订影响的 tuning versions。"""

        versions = list(self.manifest.get("tuning_versions", []))
        invalid = [
            record for record in versions
            if pd.Timestamp(str(record["fit_time"])) >= fit_time
        ]
        if not invalid:
            return
        run_dir = self._require_run_dir()
        archive = run_dir / "tuning" / ".invalidated" / uuid4().hex
        archive.mkdir(parents=True, exist_ok=True)
        invalid_names = {str(record["version"]) for record in invalid}
        for version in invalid_names:
            source = run_dir / "tuning" / version
            if source.exists():
                self._replace(source, archive / version)
        self.manifest["tuning_versions"] = [
            record for record in versions
            if str(record["version"]) not in invalid_names
        ]
        self._write_run_manifest()
        self.AppendEvent("TUNING_INVALIDATED", {"versions": sorted(invalid_names)})

    def AddTuningVersion(self, version: Mapping[str, object]) -> None:
        """把 tuning version 加入 run 时间轴。"""

        versions = list(self.manifest.get("tuning_versions", []))
        versions.append(dict(version))
        self.manifest["tuning_versions"] = versions
        self._write_run_manifest()

    def CommitWindow(
        self,
        window: Window,
        predictions: pd.DataFrame,
        model: Model,
        pipeline: TransformPipeline,
        metrics: Mapping[str, object],
        parent_bundle: Path | None,
        full_refit: bool,
    ) -> None:
        """完整写入并原子发布一个窗口。"""

        run_dir = self._require_run_dir()
        windows_dir = run_dir / "windows"
        final_dir = windows_dir / window.window_id
        if final_dir.exists():
            raise FileExistsError(f"Window already committed: {window.window_id}")
        temp_dir = windows_dir / f".tmp-{window.window_id}-{uuid4().hex}"
        bundle_dir = temp_dir / "bundle"
        try:
            bundle_dir.mkdir(parents=True)
            prediction_path = temp_dir / "predictions.parquet"
            predictions.to_parquet(prediction_path, index=False)
            model.Save(bundle_dir)
            pipeline.Save(bundle_dir)
            bundle_files = {
                path.relative_to(bundle_dir).as_posix(): self._checksum(path)
                for path in bundle_dir.rglob("*") if path.is_file()
            }
            bundle_manifest = {
                "schema_version": 1,
                "model_kind": str(getattr(model, "KIND", "unknown")),
                "files": bundle_files,
            }
            self._write_json(bundle_dir / "bundle.json", bundle_manifest)
            record = {
                **window.ToRecord(),
                "status": WindowStatus.SUCCEEDED,
                "parent_bundle": None if parent_bundle is None else parent_bundle.as_posix(),
                "full_refit": full_refit,
                "metrics": dict(metrics),
                "prediction_checksum": self._checksum(prediction_path),
                "bundle_checksum": self._checksum(bundle_dir / "bundle.json"),
                "completed_at": self._now(),
            }
            self._write_json(temp_dir / "window.json", record)
            self._replace(temp_dir, final_dir)
        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        records = [
            item for item in self.manifest.get("windows", [])
            if item.get("window_id") != window.window_id
        ]
        records.append(record)
        self.manifest["windows"] = sorted(records, key=lambda item: str(item["window_id"]))
        self._write_run_manifest()
        self.AppendEvent("WINDOW_COMMITTED", {"window_id": window.window_id})

    def AssembleFactor(self, merge_daily_info_path: str | Path) -> Path:
        """从所有成功窗口预测确定性重建 run 因子。"""

        run_dir = self._require_run_dir()
        frames: list[pd.DataFrame] = []
        for record in sorted(self.manifest.get("windows", []), key=lambda item: str(item["window_id"])):
            if record.get("status") != WindowStatus.SUCCEEDED:
                continue
            path = run_dir / "windows" / str(record["window_id"]) / "predictions.parquet"
            frames.append(pd.read_parquet(path))
        if not frames:
            raise RuntimeError("No successful window predictions are available")
        factor = pd.concat(frames, ignore_index=True)
        data_config = self.manifest["resolved_config"]["data"]
        time_col = str(data_config["time_col"])
        asset_col = str(data_config["asset_col"])
        required = {time_col, asset_col, "prediction"}
        missing = sorted(required.difference(factor.columns))
        if missing:
            raise ValueError(f"Prediction files are missing columns: {missing}")
        factor = factor[[time_col, asset_col, "prediction"]].rename(
            columns={"prediction": self.factor_name}
        )
        if factor.duplicated([time_col, asset_col]).any():
            raise ValueError("Factor assembly found duplicate (time, asset) keys")
        factor = factor.sort_values([time_col, asset_col], kind="mergesort").reset_index(drop=True)
        factor, alignment = AlignMonthlyFactor(
            factor,
            merge_daily_info_path,
            time_col,
            asset_col,
        )
        path = run_dir / "factor.parquet"
        temp = run_dir / f".factor-{uuid4().hex}.parquet"
        factor.to_parquet(temp, index=False)
        pd.read_parquet(temp, columns=[time_col, asset_col, self.factor_name])
        self._replace(temp, path)
        self.manifest["factor"] = {
            "path": path.name,
            "rows": len(factor),
            "checksum": self._checksum(path),
            "alignment": alignment,
        }
        self._write_run_manifest()
        return path

    def Publish(self, factor_path: str | Path) -> Path:
        """原子发布 run 因子及 sidecar manifest。"""

        run_dir = self._require_run_dir()
        source = run_dir / "factor.parquet"
        factor_record = self.manifest.get("factor")
        if not source.is_file() or not factor_record:
            raise FileNotFoundError("Run factor has not been assembled")
        if self._checksum(source) != factor_record.get("checksum"):
            raise RuntimeError("Run factor checksum does not match its manifest")
        target = Path(factor_path).expanduser().resolve()
        if target.suffix.lower() != ".parquet":
            raise ValueError("Published factor path must use the .parquet suffix")
        target_dir = target.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        temp = target_dir / f".{target.stem}-{uuid4().hex}.parquet"
        shutil.copyfile(source, temp)
        self._replace(temp, target)
        sidecar = {
            "experiment": self.experiment_name,
            "model": self.model_name,
            "dataset": self.dataset_name,
            "run_id": self.run_id,
            "factor_name": self.factor_name,
            "path": str(target),
            "checksum": self._checksum(target),
            "published_at": self._now(),
        }
        self._write_json(target.with_suffix(".manifest.json"), sidecar)
        self.manifest["published"] = sidecar
        self._write_run_manifest()
        self._update_experiment_index()
        return target

    def Finish(self, status: RunStatus, error: str | None = None) -> None:
        """更新 run 最终状态。"""

        self.manifest["status"] = status
        self.manifest["error"] = error
        self._write_run_manifest()
        self._update_experiment_index()
        self.AppendEvent("RUN_FINISHED", {"status": status, "error": error})

    def AppendEvent(self, event: str, fields: Mapping[str, object]) -> None:
        """追加一条结构化事件。"""

        record = {"time": self._now(), "event": event, **dict(fields)}
        with (self._require_run_dir() / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _write_run_manifest(self) -> None:
        """原子更新 run.json 并重建窗口索引。"""

        self.manifest["updated_at"] = self._now()
        run_dir = self._require_run_dir()
        self._write_json(run_dir / "run.json", self.manifest)
        rows: list[dict] = []
        for record in self.manifest.get("windows", []):
            row = {key: value for key, value in record.items() if key != "metrics"}
            for key, value in dict(record.get("metrics", {})).items():
                row[f"metric_{key}"] = value
            rows.append(row)
        pd.DataFrame(rows).to_parquet(run_dir / "windows.parquet", index=False)

    def _update_experiment_index(self) -> None:
        """更新 experiment 级 run 导航索引。"""

        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "experiment.json"
        index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "schema_version": 1,
            "experiment": self.experiment_name,
            "runs": [],
        }
        runs = [record for record in index.get("runs", []) if record.get("run_id") != self.run_id]
        runs.append({
            "run_id": self.run_id,
            "status": self.manifest.get("status"),
            "updated_at": self.manifest.get("updated_at"),
        })
        index["runs"] = sorted(runs, key=lambda record: str(record["run_id"]))
        if self.manifest.get("status") == RunStatus.SUCCEEDED:
            index["latest_successful_run"] = self.run_id
        if self.manifest.get("published"):
            index["latest_published_run"] = self.run_id
        self._write_json(path, index)

    def _write_json(self, path: Path, value: Mapping[str, object]) -> None:
        """通过同目录临时文件原子写入 JSON。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        self._replace(temp, path)

    @staticmethod
    def _replace(source: Path, target: Path) -> None:
        """以指数退避重试 Windows 上短暂被占用的原子替换。"""

        attempts = 8
        for attempt in range(attempts):
            try:
                os.replace(source, target)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                sleep(0.05 * (2**attempt))

    def _checksum(self, path: Path) -> str:
        """流式计算文件 SHA-256。"""

        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_window(self, record: Mapping[str, object]) -> bool:
        """校验成功窗口的预测和 bundle 文件。"""

        window_id = str(record.get("window_id", ""))
        directory = self._require_run_dir() / "windows" / window_id
        prediction = directory / "predictions.parquet"
        bundle_manifest_path = directory / "bundle" / "bundle.json"
        window_manifest_path = directory / "window.json"
        required = [prediction, bundle_manifest_path, window_manifest_path]
        if not all(path.is_file() for path in required):
            return False
        if self._checksum(prediction) != record.get("prediction_checksum"):
            return False
        if self._checksum(bundle_manifest_path) != record.get("bundle_checksum"):
            return False
        bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
        for relative, expected in bundle_manifest.get("files", {}).items():
            path = directory / "bundle" / relative
            if not path.is_file() or self._checksum(path) != expected:
                return False
        return True

    def _require_run_dir(self) -> Path:
        """返回已经初始化的 run 目录。"""

        if self.run_dir is None:
            raise RuntimeError("Artifact manager has no active run")
        return self.run_dir

    def _validate_run_id(self, run_id: str) -> None:
        """拒绝可能逃逸 run 根目录的 ID。"""

        safe_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not run_id or any(character not in safe_characters for character in run_id):
            raise ValueError("run_id contains unsafe characters")

    def _now(self) -> str:
        """返回带本地时区的当前时间。"""

        return datetime.now().astimezone().isoformat()


def MakeRunId() -> str:
    """生成包含微秒的本地 Run ID。"""

    return datetime.now().strftime("run-%Y%m%dT%H%M%S-%f")


__all__ = ["ArtifactManager", "MakeRunId"]
