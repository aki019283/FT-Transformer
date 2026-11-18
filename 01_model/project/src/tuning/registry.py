"""Tuning schedule 注册入口。"""

from registry import TypedRegistry
from .schedule import TuningSchedule


TUNING_SCHEDULE_REGISTRY = TypedRegistry[TuningSchedule]("tuning schedule")
for _mode in ("none", "once", "periodic", "explicit"):
    TUNING_SCHEDULE_REGISTRY.Register(
        _mode,
        lambda config, mode=_mode: TuningSchedule({**dict(config or {}), "mode": mode}),
    )


def MakeTuningSchedule(config: dict | None) -> TuningSchedule:
    """从配置创建 tuning schedule。"""

    values = dict(config or {})
    return TUNING_SCHEDULE_REGISTRY.Create(str(values.get("mode", "none")), values)


__all__ = ["TUNING_SCHEDULE_REGISTRY", "MakeTuningSchedule"]
