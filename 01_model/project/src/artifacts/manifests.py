"""运行和窗口 manifest 的稳定状态字面值。"""

from enum import StrEnum


class RunStatus(StrEnum):
    """Run 生命周期状态。"""

    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class WindowStatus(StrEnum):
    """Window 生命周期状态。"""

    PENDING = "PENDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


__all__ = ["RunStatus", "WindowStatus"]
