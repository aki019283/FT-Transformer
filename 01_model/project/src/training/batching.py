"""完整截面 batch 组织。"""

from __future__ import annotations

import numpy as np


def BuildPeriodBatches(groups: np.ndarray, periods_per_batch: int) -> list[np.ndarray]:
    """按周期分组构建 batch，保证一个截面不会被拆分。"""

    if len(groups) == 0:
        return []
    if periods_per_batch <= 0:
        return [np.arange(len(groups), dtype=np.int64)]
    unique_groups = np.unique(groups)
    np.random.shuffle(unique_groups)
    return [
        np.concatenate([
            np.flatnonzero(groups == group)
            for group in unique_groups[start:start + periods_per_batch]
        ]).astype(np.int64)
        for start in range(0, len(unique_groups), periods_per_batch)
    ]


__all__ = ["BuildPeriodBatches"]
