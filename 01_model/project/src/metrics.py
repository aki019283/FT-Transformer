"""训练和因子报告指标。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ComputeMetrics(
    frame: pd.DataFrame,
    time_col: str,
    target_col: str,
    asset_col: str | None = None,
    top_k_pct: float = 0.1,
    periods_per_year: int = 12,
    annual_risk_free_rate: float = .02,
) -> dict[str, float | int]:
    """计算窗口级验证和测试指标。"""

    if frame.empty or "prediction" not in frame or target_col not in frame:
        return {"mean_ic": float("nan"), "rows": int(len(frame))}
    correlations: list[float] = []
    rank_correlations: list[float] = []
    top_returns: list[float] = []
    top_assets: list[set[str]] = []
    for _, group in frame.groupby(time_col, sort=False):
        values = group[[target_col, "prediction"]].dropna()
        if len(values) < 2:
            continue
        correlation = values[target_col].corr(values["prediction"])
        rank_correlation = values[target_col].corr(values["prediction"], method="spearman")
        if pd.notna(correlation):
            correlations.append(float(correlation))
        if pd.notna(rank_correlation):
            rank_correlations.append(float(rank_correlation))
        top_count = max(1, int(np.ceil(len(values) * top_k_pct)))
        selected_index = values.nlargest(top_count, "prediction").index
        top_returns.append(float(group.loc[selected_index, target_col].mean()))
        if asset_col is not None:
            top_assets.append(set(group.loc[selected_index, asset_col].astype(str)))
    mean_ic = float(np.mean(correlations)) if correlations else float("nan")
    mean_rank_ic = float(np.mean(rank_correlations)) if rank_correlations else float("nan")
    ic_std = float(np.std(rank_correlations, ddof=1)) if len(rank_correlations) > 1 else float("nan")
    top_std = float(np.std(top_returns, ddof=1)) if len(top_returns) > 1 else 0.0
    turnover_values = [
        len(previous.symmetric_difference(current)) / max(len(previous), 1)
        for previous, current in zip(top_assets, top_assets[1:])
    ]
    return {
        "mean_ic": mean_ic,
        "mean_rank_ic": mean_rank_ic,
        "rank_icir": mean_rank_ic / ic_std if np.isfinite(ic_std) and ic_std > 0 else float("nan"),
        "top_return": float(np.mean(top_returns)) if top_returns else float("nan"),
        "top_sharpe": (
            (float(np.mean(top_returns)) - ((1+annual_risk_free_rate)**(1/periods_per_year)-1)) / top_std * np.sqrt(periods_per_year)
            if top_returns and top_std > 0 else float("nan")
        ),
        "turnover": float(np.mean(turnover_values)) if turnover_values else 0.0,
        "rows": int(len(frame)),
        "periods": int(frame[time_col].nunique()),
    }


def ComputeValidationLoss(
    frame: pd.DataFrame,
    time_col: str,
    asset_col: str,
    target_col: str,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float | int]]:
    """将 Rank IC、Sharpe 和 Turnover 合成为最小化损失。"""

    weights = weights or {}
    metrics = ComputeMetrics(
        frame,
        time_col,
        target_col,
        asset_col,
        float(weights.get("top_k_pct", 0.1)),
    )
    rank_ic = float(metrics["mean_rank_ic"])
    sharpe = float(metrics["top_sharpe"])
    turnover = float(metrics["turnover"])
    loss = (
        -float(weights.get("ic_weight", 1.0)) * (rank_ic if np.isfinite(rank_ic) else 0.0)
        -float(weights.get("sharpe_weight", 0.0)) * (sharpe if np.isfinite(sharpe) else 0.0)
        +float(weights.get("turnover_weight", 0.0)) * turnover
    )
    return loss, metrics


__all__ = ["ComputeMetrics", "ComputeValidationLoss"]
