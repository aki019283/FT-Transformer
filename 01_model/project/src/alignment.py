"""Align monthly factor rows to the valid stock universe on month-end trading days."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def AlignMonthlyFactor(
    factor: pd.DataFrame,
    merge_daily_info_path: str | Path,
    time_col: str,
    asset_col: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Filter covered months by their last valid trading-day stock universe.

    A prediction later than the daily-info cutoff is retained unchanged. This
    preserves a newly produced natural-month-end prediction when the daily key
    table has only been updated through an intra-month trading day.
    """

    required = {time_col, asset_col}
    missing = sorted(required.difference(factor.columns))
    if missing:
        raise ValueError(f"Factor is missing alignment columns: {missing}")

    path = Path(merge_daily_info_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"merge_daily_info file not found: {path}")
    daily_info = pd.read_parquet(path, columns=[time_col, asset_col])
    if daily_info.empty:
        raise ValueError("merge_daily_info must not be empty")
    if daily_info.duplicated([time_col, asset_col]).any():
        raise ValueError("merge_daily_info contains duplicate (time, asset) keys")

    daily_info = daily_info.copy()
    daily_info["__alignment_date"] = pd.to_datetime(
        daily_info[time_col], errors="raise"
    )
    daily_info["__alignment_asset"] = daily_info[asset_col].astype(str)
    daily_info["__alignment_month"] = daily_info["__alignment_date"].dt.to_period("M")
    cutoff = pd.Timestamp(daily_info["__alignment_date"].max())

    work = factor.copy()
    work["__alignment_date"] = pd.to_datetime(work[time_col], errors="raise")
    work["__alignment_asset"] = work[asset_col].astype(str)
    work["__alignment_month"] = work["__alignment_date"].dt.to_period("M")
    work["__alignment_order"] = range(len(work))

    historical = work[work["__alignment_date"] <= cutoff].copy()
    future = work[work["__alignment_date"] > cutoff].copy()

    latest_dates = daily_info.groupby("__alignment_month", sort=False)[
        "__alignment_date"
    ].transform("max")
    valid_keys = daily_info.loc[
        daily_info["__alignment_date"] == latest_dates,
        ["__alignment_month", "__alignment_asset"],
    ]
    missing_months = sorted(
        set(historical["__alignment_month"]).difference(valid_keys["__alignment_month"])
    )
    if missing_months:
        labels = [str(month) for month in missing_months]
        raise ValueError(f"merge_daily_info does not cover factor months: {labels}")

    aligned = historical.merge(
        valid_keys,
        on=["__alignment_month", "__alignment_asset"],
        how="inner",
        validate="many_to_one",
    )
    result = pd.concat([aligned, future], ignore_index=True)
    result = result.sort_values("__alignment_order", kind="mergesort")
    result = result[factor.columns].reset_index(drop=True)
    summary = {
        "source_path": str(path),
        "source_cutoff": cutoff.isoformat(),
        "input_rows": int(len(factor)),
        "historical_input_rows": int(len(historical)),
        "historical_output_rows": int(len(aligned)),
        "filtered_rows": int(len(historical) - len(aligned)),
        "unaligned_future_rows": int(len(future)),
        "output_rows": int(len(result)),
    }
    return result, summary


__all__ = ["AlignMonthlyFactor"]
