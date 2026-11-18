"""Replay the archived FT seed-42 book with same-date flags and scheduled closes."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / "02_backtest" / "helpers"), str(BASE / "01_model" / "project" / "src")]
import calendar_recheck as cal
from audit_months import read_slice
from valuation_fix import adjacent_returns, make_engine
from backtest.config import BacktestConfig, BenchmarkConfig, TransactionCostConfig
from backtest.engine import BacktestEngine


def run(data, output):
    output.mkdir(parents=True, exist_ok=False)
    keys = ["TradingDate", "Stkcd"]
    cal.START, cal.VALUE_END = "2024-01-01", "2025-11-30"
    lo, hi = pd.Timestamp(cal.START), pd.Timestamp("2025-12-05")
    prices = read_slice(data/"adj_close.parquet", keys+["price"], lo, hi)
    prices = prices.loc[prices.TradingDate <= cal.VALUE_END]
    flags = read_slice(data/"Suspension_Limit.parquet", keys+["Suspension", "LimitStatus"], lo, hi)
    pools = []
    for name in ["history-new.pq", "history-1800.pq"]:
        pool = pd.read_parquet(data/name)
        pool.index = pd.to_datetime(pool.index)
        aligned, _ = cal.complete_month_ends(pool.loc[cal.START:cal.VALUE_END], prices.loc[prices.price.gt(0), "TradingDate"])
        pools.append(aligned)
    cfg = BacktestConfig(cal.START, "2025-10-31", "monthly", 10, None, .02, TransactionCostConfig())
    signal = BASE/"03_predictions"/"ft_transformer_s42.parquet"
    prep = cal.make_processor(cfg, *pools).LoadAndProcess(signal, prices, flags)
    flags.Stkcd = flags.Stkcd.astype(int)
    same = prep.suspension_limit_df[keys].merge(flags, on=keys, how="left", validate="one_to_one", indicator=True)
    if not same._merge.eq("both").all():
        raise ValueError("Missing same-date trading status")
    adjusted = make_engine(BacktestEngine, prices, output)(cfg, BenchmarkConfig([]), None).Run(
        replace(prep, suspension_limit_df=same.drop(columns="_merge")))
    held = adjusted.factor_return_df.query("quantile_group == 'Group10'")[keys].merge(
        adjacent_returns(prices), on=keys, validate="one_to_one")
    held["month"] = (held.TradingDate.dt.to_period("M")+1).astype(str)
    schedule = pd.read_csv(BASE/"03_predictions"/"schedule.csv", parse_dates=["entry","exit"])
    held = held.merge(schedule[["holding_month","entry","exit"]], left_on="month", right_on="holding_month", validate="many_to_one")
    quotes = read_slice(data/"adj_close.parquet", keys+["price"], lo, hi).set_index(keys, verify_integrity=True).price
    for date in ["entry", "exit"]:
        values = quotes.reindex(pd.MultiIndex.from_arrays([held[date],held.Stkcd])).to_numpy()
        if not (np.isfinite(values) & (values > 0)).all():
            raise ValueError(f"Missing held price at {date}")
        held[date+"_price"] = values
    held["new_return"] = held.exit_price/held.entry_price-1
    book = held.groupby("month").agg(holdings=("Stkcd","size"), old_gross=("ret","mean"), new_gross=("new_return","mean"))
    legacy_net = adjusted.period_return_df.Group10.copy()
    legacy_net.index = (legacy_net.index.to_period("M")+1).astype(str)
    expected = list(pd.period_range("2024-02","2025-11",freq="M").astype(str))
    if list(book.index)!=expected or list(legacy_net.index)!=expected:
        raise ValueError("Incomplete or misaligned monthly series")
    book["legacy_cost"] = book.old_gross - legacy_net
    book["repriced_net"] = book.new_gross - book.legacy_cost
    book.to_csv(output/"monthly.csv")
    held[keys+["month","entry","exit","entry_price","exit_price","new_return"]].to_parquet(output/"positions.parquet", index=False)
    reference = pd.read_csv(BASE/"04_results"/"monthly.csv", index_col="month")
    pd.testing.assert_frame_equal(book, reference[book.columns], check_exact=False, atol=1e-12, rtol=1e-12)
    excess = book.repriced_net - (1.02**(1/12)-1)
    ratio = np.sqrt(12)*excess.mean()/excess.std(ddof=1)
    print(f"Verified {len(book)} monthly net returns; annualized Sharpe = {ratio:.12f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("F:/aki/mlp_factor/data/source_data"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.data,args.output)
