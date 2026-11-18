"""Next-session-close backtest. Missing held prices fail; no fabricated returns."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from backtest.config import BacktestConfig, BenchmarkConfig
from backtest.factor_processor import ProcessedFactor


@dataclass(frozen=True)
class BacktestResult:
    factor_name: str
    nav_df: pd.DataFrame
    period_return_df: pd.DataFrame
    benchmark_return_df: pd.DataFrame
    factor_return_df: pd.DataFrame
    meta: dict[str, Any]
    trading_df: pd.DataFrame = field(default_factory=pd.DataFrame)


class BacktestEngine:
    def __init__(self, backtest_cfg: BacktestConfig, benchmark_cfg: BenchmarkConfig,
                 index_quotes: pd.DataFrame | None):
        self.backtest_cfg = backtest_cfg
        self.benchmark_cfg = benchmark_cfg
        self.index_quotes = index_quotes

    @staticmethod
    def _pivot(frame, value):
        work = frame.copy()
        work["TradingDate"] = pd.to_datetime(work["TradingDate"])
        if work.duplicated(["TradingDate", "Stkcd"]).any():
            raise ValueError(f"Duplicate price/status keys for {value}")
        return work.pivot(index="TradingDate", columns="Stkcd", values=value).sort_index()

    def _signals(self, processed, calendar):
        factor = processed.factor_df.copy()
        factor["TradingDate"] = pd.to_datetime(factor["TradingDate"])
        if factor.duplicated(["TradingDate", "Stkcd"]).any():
            raise ValueError("Duplicate factor keys")
        freq = {"monthly": "M", "quarterly": "Q", "weekly": "W-FRI", "daily": "D"}[self.backtest_cfg.frequency]
        last_dates = pd.Series(calendar, index=calendar.to_period(freq)).groupby(level=0).max()
        snapshots = factor.groupby(factor.TradingDate.dt.to_period(freq)).TradingDate.max()
        signals = []
        for period, signal in snapshots.items():
            if signal != last_dates.get(period):
                raise ValueError(f"Signal {signal} is not the last observed trading session of {period}")
            idx = calendar.searchsorted(signal, side="right")
            if idx >= len(calendar):
                continue
            signals.append((pd.Timestamp(signal), pd.Timestamp(calendar[idx])))
        # Close the last signal's holding period on the next scheduled rebalance
        # when that entire calendar period and its execution session are observed.
        # This is a calendar boundary, not a fabricated prediction or price.
        if signals and self.backtest_cfg.frequency != "daily":
            terminal_period = signals[-1][0].to_period(freq) + 1
            if calendar.max().to_period(freq) > terminal_period:
                terminal_signal = last_dates.get(terminal_period)
                if terminal_signal is not None:
                    pos = calendar.searchsorted(terminal_signal, side="right")
                    if pos < len(calendar):
                        signals.append((pd.Timestamp(terminal_signal), pd.Timestamp(calendar[pos])))
        if len(signals) < 2:
            raise ValueError("Need two executable signal dates with observed prices")
        if self.backtest_cfg.frequency != "daily":
            periods = [s.to_period(freq).ordinal for s, _ in signals]
            if any(b - a != 1 for a, b in zip(periods, periods[1:])):
                raise ValueError("Missing rebalance periods; cannot report multi-period returns as monthly/weekly returns")
        return factor, signals

    @staticmethod
    def _returns(prices, entry, exit_date, stocks):
        stocks = pd.Index(stocks)
        a = prices.loc[entry].reindex(stocks)
        b = prices.loc[exit_date].reindex(stocks)
        bad = (~np.isfinite(a)) | (~np.isfinite(b)) | (a <= 0) | (b <= 0)
        if bad.any():
            sample = stocks[bad.to_numpy()].tolist()[:10]
            raise ValueError(f"Missing/nonpositive held or benchmark price {entry} -> {exit_date}: {sample}; supply verified prices/delisting treatment")
        return b / a - 1

    def _rebalance(self, holdings, cash, desired_stocks, suspension, limits):
        """Dollar holdings drift with prices; actual buys/sells consume cash and fees."""
        nav = float(holdings.sum() + cash)
        if nav <= 0:
            raise ValueError("Nonpositive portfolio wealth")
        universe = holdings.index.union(pd.Index(desired_stocks))
        current = holdings.reindex(universe, fill_value=0.0)
        desired = pd.Series(0.0, index=universe)
        if len(desired_stocks):
            desired.loc[desired_stocks] = nav / len(desired_stocks)
        sus, lim = suspension.reindex(universe), limits.reindex(universe)
        known = sus.eq(0) & lim.isin([-1, 0, 1])
        can_sell, can_buy = known & ~lim.eq(-1), known & ~lim.eq(1)
        sell = (current - desired).clip(lower=0).where(can_sell, 0.0)
        cost = self.backtest_cfg.transaction_cost
        sell_amount = float(sell.sum())
        cash += sell_amount * (1 - cost.sell)
        held = current - sell
        buy = (desired - held).clip(lower=0).where(can_buy, 0.0)
        requested = float(buy.sum())
        if requested:
            buy *= min(1.0, max(0.0, cash) / (requested * (1 + cost.buy)))
        buy_amount = float(buy.sum())
        cash -= buy_amount * (1 + cost.buy)
        if cash < -1e-10:
            raise ValueError("Negative cash after transaction costs")
        held += buy
        fees = buy_amount * cost.buy + sell_amount * cost.sell
        return held[held > 1e-14], max(0.0, cash), {
            "buy_turnover": buy_amount / nav, "sell_turnover": sell_amount / nav,
            "cost_fraction": fees / nav, "cash_weight_after_trade": max(0.0, cash) / nav,
            "blocked_orders": int((((current > desired) & ~can_sell) | ((desired > current) & ~can_buy)).sum()),
        }

    def Run(self, processed: ProcessedFactor) -> BacktestResult:
        prices = self._pivot(processed.price_df, "price")
        suspension = self._pivot(processed.suspension_limit_df, "Suspension")
        limits = self._pivot(processed.suspension_limit_df, "LimitStatus")
        if processed.trading_calendar is None:
            raise ValueError("Exchange calendar required; never infer sessions from filled prices")
        calendar = processed.trading_calendar
        calendar = calendar[(calendar >= prices.index.min()) & (calendar <= prices.index.max())]
        factor, signals = self._signals(processed, calendar)
        labels = [f"Group{i}" for i in range(1, self.backtest_cfg.groups + 1)]
        states = {label: (pd.Series(dtype=float), 1.0) for label in labels}
        history, rows, trades = {}, [], []
        for (signal, entry), (_, exit_date) in zip(signals, signals[1:]):
            today = factor[factor.TradingDate == signal].copy()
            today = today[np.isfinite(today[processed.factor_name])].sort_values("Stkcd")
            if len(today) < len(labels):
                raise ValueError(f"Too few eligible stocks on {signal}")
            today["quantile_group"] = pd.qcut(today[processed.factor_name].rank(method="first"), len(labels), labels=labels)
            a = prices.loc[entry].reindex(today.Stkcd).to_numpy()
            b = prices.loc[exit_date].reindex(today.Stkcd).to_numpy()
            today["ret"] = np.where((a > 0) & (b > 0), b / a - 1, np.nan)
            today["SignalDate"] = signal
            today["TradingDate"] = exit_date
            rows.append(today)
            if entry not in suspension.index or entry not in limits.index:
                raise ValueError(f"Missing execution-day status table: {entry}")
            sus, lim = suspension.loc[entry], limits.loc[entry]
            result = {}
            for label in labels:
                holdings, cash = states[label]
                before = float(holdings.sum() + cash)
                desired = today.loc[today.quantile_group == label, "Stkcd"].tolist()
                holdings, cash, diagnostic = self._rebalance(holdings, cash, desired, sus, lim)
                returns = self._returns(prices, entry, exit_date, holdings.index)
                holdings = holdings * (1 + returns)
                result[label] = float(holdings.sum() + cash) / before - 1
                states[label] = (holdings, cash)
                trades.append({"SignalDate": signal, "EntryDate": entry, "TradingDate": exit_date, "group": label, **diagnostic})
            result.update(self._benchmarks(processed, prices, signal, entry, exit_date))
            chosen = labels[-1] if self.backtest_cfg.direction == "high" else labels[0]
            for name in list(result):
                if name.startswith("Benchmark_"):
                    result[name.replace("Benchmark_", "Excess_")] = result[chosen] - result[name]
            history[exit_date] = result
        returns = pd.DataFrame.from_dict(history, orient="index").sort_index()
        if returns.empty or not np.isfinite(returns.to_numpy()).all():
            raise ValueError("Nonfinite or empty portfolio returns")
        nav = (1 + returns).cumprod()
        observations = pd.concat(rows, ignore_index=True)
        return BacktestResult(processed.factor_name, nav, returns,
                              returns[[c for c in returns if c.startswith("Benchmark_")]], observations,
                              {"factor_name": processed.factor_name, "start_date": str(returns.index.min()),
                               "end_date": str(returns.index.max()), "frequency": self.backtest_cfg.frequency,
                               "groups": len(labels), "risk_free_rate": self.backtest_cfg.risk_free_rate,
                               "advantage_group": chosen, "direction": self.backtest_cfg.direction,
                               "execution": "next_session_close_to_next_rebalance_close",
                               "missing_price_policy": "error", "initial_cost_charged": True,
                               "terminal_liquidation": False, "ic_rows": len(observations),
                               "ic_missing_returns": int(observations.ret.isna().sum()),
                               "assumptions": "Adjusted close; directional limit masks; no slippage/market impact. Excess compounds arithmetic return differences, not relative NAV."},
                              pd.DataFrame(trades))

    def _benchmarks(self, processed, prices, signal, entry, exit_date):
        result = {}
        if self.benchmark_cfg.has_pool_equal_weight:
            if processed.benchmark_pool is None:
                raise ValueError("Historical benchmark membership is required")
            pool = processed.benchmark_pool
            members = pool.loc[pd.to_datetime(pool.TradingDate) == signal, "Stkcd"].unique()
            if not len(members):
                raise ValueError(f"No benchmark membership on {signal}")
            result["Benchmark_pool"] = float(self._returns(prices, entry, exit_date, members).mean())
        for code in self.benchmark_cfg.index_codes:
            if self.index_quotes is None:
                raise ValueError("Index quotes are required")
            q = self.index_quotes.copy()
            q = q[q.Indexcd.astype(str).str.zfill(6) == str(code).zfill(6)]
            q["TradingDate"] = pd.to_datetime(q.TradingDate)
            s = q.set_index("TradingDate")["Close"]
            if s.index.has_duplicates:
                raise ValueError("Duplicate index price keys")
            a, b = s.get(entry, np.nan), s.get(exit_date, np.nan)
            if not np.isfinite([a, b]).all() or min(a, b) <= 0:
                raise ValueError(f"Missing index prices for {code}: {entry}, {exit_date}")
            result[f"Benchmark_index_{code}"] = float(b / a - 1)
        return result
