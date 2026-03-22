"""
research/backtest_engine.py — 独立回测引擎（不依赖 TqSdk）
用于参数扫描和策略验证，价格用 O-U 模型模拟或传入真实 CSV。
"""

import itertools
import numpy as np
import pandas as pd
from dataclasses import dataclass

from config import PT_UNIT, PARAM_GRID

# ---- 交易成本 ----
COMMISSION   = 10      # 元/手（单边）
SLIPPAGE_CNY = 0.5     # CNY/克，单边


@dataclass
class BacktestResult:
    lookback:      int
    z_entry:       float
    z_exit:        float
    total_ret_pct: float
    annual_ret_pct: float
    max_dd_pct:    float
    sharpe:        float
    n_trades:      int
    win_rate_pct:  float
    payoff_ratio:  float
    final_balance: float


def generate_prices(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    """生成模拟 PT / PD 价格（O-U 价差 + GBM 铂价）。"""
    rng = np.random.default_rng(seed)
    pt  = np.empty(n); pt[0] = 500.0
    eps_pt = rng.standard_normal(n)
    for i in range(1, n):
        pt[i] = pt[i-1] * np.exp((-0.5*0.015**2) + 0.015*eps_pt[i])

    spread = np.empty(n); spread[0] = 140.0
    eps_s  = rng.standard_normal(n)
    for i in range(1, n):
        spread[i] = spread[i-1] + 0.015*(140 - spread[i-1]) + 1.8*eps_s[i]

    return pd.DataFrame({"pt_close": pt, "pd_close": pt - spread, "spread": spread})


def run_backtest(
    prices: pd.DataFrame,
    lookback: int,
    z_entry: float,
    z_exit: float,
    lots: int = 2,
    init_balance: float = 3_000_000,
) -> BacktestResult:
    spread = prices["spread"].values
    n      = len(spread)
    balance = init_balance
    balance_series = np.full(n, np.nan)
    position = 0
    trades: list[dict] = []
    entry_price = 0.0
    cost = (COMMISSION + SLIPPAGE_CNY * PT_UNIT) * 2   # PT + PD 双边

    roll_mean = pd.Series(spread).rolling(lookback, min_periods=lookback//2).mean().values
    roll_std  = pd.Series(spread).rolling(lookback, min_periods=lookback//2).std().values

    for i in range(n):
        balance_series[i] = balance
        if np.isnan(roll_mean[i]) or roll_std[i] == 0:
            continue
        z = (spread[i] - roll_mean[i]) / roll_std[i]

        if position != 0 and abs(z) < z_exit:
            pnl = position * (spread[i] - entry_price) * PT_UNIT * lots - cost * lots
            balance += pnl
            trades.append({"pnl": pnl})
            position = 0

        if position == 0:
            if z > z_entry:
                position, entry_price = -1, spread[i]
                balance -= cost * lots
            elif z < -z_entry:
                position, entry_price = +1, spread[i]
                balance -= cost * lots

        balance_series[i] = balance

    # 强平
    if position != 0:
        pnl = position * (spread[-1] - entry_price) * PT_UNIT * lots - cost * lots
        balance += pnl
        trades.append({"pnl": pnl})

    return _calc_metrics(balance_series, trades, lookback, z_entry, z_exit,
                         init_balance, balance)


def _calc_metrics(bal_series, trades, lookback, z_entry, z_exit,
                  init_balance, final_balance) -> BacktestResult:
    bal = bal_series[~np.isnan(bal_series)]
    if len(bal) < 10:
        return BacktestResult(lookback, z_entry, z_exit, 0,0,0, -99, 0,0,0, init_balance)

    total_ret  = (final_balance - init_balance) / init_balance
    bars_per_yr = 252 * 6
    annual_ret  = (1 + total_ret) ** (bars_per_yr / len(bal)) - 1
    peak = np.maximum.accumulate(bal)
    max_dd = ((bal - peak) / peak).min()
    bar_rets = np.diff(bal) / bal[:-1]
    sharpe   = bar_rets.mean() / bar_rets.std() * np.sqrt(bars_per_yr) if bar_rets.std() > 0 else np.nan

    pnls     = [t["pnl"] for t in trades]
    wins     = [p for p in pnls if p > 0]
    losses   = [p for p in pnls if p <= 0]
    n_tr     = len(pnls)
    win_rate = len(wins) / n_tr * 100 if n_tr else np.nan
    payoff   = abs(np.mean(wins) / np.mean(losses)) if wins and losses else np.nan

    return BacktestResult(
        lookback       = lookback,
        z_entry        = z_entry,
        z_exit         = z_exit,
        total_ret_pct  = round(total_ret * 100, 2),
        annual_ret_pct = round(annual_ret * 100, 2),
        max_dd_pct     = round(max_dd * 100, 2),
        sharpe         = round(float(sharpe), 3) if not np.isnan(sharpe) else np.nan,
        n_trades       = n_tr,
        win_rate_pct   = round(win_rate, 1) if not np.isnan(win_rate) else np.nan,
        payoff_ratio   = round(payoff, 2)   if not np.isnan(payoff)   else np.nan,
        final_balance  = round(final_balance),
    )


def run_scan(prices: pd.DataFrame) -> pd.DataFrame:
    """网格搜索，返回按 Sharpe 排序的结果 DataFrame。"""
    results = []
    for lk, ze, zx in itertools.product(
        PARAM_GRID["lookback"], PARAM_GRID["z_entry"], PARAM_GRID["z_exit"]
    ):
        r = run_backtest(prices, lk, ze, zx)
        results.append(r.__dict__)
    return pd.DataFrame(results).sort_values("sharpe", ascending=False)


def quick_sharpe(
    spread: np.ndarray,
    lookback: int,
    z_entry: float,
    z_exit: float,
) -> float:
    """单次轻量回测，仅返回 Sharpe，供参数优化器调用。"""
    roll_mean = pd.Series(spread).rolling(lookback, min_periods=lookback//2).mean().values
    roll_std  = pd.Series(spread).rolling(lookback, min_periods=lookback//2).std().values
    bal, pos, ep, bals = 1.0, 0, 0.0, []
    cost = 0.001
    for i in range(len(spread)):
        if np.isnan(roll_mean[i]) or roll_std[i] == 0:
            continue
        z = (spread[i] - roll_mean[i]) / roll_std[i]
        if pos != 0 and abs(z) < z_exit:
            bal += pos * (spread[i] - ep) - cost; pos = 0
        if pos == 0:
            if   z >  z_entry: pos, ep = -1, spread[i]; bal -= cost
            elif z < -z_entry: pos, ep = +1, spread[i]; bal -= cost
        bals.append(bal)
    if len(bals) < 10:
        return -np.inf
    r = np.diff(bals)
    return float(r.mean() / r.std() * np.sqrt(252*6)) if r.std() > 0 else -np.inf


if __name__ == "__main__":
    prices = generate_prices()
    df     = run_scan(prices)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(df.head(10).to_string(index=False))
    df.to_csv("research/scan_results.csv", index=False)
    print("已保存至 research/scan_results.csv")
