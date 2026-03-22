"""
广期所铂钯价差套利 — 独立回测引擎（不依赖 TqSdk）
====================================================

价格模型：
  - PT 价格：几何布朗运动（GBM），起点 240 CNY/克
  - PT-PD 价差：O-U 均值回归过程（模拟跨品种价差的典型行为）
      mu    = 30 CNY/克  （铂长期相对钯的溢价）
      theta = 0.015      （均值回归速度，每小时）
      sigma = 1.8        （价差波动率，CNY/克）

交易成本：
  - 手续费：双边各 10 元/手（参考广期所标准）
  - 滑点：0.5 CNY/克 × 1000克 = 500 元/手
"""

import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass, field


# ============================================================
# 合约参数
# ============================================================

PT_UNIT      = 1000    # 克/手
PD_UNIT      = 1000    # 克/手
COMMISSION   = 10      # 元/手（单边）
SLIPPAGE_CNY = 0.5     # CNY/克，单边滑点
INIT_BALANCE = 500_000 # 元

SEED         = 42
N_HOURS      = 1_200   # 模拟小时数（约5个月交易时间）


# ============================================================
# 价格模型
# ============================================================

def generate_prices(n: int = N_HOURS, seed: int = SEED) -> pd.DataFrame:
    """
    生成模拟的 PT / PD 每小时收盘价序列。

    PT  : 几何布朗运动，年化波动率 25%
    价差 : O-U 均值回归过程
           mu=30, theta=0.015/h, sigma=1.8/h^0.5
    PD  : PT - 价差
    """
    rng = np.random.default_rng(seed)
    dt  = 1.0  # 1小时步长

    # PT 价格（GBM）
    mu_pt    = 0.0001    # 每小时漂移（约年化+2.5%）
    sigma_pt = 0.015     # 每小时波动率
    pt = np.empty(n)
    pt[0] = 240.0
    eps_pt = rng.standard_normal(n)
    for i in range(1, n):
        pt[i] = pt[i-1] * np.exp((mu_pt - 0.5 * sigma_pt**2) * dt
                                  + sigma_pt * np.sqrt(dt) * eps_pt[i])

    # 价差（O-U 过程）：模拟铂钯价差的均值回归特性
    theta = 0.015   # 均值回归速度
    mu_s  = 30.0    # 长期均值（CNY/克）
    sig_s = 1.8     # 波动率（CNY/克/h^0.5）
    spread = np.empty(n)
    spread[0] = mu_s
    eps_s = rng.standard_normal(n)
    for i in range(1, n):
        spread[i] = (spread[i-1]
                     + theta * (mu_s - spread[i-1]) * dt
                     + sig_s * np.sqrt(dt) * eps_s[i])

    pd_price = pt - spread

    return pd.DataFrame({
        "pt_close": pt,
        "pd_close": pd_price,
        "spread":   spread,
    })


# ============================================================
# 回测引擎
# ============================================================

@dataclass
class Trade:
    direction: int      # +1 做多价差, -1 做空价差
    entry_spread: float
    entry_balance: float
    exit_spread: float  = 0.0
    exit_balance: float = 0.0
    pnl: float          = 0.0
    closed: bool        = False


def run_backtest(
    prices: pd.DataFrame,
    lookback: int,
    z_entry: float,
    z_exit: float,
    lots: int = 1,
) -> dict:
    """
    在价格序列上回测价差策略，返回绩效指标。

    持仓逻辑：
      z > +z_entry → 做空价差（卖PT买PD）
      z < -z_entry → 做多价差（买PT卖PD）
      |z| < z_exit → 平仓
    """
    spread = prices["spread"].values
    n      = len(spread)

    balance = INIT_BALANCE
    balance_series = np.empty(n)
    balance_series[:] = np.nan

    position = 0     # +1 多价差, -1 空价差, 0 空仓
    trades: list[Trade] = []
    entry_balance = INIT_BALANCE

    # 每手开平仓成本
    cost_per_open  = (COMMISSION + SLIPPAGE_CNY * PT_UNIT) * 2   # PT + PD 各一手
    cost_per_close = cost_per_open

    # 滚动均值和std
    roll_mean = pd.Series(spread).rolling(lookback, min_periods=lookback // 2).mean().values
    roll_std  = pd.Series(spread).rolling(lookback, min_periods=lookback // 2).std().values

    for i in range(n):
        balance_series[i] = balance

        if np.isnan(roll_mean[i]) or roll_std[i] == 0 or np.isnan(roll_std[i]):
            continue

        z = (spread[i] - roll_mean[i]) / roll_std[i]

        # 平仓
        if position != 0 and abs(z) < z_exit:
            pnl_raw = position * (spread[i] - trades[-1].entry_spread) * PT_UNIT * lots
            pnl_net = pnl_raw - cost_per_close * lots
            balance += pnl_net
            balance_series[i] = balance
            trades[-1].exit_spread  = spread[i]
            trades[-1].exit_balance = balance
            trades[-1].pnl          = pnl_net
            trades[-1].closed       = True
            position = 0

        # 开仓（只在空仓时）
        elif position == 0:
            if z > z_entry:
                position = -1    # 做空价差
                entry_balance = balance
                balance -= cost_per_open * lots
                trades.append(Trade(-1, spread[i], entry_balance))
            elif z < -z_entry:
                position = +1    # 做多价差
                entry_balance = balance
                balance -= cost_per_open * lots
                trades.append(Trade(+1, spread[i], entry_balance))

    # 若持仓未平，按最后一根K线收盘价强平
    if position != 0 and trades:
        pnl_raw = position * (spread[-1] - trades[-1].entry_spread) * PT_UNIT * lots
        pnl_net = pnl_raw - cost_per_close * lots
        balance += pnl_net
        trades[-1].exit_spread  = spread[-1]
        trades[-1].exit_balance = balance
        trades[-1].pnl          = pnl_net
        trades[-1].closed       = True

    return _calc_metrics(balance_series, trades, lookback, z_entry, z_exit)


def _calc_metrics(
    balance_series: np.ndarray,
    trades: list[Trade],
    lookback: int,
    z_entry: float,
    z_exit: float,
) -> dict:
    bal  = balance_series[~np.isnan(balance_series)]
    if len(bal) < 10:
        return {}

    init  = bal[0]
    final = bal[-1]
    total_ret = (final - init) / init

    bars_per_year = 252 * 6     # 约252交易日，每日约6小时
    n_bars = len(bal)
    annual_ret = (1 + total_ret) ** (bars_per_year / n_bars) - 1

    # 最大回撤
    peak = np.maximum.accumulate(bal)
    dd   = (bal - peak) / peak
    max_dd = dd.min()

    # Sharpe
    bar_rets = np.diff(bal) / bal[:-1]
    sharpe   = (bar_rets.mean() / bar_rets.std() * np.sqrt(bars_per_year)
                if bar_rets.std() > 0 else np.nan)

    # 交易统计
    closed = [t for t in trades if t.closed]
    n_tr   = len(closed)
    if n_tr > 0:
        pnls     = [t.pnl for t in closed]
        wins     = [p for p in pnls if p > 0]
        losses   = [p for p in pnls if p <= 0]
        win_rate = len(wins) / n_tr * 100
        avg_win  = np.mean(wins)  if wins   else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        payoff   = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    else:
        win_rate = payoff = np.nan

    return {
        "lookback":      lookback,
        "z_entry":       z_entry,
        "z_exit":        z_exit,
        "total_ret_%":   round(total_ret  * 100, 2),
        "annual_ret_%":  round(annual_ret * 100, 2),
        "max_dd_%":      round(max_dd     * 100, 2),
        "sharpe":        round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        "n_trades":      n_tr,
        "win_rate_%":    round(win_rate, 1) if not np.isnan(win_rate) else np.nan,
        "payoff_ratio":  round(payoff,  2)  if not np.isnan(payoff)  else np.nan,
        "final_balance": round(final),
    }


# ============================================================
# 参数扫描
# ============================================================

PARAM_GRID = {
    "lookback": [60, 90, 120, 180],
    "z_entry":  [1.5, 2.0, 2.5, 3.0],
    "z_exit":   [0.3, 0.5, 0.8],
}


def run_scan(prices: pd.DataFrame) -> pd.DataFrame:
    combos = list(itertools.product(
        PARAM_GRID["lookback"],
        PARAM_GRID["z_entry"],
        PARAM_GRID["z_exit"],
    ))

    results = []
    for lookback, z_entry, z_exit in combos:
        m = run_backtest(prices, lookback, z_entry, z_exit)
        if m:
            results.append(m)

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    return df


if __name__ == "__main__":
    prices = generate_prices()
    df     = run_scan(prices)

    pd.set_option("display.max_rows", 60)
    pd.set_option("display.width",    120)
    pd.set_option("display.float_format", "{:.2f}".format)

    print("\n======================================================================")
    print("参数扫描结果（按 Sharpe 降序，共 48 组）")
    print("======================================================================")
    print(df.to_string(index=False))

    print("\n─── Top 10（Sharpe 最高）────────────────────────────────────────────")
    print(df.head(10).to_string(index=False))

    print("\n─── Bottom 5（Sharpe 最低）──────────────────────────────────────────")
    print(df.tail(5).to_string(index=False))

    df.to_csv("scan_results.csv", index=False)
    print("\n完整结果已保存至 scan_results.csv")
