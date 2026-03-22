"""
广期所铂钯价差套利 — 参数敏感性扫描
======================================

对以下参数做网格搜索，每组参数跑一次完整回测，
最后输出汇总表格，找出稳健的参数区间。

扫描参数：
  lookback : 滚动均值/std 的窗口长度（K线根数）
  z_entry  : 开仓 Z-score 阈值
  z_exit   : 平仓 Z-score 阈值

输出指标：
  总收益率、年化收益率、最大回撤、Sharpe 比率、
  交易次数、胜率、平均盈亏比

用法：
  python param_scan.py
  → 完成后在当前目录生成 scan_results.csv
"""

import itertools
import numpy as np
import pandas as pd
from datetime import date
from tqsdk import TqApi, TqAuth, TqSim, TqBacktest, TargetPosTask

# ============================================================
# 合约参数（固定）
# ============================================================

PT_SYMBOL   = "KQ.m@GFEX.pt"
PD_SYMBOL   = "KQ.m@GFEX.pd"
BAR_DURATION = 3600          # 1小时K线
INIT_BALANCE = 500_000       # 初始资金（元）
MAX_LOTS_PT  = 1             # 每次最多开 PT 的手数

BACKTEST_START = date(2024, 11, 1)
BACKTEST_END   = date(2025, 6, 1)

# ============================================================
# 扫描参数网格
# ============================================================

PARAM_GRID = {
    "lookback": [60, 90, 120, 180],    # 窗口长度（小时）
    "z_entry":  [1.5, 2.0, 2.5, 3.0], # 开仓阈值
    "z_exit":   [0.3, 0.5, 0.8],      # 平仓阈值
}


# ============================================================
# 工具函数
# ============================================================

def calc_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window // 2).mean()
    std  = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / std.replace(0, np.nan)


def calc_metrics(
    balance_series: list[float],
    trades: list[dict],
    bar_duration_sec: int = 3600,
) -> dict:
    """
    从账户净值序列和交易记录计算绩效指标。

    balance_series : 每根K线末尾采样的账户净值列表
    trades         : 每笔已平仓交易的 {"pnl": float} 字典列表
    """
    if len(balance_series) < 2:
        return {"error": "数据不足"}

    bal = np.array(balance_series, dtype=float)
    init = bal[0]
    final = bal[-1]

    # 总收益率
    total_return = (final - init) / init

    # 年化收益率（以交易小时数估算）
    n_bars = len(bal)
    bars_per_year = 252 * 6.5   # 约252交易日，每日约6.5小时
    annual_return = (1 + total_return) ** (bars_per_year / n_bars) - 1

    # 最大回撤
    rolling_max = np.maximum.accumulate(bal)
    drawdown = (bal - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # 逐 bar 收益率 → Sharpe（年化）
    bar_returns = np.diff(bal) / bal[:-1]
    if bar_returns.std() > 0:
        sharpe = (bar_returns.mean() / bar_returns.std()) * np.sqrt(bars_per_year)
    else:
        sharpe = np.nan

    # 交易统计
    n_trades = len(trades)
    if n_trades > 0:
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / n_trades
        avg_win  = np.mean(wins)  if wins   else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        payoff   = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    else:
        win_rate = avg_win = avg_loss = payoff = np.nan

    return {
        "total_return":  round(total_return * 100, 2),   # %
        "annual_return": round(annual_return * 100, 2),  # %
        "max_drawdown":  round(max_drawdown * 100, 2),   # %
        "sharpe":        round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        "n_trades":      n_trades,
        "win_rate":      round(win_rate * 100, 1) if not np.isnan(win_rate) else np.nan,  # %
        "payoff_ratio":  round(payoff, 2) if not np.isnan(payoff) else np.nan,
    }


# ============================================================
# 单次回测
# ============================================================

def run_single_backtest(
    lookback: int,
    z_entry: float,
    z_exit: float,
) -> dict:
    """
    用给定参数跑一次完整回测，返回绩效指标字典。
    """
    api = TqApi(
        TqSim(init_balance=INIT_BALANCE),
        backtest=TqBacktest(start_dt=BACKTEST_START, end_dt=BACKTEST_END),
        auth=TqAuth("快期账户", "账户密码"),
    )

    pt_klines = api.get_kline_serial(PT_SYMBOL, BAR_DURATION, data_length=lookback + 20)
    pd_klines = api.get_kline_serial(PD_SYMBOL, BAR_DURATION, data_length=lookback + 20)
    account   = api.get_account()
    pt_pos    = api.get_position(PT_SYMBOL)

    pt_task = TargetPosTask(api, PT_SYMBOL, price="ACTIVE")
    pd_task = TargetPosTask(api, PD_SYMBOL, price="ACTIVE")

    balance_series: list[float] = []
    trades: list[dict] = []
    entry_balance: float | None = None   # 开仓时的账户净值，用于计算单笔盈亏

    try:
        while True:
            api.wait_update()

            new_bar = (
                api.is_changing(pt_klines.iloc[-1], "datetime")
                or api.is_changing(pd_klines.iloc[-1], "datetime")
            )
            if not new_bar:
                continue

            # 采样净值
            balance_series.append(account.balance)

            # ---- 计算信号 ----
            gfex_spread   = pt_klines["close"] - pd_klines["close"]
            ref_spread    = gfex_spread.rolling(lookback).mean().iloc[-1]
            zscore_series = calc_zscore(gfex_spread - ref_spread, lookback)
            z = zscore_series.iloc[-1]

            if np.isnan(z):
                continue

            pt_net = pt_pos.pos_long - pt_pos.pos_short

            # ---- 开仓 ----
            if z > z_entry and pt_net == 0:
                pt_task.set_target_volume(-MAX_LOTS_PT)
                pd_task.set_target_volume(MAX_LOTS_PT)
                entry_balance = account.balance

            elif z < -z_entry and pt_net == 0:
                pt_task.set_target_volume(MAX_LOTS_PT)
                pd_task.set_target_volume(-MAX_LOTS_PT)
                entry_balance = account.balance

            # ---- 平仓 ----
            elif abs(z) < z_exit and pt_net != 0:
                pt_task.set_target_volume(0)
                pd_task.set_target_volume(0)
                if entry_balance is not None:
                    trades.append({"pnl": account.balance - entry_balance})
                    entry_balance = None

    except Exception:
        # 回测数据耗尽时 TqSdk 会抛出异常结束循环
        pass
    finally:
        api.close()

    metrics = calc_metrics(balance_series, trades)
    metrics.update({
        "lookback": lookback,
        "z_entry":  z_entry,
        "z_exit":   z_exit,
    })
    return metrics


# ============================================================
# 参数扫描主流程
# ============================================================

def run_scan():
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    print(f"开始参数扫描，共 {total} 组参数组合...\n")

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"[{i:>3}/{total}] lookback={params['lookback']:>3}  "
              f"z_entry={params['z_entry']:.1f}  z_exit={params['z_exit']:.1f}  ", end="", flush=True)

        try:
            metrics = run_single_backtest(**params)
            results.append(metrics)
            print(
                f"收益={metrics.get('total_return', 'N/A'):>6}%  "
                f"回撤={metrics.get('max_drawdown', 'N/A'):>6}%  "
                f"Sharpe={metrics.get('sharpe', 'N/A'):>5}  "
                f"交易={metrics.get('n_trades', 'N/A'):>3}次"
            )
        except Exception as e:
            print(f"失败: {e}")
            results.append({**params, "error": str(e)})

    # ---- 汇总输出 ----
    df = pd.DataFrame(results)

    # 按 Sharpe 降序排列
    if "sharpe" in df.columns:
        df = df.sort_values("sharpe", ascending=False)

    print("\n" + "=" * 80)
    print("参数扫描结果（按 Sharpe 降序）")
    print("=" * 80)

    display_cols = [
        "lookback", "z_entry", "z_exit",
        "total_return", "annual_return", "max_drawdown",
        "sharpe", "n_trades", "win_rate", "payoff_ratio",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    print(df[existing_cols].to_string(index=False))

    csv_path = "scan_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存至 {csv_path}")

    # ---- 打印 Top 5 ----
    print("\nTop 5 参数组合（Sharpe 最高）：")
    if "sharpe" in df.columns:
        top5 = df[existing_cols].dropna(subset=["sharpe"]).head(5)
        print(top5.to_string(index=False))

    return df


if __name__ == "__main__":
    run_scan()
