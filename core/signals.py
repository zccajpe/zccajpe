"""
core/signals.py — 信号计算：Z-score、Hurst 指数、市场状态识别
"""

import numpy as np
import pandas as pd


def calc_zscore(series: pd.Series, window: int) -> pd.Series:
    """滚动 Z-score，窗口内数据不足时返回 NaN。"""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std  = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / std.replace(0, np.nan)


def calc_divergence(gfex_spread: pd.Series, lme_spread_cny: float) -> pd.Series:
    """广期所价差与参考价差的偏离量。"""
    return gfex_spread - lme_spread_cny


def calc_hurst(series: pd.Series) -> float:
    """
    计算 Hurst 指数（方差法）。
      H < 0.45 → 均值回归
      H > 0.55 → 趋势持续
      0.45~0.55 → 不确定
    """
    ts = series.dropna().values
    if len(ts) < 30:
        return 0.5

    lags  = range(2, min(20, len(ts) // 3))
    tau   = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    valid = [(l, t) for l, t in zip(lags, tau) if t > 0]
    if len(valid) < 3:
        return 0.5

    h = float(np.polyfit(
        np.log([v[0] for v in valid]),
        np.log([v[1] for v in valid]),
        1
    )[0])
    return float(np.clip(h, 0.0, 1.0))


def detect_regime(spread: pd.Series, hurst_window: int = 60) -> str:
    """
    返回 'mean_reversion' / 'trend' / 'uncertain'
    """
    h = calc_hurst(spread.iloc[-hurst_window:])
    if h < 0.45:
        return "mean_reversion"
    elif h > 0.55:
        return "trend"
    return "uncertain"
