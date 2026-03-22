"""
research/cointegration.py — Engle-Granger 协整检验
"""

import pandas as pd
from statsmodels.tsa.stattools import coint


def run_cointegration_test(
    pt_prices: pd.Series,
    pd_prices: pd.Series,
    significance: float = 0.05,
) -> tuple[bool, float]:
    """
    检验 PT 和 PD 是否协整（价差不会无限漂移）。

    返回 (is_cointegrated, p_value)
      p < 0.05 → 协整成立，策略统计基础有效
      p > 0.05 → 协整不成立，建议暂停开新仓
    """
    if len(pt_prices) < 60:
        return True, 0.0   # 数据不足，默认放行

    _, pvalue, _ = coint(pt_prices.values, pd_prices.values)
    is_coint = pvalue < significance
    print(f"  🔬 协整检验 | p={pvalue:.4f}  "
          f"{'✅ 协整成立' if is_coint else '❌ 协整不成立'}")
    return is_coint, float(pvalue)
