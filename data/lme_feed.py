"""
data/lme_feed.py — LME 铂钯现货价差数据接口
TqSdk 不提供 LME 数据，通过此接口隔离外部数据源。
"""

import pandas as pd
from config import TROY_OZ_TO_GRAM


class LMEDataFeed:
    """
    接入方式：
      方案A：Wind Python API
      方案B：彭博 / 路透
      方案C：手动维护 CSV（每日更新）

    CSV 格式：
      datetime,lme_pt_usd,lme_pd_usd
      2025-01-02,950.5,920.0
      ...
    价格单位：USD/金衡盎司
    """

    def __init__(self, source: str = "csv", csv_path: str = "lme_spread.csv"):
        self._df: pd.DataFrame | None = None
        if source == "csv":
            try:
                self._df = pd.read_csv(csv_path, index_col="datetime", parse_dates=True)
                print(f"LME 数据加载成功，共 {len(self._df)} 条记录")
            except FileNotFoundError:
                print(f"警告：找不到 {csv_path}，降级为自身均值回归模式")

    def get_spread_cny_per_gram(self, usd_cny_rate: float) -> float | None:
        """
        返回最新 LME 铂钯价差（CNY/克）。
        公式：(lme_pt - lme_pd) * usd_cny / 31.1035
        """
        if self._df is None or len(self._df) == 0:
            return None
        row = self._df.iloc[-1]
        spread_usd = float(row["lme_pt_usd"]) - float(row["lme_pd_usd"])
        return spread_usd * usd_cny_rate / TROY_OZ_TO_GRAM
