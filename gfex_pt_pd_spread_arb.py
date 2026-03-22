"""
广期所铂钯跨品种价差套利策略
=====================================

策略逻辑：
  1. 计算广期所铂(PT) - 钯(PD) 的价差（CNY/克）
  2. 对比 LME 铂钯价差折算成 CNY/克（汇率调整后）
  3. 计算两个价差的偏离度并做滚动 Z-score
  4. Z-score 超过阈值时押注回归：
       偏离 > +2σ → 卖PT买PD（做空广期所相对高估的价差）
       偏离 < -2σ → 买PT卖PD（做多广期所相对低估的价差）
  5. Z-score 回归至 ±0.5σ 内时平仓

合约规格（广期所，2024年11月上市）：
  PT 铂：1000克/手，报价单位 CNY/克
  PD 钯：1000克/手，报价单位 CNY/克
  克重对冲比：1手PT 对冲 1手PD（两者合约乘数相同）

LME 数据说明：
  TqSdk 不直接提供 LME 数据，需要外部接入（彭博/Wind/CSV等）。
  代码通过 LMEDataFeed 接口隔离，无 LME 数据时自动降级为
  "GFEX 价差自身滚动均值回归"模式，仍然可以运行和回测。
"""

import numpy as np
import pandas as pd
from datetime import date, datetime
from contextlib import closing

from tqsdk import TqApi, TqAuth, TqSim, TqBacktest, TargetPosTask
from tqsdk.tools import DataDownloader


# ============================================================
# 合约参数
# ============================================================

PT_SYMBOL = "KQ.m@GFEX.pt"   # 铂主连
PD_SYMBOL = "KQ.m@GFEX.pd"   # 钯主连

PT_UNIT = 1000   # 铂：1000克/手
PD_UNIT = 1000   # 钯：1000克/手

# 克重中性对冲比：1手PT = 1手PD（两者合约乘数相同）
HEDGE_RATIO = PT_UNIT // PD_UNIT   # 1

TROY_OZ_TO_GRAM = 31.1035   # 1金衡盎司 = 31.1035克


# ============================================================
# 策略参数（可调整）
# ============================================================

BAR_DURATION = 3600    # K线周期：3600秒 = 1小时
LOOKBACK     = 120     # 滚动窗口：120根K线（约5个交易日）
Z_ENTRY      = 2.0     # 开仓 Z-score 阈值
Z_EXIT       = 0.5     # 平仓 Z-score 阈值
MAX_LOTS_PT  = 1       # 每次最多开 PT 的手数（PD = MAX_LOTS_PT × HEDGE_RATIO）


# ============================================================
# LME 数据接口（需要外部接入，TqSdk 不提供）
# ============================================================

class LMEDataFeed:
    """
    LME 铂钯现货价差数据接口。

    TqSdk 不直接提供 LME 数据，可以通过以下方式接入：
      - 方案A：Wind Python API（推荐国内用户）
      - 方案B：彭博 / 路透 数据订阅
      - 方案C：手动维护 CSV 文件，每日更新一次

    CSV 格式示例（lme_spread.csv）：
      datetime,lme_pt_usd,lme_pd_usd
      2025-01-02,950.5,920.0
      2025-01-03,955.0,918.5
      ...
    其中价格单位为 USD/金衡盎司。
    """

    def __init__(self, source: str = "csv", csv_path: str = "lme_spread.csv"):
        self.source = source
        self._df: pd.DataFrame | None = None

        if source == "csv":
            try:
                self._df = pd.read_csv(csv_path, index_col="datetime", parse_dates=True)
                print(f"LME 数据加载成功，共 {len(self._df)} 条记录")
            except FileNotFoundError:
                print(f"警告：找不到 LME 数据文件 {csv_path}，将降级为自身均值回归模式")

    def get_spread_cny_per_gram(self, usd_cny_rate: float) -> float | None:
        """
        获取最新 LME 铂钯价差，折算为 CNY/克。

        折算公式：
          spread_cny_per_gram = (lme_pt_usd - lme_pd_usd) * usd_cny_rate / 31.1035

        返回 None 表示数据不可用（降级为自身均值回归）。
        """
        if self._df is None or len(self._df) == 0:
            return None

        latest = self._df.iloc[-1]
        spread_usd_per_oz = float(latest["lme_pt_usd"]) - float(latest["lme_pd_usd"])
        return spread_usd_per_oz * usd_cny_rate / TROY_OZ_TO_GRAM


# ============================================================
# 核心计算函数
# ============================================================

def calc_gfex_spread(pt_klines: pd.DataFrame, pd_klines: pd.DataFrame) -> pd.Series:
    """
    计算广期所铂钯价差（CNY/克）。
    PT 和 PD 均以 CNY/克 报价，可直接相减。
    """
    return pt_klines["close"] - pd_klines["close"]


def calc_zscore(series: pd.Series, window: int) -> pd.Series:
    """滚动 Z-score，窗口内数据不足时返回 NaN。"""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std  = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / std.replace(0, np.nan)


def calc_divergence(gfex_spread: pd.Series, lme_spread_cny: float) -> pd.Series:
    """
    广期所价差与 LME 折算价差的偏离量。
    正值 = 广期所铂相对更贵（或钯相对更便宜）
    负值 = 反之
    """
    return gfex_spread - lme_spread_cny


# ============================================================
# 主策略
# ============================================================

def run_strategy(
    lme_feed: LMEDataFeed | None = None,
    usd_cny_rate: float = 7.25,
    backtest: bool = False,
):
    """
    主策略入口。

    参数：
      lme_feed     : LME 数据源实例，None 时降级为 GFEX 自身均值回归
      usd_cny_rate : 美元兑人民币汇率，建议从外部实时获取
      backtest     : True 时使用历史回测模式
    """
    if backtest:
        api = TqApi(
            auth=TqAuth("快期账户", "账户密码"),
            backtest=TqBacktest(
                start_dt=date(2024, 11, 1),
                end_dt=date(2025, 3, 1),
            ),
        )
    else:
        api = TqApi(
            account=TqSim(init_balance=500_000),   # 模拟账户，初始资金50万
            auth=TqAuth("快期账户", "账户密码"),
        )

    # ---- 订阅数据（只调用一次，保持引用）----
    pt_klines = api.get_kline_serial(PT_SYMBOL, BAR_DURATION, data_length=LOOKBACK + 20)
    pd_klines = api.get_kline_serial(PD_SYMBOL, BAR_DURATION, data_length=LOOKBACK + 20)
    pt_pos    = api.get_position(PT_SYMBOL)
    pd_pos    = api.get_position(PD_SYMBOL)

    # ---- 执行器：每个合约一个 TargetPosTask（不要混用手动下单）----
    pt_task = TargetPosTask(api, PT_SYMBOL, price="ACTIVE")
    pd_task = TargetPosTask(api, PD_SYMBOL, price="ACTIVE")

    print("策略启动，等待数据初始化...")

    while True:
        api.wait_update()

        # 只在新 K 线生成时重新计算（避免 tick 级别频繁计算）
        new_bar = (
            api.is_changing(pt_klines.iloc[-1], "datetime")
            or api.is_changing(pd_klines.iloc[-1], "datetime")
        )
        if not new_bar:
            continue

        # ---- 计算 GFEX 价差 ----
        gfex_spread = calc_gfex_spread(pt_klines, pd_klines)

        # ---- 获取参考价差（LME 折算 或 自身均值）----
        if lme_feed is not None:
            lme_spread_cny = lme_feed.get_spread_cny_per_gram(usd_cny_rate)
        else:
            # 降级：以 GFEX 价差的滚动均值作为"公允价差"
            lme_spread_cny = gfex_spread.rolling(LOOKBACK).mean().iloc[-1]

        if lme_spread_cny is None or np.isnan(lme_spread_cny):
            continue

        # ---- 计算偏离度与 Z-score ----
        divergence    = calc_divergence(gfex_spread, lme_spread_cny)
        zscore_series = calc_zscore(divergence, LOOKBACK)
        z = zscore_series.iloc[-1]

        if np.isnan(z):
            continue   # 数据积累不足，跳过

        # ---- 当前净持仓 ----
        pt_net = pt_pos.pos_long - pt_pos.pos_short
        pd_net = pd_pos.pos_long - pd_pos.pos_short

        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(
            f"[{ts}] GFEX价差={gfex_spread.iloc[-1]:.2f} CNY/g  "
            f"参考价差={lme_spread_cny:.2f}  偏离Z={z:.2f}  "
            f"PT持仓={pt_net}手  PD持仓={pd_net}手"
        )

        # ---- 交易信号 ----
        if z > Z_ENTRY and pt_net == 0:
            # 广期所铂钯价差偏高于参考 → 做空价差：卖PT + 买PD
            print(f"  ▼ 做空价差：目标 PT={-MAX_LOTS_PT}手 / PD=+{MAX_LOTS_PT * HEDGE_RATIO}手")
            pt_task.set_target_volume(-MAX_LOTS_PT)
            pd_task.set_target_volume(MAX_LOTS_PT * HEDGE_RATIO)

        elif z < -Z_ENTRY and pt_net == 0:
            # 广期所铂钯价差偏低于参考 → 做多价差：买PT + 卖PD
            print(f"  ▲ 做多价差：目标 PT=+{MAX_LOTS_PT}手 / PD={-MAX_LOTS_PT * HEDGE_RATIO}手")
            pt_task.set_target_volume(MAX_LOTS_PT)
            pd_task.set_target_volume(-MAX_LOTS_PT * HEDGE_RATIO)

        elif abs(z) < Z_EXIT and pt_net != 0:
            # 价差已回归均值区间，平仓
            print("  ◆ 价差回归，平仓")
            pt_task.set_target_volume(0)
            pd_task.set_target_volume(0)


# ============================================================
# 历史数据下载（研究用，需 DataDownloader 权限）
# ============================================================

def download_history(
    start: date = date(2024, 11, 1),
    end:   date = date(2025, 3, 1),
):
    """
    下载广期所铂钯 1 小时 K 线到 CSV，用于离线研究。
    DataDownloader 是天勤付费功能，需要账户有对应权限。
    """
    api = TqApi(auth=TqAuth("快期账户", "账户密码"))

    pt_task = DataDownloader(
        api, symbol_list=PT_SYMBOL, dur_sec=3600,
        start_dt=start, end_dt=end,
        csv_file_name="gfex_pt_1h.csv",
    )
    pd_task = DataDownloader(
        api, symbol_list=PD_SYMBOL, dur_sec=3600,
        start_dt=start, end_dt=end,
        csv_file_name="gfex_pd_1h.csv",
    )

    with closing(api):
        while not (pt_task.is_finished() and pd_task.is_finished()):
            api.wait_update()
            print(f"下载进度  PT: {pt_task.get_progress():.1f}%  PD: {pd_task.get_progress():.1f}%")

    print("历史数据下载完成 → gfex_pt_1h.csv / gfex_pd_1h.csv")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":

    # 第一步：下载历史数据做研究（取消注释后运行）
    # download_history()

    # 第二步A：无 LME 数据，降级为 GFEX 价差自身均值回归（模拟账户）
    run_strategy(lme_feed=None)

    # 第二步B：接入 LME CSV 数据后
    # lme = LMEDataFeed(source="csv", csv_path="lme_spread.csv")
    # run_strategy(lme_feed=lme, usd_cny_rate=7.25)

    # 第三步：历史回测模式
    # run_strategy(lme_feed=None, backtest=True)
