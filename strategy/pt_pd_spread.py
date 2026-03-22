"""
strategy/pt_pd_spread.py — 铂钯价差套利主策略
只负责：订阅数据 → 触发信号 → 调用 core/ 执行下单
所有计算逻辑在 core/signals.py / core/risk.py / core/performance.py
"""

import numpy as np
from datetime import datetime

from tqsdk import TqApi, TqAuth, TqSim, TqBacktest, TargetPosTask

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    PT_SYMBOL, PD_SYMBOL, BAR_DURATION, LOOKBACK, HURST_WINDOW,
    Z_ENTRY_1, Z_ENTRY_2, Z_STOP, Z_EXIT,
    LOTS_LEVEL_1, LOTS_LEVEL_2, HEDGE_RATIO,
    PARAM_OPT_INTERVAL_HOURS, COINT_CHECK_INTERVAL_HOURS,
)
from core.signals    import calc_zscore, calc_divergence, calc_hurst, detect_regime
from core.risk       import check_limit_protection, check_close_protection, calc_margin_usage
from core.performance import PerformanceTracker, TradeRecord
from data.lme_feed   import LMEDataFeed
from research.cointegration import run_cointegration_test
from research.backtest_engine import quick_sharpe

import itertools


def _optimize_params(spread_vals, param_grid) -> dict:
    best_sh, best_p = -float("inf"), {"lookback": 120, "z_entry": 2.0, "z_exit": 0.5}
    if len(spread_vals) < 200:
        return best_p
    for lk, ze, zx in itertools.product(
        param_grid["lookback"], param_grid["z_entry"], param_grid["z_exit"]
    ):
        sh = quick_sharpe(spread_vals, lk, ze, zx)
        if sh > best_sh:
            best_sh, best_p = sh, {"lookback": lk, "z_entry": ze, "z_exit": zx}
    print(f"  ⚙️  参数优化 | lookback={best_p['lookback']} "
          f"z_entry={best_p['z_entry']} z_exit={best_p['z_exit']} Sharpe={best_sh:.2f}")
    return best_p


def run(
    lme_feed: LMEDataFeed | None = None,
    usd_cny_rate: float = 7.25,
    backtest: bool = False,
    auth_user: str = "快期账户",
    auth_pass: str = "账户密码",
):
    from config import PARAM_GRID

    if backtest:
        from datetime import date
        api = TqApi(
            TqSim(init_balance=3_000_000),
            backtest=TqBacktest(start_dt=date(2024, 11, 1), end_dt=date(2025, 6, 1)),
            auth=TqAuth(auth_user, auth_pass),
        )
    else:
        api = TqApi(
            TqSim(init_balance=3_000_000),
            auth=TqAuth(auth_user, auth_pass),
        )

    pt_klines = api.get_kline_serial(PT_SYMBOL, BAR_DURATION, data_length=LOOKBACK + 20)
    pd_klines = api.get_kline_serial(PD_SYMBOL, BAR_DURATION, data_length=LOOKBACK + 20)
    pt_quote  = api.get_quote(PT_SYMBOL)
    pd_quote  = api.get_quote(PD_SYMBOL)
    pt_pos    = api.get_position(PT_SYMBOL)
    account   = api.get_account()

    pt_task = TargetPosTask(api, PT_SYMBOL, price="ACTIVE")
    pd_task = TargetPosTask(api, PD_SYMBOL, price="ACTIVE")

    tracker       = PerformanceTracker()
    current_trade: TradeRecord | None = None
    pos_direction = 0
    pos_level     = 0

    dyn_lookback = LOOKBACK
    dyn_z_entry  = Z_ENTRY_1
    dyn_z_exit   = Z_EXIT

    last_coint = datetime.min
    last_opt   = datetime.min

    print("策略启动...")

    while True:
        api.wait_update()
        if not (api.is_changing(pt_klines.iloc[-1], "datetime") or
                api.is_changing(pd_klines.iloc[-1], "datetime")):
            continue

        now = datetime.now()

        # ---- 协整检验（每24h）----
        if (now - last_coint).total_seconds() > COINT_CHECK_INTERVAL_HOURS * 3600:
            is_coint, _ = run_cointegration_test(pt_klines["close"], pd_klines["close"])
            last_coint  = now
            if not is_coint and pos_level == 0:
                print("  ⚠️  协整检验未通过，暂停开新仓")
                continue

        # ---- 参数优化（每7天）----
        if (now - last_opt).total_seconds() > PARAM_OPT_INTERVAL_HOURS * 3600:
            spread_tmp  = (pt_klines["close"] - pd_klines["close"]).dropna().values
            best        = _optimize_params(spread_tmp, PARAM_GRID)
            dyn_lookback = best["lookback"]
            dyn_z_entry  = best["z_entry"]
            dyn_z_exit   = best["z_exit"]
            last_opt     = now

        # ---- 价差 & 信号 ----
        gfex_spread = pt_klines["close"] - pd_klines["close"]

        if lme_feed is not None:
            ref = lme_feed.get_spread_cny_per_gram(usd_cny_rate)
        else:
            ref = gfex_spread.rolling(LOOKBACK).mean().iloc[-1]

        if ref is None or np.isnan(ref):
            continue

        divergence = calc_divergence(gfex_spread, ref)
        z = calc_zscore(divergence, dyn_lookback).iloc[-1]
        if np.isnan(z):
            continue

        regime     = detect_regime(gfex_spread, HURST_WINDOW)
        h_val      = calc_hurst(gfex_spread.iloc[-HURST_WINDOW:])
        spread_now = float(gfex_spread.iloc[-1])
        pt_net     = pt_pos.pos_long - pt_pos.pos_short

        # ---- 保证金监控 ----
        mm = calc_margin_usage(pt_quote.last_price, pd_quote.last_price, pos_level)
        warn = " ⚠️ 保证金预警" if mm["warning"] else ""
        print(f"[{now:%H:%M}] 价差={spread_now:.1f} Z={z:.2f} H={h_val:.2f} "
              f"状态={regime} 保证金={mm['margin_pct']}%{warn}")

        adverse_z = z * pos_direction * -1 if pos_direction != 0 else 0

        # ---- 内部辅助 ----
        def _open(direction: int, lots: int, reason: str):
            nonlocal pos_direction, pos_level, current_trade
            ok, msg = check_limit_protection(direction, pt_quote, pd_quote)
            if not ok:
                print(f"  🚫 {msg}"); return
            pt_task.set_target_volume(direction * lots)
            pd_task.set_target_volume(-direction * lots * HEDGE_RATIO)
            pos_direction = direction
            pos_level     = 1 if pos_level == 0 else 2
            current_trade = tracker.open_trade(direction, z, spread_now, pos_level, lots)
            print(f"  {'▲' if direction>0 else '▼'} [{reason}] Z={z:.2f} {lots}手")

        def _close(reason: str):
            nonlocal pos_direction, pos_level, current_trade
            ok, msg = check_close_protection(pos_direction, pt_quote, pd_quote)
            if not ok:
                print(f"  🚫 {msg} — 请手动平仓！"); return
            pt_task.set_target_volume(0); pd_task.set_target_volume(0)
            if current_trade:
                tracker.close_trade(current_trade, z, spread_now, reason, pos_direction)
                current_trade = None
            pos_direction, pos_level = 0, 0

        # ---- 信号执行 ----
        if regime == "uncertain" and pos_level > 0:
            _close("regime")

        elif regime == "mean_reversion":
            if pos_level > 0 and adverse_z >= Z_STOP:
                _close("stop_loss")
            elif pos_level == 1 and adverse_z >= (Z_ENTRY_2 - Z_ENTRY_1):
                _open(pos_direction, LOTS_LEVEL_1 + LOTS_LEVEL_2, "加仓L2")
            elif pos_level > 0 and abs(z) < dyn_z_exit:
                _close("profit")
            elif pos_level == 0 and z >  dyn_z_entry:
                _open(-1, LOTS_LEVEL_1, "空价差L1")
            elif pos_level == 0 and z < -dyn_z_entry:
                _open(+1, LOTS_LEVEL_1, "多价差L1")

        elif regime == "trend":
            mean = gfex_spread.rolling(dyn_lookback).mean().iloc[-1]
            if spread_now > mean and pos_direction <= 0:
                if pos_level > 0: _close("趋势反转")
                _open(+1, LOTS_LEVEL_1, "趋势多")
            elif spread_now <= mean and pos_direction >= 0:
                if pos_level > 0: _close("趋势反转")
                _open(-1, LOTS_LEVEL_1, "趋势空")
