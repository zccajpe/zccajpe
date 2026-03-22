"""
research/param_scan.py — TqSdk 版参数扫描（需要天勤账号 + DataDownloader 权限）
每组参数跑完整 TqBacktest，结果保存到 research/scan_results_tqsdk.csv
"""

import itertools
import numpy as np
import pandas as pd
from datetime import date

from tqsdk import TqApi, TqAuth, TqSim, TqBacktest, TargetPosTask

from config import (
    PT_SYMBOL, PD_SYMBOL, BAR_DURATION,
    LOTS_LEVEL_1, PT_UNIT, PARAM_GRID,
)

INIT_BALANCE   = 3_000_000
BACKTEST_START = date(2024, 11, 1)
BACKTEST_END   = date(2025, 6, 1)


def run_single(lookback: int, z_entry: float, z_exit: float) -> dict:
    api = TqApi(
        TqSim(init_balance=INIT_BALANCE),
        backtest=TqBacktest(start_dt=BACKTEST_START, end_dt=BACKTEST_END),
        auth=TqAuth("快期账户", "账户密码"),
    )
    pt_klines = api.get_kline_serial(PT_SYMBOL, BAR_DURATION, data_length=lookback+20)
    pd_klines = api.get_kline_serial(PD_SYMBOL, BAR_DURATION, data_length=lookback+20)
    account   = api.get_account()
    pt_pos    = api.get_position(PT_SYMBOL)
    pt_task   = TargetPosTask(api, PT_SYMBOL, price="ACTIVE")
    pd_task   = TargetPosTask(api, PD_SYMBOL, price="ACTIVE")

    bal_series, trades, entry_bal = [], [], INIT_BALANCE
    position = 0

    try:
        while True:
            api.wait_update()
            if not (api.is_changing(pt_klines.iloc[-1], "datetime") or
                    api.is_changing(pd_klines.iloc[-1], "datetime")):
                continue
            bal_series.append(account.balance)
            spread = pt_klines["close"] - pd_klines["close"]
            ref    = spread.rolling(lookback).mean().iloc[-1]
            std    = spread.rolling(lookback).std().iloc[-1]
            if np.isnan(ref) or std == 0:
                continue
            z    = (spread.iloc[-1] - ref) / std
            pnet = pt_pos.pos_long - pt_pos.pos_short

            if position != 0 and abs(z) < z_exit:
                pt_task.set_target_volume(0); pd_task.set_target_volume(0)
                trades.append({"pnl": account.balance - entry_bal})
                position = 0
            elif position == 0 and z > z_entry:
                pt_task.set_target_volume(-LOTS_LEVEL_1)
                pd_task.set_target_volume(LOTS_LEVEL_1)
                entry_bal = account.balance; position = -1
            elif position == 0 and z < -z_entry:
                pt_task.set_target_volume(LOTS_LEVEL_1)
                pd_task.set_target_volume(-LOTS_LEVEL_1)
                entry_bal = account.balance; position = +1
    except Exception:
        pass
    finally:
        api.close()

    bal  = np.array(bal_series)
    if len(bal) < 10:
        return {"lookback": lookback, "z_entry": z_entry, "z_exit": z_exit}
    tr   = (bal[-1] - INIT_BALANCE) / INIT_BALANCE
    dd   = ((bal - np.maximum.accumulate(bal)) / np.maximum.accumulate(bal)).min()
    rets = np.diff(bal) / bal[:-1]
    sh   = rets.mean() / rets.std() * np.sqrt(252*6) if rets.std() > 0 else np.nan
    pnls = [t["pnl"] for t in trades]
    wr   = len([p for p in pnls if p > 0]) / len(pnls) * 100 if pnls else np.nan
    return {
        "lookback": lookback, "z_entry": z_entry, "z_exit": z_exit,
        "total_ret_%": round(tr*100,2), "max_dd_%": round(dd*100,2),
        "sharpe": round(float(sh),3), "n_trades": len(trades),
        "win_rate_%": round(wr,1),
    }


if __name__ == "__main__":
    combos  = list(itertools.product(
        PARAM_GRID["lookback"], PARAM_GRID["z_entry"], PARAM_GRID["z_exit"]
    ))
    results = []
    for i, (lk, ze, zx) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] lookback={lk} z_entry={ze} z_exit={zx}")
        results.append(run_single(lk, ze, zx))

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print(df.to_string(index=False))
    df.to_csv("research/scan_results_tqsdk.csv", index=False)
