"""
core/performance.py — 绩效实时记录，每笔交易写入 trade_log.csv
"""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import PT_UNIT


@dataclass
class TradeRecord:
    trade_id:     int
    direction:    str        # "多价差" / "空价差"
    entry_time:   str
    entry_z:      float
    entry_spread: float
    entry_level:  int
    exit_time:    str   = ""
    exit_z:       float = 0.0
    exit_spread:  float = 0.0
    exit_reason:  str   = ""  # profit / stop_loss / regime / 趋势反转
    lots:         int   = 0
    pnl:          float = 0.0


class PerformanceTracker:
    LOG_FILE = Path("trade_log.csv")
    FIELDS   = list(TradeRecord.__dataclass_fields__.keys())

    def __init__(self):
        self.trades: list[TradeRecord] = []
        self._next_id = 1
        if not self.LOG_FILE.exists():
            with open(self.LOG_FILE, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def open_trade(
        self, direction: int, z: float, spread: float, level: int, lots: int
    ) -> TradeRecord:
        rec = TradeRecord(
            trade_id     = self._next_id,
            direction    = "多价差" if direction == +1 else "空价差",
            entry_time   = datetime.now().strftime("%Y-%m-%d %H:%M"),
            entry_z      = round(z, 3),
            entry_spread = round(spread, 3),
            entry_level  = level,
            lots         = lots,
        )
        self.trades.append(rec)
        self._next_id += 1
        return rec

    def close_trade(
        self, rec: TradeRecord, z: float, spread: float,
        reason: str, pos_direction: int,
    ):
        rec.exit_time   = datetime.now().strftime("%Y-%m-%d %H:%M")
        rec.exit_z      = round(z, 3)
        rec.exit_spread = round(spread, 3)
        rec.exit_reason = reason
        rec.pnl = round(
            pos_direction * (rec.exit_spread - rec.entry_spread) * rec.lots * PT_UNIT, 2
        )
        with open(self.LOG_FILE, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(rec.__dict__)
        self._print_summary()

    def _print_summary(self):
        closed = [t for t in self.trades if t.exit_time]
        if not closed:
            return
        pnls     = [t.pnl for t in closed]
        wins     = [p for p in pnls if p > 0]
        total    = sum(pnls)
        win_rate = len(wins) / len(closed) * 100
        print(
            f"  📊 绩效 | 累计{len(closed)}笔  总盈亏{total/10000:.2f}万  "
            f"胜率{win_rate:.0f}%  "
            f"最近:{'+' if pnls[-1]>=0 else ''}{pnls[-1]/10000:.2f}万"
            f"({closed[-1].exit_reason})"
        )
