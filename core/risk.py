"""
core/risk.py — 风险控制：涨跌停保护、保证金监控
"""

from config import (
    PT_UNIT, MARGIN_RATIO, ACCOUNT_BALANCE,
    LOTS_LEVEL_1, MARGIN_WARN_PCT,
)


def check_limit_protection(direction: int, pt_quote, pd_quote) -> tuple[bool, str]:
    """
    开仓前检查涨跌停。
      direction = +1：买PT 卖PD
      direction = -1：卖PT 买PD
    返回 (可以交易, 原因说明)
    """
    pt_up   = pt_quote.last_price >= pt_quote.upper_limit
    pt_down = pt_quote.last_price <= pt_quote.lower_limit
    pd_up   = pd_quote.last_price >= pd_quote.upper_limit
    pd_down = pd_quote.last_price <= pd_quote.lower_limit

    if direction == +1:
        if pt_up:   return False, "PT涨停，无法买入"
        if pd_down: return False, "PD跌停，无法卖出"
    else:
        if pt_down: return False, "PT跌停，无法卖出"
        if pd_up:   return False, "PD涨停，无法买入"

    return True, ""


def check_close_protection(pos_direction: int, pt_quote, pd_quote) -> tuple[bool, str]:
    """平仓方向与开仓相反，复用 check_limit_protection。"""
    return check_limit_protection(-pos_direction, pt_quote, pd_quote)


def calc_margin_usage(pt_price: float, pd_price: float, pos_level: int) -> dict:
    """
    返回当前持仓的保证金占用情况。
    """
    lots           = LOTS_LEVEL_1 * pos_level if pos_level > 0 else 0
    margin_per_lot = (pt_price + pd_price) * PT_UNIT * MARGIN_RATIO
    margin_used    = lots * margin_per_lot
    margin_pct     = margin_used / ACCOUNT_BALANCE * 100

    return {
        "lots":       lots,
        "margin_万":  round(margin_used / 10_000, 1),
        "margin_pct": round(margin_pct, 1),
        "warning":    margin_pct >= MARGIN_WARN_PCT * 100,
    }
