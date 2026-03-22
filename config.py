# ============================================================
# config.py — 全局参数，所有模块从这里导入，不要分散硬编码
# ============================================================

# ---- 合约 ----
PT_SYMBOL      = "KQ.m@GFEX.pt"
PD_SYMBOL      = "KQ.m@GFEX.pd"
PT_UNIT        = 1000          # 克/手
PD_UNIT        = 1000          # 克/手
HEDGE_RATIO    = PT_UNIT // PD_UNIT   # 1
TROY_OZ_TO_GRAM = 31.1035

# ---- K线 ----
BAR_DURATION   = 3600          # 1小时

# ---- 信号 ----
LOOKBACK       = 120           # 默认滚动窗口（动态优化后会覆盖）
HURST_WINDOW   = 60
Z_ENTRY_1      = 2.0
Z_ENTRY_2      = 3.0
Z_STOP         = 4.0
Z_EXIT         = 0.5

# ---- 资金 ----
ACCOUNT_BALANCE  = 3_000_000
MARGIN_RATIO     = 0.22
LOTS_LEVEL_1     = 2
LOTS_LEVEL_2     = 2
MIN_LOTS         = 2
MARGIN_WARN_PCT  = 0.20

# ---- 研究 ----
PARAM_GRID = {
    "lookback": [60, 90, 120, 180],
    "z_entry":  [1.5, 2.0, 2.5],
    "z_exit":   [0.3, 0.5, 0.8],
}
PARAM_OPT_INTERVAL_HOURS = 7 * 24   # 每7天重新优化
COINT_CHECK_INTERVAL_HOURS = 24     # 每24小时协整检验
