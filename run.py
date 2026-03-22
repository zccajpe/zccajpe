"""
run.py — 统一入口
用法：
  python run.py              # 实盘/模拟
  python run.py --backtest   # 历史回测
  python run.py --download   # 下载历史数据
  python run.py --scan       # 参数扫描（不依赖 TqSdk）
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="铂钯价差套利策略")
    parser.add_argument("--backtest", action="store_true", help="历史回测模式")
    parser.add_argument("--download", action="store_true", help="下载历史K线")
    parser.add_argument("--scan",     action="store_true", help="独立参数扫描")
    parser.add_argument("--lme",      type=str, default=None, help="LME CSV 文件路径")
    parser.add_argument("--usdcny",   type=float, default=7.25, help="美元兑人民币汇率")
    args = parser.parse_args()

    if args.download:
        from data.downloader import download_history
        download_history()
        return

    if args.scan:
        from research.backtest_engine import generate_prices, run_scan
        import pandas as pd
        prices = generate_prices()
        df     = run_scan(prices)
        pd.set_option("display.float_format", "{:.2f}".format)
        print(df.head(10).to_string(index=False))
        df.to_csv("research/scan_results.csv", index=False)
        print("已保存至 research/scan_results.csv")
        return

    from strategy.pt_pd_spread import run
    lme_feed = None
    if args.lme:
        from data.lme_feed import LMEDataFeed
        lme_feed = LMEDataFeed(csv_path=args.lme)

    run(lme_feed=lme_feed, usd_cny_rate=args.usdcny, backtest=args.backtest)


if __name__ == "__main__":
    main()
