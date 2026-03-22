"""
data/downloader.py — 历史 K 线下载（需要天勤 DataDownloader 权限）
"""

from contextlib import closing
from datetime import date

from tqsdk import TqApi, TqAuth
from tqsdk.tools import DataDownloader

from config import PT_SYMBOL, PD_SYMBOL


def download_history(
    start: date = date(2024, 11, 1),
    end:   date = date(2025, 3, 1),
    auth_user: str = "快期账户",
    auth_pass: str = "账户密码",
):
    """下载广期所铂钯 1 小时 K 线到 CSV。"""
    api = TqApi(auth=TqAuth(auth_user, auth_pass))

    pt_task = DataDownloader(
        api, symbol_list=PT_SYMBOL, dur_sec=3600,
        start_dt=start, end_dt=end,
        csv_file_name="data/gfex_pt_1h.csv",
    )
    pd_task = DataDownloader(
        api, symbol_list=PD_SYMBOL, dur_sec=3600,
        start_dt=start, end_dt=end,
        csv_file_name="data/gfex_pd_1h.csv",
    )

    with closing(api):
        while not (pt_task.is_finished() and pd_task.is_finished()):
            api.wait_update()
            print(f"PT: {pt_task.get_progress():.1f}%  PD: {pd_task.get_progress():.1f}%")

    print("下载完成 → data/gfex_pt_1h.csv / data/gfex_pd_1h.csv")
