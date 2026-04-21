"""
update_monthly_rev.py
月營收資料庫建立腳本
用途：從 FinMind 抓取全市場月營收（比季財報更新更快），
      建立 tw_monthly_rev.parquet，供 app.py 計算精確的累計年增率。
執行頻率：建議每月底或每月 10 號後更新一次即可。
"""

import pandas as pd
import os
import time
from FinMind.data import DataLoader
from datetime import datetime

print("📅 啟動月營收資料庫建立程序...")

API_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMy0yOCAyMDo0MjowNCIsInVzZXJfaWQiOiJlcmljY2hpIiwiZW1haWwiOiJlcmljLmNoaTE5ODhAZ21haWwuY29tIiwiaXAiOiI4Ny4yNDkuMTM4Ljc5In0.Kr93MomHC6qtFsyVhYBy5sgADiyW_zRM6i21UNkkpvw"
# 抓取兩年月營收，確保 YoY 比較有完整的去年同期資料
MR_START  = "2024-01-01"
TEMP_FILE = "temp_new_mr.csv"
HIST_FILE = "tw_historical_mr.csv"
PARQ_FILE = "tw_monthly_rev.parquet"

api = DataLoader()
api.login_by_token(api_token=API_TOKEN)

# 讀取股票清單
if not os.path.exists("tw_stock_list.csv"):
    print("❌ 找不到 tw_stock_list.csv，請確認檔案存在！")
    exit(1)

tickers = pd.read_csv("tw_stock_list.csv")["Code"].astype(str).tolist()
print(f"📋 股票清單：{len(tickers)} 檔")

# 斷點續傳
done = set()
if os.path.exists(TEMP_FILE):
    try:
        done.update(pd.read_csv(TEMP_FILE)["stock_id"].astype(str).unique())
        print(f"✅ 斷點續傳：已完成 {len(done)} 檔，繼續補抓剩餘。")
    except: pass

remaining = [t for t in tickers if t not in done]
print(f"🚀 待抓取：{len(remaining)} 檔\n")

api_count = 0
MAX_PER_HOUR = 580

for i, ticker in enumerate(remaining):
    if api_count >= MAX_PER_HOUR - 1:
        print(f"\n💤 [{datetime.now().strftime('%H:%M')}] 達到速率上限，休眠 60 分鐘...")
        time.sleep(3600)
        api_count = 0
        print("☀️ 繼續抓取！\n")

    try:
        df = api.taiwan_stock_month_revenue(stock_id=ticker, start_date=MR_START)
        api_count += 1

        if not df.empty:
            # 只保留關鍵欄位：stock_id, date, revenue
            keep = [c for c in ["stock_id", "date", "revenue"] if c in df.columns]
            df_keep = df[keep].copy()
            if "stock_id" in df_keep.columns:
                df_keep["stock_id"] = df_keep["stock_id"].astype(str).str.replace('*', '', regex=False)
            
            df_keep.to_csv(
                TEMP_FILE,
                mode="a",
                header=not os.path.exists(TEMP_FILE),
                index=False,
                encoding="utf-8-sig",
            )
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(remaining)}] ✅ {ticker}")
        time.sleep(0.25)

    except Exception as e:
        print(f"  [{i+1}/{len(remaining)}] ❌ {ticker}: {e}")
        time.sleep(1)

print("\n✅ 月營收數據下載完成！開始整合...")

# 合併歷史 + 新增
frames = []
if os.path.exists(HIST_FILE):
    frames.append(pd.read_csv(HIST_FILE))
if os.path.exists(TEMP_FILE):
    frames.append(pd.read_csv(TEMP_FILE))

if frames:
    df_all = pd.concat(frames, ignore_index=True)
    df_all.drop_duplicates(subset=["stock_id", "date"], keep="last", inplace=True)
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all.sort_values(["stock_id", "date"], inplace=True)

    # 儲存合併後的 CSV 及 Parquet
    df_all.to_csv(HIST_FILE, index=False, encoding="utf-8-sig")
    df_all.to_parquet(PARQ_FILE, index=False)

    print(f"🎊 月營收 Parquet 已建立：{PARQ_FILE}")
    print(f"   共 {len(df_all)} 筆 / {df_all['stock_id'].nunique()} 檔股票")
    print(f"   橫跨期間：{df_all['date'].min().strftime('%Y-%m')} ~ {df_all['date'].max().strftime('%Y-%m')}")

    # 清理暫存
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)
else:
    print("⚠️ 沒有任何資料，請確認 API Token 有效。")
