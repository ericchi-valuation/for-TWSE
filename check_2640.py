import pandas as pd
import os

TARGET = "2640"

for name, path in [("IS", "tw_is_lite.parquet"), ("BS", "tw_bs_lite.parquet"), ("CF", "tw_cf_lite.parquet")]:
    if not os.path.exists(path):
        print(f"{name}: 檔案不存在")
        continue
    df = pd.read_parquet(path)
    rows = df[df["stock_id"].astype(str) == TARGET]
    if not rows.empty:
        rows["date"] = pd.to_datetime(rows["date"])
        print(f"{name}: 有 {len(rows)} 筆，日期 {rows['date'].min().strftime('%Y-%m')} ~ {rows['date'].max().strftime('%Y-%m')}")
    else:
        print(f"{name}: 無資料")

if os.path.exists("tw_monthly_rev.parquet"):
    mr = pd.read_parquet("tw_monthly_rev.parquet")
    rows = mr[mr["stock_id"].astype(str) == TARGET]
    if not rows.empty:
        rows["date"] = pd.to_datetime(rows["date"])
        print(f"月營收: 有 {len(rows)} 筆，日期 {rows['date'].min().strftime('%Y-%m')} ~ {rows['date'].max().strftime('%Y-%m')}")
    else:
        print("月營收: 無資料")
else:
    print("月營收: tw_monthly_rev.parquet 尚未建立")
