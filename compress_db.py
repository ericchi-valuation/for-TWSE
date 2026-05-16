import pandas as pd

print("🔥 啟動資料庫終極瘦身與壓縮程序...\n")

# 定義我們 V7.0 估值模型「真正需要」的黃金欄位
is_cols = ['Revenue', 'OperatingIncome', 'EPS', 'NetIncome']
bs_cols = ['EquityAttributableToOwnersOfParent', 'TotalEquity', 'CurrentLiabilities', 'NoncurrentLiabilities', 'CashAndCashEquivalents', 'CashAndCashEquivalents_per']
cf_cols = ['CashFlowsFromOperatingActivities', 'NetCashInflowFromOperatingActivities', 'CashProvidedByInvestingActivities', 'Depreciation']

try:
    # 1. 處理綜合損益表 (IS)
    print("📦 正在處理 綜合損益表 (IS)...")
    df_is = pd.read_csv("tw_historical_is.csv")
    df_is_lite = df_is[df_is['type'].isin(is_cols)]
    df_is_lite.to_parquet("tw_is_lite.parquet", index=False)
    print(f"   ✅ 完成！原資料筆數: {len(df_is)} -> 瘦身後: {len(df_is_lite)}")

    # 2. 處理資產負債表 (BS)
    print("📦 正在處理 資產負債表 (BS)...")
    df_bs = pd.read_csv("tw_historical_bs.csv")
    df_bs_lite = df_bs[df_bs['type'].isin(bs_cols)]
    df_bs_lite.to_parquet("tw_bs_lite.parquet", index=False)
    print(f"   ✅ 完成！原資料筆數: {len(df_bs)} -> 瘦身後: {len(df_bs_lite)}")

    # 3. 處理現金流量表 (CF)
    print("📦 正在處理 現金流量表 (CF)...")
    df_cf = pd.read_csv("tw_historical_cf.csv")
    df_cf_lite = df_cf[df_cf['type'].isin(cf_cols)]
    df_cf_lite.to_parquet("tw_cf_lite.parquet", index=False)
    print(f"   ✅ 完成！原資料筆數: {len(df_cf)} -> 瘦身後: {len(df_cf_lite)}")

    print("\n🎉 瘦身大成功！請查看資料夾，您現在擁有了三個輕量級的 .parquet 檔案！")
    print("⚠️ (您可以把原本龐大的 .csv 檔案刪除或移到別的資料夾了)")

except Exception as e:
    print(f"\n❌ 發生錯誤: {e}")