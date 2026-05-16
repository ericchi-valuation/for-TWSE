import pandas as pd
import os
from FinMind.data import DataLoader

print("🚑 啟動單檔股票救援程序...")

# ==========================================
# 1. 🔑 請填入您的 API Token 與要救援的代碼
# ==========================================
API_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMy0yOCAyMDo0MjowNCIsInVzZXJfaWQiOiJlcmljY2hpIiwiZW1haWwiOiJlcmljLmNoaTE5ODhAZ21haWwuY29tIiwiaXAiOiI4Ny4yNDkuMTM4Ljc5In0.Kr93MomHC6qtFsyVhYBy5sgADiyW_zRM6i21UNkkpvw" 
TARGET_TICKER = "2640"
START_DATE = "2014-01-01"

api = DataLoader()
api.login_by_token(api_token=API_TOKEN)

# ==========================================
# 2. 強制抓取三大財報並寫入 CSV
# ==========================================
print(f"📥 正在單獨補抓 {TARGET_TICKER} 的 10 年財報...")
try:
    df_is = api.taiwan_stock_financial_statement(stock_id=TARGET_TICKER, start_date=START_DATE)
    df_bs = api.taiwan_stock_balance_sheet(stock_id=TARGET_TICKER, start_date=START_DATE)
    df_cf = api.taiwan_stock_cash_flows_statement(stock_id=TARGET_TICKER, start_date=START_DATE)

    if not df_is.empty:
        df_is.to_csv("tw_historical_is.csv", mode='a', header=not os.path.exists("tw_historical_is.csv"), index=False, encoding='utf-8-sig')
    if not df_bs.empty:
        df_bs.to_csv("tw_historical_bs.csv", mode='a', header=not os.path.exists("tw_historical_bs.csv"), index=False, encoding='utf-8-sig')
    if not df_cf.empty:
        df_cf.to_csv("tw_historical_cf.csv", mode='a', header=not os.path.exists("tw_historical_cf.csv"), index=False, encoding='utf-8-sig')
        
    print("✅ CSV 補漏完成！")

except Exception as e:
    print(f"❌ 抓取失敗，請確認 Token 或網路狀態: {e}")

# ==========================================
# 3. 重新編譯 Parquet 輕量金庫 (讓 App 能讀到)
# ==========================================
print("📦 正在重新編譯 Parquet 輕量金庫...")

is_cols = ['Revenue', 'OperatingIncome', 'EPS', 'NetIncome']
bs_cols = ['EquityAttributableToOwnersOfParent', 'TotalEquity', 'CurrentLiabilities', 'NoncurrentLiabilities', 'CashAndCashEquivalents', 'CashAndCashEquivalents_per', 'OrdinaryShare', 'CapitalStock', 'OrdinaryShare_per', 'CapitalStock_per']
cf_cols = ['CashFlowsFromOperatingActivities', 'NetCashInflowFromOperatingActivities', 'CashProvidedByInvestingActivities', 'Depreciation', 'InterestExpense', 'PayTheInterest']

try:
    pd.read_csv("tw_historical_is.csv").drop_duplicates(subset=['stock_id', 'date', 'type'], keep='last')[lambda x: x['type'].isin(is_cols)].to_parquet("tw_is_lite.parquet", index=False)
    pd.read_csv("tw_historical_bs.csv").drop_duplicates(subset=['stock_id', 'date', 'type'], keep='last')[lambda x: x['type'].isin(bs_cols)].to_parquet("tw_bs_lite.parquet", index=False)
    pd.read_csv("tw_historical_cf.csv").drop_duplicates(subset=['stock_id', 'date', 'type'], keep='last')[lambda x: x['type'].isin(cf_cols)].to_parquet("tw_cf_lite.parquet", index=False)
    
    print("🎊 救援成功！您的 2640 已經正式歸隊，請重整您的網頁試試看！")
except Exception as e:
    print(f"❌ 打包 Parquet 失敗: {e}")