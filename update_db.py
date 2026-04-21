import pandas as pd
import time
import os
from FinMind.data import DataLoader
from datetime import datetime

print("🔥 啟動 V8.0 智慧增量更新引擎 (下載 -> 合併 -> 壓縮)...\n")

# ==========================================
# 1. 🔑 API 與更新參數設定
# ==========================================
API_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMy0yOCAyMDo0MjowNCIsInVzZXJfaWQiOiJlcmljY2hpIiwiZW1haWwiOiJlcmljLmNoaTE5ODhAZ21haWwuY29tIiwiaXAiOiI4Ny4yNDkuMTM4Ljc5In0.Kr93MomHC6qtFsyVhYBy5sgADiyW_zRM6i21UNkkpvw" 
api = DataLoader()
api.login_by_token(api_token=API_TOKEN)

# 設定增量更新的起點 (涵蓋 2024 Q4 與 2025 全年財報)
START_DATE = "2024-11-01"

# 檔案路徑設定
TEMP_IS = "temp_new_is.csv"
TEMP_BS = "temp_new_bs.csv"
TEMP_CF = "temp_new_cf.csv"

# ==========================================
# 2. 取得代碼清單與斷點續傳檢查
# ==========================================
def get_tw_stock_list():
    if os.path.exists('tw_stock_list.csv'):
        return pd.read_csv('tw_stock_list.csv')['Code'].astype(str).tolist()
    else:
        print("❌ 找不到 tw_stock_list.csv，請確認檔案存在！")
        return []

target_tickers = get_tw_stock_list()
done_tickers = set()

if os.path.exists(TEMP_IS):
    try:
        df_exist = pd.read_csv(TEMP_IS)
        done_tickers.update(df_exist['stock_id'].astype(str).unique())
        print(f"✅ 斷點續傳啟動：發現已抓取 {len(done_tickers)} 檔最新財報，將自動跳過。")
    except: pass

remaining_tickers = sorted([t for t in target_tickers if t not in done_tickers])
print(f"🚀 準備抓取最新財報！剩餘待更新股票：{len(remaining_tickers)} 檔")

# ==========================================
# 3. 增量下載迴圈 (掛機專用)
# ==========================================
api_calls_this_hour = 0
MAX_CALLS_PER_HOUR = 550 

for i, ticker in enumerate(remaining_tickers):
    if api_calls_this_hour >= (MAX_CALLS_PER_HOUR - 3):
        print(f"\n💤 [{datetime.now().strftime('%H:%M:%S')}] 達到安全上限。進入休眠 60 分鐘...")
        time.sleep(3600)
        api_calls_this_hour = 0
        print(f"☀️ [{datetime.now().strftime('%H:%M:%S')}] 滿血復活，繼續抓取！\n")

    print(f"[{i+1}/{len(remaining_tickers)}] 📥 抓取 {ticker} 最新財報...", end=" ")
    
    try:
        df_is = api.taiwan_stock_financial_statement(stock_id=ticker, start_date=START_DATE)
        api_calls_this_hour += 1; time.sleep(0.3)
        
        df_bs = api.taiwan_stock_balance_sheet(stock_id=ticker, start_date=START_DATE)
        api_calls_this_hour += 1; time.sleep(0.3)

        df_cf = api.taiwan_stock_cash_flows_statement(stock_id=ticker, start_date=START_DATE)
        api_calls_this_hour += 1; time.sleep(0.3)

        if not df_is.empty: 
            if 'stock_id' in df_is.columns: df_is['stock_id'] = df_is['stock_id'].astype(str).str.replace('*', '', regex=False)
            df_is.to_csv(TEMP_IS, mode='a', header=not os.path.exists(TEMP_IS), index=False, encoding='utf-8-sig')
        if not df_bs.empty: 
            if 'stock_id' in df_bs.columns: df_bs['stock_id'] = df_bs['stock_id'].astype(str).str.replace('*', '', regex=False)
            df_bs.to_csv(TEMP_BS, mode='a', header=not os.path.exists(TEMP_BS), index=False, encoding='utf-8-sig')
        if not df_cf.empty: 
            if 'stock_id' in df_cf.columns: df_cf['stock_id'] = df_cf['stock_id'].astype(str).str.replace('*', '', regex=False)
            df_cf.to_csv(TEMP_CF, mode='a', header=not os.path.exists(TEMP_CF), index=False, encoding='utf-8-sig')
            
        print("成功 ✅")
    except Exception as e:
        print(f"失敗 ❌ ({str(e)})")
        time.sleep(2)

print("\n🎉 第一階段：全市場 2025 最新財報增量下載完畢！\n")

# ==========================================
# 4. 舊資料合併與去重複 (Merge & Deduplicate)
# ==========================================
print("🔄 第二階段：開始將新資料無縫合併至 10 年歷史金庫...")

def merge_and_save(old_file, temp_file):
    if not os.path.exists(old_file) or not os.path.exists(temp_file): return
    print(f"   -> 正在合併 {old_file} ...")
    df_old = pd.read_csv(old_file)
    df_new = pd.read_csv(temp_file)
    
    # 合併並依照 stock_id, date, type 進行去重複 (保留最新抓取的資料)
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined.drop_duplicates(subset=['stock_id', 'date', 'type'], keep='last', inplace=True)
    
    # 覆蓋舊的 CSV 備份檔
    df_combined.to_csv(old_file, index=False, encoding='utf-8-sig')
    
merge_and_save("tw_historical_is.csv", TEMP_IS)
merge_and_save("tw_historical_bs.csv", TEMP_BS)
merge_and_save("tw_historical_cf.csv", TEMP_CF)

# ==========================================
# 5. 終極壓縮引擎 (Parquet)
# ==========================================
print("\n📦 第三階段：啟動資料庫終極瘦身與壓縮程序...")

is_cols = ['Revenue', 'OperatingIncome', 'EPS', 'NetIncome']
bs_cols = ['EquityAttributableToOwnersOfParent', 'TotalEquity', 'CurrentLiabilities', 'NoncurrentLiabilities', 'CashAndCashEquivalents', 'CashAndCashEquivalents_per', 'OrdinaryShare', 'CapitalStock', 'OrdinaryShare_per', 'CapitalStock_per']
cf_cols = ['CashFlowsFromOperatingActivities', 'NetCashInflowFromOperatingActivities', 'CashProvidedByInvestingActivities', 'Depreciation', 'InterestExpense', 'PayTheInterest']

try:
    df_is = pd.read_csv("tw_historical_is.csv")
    df_is[df_is['type'].isin(is_cols)].to_parquet("tw_is_lite.parquet", index=False)
    
    df_bs = pd.read_csv("tw_historical_bs.csv")
    df_bs[df_bs['type'].isin(bs_cols)].to_parquet("tw_bs_lite.parquet", index=False)
    
    df_cf = pd.read_csv("tw_historical_cf.csv")
    df_cf[df_cf['type'].isin(cf_cols)].to_parquet("tw_cf_lite.parquet", index=False)
    
    print("\n🎊 任務圓滿結束！您的 V7.3 系統現在已經充滿 2025 年的最新火力了！")
    
    # 清理暫存檔
    for f in [TEMP_IS, TEMP_BS, TEMP_CF]:
        if os.path.exists(f): os.remove(f)

except Exception as e:
    print(f"\n❌ 壓縮階段發生錯誤: {e}")