import pandas as pd
import time
import os
import requests
from bs4 import BeautifulSoup
from FinMind.data import DataLoader
from datetime import datetime

# ==========================================
# 1. 🔑 請在這裡貼上您的 FinMind API Token
# ==========================================
API_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMy0wMSAwNjozMToxNCIsInVzZXJfaWQiOiJlcmljY2hpIiwiZW1haWwiOiJlcmljLmNoaTE5ODhAZ21haWwuY29tIiwiaXAiOiIzMS40Ni4yNDUuMTAxIn0.7QLdFe1Pc6i-ZTdjYzV_672LV7Klfu7vCGJ_BlfpKX8" 

# ==========================================
# 2. 自動獲取台股全市場代碼
# ==========================================
def get_tw_stock_list():
    if os.path.exists('tw_stock_list.csv'):
        print("📂 找到本地 tw_stock_list.csv，直接讀取...")
        return pd.read_csv('tw_stock_list.csv')['Code'].astype(str).tolist()
    
    print("🌐 未找到本地清單，開始自動抓取全台股代碼...")
    data = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    for mode in [2, 4]:
        url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
        res = requests.get(url, headers=headers)
        res.encoding = "big5"
        soup = BeautifulSoup(res.text, "html.parser")
        for row in soup.find("table", {"class": "h4"}).find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) == 7 and "　" in cells[0].text:
                code = cells[0].text.strip().split("　")[0]
                if len(code) == 4: data.append(code)
    return list(set(data))

# ==========================================
# 3. 啟動 API 與參數設定
# ==========================================
api = DataLoader()
api.login_by_token(api_token=API_TOKEN)

target_tickers = get_tw_stock_list()
start_date = "2014-01-01" 

# 設定存檔路徑 (產出三大財務報表)
IS_FILE = "tw_historical_is.csv" # 綜合損益表
BS_FILE = "tw_historical_bs.csv" # 資產負債表
CF_FILE = "tw_historical_cf.csv" # 現金流量表

# ==========================================
# 4. 斷點續傳機制
# ==========================================
done_tickers = set()
if os.path.exists(IS_FILE):
    try:
        df_exist = pd.read_csv(IS_FILE)
        done_tickers.update(df_exist['stock_id'].astype(str).unique())
        print(f"✅ 斷點續傳啟動：發現已完成 {len(done_tickers)} 檔股票，將自動跳過。")
    except: pass

remaining_tickers = [t for t in target_tickers if t not in done_tickers]
print(f"🚀 準備開始下載！剩餘待抓取股票：{len(remaining_tickers)} 檔")

# ==========================================
# 5. 安全限速下載迴圈
# ==========================================
api_calls_this_hour = 0
MAX_CALLS_PER_HOUR = 500 # 預留安全邊際

for i, ticker in enumerate(remaining_tickers):
    if api_calls_this_hour >= (MAX_CALLS_PER_HOUR - 3):
        print(f"\n💤 [{datetime.now().strftime('%H:%M:%S')}] 達到安全上限({api_calls_this_hour}次)。")
        print("程式自動進入休眠 60 分鐘... 請勿關閉電腦與 VS Code...")
        time.sleep(3600)
        api_calls_this_hour = 0
        print(f"☀️ [{datetime.now().strftime('%H:%M:%S')}] 滿血復活，繼續抓取！\n")

    print(f"[{i+1}/{len(remaining_tickers)}] 📥 下載 {ticker} 三大財報...", end=" ")
    
    try:
        df_is = api.taiwan_stock_financial_statement(stock_id=ticker, start_date=start_date)
        api_calls_this_hour += 1
        time.sleep(0.3)
        
        df_bs = api.taiwan_stock_balance_sheet(stock_id=ticker, start_date=start_date)
        api_calls_this_hour += 1
        time.sleep(0.3)

        df_cf = api.taiwan_stock_cash_flows_statement(stock_id=ticker, start_date=start_date)
        api_calls_this_hour += 1
        time.sleep(0.3)

        if not df_is.empty:
            df_is.to_csv(IS_FILE, mode='a', header=not os.path.exists(IS_FILE), index=False, encoding='utf-8-sig')
        if not df_bs.empty:
            df_bs.to_csv(BS_FILE, mode='a', header=not os.path.exists(BS_FILE), index=False, encoding='utf-8-sig')
        if not df_cf.empty:
            df_cf.to_csv(CF_FILE, mode='a', header=not os.path.exists(CF_FILE), index=False, encoding='utf-8-sig')
            
        print("成功 ✅")
        
    except Exception as e:
        print(f"失敗 ❌ ({str(e)})")
        time.sleep(5) 

print("\n🎉 全市場歷史【三大財報】下載完畢！")