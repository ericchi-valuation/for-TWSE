import pandas as pd

print("🔍 正在掃描您的三大財報金庫欄位...\n")

# 定義要檢查的檔案
files_to_check = {
    "📊 綜合損益表 (IS)": "tw_historical_is.csv",
    "⚖️ 資產負債表 (BS)": "tw_historical_bs.csv",
    "💵 現金流量表 (CF)": "tw_historical_cf.csv"
}

for name, filename in files_to_check.items():
    try:
        df = pd.read_csv(filename)
        # 抓取 type 欄位中所有不重複的名稱，並轉成列表方便閱讀
        unique_types = df['type'].unique().tolist()
        
        print(f"{name} 包含的 type 有：")
        # 印出前 30 個，確保我們能看到核心的營收、淨利、淨值、現金流名稱
        print(unique_types[:30]) 
        print("-" * 60)
    except Exception as e:
        print(f"❌ 讀取 {filename} 失敗: {e}")
        print("-" * 60)

print("✅ 掃描完畢！請將上方印出的英文單字複製貼給 AI。")