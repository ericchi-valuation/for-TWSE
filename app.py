import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import warnings

# ==========================================
# 頁面與基本設定
# ==========================================
st.set_page_config(page_title="V5.5 Eric Chi估值模型", page_icon="📊", layout="wide")
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 0. 基礎爬蟲 (OpenAPI 防止封鎖)
# ==========================================
@st.cache_data(ttl=86400)
# [V6.0 積木 1] 強化版產業抓取器 (加入全產業結構備援)
@st.cache_data(ttl=86400)
def fetch_twse_isin():
    data = []
    try:
        res_l = requests.get("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", timeout=5)
        if res_l.status_code == 200:
            for item in res_l.json():
                if len(item.get("公司代號", "")) == 4:
                    data.append({"Code": item["公司代號"], "Name": item["公司名稱"], "Industry": item["產業別"], "Ticker": f"{item['公司代號']}.TW"})
    except: pass

    try:
        res_o = requests.get("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", timeout=5)
        if res_o.status_code == 200:
            for item in res_o.json():
                if len(item.get("公司代號", "")) == 4:
                    data.append({"Code": item["公司代號"], "Name": item["公司名稱"], "Industry": item["產業別"], "Ticker": f"{item['公司代號']}.TWO"})
    except: pass

    df = pd.DataFrame(data)
    
    # 若 OpenAPI 雙雙陣亡，啟用「完整產業地圖」備援機制
    if df.empty or len(df) < 100:
        st.toast("⚠️ 政府 OpenAPI 連線不穩，已啟動離線產業地圖。", icon="📡")
        # 建立涵蓋所有主要產業的基礎清單 (簡化版示範)
        backup_list = [
            ("2330", "台積電", "半導體業"), ("2454", "聯發科", "半導體業"),
            ("2317", "鴻海", "其他電子業"), ("2382", "廣達", "電腦及週邊設備業"),
            ("3017", "奇鋐", "電腦及週邊設備業"), ("1519", "華城", "電機機械"),
            ("2881", "富邦金", "金融保險業"), ("2882", "國泰金", "金融保險業"),
            ("2002", "中鋼", "鋼鐵工業"), ("2603", "長榮", "航運業")
        ]
        df = pd.DataFrame([{"Code": c, "Name": n, "Industry": i, "Ticker": f"{c}.TW"} for c, n, i in backup_list])
    
    return df[df['Industry'] != '']
def get_tw_yahoo_cum_growth(symbol):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        clean_code = symbol.split('.')[0]
        url = f"https://tw.stock.yahoo.com/quote/{clean_code}.TW/revenue"
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        for row in soup.select('li.List\(n\)'):
            label = row.select_one('div > span')
            if label and '累計營收年增率' in label.text:
                val = row.select('div > span')[-1].text.replace('%', '').replace(',', '').strip()
                return float(val) / 100.0
        return None
    except: return None

# [V6.0 積木 2] 時點財務過濾器 (Point-in-Time Engine)
def get_pit_financials(stock, target_date):
    """
    輸入目標日期，回傳該日期當下「最新已公布」的財務數據。
    排除未來數據 (Look-ahead bias)。
    """
    try:
        # 將目標日期轉換為 pandas datetime
        target_dt = pd.to_datetime(target_date).tz_localize(None)
        
        q_fin = stock.quarterly_financials.T
        if q_fin.empty: return None, None
        
        # 統一處理時區，確保能比對
        q_fin.index = pd.to_datetime(q_fin.index).tz_localize(None)
        
        # 核心邏輯：財報公布通常有延遲 (約 45 天)。
        # 為了嚴謹回測，我們假設「財報結算日 + 45天」才是市場真正看到數據的日子。
        # 篩選出「公布日」早於「我們回測目標日」的財報。
        available_reports = q_fin[q_fin.index + pd.Timedelta(days=45) <= target_dt]
        
        if available_reports.empty:
            return None, None
            
        # 取得當時最新的一季數據
        latest_pit_report = available_reports.iloc[0]
        
        # 取得當時的近四季 (TTM) EPS 用於計算 P/E
        pit_eps_ttm = available_reports['Basic EPS'].head(4).sum() if 'Basic EPS' in available_reports.columns else 0
        
        # 取得當時最新一季的 EBITDA 用於後續計算
        pit_ebitda = latest_pit_report.get('EBITDA', latest_pit_report.get('EBIT', 0))
        
        return pit_eps_ttm, pit_ebitda
    except Exception as e:
        return None, None
# ==========================================
# 1. 歷史區間計算 (完整還原 V5.0)
# ==========================================
def get_historical_metrics(stock, hist_data):
    try:
        if hist_data.empty: return "-", "-", "-", "-", 0
        hist_data.index = hist_data.index.tz_localize(None)
        
        fin = stock.quarterly_financials.T
        bs = stock.quarterly_balance_sheet.T
        if fin.empty or bs.empty: return "-", "-", "-", "-", 0
        
        fin.index = pd.to_datetime(fin.index).tz_localize(None)
        bs.index = pd.to_datetime(bs.index).tz_localize(None)
        
        pe_vals, pb_vals, ps_vals, evebitda_vals = [], [], [], []
        shares = stock.info.get('sharesOutstanding', 1)
        
        for rpt_date in fin.index:
            if rpt_date not in hist_data.index:
                nearest_idx = hist_data.index.get_indexer([rpt_date], method='nearest')[0]
                if nearest_idx == -1: continue
                price = hist_data.iloc[nearest_idx]['Close']
            else:
                price = hist_data.loc[rpt_date]['Close']
            
            # EV/EBITDA
            if rpt_date in bs.index:
                total_debt = bs.loc[rpt_date, 'Total Debt'] if 'Total Debt' in bs.columns else 0
                cash = bs.loc[rpt_date, 'Cash And Cash Equivalents'] if 'Cash And Cash Equivalents' in bs.columns else 0
                ev = (price * shares) + total_debt - cash
                ebitda = 0
                if 'EBITDA' in fin.columns: ebitda = fin.loc[rpt_date, 'EBITDA']
                elif 'EBIT' in fin.columns: ebitda = fin.loc[rpt_date, 'EBIT']
                if ebitda > 0:
                    ratio = ev / (ebitda * 4) 
                    if 0 < ratio < 100: evebitda_vals.append(ratio)
            
            # P/E
            if 'Basic EPS' in fin.columns:
                eps = fin.loc[rpt_date, 'Basic EPS']
                if eps > 0: pe_vals.append(price / (eps * 4))
            
            # P/S
            if 'Total Revenue' in fin.columns:
                rev = fin.loc[rpt_date, 'Total Revenue']
                if rev > 0: ps_vals.append(price / ((rev/shares) * 4))
                
            # P/B
            if rpt_date in bs.index and 'Stockholders Equity' in bs.columns:
                bv = bs.loc[rpt_date, 'Stockholders Equity']
                if bv > 0: pb_vals.append(price / (bv/shares))
                
        def fmt_rng(vals):
            clean = [v for v in vals if 0 < v < 150]
            if not clean: return "-"
            return f"{min(clean):.1f}-{max(clean):.1f}"
            
        return fmt_rng(pe_vals), fmt_rng(pb_vals), fmt_rng(ps_vals), fmt_rng(evebitda_vals), (np.mean(pe_vals) if pe_vals else 0)
    except: return "-", "-", "-", "-", 0

# ==========================================
# 2. 估值核心 (3-Stage DCF)
# ==========================================
def get_3_stage_valuation(stock, is_finance, real_growth):
    try:
        info = stock.info; shares = info.get('sharesOutstanding', 1)
        bs = stock.balance_sheet.fillna(0); fin = stock.financials.fillna(0)
        if bs.empty or fin.empty: return 0, 0, 0.1, 0
        
        beta = info.get('beta', 1.0) or 1.0
        ke = max(0.035 + beta * 0.06, 0.07)
        equity = bs.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in bs.index else 1
        debt = bs.loc['Total Debt'].iloc[0] if 'Total Debt' in bs.index else 0
        cash = bs.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs.index else 0
        ebit = fin.loc['EBIT'].iloc[0] if 'EBIT' in fin.index else 0
        
        invested_capital = equity + debt - cash
        roic = (ebit * 0.8 / invested_capital) if invested_capital > 0 else 0.05
        wacc = max((equity/(equity+debt))*ke + (debt/(equity+debt))*0.025, 0.08) if is_finance else (equity/(equity+debt))*ke + (debt/(equity+debt))*0.025
        
        g1 = min(max(real_growth * 0.8, 0.02), 0.25)
        g_term = 0.025; g2 = (g1 + g_term) / 2
        
        if is_finance: base_cf = (info.get('netIncomeToCommon', 0) * 0.6)
        else:
            nopat = ebit * 0.8
            if nopat <= 0: return 0, g1, wacc, roic
            base_cf = nopat * 0.7 
            
        dcf_sum = 0; cf = base_cf
        for i in range(1, 4): cf *= (1 + g1); dcf_sum += cf / ((1 + wacc)**i)
        for i in range(4, 6): cf *= (1 + g2); dcf_sum += cf / ((1 + wacc)**i)
        tv = (cf * (1 + g_term)) / (wacc - g_term)
        dcf_sum += tv / ((1 + wacc)**5)
        
        equity_val = dcf_sum - (debt if not is_finance else 0) + (cash if not is_finance else 0)
        return max(equity_val / shares, 0), g1, wacc, roic
    except: return 0, 0, 0.1, 0

# ==========================================
# 3. 評分與資料整合
# ==========================================
def compile_stock_data(symbol, industry_name, stock, info, price, real_growth, qoq_growth, wacc, roic, 
                       pe_rng, pb_rng, ps_rng, ev_rng, avg_pe, cur_pe, cur_ev_ebitda, 
                       intrinsic, upside, eps, median_pe, is_finance):
    
    # 評分邏輯
    scores = {'Q': 0, 'V': 0, 'G': 0, 'Total': 0, 'Msg': []}
    if real_growth > 0.15: w_q, w_v, w_g = 0.2, 0.3, 0.5; lifecycle = "Growth"
    elif real_growth < 0.05: w_q, w_v, w_g = 0.5, 0.4, 0.1; lifecycle = "Mature"
    else: w_q, w_v, w_g = 0.3, 0.4, 0.3; lifecycle = "Stable"

    try: icr = stock.financials.fillna(0).loc['EBIT'].iloc[0] / abs(stock.financials.fillna(0).loc['Interest Expense'].iloc[0])
    except: icr = 10
    if icr > 5: scores['Q'] += 4
    elif icr < 1.5: scores['Q'] -= 5; scores['Msg'].append("高財務風險")
    else: scores['Q'] += 1
    
    if roic > wacc: scores['Q'] += 4
    else: scores['Q'] -= 2; scores['Msg'].append("ROIC<WACC")

    if upside > 0.15: scores['V'] += 4
    elif upside > 0.0: scores['V'] += 2
    if avg_pe > 0 and 0 < cur_pe < (avg_pe * 1.1): scores['V'] += 3
    if median_pe > 0 and 0 < cur_pe < median_pe: scores['V'] += 3
    if 0 < cur_ev_ebitda < 18: scores['V'] += 3

    if real_growth > 0.10 and roic < wacc: scores['G'] -= 5; scores['Msg'].append("無效成長")
    else:
        if real_growth > 0.20: scores['G'] += 5
        elif real_growth > 0.10: scores['G'] += 3
    if qoq_growth > 0.05: scores['G'] += 3
    elif qoq_growth < -0.05: scores['G'] -= 3; scores['Msg'].append("動能轉弱")

    scores['Total'] = (scores['Q'] * w_q * 10) + (scores['V'] * w_v * 10) + (scores['G'] * w_g * 10)
    
    status = f"{lifecycle} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}" + (f" | ⚠️{' '.join(scores['Msg'])}" if scores['Msg'] else "")
    logic = f"Score: {int(scores['Total'])}" + (" (首選)" if scores['Total'] >= 70 else "")

    # 完整表格欄位
    est_eps = eps * (1 + min(real_growth, 0.1))
    ratios = {'op': info.get('operatingMargins', 0), 'net': info.get('profitMargins', 0)}
    ev_str = f"{cur_ev_ebitda:.1f}" if cur_ev_ebitda > 0 else "-"

    return {
        '產業別': industry_name,
        '股票代碼': symbol,
        '名稱': info.get('shortName', symbol),
        '現價': price,
        '營收成長率': f"{real_growth*100:.1f}%",
        '營業利益率': f"{ratios['op']*100:.1f}%" if ratios['op'] else "-",
        '淨利率': f"{ratios['net']*100:.1f}%" if ratios['net'] else "-",
        '預估EPS': round(est_eps, 2),
        'P/E (TTM)': round(cur_pe, 1) if cur_pe else "-",
        'P/B (Lag)': round(info.get('priceToBook', 0) or 0, 2),
        'P/S (Lag)': round(info.get('priceToSalesTrailing12Months', 0) or 0, 2),
        'EV/EBITDA': ev_str,
        '預估範圍P/E': pe_rng,
        '預估範圍P/B': pb_rng,
        '預估範圍P/S': ps_rng,
        '預估範圍EV/EBITDA': ev_rng,
        'DCF/DDM合理價': round(intrinsic, 1),
        '狀態': status,
        'vs產業PE': "低於同業" if cur_pe < median_pe else "高於同業",
        '選股邏輯': logic,
        'Total_Score': scores['Total']
    }

# ==========================================
# UI 介面
# ==========================================
st.title("V5.5 Eric Chi估值模型")

tab1, tab2, tab3 = st.tabs(["全產業掃描", "單股查詢", "歷史回測"])

# --- Tab 1: 產業掃描 (V4.6 全掃描邏輯回歸) ---
with tab1:
    with st.spinner("載入產業清單中..."):
        df_all = fetch_twse_isin()
        
    if not df_all.empty:
        valid_industries = sorted([i for i in df_all['Industry'].unique()])
        st.info(f"系統共偵測到 {len(valid_industries)} 個產業。全市場掃描將動態印出各產業 Top 6 企業，整體耗時較長，請保持網頁開啟。")
        
        if st.button("執行全產業掃描", type="primary"):
            pb = st.progress(0)
            status_text = st.empty()
            
            # 建立一個容器，用來動態裝載印出來的各產業表格
            results_container = st.container()
            
            total_inds = len(valid_industries)
            cols_display = ['股票代碼', '名稱', '現價', '營收成長率', '營業利益率', '淨利率', 
                            '預估EPS', 'P/E (TTM)', 'P/B (Lag)', 'P/S (Lag)', 'EV/EBITDA',
                            '預估範圍P/E', '預估範圍P/B', '預估範圍P/S', '預估範圍EV/EBITDA',
                            'DCF/DDM合理價', '狀態', 'vs產業PE', '選股邏輯']
            
            for idx, ind in enumerate(valid_industries):
                status_text.text(f"進度: {idx+1}/{total_inds} | 正在精算 [{ind}]...")
                
                tickers = df_all[df_all["Industry"] == ind]["Ticker"].tolist()
                if not tickers:
                    pb.progress((idx + 1) / total_inds)
                    continue
                    
                # 市值初篩 (保留前 50% 加快運算速度)
                caps = []
                for t in tickers:
                    try: caps.append((t, yf.Ticker(t).fast_info['market_cap']))
                    except: pass
                caps.sort(key=lambda x: x[1], reverse=True)
                targets = [x[0] for x in caps[:max(len(caps)//2, 1)]]
                
                ind_pes = []
                raw_data = []
                
                for sym in targets:
                    try:
                        stock = yf.Ticker(sym); info = stock.info
                        price = info.get('currentPrice') or info.get('previousClose')
                        if not price: continue
                        
                        real_g = get_tw_yahoo_cum_growth(sym) or info.get('revenueGrowth', 0.0)
                        
                        q_fin = stock.quarterly_financials
                        qoq_g = (q_fin.loc['Total Revenue'].iloc[0] - q_fin.loc['Total Revenue'].iloc[1]) / q_fin.loc['Total Revenue'].iloc[1] if not q_fin.empty and len(q_fin.columns) >= 2 else 0
                        
                        hist = stock.history(period="10y")
                        pe_rng, pb_rng, ps_rng, ev_rng, avg_pe = get_historical_metrics(stock, hist)
                        
                        eps = info.get('trailingEps', 0)
                        cur_pe = price / eps if eps > 0 else 0
                        if 0 < cur_pe < 120: ind_pes.append(cur_pe)
                        
                        cur_ev = info.get('enterpriseToEbitda', 0)
                        if not cur_ev:
                            mcap = price * info.get('sharesOutstanding', 1)
                            cur_ev = (mcap + info.get('totalDebt', 0) - info.get('totalCash', 0)) / info.get('ebitda', 1)
                            
                        is_fin = any(x in ind for x in ["金融", "保險"])
                        intrinsic, g_used, wacc, roic = get_3_stage_valuation(stock, is_fin, real_g)
                        upside = (intrinsic - price) / price if intrinsic > 0 else -1
                        
                        raw_data.append((sym, ind, stock, info, price, real_g, qoq_g, wacc, roic, pe_rng, pb_rng, ps_rng, ev_rng, avg_pe, cur_pe, cur_ev, intrinsic, upside, eps, is_fin))
                    except: pass
                
                # 計算該產業中位數PE
                pe_med = np.median(ind_pes) if ind_pes else 22.0
                
                # 彙整該產業所有股票評分
                ind_results = []
                for d in raw_data:
                    ind_results.append(compile_stock_data(*d[:19], pe_med, d[19]))
                    
                # 排序並印出 Top 6
                if ind_results:
                    df_res = pd.DataFrame(ind_results)
                    df_ind = df_res.sort_values(by='Total_Score', ascending=False).head(6)
                    
                    # 將結果即時繪製到畫面上，讓使用者不需要乾等
                    with results_container:
                        st.markdown(f"### 🏆 {ind} (精選 Top 6)")
                        st.dataframe(df_ind[cols_display], use_container_width=True)
                        
                pb.progress((idx + 1) / total_inds)
                
            status_text.text("✅ 全市場產業掃描完成！")

# --- Tab 2: 單股查詢 ---
with tab2:
    col_input, col_info = st.columns([1, 2])
    with col_input:
        stock_code = st.text_input("輸入股票代碼:", value="2330")
        if st.button("查詢", type="primary"):
            sym = f"{stock_code}.TW"
            with st.spinner("查詢中..."):
                try:
                    stock = yf.Ticker(sym); info = stock.info
                    price = info.get('currentPrice') or info.get('previousClose')
                    real_g = get_tw_yahoo_cum_growth(sym) or info.get('revenueGrowth', 0.0)
                    pe_rng, pb_rng, ps_rng, ev_rng, avg_pe = get_historical_metrics(stock, stock.history(period="10y"))
                    eps = info.get('trailingEps', 0); cur_pe = price/eps if eps>0 else 0
                    cur_ev = info.get('enterpriseToEbitda', 0)
                    is_fin = "Financial" in info.get('sector', '')
                    intrinsic, g_used, wacc, roic = get_3_stage_valuation(stock, is_fin, real_g)
                    upside = (intrinsic - price) / price if intrinsic > 0 else -1
                    
                    data = compile_stock_data(sym, info.get('industry', 'N/A'), stock, info, price, real_g, 0, wacc, roic, pe_rng, pb_rng, ps_rng, ev_rng, avg_pe, cur_pe, cur_ev, intrinsic, upside, eps, 22.0, is_fin)
                    
                    st.metric("現價", f"{price} TWD")
                    st.metric("合理價", f"{intrinsic:.1f} TWD", f"{upside:.1%} 潛在空間")
                    st.progress(data['Total_Score']/100, text=f"模型評分: {int(data['Total_Score'])}")
                    st.info(data['狀態'])
                    
                    with col_info: 
                        st.dataframe(pd.DataFrame([data]).drop(columns=['Total_Score', '產業別']).T, use_container_width=True)
                except Exception as e: 
                    st.error("查無資料或發生錯誤")

# --- Tab 3: 歷史回測 ---
with tab3:
    c1, c2 = st.columns(2)
    with c1: t_input = st.text_area("測試代碼 (逗號分隔):", "1519.TW, 3017.TW, 2330.TW")
    with c2: s_date = st.date_input("進場日:", datetime(2023, 11, 27)); run_bt = st.button("回測", type="primary")
    
    if run_bt:
        res_bt = []; pb = st.progress(0); t_list = [t.strip() for t in t_input.split(',')]
        for i, sym in enumerate(t_list):
            try:
                stock = yf.Ticker(sym); hist = stock.history(start=s_date); name = stock.info.get('shortName', sym)
                if not hist.empty:
                    ep = hist['Close'].iloc[0]; cp = hist['Close'].iloc[-1]
                    dates = hist.index
                    
                    def get_ret(days):
                        target_date = dates[0] + timedelta(days=days)
                        idx = dates.searchsorted(target_date)
                        if idx < len(dates):
                            p = hist['Close'].iloc[idx]
                            return (p - ep) / ep
                        return None

                    ret_3m = get_ret(90); ret_6m = get_ret(180); ret_12m = get_ret(365)
                    total_ret = (cp - ep) / ep
                    
                    res_bt.append({
                        '代碼': sym, '名稱': name, '進場價': round(ep,1), '現價': round(cp,1),
                        '3個月': f"{ret_3m*100:.1f}%" if ret_3m else "-",
                        '6個月': f"{ret_6m*100:.1f}%" if ret_6m else "-",
                        '12個月': f"{ret_12m*100:.1f}%" if ret_12m else "-",
                        '至今報酬': f"{total_ret*100:.1f}%", 'Raw': total_ret
                    })
            except: pass
            pb.progress((i+1)/len(t_list))
            
        if res_bt:
            df_bt = pd.DataFrame(res_bt)
            st.metric("投資組合平均至今報酬率", f"{df_bt['Raw'].mean()*100:.1f}%")
            st.dataframe(df_bt.drop(columns=['Raw']), use_container_width=True)