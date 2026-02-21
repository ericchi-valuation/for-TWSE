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
st.set_page_config(page_title="V6.1 Eric Chi估值模型", page_icon="📊", layout="wide")
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 0. 基礎爬蟲 (函數更名以強制清除舊的錯誤快取)
# ==========================================
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_industry_list_v6():
    """絕對乾淨的資料抓取層，內部絕對不可包含任何 st. 語法"""
    data = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    # 引擎 1: 證交所 ISIN 網頁爬蟲 (抓取最完整 27 產業)
    try:
        for mode in [2, 4]:
            url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
            res = requests.get(url, headers=headers, timeout=10)
            res.encoding = "big5"
            soup = BeautifulSoup(res.text, "html.parser")
            rows = soup.find("table", {"class": "h4"}).find_all("tr")[1:]
            for row in rows:
                cells = row.find_all("td")
                if len(cells) != 7: continue
                raw = cells[0].text.strip()
                if "　" in raw:
                    code, name = raw.split("　", 1)
                    if len(code) == 4:
                        industry = cells[4].text.strip()
                        if industry:
                            ticker = f"{code}.TW" if mode == 2 else f"{code}.TWO"
                            data.append({"Code": code, "Name": name, "Industry": industry, "Ticker": ticker})
        if len(data) > 100: return pd.DataFrame(data).drop_duplicates(subset=['Code'])
    except: pass

    # 引擎 2: OpenAPI 備援
    try:
        res_l = requests.get("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", timeout=10)
        if res_l.status_code == 200:
            for item in res_l.json():
                if len(item.get("公司代號", "")) == 4:
                    data.append({"Code": item["公司代號"], "Name": item["公司名稱"], "Industry": item["產業別"], "Ticker": f"{item['公司代號']}.TW"})
        res_o = requests.get("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", timeout=10)
        if res_o.status_code == 200:
            for item in res_o.json():
                if len(item.get("公司代號", "")) == 4:
                    data.append({"Code": item["公司代號"], "Name": item["公司名稱"], "Industry": item["產業別"], "Ticker": f"{item['公司代號']}.TWO"})
        if len(data) > 100: return pd.DataFrame(data).drop_duplicates(subset=['Code'])
    except: pass

    return pd.DataFrame() 

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

# ==========================================
# 1. 歷史區間計算
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
            
            if rpt_date in bs.index:
                ev = (price * shares) + bs.loc[rpt_date].get('Total Debt', 0) - bs.loc[rpt_date].get('Cash And Cash Equivalents', 0)
                ebitda = fin.loc[rpt_date].get('EBITDA', fin.loc[rpt_date].get('EBIT', 0))
                if ebitda > 0:
                    ratio = ev / (ebitda * 4) 
                    if 0 < ratio < 100: evebitda_vals.append(ratio)
            
            eps = fin.loc[rpt_date].get('Basic EPS', 0)
            if eps > 0: pe_vals.append(price / (eps * 4))
            
            rev = fin.loc[rpt_date].get('Total Revenue', 0)
            if rev > 0: ps_vals.append(price / ((rev/shares) * 4))
                
            bv = bs.loc[rpt_date].get('Stockholders Equity', 0) if rpt_date in bs.index else 0
            if bv > 0: pb_vals.append(price / (bv/shares))
                
        def fmt_rng(vals):
            clean = [v for v in vals if 0 < v < 150]
            return f"{min(clean):.1f}-{max(clean):.1f}" if clean else "-"
            
        return fmt_rng(pe_vals), fmt_rng(pb_vals), fmt_rng(ps_vals), fmt_rng(evebitda_vals), (np.mean(pe_vals) if pe_vals else 0)
    except: return "-", "-", "-", "-", 0

# ==========================================
# 2. 估值核心
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
        
        base_cf = (info.get('netIncomeToCommon', 0) * 0.6) if is_finance else (ebit * 0.8 * 0.7)
        if base_cf <= 0: return 0, g1, wacc, roic
            
        dcf_sum = sum([base_cf * ((1 + g1)**i) / ((1 + wacc)**i) for i in range(1, 4)])
        dcf_sum += sum([(base_cf * ((1 + g1)**3)) * ((1 + g2)**(i-3)) / ((1 + wacc)**i) for i in range(4, 6)])
        tv = ((base_cf * ((1 + g1)**3) * ((1 + g2)**2)) * (1 + g_term)) / (wacc - g_term)
        dcf_sum += tv / ((1 + wacc)**5)
        
        equity_val = dcf_sum - (debt if not is_finance else 0) + (cash if not is_finance else 0)
        return max(equity_val / shares, 0), g1, wacc, roic
    except: return 0, 0, 0.1, 0

# ==========================================
# 3. 評分邏輯
# ==========================================
def calculate_scores(info, financials, growth_rate, qoq_growth, valuation_upside, cur_pe, cur_ev_ebitda, hist_avg_pe, industry_pe_median, wacc, roic):
    scores = {'Q': 0, 'V': 0, 'G': 0, 'Total': 0, 'Msg': []}
    w_q, w_v, w_g = (0.2, 0.3, 0.5) if growth_rate > 0.15 else ((0.5, 0.4, 0.1) if growth_rate < 0.05 else (0.3, 0.4, 0.3))
    scores['Lifecycle'] = "Growth" if growth_rate > 0.15 else ("Mature" if growth_rate < 0.05 else "Stable")

    try: icr = financials.loc['EBIT'].iloc[0] / abs(financials.loc['Interest Expense'].iloc[0])
    except: icr = 10
    if icr > 5: scores['Q'] += 4
    elif icr < 1.5: scores['Q'] -= 5; scores['Msg'].append("高財務風險")
    else: scores['Q'] += 1
    
    if roic > wacc: scores['Q'] += 4
    else: scores['Q'] -= 2; scores['Msg'].append("ROIC<WACC")

    if valuation_upside > 0.15: scores['V'] += 4
    elif valuation_upside > 0.0: scores['V'] += 2
    if hist_avg_pe > 0 and 0 < cur_pe < (hist_avg_pe * 1.1): scores['V'] += 3
    if industry_pe_median > 0 and 0 < cur_pe < industry_pe_median: scores['V'] += 3
    if 0 < cur_ev_ebitda < 18: scores['V'] += 3

    if growth_rate > 0.10 and roic < wacc: scores['G'] -= 5; scores['Msg'].append("無效成長")
    else:
        if growth_rate > 0.20: scores['G'] += 5
        elif growth_rate > 0.10: scores['G'] += 3
    if qoq_growth > 0.05: scores['G'] += 3
    elif qoq_growth < -0.05: scores['G'] -= 3; scores['Msg'].append("動能轉弱")
    
    if 0 < info.get('pegRatio', 0) < 1.5: scores['G'] += 2

    scores['Total'] = (scores['Q'] * w_q * 10) + (scores['V'] * w_v * 10) + (scores['G'] * w_g * 10)
    return scores

def compile_stock_data(symbol, ind, stock, info, price, real_g, qoq_g, wacc, roic, ranges, avg_pe, cur_pe, cur_ev, intrinsic, upside, eps, med_pe, is_fin):
    scores = calculate_scores(info, stock.financials.fillna(0), real_g, qoq_g, upside, cur_pe, cur_ev, avg_pe, med_pe, wacc, roic)
    status = f"{scores['Lifecycle']} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}" + (f" | ⚠️{' '.join(scores['Msg'])}" if scores['Msg'] else "")
    logic = f"Score: {int(scores['Total'])}" + (" (首選)" if scores['Total'] >= 70 else "")
    
    return {
        '產業別': ind, '股票代碼': symbol, '名稱': info.get('shortName', symbol), '現價': price,
        '營收成長率': f"{real_g*100:.1f}%", '營業利益率': f"{info.get('operatingMargins', 0)*100:.1f}%", '淨利率': f"{info.get('profitMargins', 0)*100:.1f}%",
        '預估EPS': round(eps * (1 + min(real_g, 0.1)), 2), 'P/E (TTM)': round(cur_pe, 1) if cur_pe else "-",
        'P/B (Lag)': round(info.get('priceToBook', 0) or 0, 2), 'P/S (Lag)': round(info.get('priceToSalesTrailing12Months', 0) or 0, 2),
        'EV/EBITDA': f"{cur_ev:.1f}" if cur_ev > 0 else "-",
        '預估範圍P/E': ranges[0], '預估範圍P/B': ranges[1], '預估範圍P/S': ranges[2], '預估範圍EV/EBITDA': ranges[3],
        'DCF/DDM合理價': round(intrinsic, 1), '狀態': status, 'vs產業PE': "低於同業" if cur_pe < med_pe else "高於同業",
        '選股邏輯': logic, 'Total_Score': scores['Total']
    }

# ==========================================
# 4. 時點回測引擎 (Point-in-Time Engine)
# ==========================================
def run_pit_backtest(sym, stock, target_date, is_finance):
    try:
        target_dt = pd.to_datetime(target_date).tz_localize(None)
        hist = stock.history(start=target_dt - pd.Timedelta(days=3650), end=datetime.today())
        if hist.empty or hist[hist.index >= target_dt].empty: return None

        entry_price = hist[hist.index >= target_dt]['Close'].iloc[0]
        current_price = hist['Close'].iloc[-1]

        q_fin = stock.quarterly_financials.T
        q_bs = stock.quarterly_balance_sheet.T
        if q_fin.empty or q_bs.empty: return None
        
        q_fin.index = pd.to_datetime(q_fin.index).tz_localize(None)
        q_bs.index = pd.to_datetime(q_bs.index).tz_localize(None)
        
        valid_dates = q_fin.index[q_fin.index + pd.Timedelta(days=45) <= target_dt]
        if len(valid_dates) < 4: return None

        latest_date = valid_dates[0]
        eps_ttm = q_fin.loc[valid_dates[:4], 'Basic EPS'].sum() if 'Basic EPS' in q_fin.columns else 0
        rev_ttm = q_fin.loc[valid_dates[:4], 'Total Revenue'].sum() if 'Total Revenue' in q_fin.columns else 0
        prev_rev_ttm = q_fin.loc[valid_dates[4:8], 'Total Revenue'].sum() if 'Total Revenue' in q_fin.columns and len(valid_dates) >= 8 else 0
        
        real_growth = (rev_ttm - prev_rev_ttm) / prev_rev_ttm if prev_rev_ttm > 0 else 0.05
        qoq_growth = (q_fin.loc[valid_dates[0], 'Total Revenue'] - q_fin.loc[valid_dates[1], 'Total Revenue']) / q_fin.loc[valid_dates[1], 'Total Revenue'] if len(valid_dates) > 1 else 0

        ebit = q_fin.loc[latest_date].get('EBIT', 0)
        ebitda = q_fin.loc[latest_date].get('EBITDA', ebit)
        equity = q_bs.loc[latest_date].get('Stockholders Equity', 1)
        debt = q_bs.loc[latest_date].get('Total Debt', 0)
        cash = q_bs.loc[latest_date].get('Cash And Cash Equivalents', 0)
        shares = stock.info.get('sharesOutstanding', 1)

        cur_pe = entry_price / eps_ttm if eps_ttm > 0 else 0
        cur_ev_ebitda = ((entry_price * shares) + debt - cash) / (ebitda * 4) if ebitda > 0 else 0

        beta = stock.info.get('beta', 1.0)
        ke = max(0.035 + beta * 0.06, 0.07)
        invested_capital = equity + debt - cash
        roic = (ebit * 0.8 * 4 / invested_capital) if invested_capital > 0 else 0.05
        wacc = max((equity/(equity+debt))*ke + (debt/(equity+debt))*0.025, 0.08) if is_finance else (equity/(equity+debt))*ke + (debt/(equity+debt))*0.025

        g1 = min(max(real_growth * 0.8, 0.02), 0.25); g_term = 0.025; g2 = (g1 + g_term) / 2
        base_cf = (q_fin.loc[latest_date].get('Net Income', 0) * 4 * 0.6) if is_finance else (ebit * 4 * 0.8 * 0.7)
        
        if base_cf <= 0: intrinsic = 0
        else:
            dcf_sum = sum([base_cf * ((1 + g1)**i) / ((1 + wacc)**i) for i in range(1, 4)])
            dcf_sum += sum([(base_cf * ((1 + g1)**3)) * ((1 + g2)**(i-3)) / ((1 + wacc)**i) for i in range(4, 6)])
            tv = ((base_cf * ((1 + g1)**3) * ((1 + g2)**2)) * (1 + g_term)) / (wacc - g_term)
            dcf_sum += tv / ((1 + wacc)**5)
            intrinsic = max((dcf_sum - (debt if not is_finance else 0) + (cash if not is_finance else 0)) / shares, 0)

        upside = (intrinsic - entry_price) / entry_price if intrinsic > 0 else -1

        pe_vals = []
        for d in valid_dates[:20]:
            try:
                p = hist.loc[hist.index <= d]['Close'].iloc[-1]
                e = q_fin.loc[d, 'Basic EPS']
                if e > 0: pe_vals.append(p / (e * 4))
            except: pass
        avg_pe = np.mean(pe_vals) if pe_vals else 0

        mock_fin = pd.DataFrame({'EBIT': [ebit], 'Interest Expense': [abs(q_fin.loc[latest_date].get('Interest Expense', ebit*0.1))]})
        scores = calculate_scores(stock.info, mock_fin, real_growth, qoq_growth, upside, cur_pe, cur_ev_ebitda, avg_pe, 22.0, wacc, roic)

        dates = hist[hist.index >= target_dt].index
        def get_ret(days):
            td = dates[0] + pd.Timedelta(days=days)
            idx = dates.searchsorted(td)
            if idx < len(dates): return (hist['Close'].iloc[idx] - entry_price) / entry_price
            return None

        return {
            '代碼': sym, '名稱': stock.info.get('shortName', sym), '進場日': target_dt.strftime('%Y-%m-%d'),
            '進場價': round(entry_price, 1), '現價': round(current_price, 1),
            '當時總分': int(scores['Total']), '當時狀態': f"{scores['Lifecycle']} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}",
            '當時合理價': round(intrinsic, 1), '當時PE': round(cur_pe, 1),
            '3個月': f"{get_ret(90)*100:.1f}%" if get_ret(90) else "-",
            '6個月': f"{get_ret(180)*100:.1f}%" if get_ret(180) else "-",
            '12個月': f"{get_ret(365)*100:.1f}%" if get_ret(365) else "-",
            '至今報酬': f"{(current_price - entry_price)/entry_price*100:.1f}%", 'Raw': (current_price - entry_price)/entry_price
        }
    except Exception as e: return None

# ==========================================
# UI 介面
# ==========================================
st.title("V6.1 Eric Chi估值模型")
tab1, tab2, tab3 = st.tabs(["全產業掃描", "單股查詢", "真·時光機回測"])

# --- Tab 1: 全產業掃描 ---
with tab1:
    with st.spinner("載入產業清單中..."):
        df_all = fetch_industry_list_v6()
        
    if df_all.empty:
        st.error("❌ 無法連線至證交所抓取產業清單。這通常是因為 Streamlit 雲端主機 IP 遭到台灣證交所防火牆阻擋，請稍後再試。")
    else:
        valid_industries = sorted([i for i in df_all['Industry'].unique()])
        st.info(f"偵測到 {len(valid_industries)} 個產業。掃描將動態印出各產業 Top 6，請保持網頁開啟。")
        if st.button("執行全產業掃描", type="primary"):
            pb = st.progress(0); status_text = st.empty(); results_container = st.container()
            total_inds = len(valid_industries)
            cols_display = ['股票代碼', '名稱', '現價', '營收成長率', '營業利益率', '淨利率', '預估EPS', 'P/E (TTM)', 'P/B (Lag)', 'P/S (Lag)', 'EV/EBITDA', '預估範圍P/E', '預估範圍P/B', '預估範圍P/S', '預估範圍EV/EBITDA', 'DCF/DDM合理價', '狀態', 'vs產業PE', '選股邏輯']
            
            for idx, ind in enumerate(valid_industries):
                status_text.text(f"進度: {idx+1}/{total_inds} | 正在精算 [{ind}]...")
                tickers = df_all[df_all["Industry"] == ind]["Ticker"].tolist()
                if not tickers: pb.progress((idx + 1) / total_inds); continue
                
                caps = []
                for t in tickers:
                    try: caps.append((t, yf.Ticker(t).fast_info['market_cap']))
                    except: pass
                caps.sort(key=lambda x: x[1], reverse=True)
                targets = [x[0] for x in caps[:max(len(caps)//2, 1)]]
                
                ind_pes = []; raw_data = []
                for sym in targets:
                    try:
                        stock = yf.Ticker(sym); info = stock.info
                        price = info.get('currentPrice') or info.get('previousClose')
                        if not price: continue
                        real_g = get_tw_yahoo_cum_growth(sym) or info.get('revenueGrowth', 0.0)
                        q_fin = stock.quarterly_financials
                        qoq_g = (q_fin.loc['Total Revenue'].iloc[0] - q_fin.loc['Total Revenue'].iloc[1]) / q_fin.loc['Total Revenue'].iloc[1] if not q_fin.empty and len(q_fin.columns) >= 2 else 0
                        ranges, avg_pe = get_historical_metrics(stock, stock.history(period="10y"))
                        eps = info.get('trailingEps', 0); cur_pe = price / eps if eps > 0 else 0
                        if 0 < cur_pe < 120: ind_pes.append(cur_pe)
                        cur_ev = info.get('enterpriseToEbitda', 0)
                        if not cur_ev: cur_ev = ((price * info.get('sharesOutstanding', 1)) + info.get('totalDebt', 0) - info.get('totalCash', 0)) / info.get('ebitda', 1)
                        is_fin = any(x in ind for x in ["金融", "保險"])
                        intrinsic, g_used, wacc, roic = get_3_stage_valuation(stock, is_fin, real_g)
                        upside = (intrinsic - price) / price if intrinsic > 0 else -1
                        raw_data.append((sym, ind, stock, info, price, real_g, qoq_g, wacc, roic, ranges, avg_pe, cur_pe, cur_ev, intrinsic, upside, eps, is_fin))
                    except: pass
                
                pe_med = np.median(ind_pes) if ind_pes else 22.0
                ind_results = [compile_stock_data(*d[:16], pe_med, d[16]) for d in raw_data]
                if ind_results:
                    df_ind = pd.DataFrame(ind_results).sort_values(by='Total_Score', ascending=False).head(6)
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
                    with col_info: st.dataframe(pd.DataFrame([data]).drop(columns=['Total_Score', '產業別']).T, use_container_width=True)
                except Exception as e: st.error("查無資料或發生錯誤")

# --- Tab 3: 真·時光機回測 ---
with tab3:
    st.markdown("⚠️ **V6.0 真·時點回測**：過濾進場日之後的「未來財報」，模擬當時真實的估值與得分。")
    c1, c2 = st.columns(2)
    with c1: t_input = st.text_area("測試代碼 (逗號分隔):", "1519.TW, 3017.TW, 2330.TW")
    with c2: s_date = st.date_input("進場日 (時光機日期):", datetime(2023, 11, 27)); run_bt = st.button("執行時點回測", type="primary")
    
    if run_bt:
        res_bt = []; pb = st.progress(0); t_list = [t.strip() for t in t_input.split(',')]
        for i, sym in enumerate(t_list):
            try:
                stock = yf.Ticker(sym)
                is_fin = "Financial" in stock.info.get('sector', '')
                pit_data = run_pit_backtest(sym, stock, s_date.strftime('%Y-%m-%d'), is_fin)
                if pit_data: res_bt.append(pit_data)
            except: pass
            pb.progress((i+1)/len(t_list))
            
        if res_bt:
            df_bt = pd.DataFrame(res_bt)
            st.metric("投資組合平均至今報酬率", f"{df_bt['Raw'].mean()*100:.1f}%")
            cols_show = ['代碼', '名稱', '進場日', '進場價', '當時PE', '當時合理價', '當時總分', '當時狀態', '3個月', '6個月', '12個月', '至今報酬']
            st.dataframe(df_bt[cols_show], use_container_width=True)