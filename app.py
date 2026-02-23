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
st.set_page_config(page_title="V6.3 Eric Chi估值模型", page_icon="📈", layout="wide")
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 0. 基礎資料庫 (讀取本地 CSV)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_industry_list_v6():
    try: return pd.read_csv('tw_stock_list.csv')
    except: return pd.DataFrame() 

def get_growth_data(stock, symbol):
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
    except: pass
    return stock.info.get('revenueGrowth', 0.0)

# ==========================================
# 1. 歷史區間計算 (V6.3 新增 PB, PS 最低點提取)
# ==========================================
def get_historical_metrics(stock, hist_data):
    try:
        if hist_data.empty: return "-", "-", "-", "-", 0, 0, 0
        hist_data.index = hist_data.index.tz_localize(None)
        
        fin = stock.quarterly_financials.T
        bs = stock.quarterly_balance_sheet.T
        if fin.empty or bs.empty: return "-", "-", "-", "-", 0, 0, 0
        
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
                total_debt = bs.loc[rpt_date].get('Total Debt', 0)
                cash = bs.loc[rpt_date].get('Cash And Cash Equivalents', 0)
                ev = (price * shares) + total_debt - cash
                ebitda = fin.loc[rpt_date].get('EBITDA', fin.loc[rpt_date].get('EBIT', 0))
                if ebitda > 0:
                    ratio = ev / (ebitda * 4) 
                    if 0 < ratio < 100: evebitda_vals.append(ratio)
            
            eps = fin.loc[rpt_date].get('Basic EPS', 0)
            if eps > 0: pe_vals.append(price / (eps * 4))
            
            rev = fin.loc[rpt_date].get('Total Revenue', 0)
            if rev > 0: ps_vals.append(price / ((rev/shares) * 4))
                
            bv = bs.loc[rpt_date].get('Stockholders Equity', 0)
            if bv > 0: pb_vals.append(price / (bv/shares))
                
        def get_clean(vals): return [v for v in vals if 0 < v < 150]
        def fmt_rng(clean): return f"{min(clean):.1f}-{max(clean):.1f}" if clean else "-"
        
        c_pe, c_pb, c_ps, c_ev = get_clean(pe_vals), get_clean(pb_vals), get_clean(ps_vals), get_clean(evebitda_vals)
        avg_pe = np.mean(c_pe) if c_pe else 0
        min_pb = min(c_pb) if c_pb else 0
        min_ps = min(c_ps) if c_ps else 0
        
        return fmt_rng(c_pe), fmt_rng(c_pb), fmt_rng(c_ps), fmt_rng(c_ev), avg_pe, min_pb, min_ps
    except: return "-", "-", "-", "-", 0, 0, 0

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
# 3. 評分與資料整合 (V6.3 全新評分引擎)
# ==========================================
def calculate_scores(info, real_growth, qoq_growth, upside, cur_pe, cur_ev_ebitda, hist_avg_pe, cur_pb, cur_ps, min_pb, min_ps, wacc, roic, debt_to_ebitda, op_margins):
    scores = {'Q': 0, 'V': 0, 'G': 0, 'Total': 0, 'Msg': []}
    w_q, w_v, w_g = (0.2, 0.3, 0.5) if real_growth > 0.15 else ((0.5, 0.4, 0.1) if real_growth < 0.05 else (0.3, 0.4, 0.3))
    scores['Lifecycle'] = "Growth" if real_growth > 0.15 else ("Mature" if real_growth < 0.05 else "Stable")

    # --- Quality (滿分 10 分) ---
    if debt_to_ebitda > 0:
        if debt_to_ebitda < 4.0: scores['Q'] += 3
        elif debt_to_ebitda > 4.0: scores['Q'] -= 5; scores['Msg'].append("高財務風險(Debt/EBITDA>4)")
        
    if roic > wacc: scores['Q'] += 4
    else: scores['Q'] -= 2; scores['Msg'].append("ROIC<WACC")

    # op_margins 排序為 [Q1(最新), Q2, Q3, Q4]
    if len(op_margins) >= 4 and all(op_margins[i] > op_margins[i+1] for i in range(3)):
        scores['Q'] += 3
    elif len(op_margins) >= 2 and op_margins[0] > op_margins[1]:
        scores['Q'] += 2
    elif len(op_margins) >= 2 and op_margins[0] < op_margins[1]:
        scores['Q'] -= 1; scores['Msg'].append("營益率下滑")

    # --- Value (滿分 10 分) ---
    if upside > 0.30: scores['V'] += 4
    elif upside > 0.0: scores['V'] += 2

    if hist_avg_pe > 0 and 0 < cur_pe < (hist_avg_pe * 1.1): scores['V'] += 2
    if min_ps > 0 and 0 < cur_ps < (min_ps * 1.1): scores['V'] += 1
    if 0 < cur_ev_ebitda < 15: scores['V'] += 1
    if min_pb > 0 and 0 < cur_pb < (min_pb * 1.1): scores['V'] += 2

    # --- Growth (滿分 10 分) ---
    if real_growth > 0.10 and roic < wacc: scores['G'] -= 5; scores['Msg'].append("無效成長")
    else:
        if real_growth > 0.25: scores['G'] += 5
        elif real_growth > 0.10: scores['G'] += 3
        
    if qoq_growth > 0.05: scores['G'] += 3
    elif qoq_growth < -0.05: scores['G'] -= 3; scores['Msg'].append("動能轉弱")
    
    if 0 < info.get('pegRatio', 0) < 1.5: scores['G'] += 2

    scores['Total'] = (scores['Q'] * w_q * 10) + (scores['V'] * w_v * 10) + (scores['G'] * w_g * 10)
    return scores

def compile_stock_data(symbol, ind, stock, info, price, real_g, qoq_g, wacc, roic, ranges, avg_pe, min_pb, min_ps, cur_pe, cur_ev, intrinsic, upside, eps, is_fin):
    q_fin = stock.quarterly_financials
    q_bs = stock.quarterly_balance_sheet
    
    # 提取 Operating Margins
    op_margins = []
    if not q_fin.empty and 'Total Revenue' in q_fin.index:
        op_key = 'Operating Income' if 'Operating Income' in q_fin.index else ('EBIT' if 'EBIT' in q_fin.index else None)
        if op_key:
            for col in q_fin.columns[:4]:
                r = q_fin.loc['Total Revenue', col]
                o = q_fin.loc[op_key, col]
                op_margins.append(o/r if r > 0 else 0)
                
    # 提取 Debt / TTM EBITDA
    debt = q_bs.loc['Total Debt'].iloc[0] if not q_bs.empty and 'Total Debt' in q_bs.index else 0
    ebitda_key = 'EBITDA' if not q_fin.empty and 'EBITDA' in q_fin.index else ('EBIT' if not q_fin.empty and 'EBIT' in q_fin.index else None)
    ttm_ebitda = q_fin.loc[ebitda_key].iloc[:4].sum() if ebitda_key else 0
    debt_to_ebitda = debt / ttm_ebitda if ttm_ebitda > 0 else 0
    
    # 提取 PB, PS
    cur_pb = info.get('priceToBook', 0) or 0
    cur_ps = info.get('priceToSalesTrailing12Months', 0) or 0

    scores = calculate_scores(info, real_g, qoq_g, upside, cur_pe, cur_ev, avg_pe, cur_pb, cur_ps, min_pb, min_ps, wacc, roic, debt_to_ebitda, op_margins)
    
    status = f"{scores['Lifecycle']} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}" + (f" | ⚠️{' '.join(scores['Msg'])}" if scores['Msg'] else "")
    logic = f"Score: {int(scores['Total'])}" + (" (首選)" if scores['Total'] >= 70 else "")
    
    return {
        '產業別': ind, '股票代碼': symbol, '名稱': info.get('shortName', symbol), '現價': price,
        '營收成長率': f"{real_g*100:.1f}%", '預估EPS': round(eps * (1 + min(real_g, 0.1)), 2),
        '營業利益率': f"{info.get('operatingMargins', 0)*100:.1f}%", '淨利率': f"{info.get('profitMargins', 0)*100:.1f}%",
        'P/E (TTM)': round(cur_pe, 1) if cur_pe else "-", 'P/B (Lag)': round(cur_pb, 2),
        'EV/EBITDA': f"{cur_ev:.1f}" if cur_ev > 0 else "-",
        '預估範圍P/E': ranges[0], '預估範圍P/B': ranges[1], '預估範圍P/S': ranges[2], '預估範圍EV/EBITDA': ranges[3],
        'DCF/DDM合理價': round(intrinsic, 1), '狀態': status, '選股邏輯': logic, 'Total_Score': scores['Total']
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

        # 計算 OP Margins, Debt/EBITDA, PB, PS
        op_margins = []
        for d in valid_dates[:4]:
            r = q_fin.loc[d].get('Total Revenue', 0)
            o = q_fin.loc[d].get('Operating Income', q_fin.loc[d].get('EBIT', 0))
            op_margins.append(o/r if r > 0 else 0)

        ebit = q_fin.loc[latest_date].get('EBIT', 0)
        ebitda = q_fin.loc[latest_date].get('EBITDA', ebit)
        equity = q_bs.loc[latest_date].get('Stockholders Equity', 1)
        debt = q_bs.loc[latest_date].get('Total Debt', 0)
        cash = q_bs.loc[latest_date].get('Cash And Cash Equivalents', 0)
        shares = stock.info.get('sharesOutstanding', 1)

        ttm_ebitda = q_fin.loc[valid_dates[:4], 'EBITDA'].sum() if 'EBITDA' in q_fin.columns else (q_fin.loc[valid_dates[:4], 'EBIT'].sum() if 'EBIT' in q_fin.columns else 0)
        debt_to_ebitda = debt / ttm_ebitda if ttm_ebitda > 0 else 0

        bv = q_bs.loc[latest_date].get('Stockholders Equity', 0)
        cur_pb = entry_price / (bv / shares) if bv > 0 and shares > 0 else 0
        cur_ps = entry_price / (rev_ttm / shares) if rev_ttm > 0 and shares > 0 else 0

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

        pe_vals, pb_vals, ps_vals = [], [], []
        for d in valid_dates[:20]:
            try:
                p = hist.loc[hist.index <= d]['Close'].iloc[-1]
                e = q_fin.loc[d, 'Basic EPS']
                if e > 0: pe_vals.append(p / (e * 4))
                r = q_fin.loc[d, 'Total Revenue'] if 'Total Revenue' in q_fin.columns else 0
                if r > 0: ps_vals.append(p / ((r/shares) * 4))
                b = q_bs.loc[d, 'Stockholders Equity'] if d in q_bs.index and 'Stockholders Equity' in q_bs.columns else 0
                if b > 0: pb_vals.append(p / (b/shares))
            except: pass
            
        avg_pe = np.mean([v for v in pe_vals if 0<v<150]) if pe_vals else 0
        min_ps = min([v for v in ps_vals if 0<v<150]) if ps_vals else 0
        min_pb = min([v for v in pb_vals if 0<v<150]) if pb_vals else 0

        scores = calculate_scores(stock.info, real_growth, qoq_growth, upside, cur_pe, cur_ev_ebitda, avg_pe, cur_pb, cur_ps, min_pb, min_ps, wacc, roic, debt_to_ebitda, op_margins)

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
    except: return None

# ==========================================
# UI 介面
# ==========================================
st.title("V6.3 Eric Chi估值模型")
tab1, tab2, tab3 = st.tabs(["全產業掃描", "單股查詢", "真·時光機回測"])

# --- Tab 1: 全產業掃描 ---
with tab1:
    with st.spinner("讀取本地清單中..."):
        df_all = fetch_industry_list_v6()
    
    if df_all.empty:
        st.error("❌ 找不到 tw_stock_list.csv，請確認已上傳。")
    else:
        valid_industries = sorted([i for i in df_all['Industry'].unique()])
        st.info(f"偵測到 {len(valid_industries)} 個產業。掃描將動態印出 Top 6。")
        if st.button("執行全產業掃描", type="primary"):
            pb = st.progress(0); status_text = st.empty(); results_container = st.container()
            total_inds = len(valid_industries)
            cols_display = ['股票代碼', '名稱', '現價', '營收成長率', '預估EPS', 'P/E (TTM)', 'EV/EBITDA', 'DCF/DDM合理價', '狀態', '選股邏輯']
            
            for idx, ind in enumerate(valid_industries):
                status_text.text(f"精算 [{ind}]...")
                tickers = df_all[df_all["Industry"] == ind]["Ticker"].tolist()
                caps = []
                for t in tickers:
                    try: caps.append((t, yf.Ticker(t).fast_info['market_cap']))
                    except: pass
                caps.sort(key=lambda x: x[1], reverse=True)
                targets = [x[0] for x in caps[:max(len(caps)//2, 1)]]
                
                raw_data = []
                for sym in targets:
                    try:
                        stock = yf.Ticker(sym); info = stock.info
                        price = info.get('currentPrice') or info.get('previousClose')
                        real_g = get_growth_data(stock, sym)
                        
                        q_fin = stock.quarterly_financials
                        qoq_g = (q_fin.loc['Total Revenue'].iloc[0] - q_fin.loc['Total Revenue'].iloc[1]) / q_fin.loc['Total Revenue'].iloc[1] if not q_fin.empty and len(q_fin.columns) >= 2 else 0
                        
                        ranges, avg_pe, min_pb, min_ps = get_historical_metrics(stock, stock.history(period="10y"))
                        eps = info.get('trailingEps', 0); cur_pe = price / eps if eps > 0 else 0
                        
                        cur_ev = info.get('enterpriseToEbitda', 0)
                        if not cur_ev:
                            mcap = price * info.get('sharesOutstanding', 1)
                            cur_ev = (mcap + info.get('totalDebt', 0) - info.get('totalCash', 0)) / info.get('ebitda', 1)
                            
                        is_fin = any(x in ind for x in ["金融", "保險"])
                        intrinsic, g_used, wacc, roic = get_3_stage_valuation(stock, is_fin, real_g)
                        upside = (intrinsic - price) / price if intrinsic > 0 else -1
                        
                        raw_data.append((sym, ind, stock, info, price, real_g, qoq_g, wacc, roic, ranges, avg_pe, min_pb, min_ps, cur_pe, cur_ev, intrinsic, upside, eps, is_fin))
                    except: pass
                
                ind_results = [compile_stock_data(*d) for d in raw_data]
                if ind_results:
                    df_ind = pd.DataFrame(ind_results).sort_values(by='Total_Score', ascending=False).head(6)
                    with results_container:
                        st.markdown(f"### 🏆 {ind}")
                        st.dataframe(df_ind[cols_display], use_container_width=True)
                pb.progress((idx + 1) / total_inds)
            status_text.text("✅ 掃描完成！")

# --- Tab 2: 單股查詢 ---
with tab2:
    col_input, col_info = st.columns([1, 2])
    with col_input:
        stock_code = st.text_input("輸入代碼 (例如: 2330):", value="2330")
        if st.button("查詢", type="primary"):
            sym = f"{stock_code}.TW"
            with st.spinner("查詢中..."):
                try:
                    stock = yf.Ticker(sym); info = stock.info
                    price = info.get('currentPrice') or info.get('previousClose')
                    if not price: 
                        st.error("❌ 抓不到股價，API 可能暫時超時。")
                    else:
                        real_g = get_growth_data(stock, sym)
                        
                        q_fin = stock.quarterly_financials
                        qoq_g = (q_fin.loc['Total Revenue'].iloc[0] - q_fin.loc['Total Revenue'].iloc[1]) / q_fin.loc['Total Revenue'].iloc[1] if not q_fin.empty and len(q_fin.columns) >= 2 else 0
                        
                        ranges, avg_pe, min_pb, min_ps = get_historical_metrics(stock, stock.history(period="10y"))
                        eps = info.get('trailingEps', 0); cur_pe = price/eps if eps>0 else 0
                        cur_ev = info.get('enterpriseToEbitda', 0)
                        is_fin = "Financial" in info.get('sector', '')
                        
                        intrinsic, g_used, wacc, roic = get_3_stage_valuation(stock, is_fin, real_g)
                        upside = (intrinsic - price) / price if intrinsic > 0 else -1
                        
                        data = compile_stock_data(sym, info.get('industry', 'N/A'), stock, info, price, real_g, qoq_g, wacc, roic, ranges, avg_pe, min_pb, min_ps, cur_pe, cur_ev, intrinsic, upside, eps, is_fin)
                        
                        st.metric("合理價", f"{intrinsic:.1f} TWD", f"{upside:.1%} 空間")
                        st.success(data['狀態'])
                        with col_info: st.dataframe(pd.DataFrame([data]).drop(columns=['Total_Score', '產業別']).T, use_container_width=True)
                except Exception as e: 
                    st.error(f"❌ 發生錯誤: {e}")

# --- Tab 3: 真·時光機回測 ---
with tab3:
    c1, c2 = st.columns(2)
    with c1: t_input = st.text_area("代碼:", "1519.TW, 3017.TW, 2330.TW")
    with c2: s_date = st.date_input("日期:", datetime(2023, 11, 27)); run_bt = st.button("執行", type="primary")
    if run_bt:
        res_bt = []; pb = st.progress(0); t_list = [t.strip() for t in t_input.split(',')]
        for i, sym in enumerate(t_list):
            stock = yf.Ticker(sym)
            pit_data = run_pit_backtest(sym, stock, s_date.strftime('%Y-%m-%d'), "Financial" in stock.info.get('sector', ''))
            if pit_data: res_bt.append(pit_data)
            pb.progress((i+1)/len(t_list))
        if res_bt:
            df_bt = pd.DataFrame(res_bt)
            st.metric("平均報酬", f"{df_bt['Raw'].mean()*100:.1f}%")
            st.dataframe(df_bt.drop(columns=['Raw']), use_container_width=True)