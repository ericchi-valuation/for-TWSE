import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import warnings
import time

# ==========================================
# 頁面與基本設定
# ==========================================
st.set_page_config(page_title="V6.14 Eric Chi估值模型", page_icon="📊", layout="wide")
warnings.simplefilter(action='ignore', category=FutureWarning)

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []

# ==========================================
# 核心防護工具區
# ==========================================
def strip_tz(dt_index):
    try:
        return pd.to_datetime(dt_index).tz_localize(None)
    except:
        return pd.to_datetime(dt_index) 

def safe_get(df_series, col, default=0):
    try:
        val = df_series.get(col, default)
        if isinstance(val, pd.Series): val = val.iloc[0]
        num_val = float(val)
        return num_val if pd.notna(num_val) and num_val != 0 else default
    except:
        return default

# 動態產業 PE 預設表
DEFAULT_PE_MAP = {
    "半導體業": 25.0, "金融保險業": 12.0, "電腦及週邊設備業": 20.0, 
    "光電業": 18.0, "電子零組件業": 18.0, "通信網路業": 18.0,
    "航運業": 10.0, "鋼鐵工業": 15.0, "塑膠工業": 15.0, "建材營造": 12.0,
    "電機機械": 20.0, "生技醫療業": 25.0
}

# ==========================================
# 0. 基礎資料庫
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_industry_list_v6():
    try:
        return pd.read_csv('tw_stock_list.csv')
    except:
        return pd.DataFrame() 

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
    return safe_get(stock.info, 'revenueGrowth', 0.05)

# ==========================================
# 1. 歷史區間與估值
# ==========================================
def get_historical_metrics(stock, hist_data):
    try:
        if hist_data.empty: return ["-", "-", "-", "-"], 0
        hist_data.index = strip_tz(hist_data.index)
        hist_data = hist_data.sort_index()
        fin = stock.quarterly_financials.T
        bs = stock.quarterly_balance_sheet.T
        if fin.empty: fin = stock.financials.T; bs = stock.balance_sheet.T
        if fin.empty: return ["-", "-", "-", "-"], 0
        fin.index = strip_tz(fin.index); bs.index = strip_tz(bs.index)
        
        pe_vals, pb_vals, ps_vals, evebitda_vals = [], [], [], []
        shares = safe_get(stock.info, 'sharesOutstanding', 1)
        
        for rpt_date in fin.index:
            try:
                if rpt_date not in hist_data.index:
                    nearest_idx = hist_data.index.get_indexer([rpt_date], method='nearest')[0]
                    if nearest_idx == -1: continue
                    price = float(hist_data.iloc[nearest_idx]['Close'])
                else:
                    price = float(hist_data.loc[rpt_date]['Close'])
                
                if isinstance(price, pd.Series): price = price.iloc[0]
                
                if rpt_date in bs.index:
                    bs_row = bs.loc[rpt_date]
                    total_debt = safe_get(bs_row, 'Total Debt', 0)
                    cash = safe_get(bs_row, 'Cash And Cash Equivalents', 0)
                    ev = (price * shares) + total_debt - cash
                    
                    fin_row = fin.loc[rpt_date]
                    ebit = safe_get(fin_row, 'EBIT', 0)
                    ebitda = safe_get(fin_row, 'EBITDA', ebit)
                    if ebitda > 0:
                        ratio = ev / (ebitda * 4) 
                        if 0 < ratio < 100: evebitda_vals.append(ratio)
                
                fin_row_2 = fin.loc[rpt_date]
                eps = safe_get(fin_row_2, 'Basic EPS', 0)
                if eps > 0: pe_vals.append(price / (eps * 4))
                
                rev = safe_get(fin_row_2, 'Total Revenue', 0)
                if rev > 0: ps_vals.append(price / ((rev/shares) * 4))
                
                if rpt_date in bs.index:
                    bv = safe_get(bs.loc[rpt_date], 'Stockholders Equity', 0)
                    if bv > 0: pb_vals.append(price / (bv/shares))
            except: continue
                
        def fmt_rng(vals):
            clean = [v for v in vals if not pd.isna(v) and 0 < v < 150]
            return f"{min(clean):.1f}-{max(clean):.1f}" if clean else "-"
            
        avg_pe = np.mean([v for v in pe_vals if not pd.isna(v) and 0 < v < 150]) if pe_vals else 0
        return [fmt_rng(pe_vals), fmt_rng(pb_vals), fmt_rng(ps_vals), fmt_rng(evebitda_vals)], avg_pe
    except: return ["-", "-", "-", "-"], 0

def get_3_stage_valuation(stock, is_finance, real_growth):
    try:
        info = stock.info; shares = safe_get(info, 'sharesOutstanding', 1)
        bs = stock.balance_sheet.fillna(0); fin = stock.financials.fillna(0)
        equity = safe_get(bs.loc['Stockholders Equity'], 0, 1) if 'Stockholders Equity' in bs.index else 1
        debt = safe_get(bs.loc['Total Debt'], 0, 0) if 'Total Debt' in bs.index else 0
        cash = safe_get(bs.loc['Cash And Cash Equivalents'], 0, 0) if 'Cash And Cash Equivalents' in bs.index else 0
        ebit = safe_get(fin.loc['EBIT'], 0, 0) if 'EBIT' in fin.index else 0
        
        beta = safe_get(info, 'beta', 1.0); ke = max(0.035 + beta * 0.06, 0.07)
        roic = (ebit * 0.8 / (equity + debt - cash)) if (equity + debt - cash) > 0 else 0.05
        wacc = max((equity/(equity+debt))*ke + (debt/(equity+debt))*0.025, 0.08) if is_finance else (equity/(equity+debt))*ke + (debt/(equity+debt))*0.025
        
        g1 = min(max(real_growth * 0.8, 0.02), 0.25); g_term = 0.025; g2 = (g1 + g_term) / 2
        base_cf = (safe_get(info, 'netIncomeToCommon', 0) * 0.6) if is_finance else (ebit * 0.8 * 0.7)
        
        if base_cf <= 0: return 0, g1, wacc, roic
        dcf_sum = sum([base_cf * ((1 + g1)**i) / ((1 + wacc)**i) for i in range(1, 4)])
        dcf_sum += sum([(base_cf * ((1 + g1)**3)) * ((1 + g2)**(i-3)) / ((1 + wacc)**i) for i in range(4, 6)])
        spread = max(wacc - g_term, 0.03)
        tv = (base_cf * ((1 + g1)**3) * ((1 + g2)**2)) * (1 + g_term) / spread
        dcf_sum += tv / ((1 + wacc)**5)
        
        return max((dcf_sum - (debt if not is_finance else 0) + (cash if not is_finance else 0)) / shares, 0), g1, wacc, roic
    except: return 0, 0, 0.1, 0

# ==========================================
# 3. 評分邏輯
# ==========================================
def calculate_raw_scores(info, financials, growth_rate, qoq_growth, valuation_upside, cur_pe, cur_ev_ebitda, hist_avg_pe, industry_pe_median, wacc, roic):
    scores = {'Q': 0, 'V': 0, 'G': 0, 'Msg': []}
    w_q, w_v, w_g = (0.3, 0.4, 0.3) if growth_rate < 0.15 else (0.2, 0.3, 0.5)
    
    try:
        ebit = safe_get(financials.loc['EBIT'], 0, safe_get(financials.loc['Operating Income'], 0, 0))
        interest = abs(safe_get(financials.loc['Interest Expense'], 0, 1))
        icr = ebit / interest if interest > 0 else 10
    except: icr = 10
    if icr > 5: scores['Q'] += 4
    elif icr < 1.5: scores['Q'] -= 5; scores['Msg'].append("高財務風險")
    else: scores['Q'] += 1
    
    if roic > wacc + 0.05: scores['Q'] += 5
    elif roic > wacc: scores['Q'] += 1
    else: scores['Msg'].append("ROIC<WACC")

    if valuation_upside > 0.15: scores['V'] += 4
    elif valuation_upside > 0.0: scores['V'] += 2
    elif valuation_upside < -0.20: scores['V'] -= 4; scores['Msg'].append("估值過熱")
    
    if hist_avg_pe > 0 and 0 < cur_pe < (hist_avg_pe * 1.1): scores['V'] += 3
    if industry_pe_median > 0 and 0 < cur_pe < industry_pe_median: scores['V'] += 3
    if 0 < cur_ev_ebitda < 15: scores['V'] += 3

    if growth_rate > 0.10 and roic < wacc: 
        scores['G'] -= 5; scores['Msg'].append("無效成長")
    else:
        if growth_rate > 0.25: scores['G'] += 5
        elif growth_rate > 0.15: scores['G'] += 3
        
    try: 
        op_now = safe_get(financials.loc['Operating Income'], 0) / safe_get(financials.loc['Total Revenue'], 0, 1)
        op_prev = safe_get(financials.loc['Operating Income'], 1) / safe_get(financials.loc['Total Revenue'], 1, 1)
        if op_now < op_prev * 0.95 and growth_rate > 0.1:
            scores['G'] -= 5; scores['Msg'].append("利潤率下滑")
    except: pass

    if qoq_growth > 0.05: scores['G'] += 3
    elif qoq_growth < -0.05: scores['G'] -= 3; scores['Msg'].append("動能轉弱")
    if 0 < safe_get(info, 'pegRatio', 0) < 1.5: scores['G'] += 2

    # Clamp [-10, 10]
    for k in ['Q', 'V', 'G']: scores[k] = max(-10, min(scores[k], 10))
    raw_total = (scores['Q'] * w_q * 10) + (scores['V'] * w_v * 10) + (scores['G'] * w_g * 10)
    if roic < wacc: raw_total *= 0.7 
    scores['Raw_Total'] = raw_total
    return scores

def compile_stock_data(symbol, ind, stock, info, price, real_g, qoq_g, wacc, roic, ranges, avg_pe, cur_pe, cur_ev, intrinsic, upside, eps, med_pe, is_fin, override_score=None):
    scores = calculate_raw_scores(info, stock.financials.fillna(0), real_g, qoq_g, upside, cur_pe, cur_ev, avg_pe, med_pe, wacc, roic)
    final_score = override_score if override_score is not None else min(scores['Raw_Total'], 100)
    status = f"Q:{scores['Q']} V:{scores['V']} G:{scores['G']}" + (f" | ⚠️{' '.join(scores['Msg'])}" if scores['Msg'] else "")
    
    return {
        '產業別': ind, '股票代碼': symbol, '名稱': info.get('shortName', symbol), '現價': price,
        '營收成長率': f"{real_g*100:.1f}%", '營業利益率': f"{safe_get(info, 'operatingMargins', 0)*100:.1f}%", '淨利率': f"{safe_get(info, 'profitMargins', 0)*100:.1f}%",
        '預估EPS': round(eps * (1 + min(real_g, 0.1)), 2), 'P/E (TTM)': round(cur_pe, 1) if cur_pe else "-",
        'P/B (Lag)': round(safe_get(info, 'priceToBook', 0), 2), 'P/S (Lag)': round(safe_get(info, 'priceToSalesTrailing12Months', 0), 2),
        'EV/EBITDA': f"{cur_ev:.1f}" if cur_ev > 0 else "-",
        '預估範圍P/E': ranges[0], '預估範圍P/B': ranges[1], '預估範圍P/S': ranges[2], '預估範圍EV/EBITDA': ranges[3],
        'DCF合理價': round(intrinsic, 1), '狀態': status, 'vs產業PE': "低於同業" if cur_pe < med_pe else "高於同業",
        '選股邏輯': f"Score: {int(final_score)}" + (" (首選)" if final_score >= 80 else ""),
        'Total_Score': final_score
    }

# ==========================================
# 4. 時點回測引擎 (PIT)
# ==========================================
def run_pit_backtest(sym, stock, target_date, is_finance, med_pe=18.0):
    try:
        target_dt = pd.to_datetime(target_date).tz_localize(None)
        hist = stock.history(start=target_dt - pd.Timedelta(days=3650), end=datetime.today())
        if hist.empty: return None
        hist.index = strip_tz(hist.index)
        future_prices = hist[hist.index >= target_dt]
        if future_prices.empty: return None
        entry_price = float(future_prices['Close'].iloc[0])
        current_price = float(hist['Close'].iloc[-1])

        q_fin = stock.quarterly_financials.T
        q_bs = stock.quarterly_balance_sheet.T
        if not q_fin.empty: q_fin.index = strip_tz(q_fin.index)
        if not q_bs.empty: q_bs.index = strip_tz(q_bs.index)
        
        valid_q_dates = q_fin.index[q_fin.index + pd.Timedelta(days=45) <= target_dt] if not q_fin.empty else []
        
        use_annual = False
        if len(valid_q_dates) < 4:
            a_fin = stock.financials.T
            a_bs = stock.balance_sheet.T
            if not a_fin.empty: a_fin.index = strip_tz(a_fin.index)
            if not a_bs.empty: a_bs.index = strip_tz(a_bs.index)
            
            valid_a_dates = a_fin.index[a_fin.index + pd.Timedelta(days=90) <= target_dt] if not a_fin.empty else []
            if len(valid_a_dates) == 0: return None
            
            use_annual = True; valid_dates = valid_a_dates; fin_df = a_fin; bs_df = a_bs
        else:
            valid_dates = valid_q_dates; fin_df = q_fin; bs_df = q_bs

        latest_date = valid_dates[0]
        annual_multiplier = 1 if use_annual else 4

        if use_annual:
            eps_ttm = safe_get(fin_df.loc[latest_date], 'Basic EPS', 0)
            rev_now = safe_get(fin_df.loc[latest_date], 'Total Revenue', 0)
            prev_date = valid_dates[1] if len(valid_dates) > 1 else latest_date
            rev_prev = safe_get(fin_df.loc[prev_date], 'Total Revenue', rev_now)
            qoq_growth = 0
        else:
            eps_ttm = float(fin_df.loc[valid_dates[:4], 'Basic EPS'].sum())
            rev_now = float(fin_df.loc[valid_dates[:4], 'Total Revenue'].sum())
            rev_prev = float(fin_df.loc[valid_dates[4:8], 'Total Revenue'].sum()) if len(valid_dates) >= 8 else rev_now
            rev_q1 = safe_get(fin_df.loc[valid_dates[0]], 'Total Revenue', 0)
            rev_q2 = safe_get(fin_df.loc[valid_dates[1]], 'Total Revenue', rev_q1) if len(valid_dates) > 1 else rev_q1
            qoq_growth = (rev_q1 - rev_q2) / rev_q2 if rev_q2 > 0 else 0

        real_growth = (rev_now - rev_prev) / rev_prev if rev_prev > 0 else 0.05
        ebit = safe_get(fin_df.loc[latest_date], 'EBIT', 0)
        ebitda = safe_get(fin_df.loc[latest_date], 'EBITDA', ebit)
        equity = safe_get(bs_df.loc[latest_date], 'Stockholders Equity', 1)
        debt = safe_get(bs_df.loc[latest_date], 'Total Debt', 0)
        cash = safe_get(bs_df.loc[latest_date], 'Cash And Cash Equivalents', 0)
        shares = safe_get(stock.info, 'sharesOutstanding', 1)

        cur_pe = entry_price / eps_ttm if eps_ttm > 0 else 0
        cur_ev_ebitda = ((entry_price * shares) + debt - cash) / (ebitda * annual_multiplier) if ebitda > 0 else 0

        beta = safe_get(stock.info, 'beta', 1.0)
        ke = max(0.035 + beta * 0.06, 0.07)
        invested_capital = equity + debt - cash
        roic = (ebit * annual_multiplier * 0.8 / invested_capital) if invested_capital > 0 else 0.05
        wacc = max((equity/(equity+debt))*ke + (debt/(equity+debt))*0.025, 0.08) if is_finance else (equity/(equity+debt))*ke + (debt/(equity+debt))*0.025

        g1 = min(max(real_growth * 0.8, 0.02), 0.25); g_term = 0.025; g2 = (g1 + g_term) / 2
        base_cf = (safe_get(fin_df.loc[latest_date], 'Net Income', 0) * annual_multiplier * 0.6) if is_finance else (ebit * annual_multiplier * 0.8 * 0.7)
        
        if base_cf <= 0: intrinsic = 0
        else:
            dcf_sum = sum([base_cf * ((1 + g1)**i) / ((1 + wacc)**i) for i in range(1, 4)])
            dcf_sum += sum([(base_cf * ((1 + g1)**3)) * ((1 + g2)**(i-3)) / ((1 + wacc)**i) for i in range(4, 6)])
            spread = max(wacc - g_term, 0.03)
            tv = (base_cf * ((1 + g1)**3) * ((1 + g2)**2)) * (1 + g_term) / spread
            dcf_sum += tv / ((1 + wacc)**5)
            intrinsic = max((dcf_sum - (debt if not is_finance else 0) + (cash if not is_finance else 0)) / shares, 0)

        upside = (intrinsic - entry_price) / entry_price if intrinsic > 0 else -1

        pe_vals = []
        for d in valid_dates[:10]:
            try:
                p_hist = hist.loc[hist.index <= d]['Close']
                if not p_hist.empty:
                    p = float(p_hist.iloc[-1])
                    e = safe_get(fin_df.loc[d], 'Basic EPS', 0) * annual_multiplier
                    if e > 0: pe_vals.append(p / e)
            except: pass
        avg_pe = np.mean(pe_vals) if pe_vals else 0

        pit_financials = fin_df.loc[valid_dates].T
        scores = calculate_raw_scores(stock.info, pit_financials, real_growth, qoq_growth, upside, cur_pe, cur_ev_ebitda, avg_pe, med_pe, wacc, roic)

        def get_ret(days):
            td = future_prices.index[0] + pd.Timedelta(days=days)
            idx = future_prices.index.searchsorted(td)
            if idx < len(future_prices): return (future_prices['Close'].iloc[idx] - entry_price) / entry_price
            return None

        return {
            '代碼': sym, '名稱': stock.info.get('shortName', sym), '進場日': target_dt.strftime('%Y-%m-%d'),
            '進場價': round(entry_price, 1), '現價': round(current_price, 1),
            '當時總分': int(min(scores['Raw_Total'], 100)), '當時狀態': f"Q:{scores['Q']} V:{scores['V']} G:{scores['G']}",
            '當時合理價': round(intrinsic, 1), '當時PE': round(cur_pe, 1),
            '3個月': f"{get_ret(90)*100:.1f}%" if get_ret(90) else "-",
            '6個月': f"{get_ret(180)*100:.1f}%" if get_ret(180) else "-",
            '12個月': f"{get_ret(365)*100:.1f}%" if get_ret(365) else "-",
            '至今報酬': f"{(current_price - entry_price)/entry_price*100:.1f}%", 'Raw': (current_price - entry_price)/entry_price
        }
    except Exception as e: 
        return None

# ==========================================
# UI 介面
# ==========================================
st.title("V6.14 Eric Chi估值模型")
tab1, tab2, tab3 = st.tabs(["產業精準掃描", "單股查詢", "真·時光機回測"])

# --- Tab 1: 產業精準掃描 (市值前50%回歸版) ---
with tab1:
    df_all = fetch_industry_list_v6()
    if df_all.empty:
        st.error("❌ 找不到 tw_stock_list.csv")
    else:
        valid_industries = sorted([i for i in df_all['Industry'].unique()])
        st.info("💡 **過濾器升級**：已重啟「市值前50%」嚴選機制。為防斷線，建議每次勾選 1~3 個產業。")
        
        selected_inds = st.multiselect(
            "請選擇要掃描的產業：", 
            options=valid_industries, 
            default=valid_industries[:2]
        )
        
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("🚀 執行所選產業掃描", type="primary"):
                if not selected_inds:
                    st.warning("請至少選擇一個產業！")
                else:
                    pb = st.progress(0); status_text = st.empty(); results_container = st.container()
                    total_inds = len(selected_inds)
                    cols_display = ['股票代碼', '名稱', '現價', '營收成長率', '營業利益率', '淨利率', '預估EPS', 'P/E (TTM)', 'P/B (Lag)', 'P/S (Lag)', 'EV/EBITDA', '預估範圍P/E', '預估範圍P/B', '預估範圍P/S', '預估範圍EV/EBITDA', 'DCF合理價', '狀態', 'vs產業PE', '選股邏輯']
                    
                    for idx, ind in enumerate(selected_inds):
                        status_text.text(f"⏳ [{ind}] ({idx+1}/{total_inds}) 階段一：過濾市值前 50%...")
                        tickers = df_all[df_all["Industry"] == ind]["Ticker"].tolist()
                        
                        caps = []
                        for t in tickers:
                            try:
                                tk = yf.Ticker(t)
                                mcap = tk.fast_info.get('marketCap') or tk.fast_info.get('market_cap')
                                if not mcap: mcap = tk.info.get('marketCap', 0)
                                if mcap and float(mcap) > 0: 
                                    caps.append((t, float(mcap)))
                            except: pass
                        
                        if caps:
                            caps.sort(key=lambda x: x[1], reverse=True)
                            half_len = max(len(caps) // 2, 1)
                            targets = [x[0] for x in caps[:half_len]]
                        else:
                            targets = tickers[:15]
                            
                        status_text.text(f"⏳ [{ind}] ({idx+1}/{total_inds}) 階段二：精算 {len(targets)} 檔權值股財報...")
                        
                        ind_pes = []; raw_data = []
                        for sym in targets:
                            try:
                                stock = yf.Ticker(sym); info = stock.info
                                price = info.get('currentPrice') or info.get('previousClose')
                                if not price: continue
                                real_g = get_growth_data(stock, sym)
                                
                                q_fin = stock.quarterly_financials
                                if not q_fin.empty and len(q_fin.columns) >= 2:
                                    rev_q1 = safe_get(q_fin.iloc[:, 0], 'Total Revenue')
                                    rev_q2 = safe_get(q_fin.iloc[:, 1], 'Total Revenue', rev_q1)
                                    qoq_g = (rev_q1 - rev_q2) / rev_q2 if rev_q2 > 0 else 0
                                else:
                                    qoq_g = 0
                                    
                                ranges, avg_pe = get_historical_metrics(stock, stock.history(period="10y"))
                                eps = safe_get(info, 'trailingEps', 0); cur_pe = price / eps if eps > 0 else 0
                                if 0 < cur_pe < 120: ind_pes.append(cur_pe)
                                
                                cur_ev = safe_get(info, 'enterpriseToEbitda', safe_get(info, 'enterpriseValue', 1)/safe_get(info, 'ebitda', 1))
                                is_fin = any(x in ind for x in ["金融", "保險"])
                                intrinsic, _, wacc, roic = get_3_stage_valuation(stock, is_fin, real_g)
                                
                                raw_data.append({'sym': sym, 'ind': ind, 'stock': stock, 'info': info, 'price': price, 'real_g': real_g, 'qoq_g': qoq_g, 'wacc': wacc, 'roic': roic, 'ranges': ranges, 'avg_pe': avg_pe, 'cur_pe': cur_pe, 'cur_ev': cur_ev, 'intrinsic': intrinsic, 'eps': eps, 'is_fin': is_fin})
                                time.sleep(0.3) 
                            except: pass
                        
                        clean_pes = [pe for pe in ind_pes if 5 < pe < 60]
                        pe_med = np.median(clean_pes) if clean_pes else 22.0
                        
                        raw_scores = []
                        for d in raw_data:
                            s = calculate_raw_scores(d['info'], d['stock'].financials.fillna(0), d['real_g'], d['qoq_g'], (d['intrinsic']-d['price'])/d['price'], d['cur_pe'], d['cur_ev'], d['avg_pe'], pe_med, d['wacc'], d['roic'])
                            raw_scores.append(s['Raw_Total'])
                        
                        if len(raw_scores) > 1:
                            ranks = pd.Series(raw_scores).rank(pct=True) 
                            multiplier = 0.8 + (ranks * 0.4) 
                            adjusted_scores = pd.Series(raw_scores) * multiplier
                            adjusted_scores = adjusted_scores.clip(upper=100) 
                        else:
                            adjusted_scores = pd.Series(raw_scores)

                        ind_results = []
                        for i, d in enumerate(raw_data):
                            res = compile_stock_data(d['sym'], d['ind'], d['stock'], d['info'], d['price'], d['real_g'], d['qoq_g'], d['wacc'], d['roic'], d['ranges'], d['avg_pe'], d['cur_pe'], d['cur_ev'], d['intrinsic'], (d['intrinsic']-d['price'])/d['price'], d['eps'], pe_med, d['is_fin'], override_score=adjusted_scores.iloc[i])
                            ind_results.append(res)
                        
                        if ind_results:
                            df_ind = pd.DataFrame(ind_results).sort_values(by='Total_Score', ascending=False).head(6)
                            if not any(x[0] == ind for x in st.session_state.scan_results):
                                st.session_state.scan_results.append((ind, df_ind))
                        pb.progress((idx + 1) / total_inds)
                    status_text.text("✅ 本次勾選之產業掃描完成！可繼續勾選其他產業累積名單。")

        with c2:
            if st.button("🗑️ 清空所有掃描暫存"):
                st.session_state.scan_results = []
                st.rerun()

        if st.session_state.scan_results:
            st.markdown("---")
            full_df = pd.concat([x[1] for x in st.session_state.scan_results])
            st.download_button("💾 下載目前累積的所有報告 (CSV)", data=full_df.to_csv(index=False).encode('utf-8-sig'), file_name=f"TW_Stock_Scan_Accumulated_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
            for ind, df_ind in st.session_state.scan_results:
                with st.expander(f"🏆 {ind} (市值前50% 嚴選 Top 6)", expanded=True):
                    st.dataframe(df_ind.drop(columns=['Total_Score']), use_container_width=True)

# --- Tab 2: 單股查詢 ---
with tab2:
    col_input, col_info = st.columns([1, 2])
    with col_input:
        stock_code = st.text_input("輸入代碼 (例如: 2330):", value="2330")
        if st.button("查詢", type="primary", key="single_search"):
            sym = stock_code.strip().upper()
            if not sym.endswith('.TW') and not sym.endswith('.TWO'):
                sym = f"{sym}.TW"
                
            with st.spinner("查詢中..."):
                try:
                    df_all = fetch_industry_list_v6()
                    ind = df_all.loc[df_all['Ticker'] == sym, 'Industry'].iloc[0] if (not df_all.empty and sym in df_all['Ticker'].values) else "未知產業"
                    med_pe = DEFAULT_PE_MAP.get(ind, 18.0) 
                    
                    stock = yf.Ticker(sym); info = stock.info
                    price = info.get('currentPrice') or info.get('previousClose')
                    if not price: 
                        st.error("❌ 抓不到股價，API 可能暫時超時。")
                    else:
                        real_g = get_growth_data(stock, sym)
                        q_fin = stock.quarterly_financials
                        if not q_fin.empty and len(q_fin.columns) >= 2:
                            rev_q1 = safe_get(q_fin.iloc[:, 0], 'Total Revenue')
                            rev_q2 = safe_get(q_fin.iloc[:, 1], 'Total Revenue', rev_q1)
                            qoq_g = (rev_q1 - rev_q2) / rev_q2 if rev_q2 > 0 else 0
                        else:
                            qoq_g = 0
                            
                        ranges, avg_pe = get_historical_metrics(stock, stock.history(period="10y"))
                        eps = safe_get(info, 'trailingEps', 0); cur_pe = price/eps if eps>0 else 0
                        cur_ev = safe_get(info, 'enterpriseToEbitda', 0)
                        is_fin = any(x in ind for x in ["金融", "保險"])
                        intrinsic, g_used, wacc, roic = get_3_stage_valuation(stock, is_fin, real_g)
                        upside = (intrinsic - price) / price if intrinsic > 0 else -1
                        data = compile_stock_data(sym, ind, stock, info, price, real_g, qoq_g, wacc, roic, ranges, avg_pe, cur_pe, cur_ev, intrinsic, upside, eps, med_pe, is_fin, override_score=None)
                        
                        st.metric("合理價", f"{intrinsic:.1f} TWD", f"{upside:.1%} 空間")
                        st.success(data['狀態'])
                        with col_info: st.dataframe(pd.DataFrame([data]).drop(columns=['Total_Score', '產業別']).T, use_container_width=True)
                except Exception as e: 
                    st.error(f"❌ 發生錯誤: {e}")

# --- Tab 3: 時光機回測 ---
with tab3:
    c1, c2 = st.columns(2)
    with c1: t_input = st.text_area("回測代碼:", "1519, 3017, 2330")
    with c2: s_date = st.date_input("回測日期:", datetime(2023, 11, 27))
    if st.button("啟動時光機", type="primary"):
        res_bt = []; pb = st.progress(0)
        t_list = [t.strip().upper() for t in t_input.split(',')]
        df_all = fetch_industry_list_v6()
        
        for i, raw_sym in enumerate(t_list):
            try:
                sym = raw_sym if (raw_sym.endswith('.TW') or raw_sym.endswith('.TWO')) else f"{raw_sym}.TW"
                stock = yf.Ticker(sym)
                ind = df_all.loc[df_all['Ticker'] == sym, 'Industry'].iloc[0] if (not df_all.empty and sym in df_all['Ticker'].values) else ""
                med_pe = DEFAULT_PE_MAP.get(ind, 18.0)
                is_fin = any(x in ind for x in ["金融", "保險"])
                
                pit_data = run_pit_backtest(sym, stock, s_date.strftime('%Y-%m-%d'), is_fin, med_pe)
                if pit_data: res_bt.append(pit_data)
                time.sleep(0.3)
            except: pass
            pb.progress((i+1)/len(t_list))
            
        if res_bt:
            df_bt = pd.DataFrame(res_bt)
            st.metric("平均至今報酬", f"{df_bt['Raw'].mean()*100:.1f}%")
            cols_show = ['代碼', '名稱', '進場日', '進場價', '當時PE', '當時合理價', '當時總分', '當時狀態', '3個月', '6個月', '12個月', '至今報酬']
            st.dataframe(df_bt[cols_show], use_container_width=True)
        else:
            st.warning("⚠️ 查無歷史數據。原因：免費版 API 僅提供近1年季報與近4年年報。")