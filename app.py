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
st.set_page_config(page_title="V6.6 Eric Chi估值模型", page_icon="📊", layout="wide")
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 0. 基礎資料庫 (讀取上傳的 CSV)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_industry_list_v6():
    try:
        df = pd.read_csv('tw_stock_list.csv')
        return df
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
    except:
        pass
    return stock.info.get('revenueGrowth', 0.0)

# ==========================================
# 1. 歷史區間計算
# ==========================================
def get_historical_metrics(stock, hist_data):
    try:
        if hist_data.empty: return ["-", "-", "-", "-"], 0
        hist_data.index = pd.to_datetime(hist_data.index).tz_localize(None)
        hist_data = hist_data.sort_index()
        
        fin = stock.quarterly_financials.T
        bs = stock.quarterly_balance_sheet.T
        
        if fin.empty or bs.empty:
            fin = stock.financials.T
            bs = stock.balance_sheet.T
            if fin.empty or bs.empty:
                return ["-", "-", "-", "-"], 0
                
        fin.index = pd.to_datetime(fin.index).tz_localize(None)
        bs.index = pd.to_datetime(bs.index).tz_localize(None)
        
        pe_vals, pb_vals, ps_vals, evebitda_vals = [], [], [], []
        shares = stock.info.get('sharesOutstanding', 1)
        
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
                    if isinstance(bs_row, pd.DataFrame): bs_row = bs_row.iloc[0]
                    total_debt = float(bs_row.get('Total Debt', 0) or 0)
                    cash = float(bs_row.get('Cash And Cash Equivalents', 0) or 0)
                    ev = (price * shares) + total_debt - cash
                    
                    fin_row = fin.loc[rpt_date]
                    if isinstance(fin_row, pd.DataFrame): fin_row = fin_row.iloc[0]
                    ebitda = float(fin_row.get('EBITDA', fin_row.get('EBIT', 0)) or 0)
                    if ebitda > 0:
                        ratio = ev / (ebitda * 4) 
                        if 0 < ratio < 100: evebitda_vals.append(ratio)
                
                fin_row_2 = fin.loc[rpt_date]
                if isinstance(fin_row_2, pd.DataFrame): fin_row_2 = fin_row_2.iloc[0]
                eps = float(fin_row_2.get('Basic EPS', 0) or 0)
                if eps > 0: pe_vals.append(price / (eps * 4))
                
                rev = float(fin_row_2.get('Total Revenue', 0) or 0)
                if rev > 0: ps_vals.append(price / ((rev/shares) * 4))
                    
                if rpt_date in bs.index:
                    bv = float(bs_row.get('Stockholders Equity', 0) or 0)
                    if bv > 0: pb_vals.append(price / (bv/shares))
            except: continue
                
        def fmt_rng(vals):
            clean = [v for v in vals if not pd.isna(v) and 0 < v < 150]
            return f"{min(clean):.1f}-{max(clean):.1f}" if clean else "-"
            
        avg_pe = np.mean([v for v in pe_vals if not pd.isna(v) and 0 < v < 150]) if pe_vals else 0
        return [fmt_rng(pe_vals), fmt_rng(pb_vals), fmt_rng(ps_vals), fmt_rng(evebitda_vals)], avg_pe
    except: return ["-", "-", "-", "-"], 0

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
        
        g1 = min(max(real_growth * 0.8, 0.02), 0.25); g_term = 0.025; g2 = (g1 + g_term) / 2
        
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
# 3. 評分邏輯 (V6.6: IB 與 Quant 嚴格審核機制)
# ==========================================
def calculate_raw_scores(info, financials, growth_rate, qoq_growth, valuation_upside, cur_pe, cur_ev_ebitda, hist_avg_pe, industry_pe_median, wacc, roic):
    scores = {'Q': 0, 'V': 0, 'G': 0, 'Msg': []}
    w_q, w_v, w_g = (0.2, 0.3, 0.5) if growth_rate > 0.15 else ((0.5, 0.4, 0.1) if growth_rate < 0.05 else (0.3, 0.4, 0.3))
    scores['Lifecycle'] = "Growth" if growth_rate > 0.15 else ("Mature" if growth_rate < 0.05 else "Stable")

    # Quality Check
    try: 
        ebit = financials.loc['EBIT'].iloc[0] if 'EBIT' in financials.index else financials.loc['Operating Income'].iloc[0]
        icr = ebit / abs(financials.loc['Interest Expense'].iloc[0])
    except: icr = 10
    if icr > 5: scores['Q'] += 4
    elif icr < 1.5: scores['Q'] -= 5; scores['Msg'].append("高財務風險")
    else: scores['Q'] += 1
    
    if roic > wacc + 0.05: scores['Q'] += 5 # IB: 需超越 WACC 5% 才拿滿分
    elif roic > wacc: scores['Q'] += 1
    else: scores['Msg'].append("ROIC<WACC")

    # Value Check (Damodaran 估值過熱懲罰)
    if valuation_upside > 0.15: scores['V'] += 4
    elif valuation_upside > 0.0: scores['V'] += 2
    elif valuation_upside < -0.20: scores['V'] -= 4; scores['Msg'].append("估值過熱")
        
    if hist_avg_pe > 0 and 0 < cur_pe < (hist_avg_pe * 1.1): scores['V'] += 3
    if industry_pe_median > 0 and 0 < cur_pe < industry_pe_median: scores['V'] += 3
    if 0 < cur_ev_ebitda < 18: scores['V'] += 3

    # Growth Check (IB 成長門檻與利潤率雙重檢查)
    if growth_rate > 0.10 and roic < wacc: 
        scores['G'] -= 5; scores['Msg'].append("無效成長")
    else:
        if growth_rate > 0.25: scores['G'] += 5 # IB: 門檻提升至 25%
        elif growth_rate > 0.15: scores['G'] += 3
        
    try: # IB: 賠錢賺吆喝檢查
        op_now = financials.loc['Operating Income'].iloc[0] / financials.loc['Total Revenue'].iloc[0]
        op_prev = financials.loc['Operating Income'].iloc[1] / financials.loc['Total Revenue'].iloc[1]
        if op_now < op_prev * 0.95 and growth_rate > 0.1:
            scores['G'] -= 5; scores['Msg'].append("利潤率下滑")
    except: pass

    if qoq_growth > 0.05: scores['G'] += 3
    elif qoq_growth < -0.05: scores['G'] -= 3; scores['Msg'].append("動能轉弱")
    
    if 0 < info.get('pegRatio', 0) < 1.5: scores['G'] += 2

    raw_total = (scores['Q'] * w_q * 10) + (scores['V'] * w_v * 10) + (scores['G'] * w_g * 10)
    
    # Quant: ROIC < WACC 總分打 7 折一票否決
    if roic < wacc: raw_total *= 0.7 
        
    scores['Raw_Total'] = raw_total
    return scores

def compile_stock_data(symbol, ind, stock, info, price, real_g, qoq_g, wacc, roic, ranges, avg_pe, cur_pe, cur_ev, intrinsic, upside, eps, med_pe, is_fin, override_score=None):
    scores = calculate_raw_scores(info, stock.financials.fillna(0), real_g, qoq_g, upside, cur_pe, cur_ev, avg_pe, med_pe, wacc, roic)
    
    # 決定最終分數 (如果是產業掃描，會傳入經過相對排序的 override_score)
    final_score = override_score if override_score is not None else min(scores['Raw_Total'], 100)
    
    status = f"{scores['Lifecycle']} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}" + (f" | ⚠️{' '.join(scores['Msg'])}" if scores['Msg'] else "")
    logic = f"Score: {int(final_score)}" + (" (首選)" if final_score >= 85 else "")
    
    return {
        '產業別': ind, '股票代碼': symbol, '名稱': info.get('shortName', symbol), '現價': price,
        '營收成長率': f"{real_g*100:.1f}%", '營業利益率': f"{info.get('operatingMargins', 0)*100:.1f}%", '淨利率': f"{info.get('profitMargins', 0)*100:.1f}%",
        '預估EPS': round(eps * (1 + min(real_g, 0.1)), 2), 'P/E (TTM)': round(cur_pe, 1) if cur_pe else "-",
        'P/B (Lag)': round(info.get('priceToBook', 0) or 0, 2), 'P/S (Lag)': round(info.get('priceToSalesTrailing12Months', 0) or 0, 2),
        'EV/EBITDA': f"{cur_ev:.1f}" if cur_ev > 0 else "-",
        '預估範圍P/E': ranges[0], '預估範圍P/B': ranges[1], '預估範圍P/S': ranges[2], '預估範圍EV/EBITDA': ranges[3],
        'DCF/DDM合理價': round(intrinsic, 1), '狀態': status, 'vs產業PE': "低於同業" if cur_pe < med_pe else "高於同業",
        '選股邏輯': logic, 'Total_Score': final_score
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
        scores = calculate_raw_scores(stock.info, mock_fin, real_growth, qoq_growth, upside, cur_pe, cur_ev_ebitda, avg_pe, 22.0, wacc, roic)

        dates = hist[hist.index >= target_dt].index
        def get_ret(days):
            td = dates[0] + pd.Timedelta(days=days)
            idx = dates.searchsorted(td)
            if idx < len(dates): return (hist['Close'].iloc[idx] - entry_price) / entry_price
            return None

        return {
            '代碼': sym, '名稱': stock.info.get('shortName', sym), '進場日': target_dt.strftime('%Y-%m-%d'),
            '進場價': round(entry_price, 1), '現價': round(current_price, 1),
            '當時總分': int(min(scores['Raw_Total'], 100)), '當時狀態': f"{scores['Lifecycle']} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}",
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
st.title("V6.6 Eric Chi估值模型")
tab1, tab2, tab3 = st.tabs(["全產業掃描", "單股查詢", "真·時光機回測"])

# --- Tab 1: 全產業掃描 ---
with tab1:
    with st.spinner("讀取本地清單中..."):
        df_all = fetch_industry_list_v6()
    
    if df_all.empty:
        st.error("❌ 找不到 tw_stock_list.csv，請確認已上傳。")
    else:
        valid_industries = sorted([i for i in df_all['Industry'].unique()])
        st.info(f"偵測到 {len(valid_industries)} 個產業。系統已啟動 Quant 相對排序與 IB 嚴篩機制。")
        if st.button("執行全產業掃描", type="primary"):
            pb = st.progress(0); status_text = st.empty(); results_container = st.container()
            total_inds = len(valid_industries)
            # V6.6 完整 19 欄位歸隊
            cols_display = ['股票代碼', '名稱', '現價', '營收成長率', '營業利益率', '淨利率', '預估EPS', 'P/E (TTM)', 'P/B (Lag)', 'P/S (Lag)', 'EV/EBITDA', '預估範圍P/E', '預估範圍P/B', '預估範圍P/S', '預估範圍EV/EBITDA', 'DCF/DDM合理價', '狀態', 'vs產業PE', '選股邏輯']
            
            for idx, ind in enumerate(valid_industries):
                status_text.text(f"進度: {idx+1}/{total_inds} | 正在精算 [{ind}]...")
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
                    targets = [x[0] for x in caps[:max(len(caps)//2, 1)]]
                else:
                    targets = tickers[:15]
                
                ind_pes = []; raw_data = []
                for sym in targets:
                    try:
                        stock = yf.Ticker(sym); info = stock.info
                        price = info.get('currentPrice') or info.get('previousClose')
                        if not price: continue
                        real_g = get_growth_data(stock, sym)
                        q_fin = stock.quarterly_financials
                        qoq_g = (q_fin.loc['Total Revenue'].iloc[0] - q_fin.loc['Total Revenue'].iloc[1]) / q_fin.loc['Total Revenue'].iloc[1] if not q_fin.empty and len(q_fin.columns) >= 2 else 0
                        ranges, avg_pe = get_historical_metrics(stock, stock.history(period="10y"))
                        eps = info.get('trailingEps', 0); cur_pe = price / eps if eps > 0 else 0
                        if 0 < cur_pe < 120: ind_pes.append(cur_pe)
                        cur_ev = info.get('enterpriseToEbitda', 0)
                        if not cur_ev:
                            mcap = price * info.get('sharesOutstanding', 1)
                            cur_ev = (mcap + info.get('totalDebt', 0) - info.get('totalCash', 0)) / info.get('ebitda', 1)
                        is_fin = any(x in ind for x in ["金融", "保險"])
                        intrinsic, g_used, wacc, roic = get_3_stage_valuation(stock, is_fin, real_g)
                        upside = (intrinsic - price) / price if intrinsic > 0 else -1
                        
                        raw_data.append({'sym': sym, 'ind': ind, 'stock': stock, 'info': info, 'price': price, 'real_g': real_g, 'qoq_g': qoq_g, 'wacc': wacc, 'roic': roic, 'ranges': ranges, 'avg_pe': avg_pe, 'cur_pe': cur_pe, 'cur_ev': cur_ev, 'intrinsic': intrinsic, 'upside': upside, 'eps': eps, 'is_fin': is_fin})
                    except: pass
                
                pe_med = np.median(ind_pes) if ind_pes else 22.0
                
                # Quant: 產業內強制百分位數排名 (Percentile Ranking)
                raw_scores = []
                for d in raw_data:
                    s = calculate_raw_scores(d['info'], d['stock'].financials.fillna(0), d['real_g'], d['qoq_g'], d['upside'], d['cur_pe'], d['cur_ev'], d['avg_pe'], pe_med, d['wacc'], d['roic'])
                    raw_scores.append(s['Raw_Total'])
                
                if len(raw_scores) > 1:
                    ranks = pd.Series(raw_scores).rank(pct=True)
                    adjusted_scores = 40 + (ranks * 60) # 強制將分數分配在 40~100 之間，確保鑑別度
                else:
                    adjusted_scores = pd.Series(raw_scores)

                ind_results = []
                for i, d in enumerate(raw_data):
                    final_sc = adjusted_scores.iloc[i]
                    res = compile_stock_data(d['sym'], d['ind'], d['stock'], d['info'], d['price'], d['real_g'], d['qoq_g'], d['wacc'], d['roic'], d['ranges'], d['avg_pe'], d['cur_pe'], d['cur_ev'], d['intrinsic'], d['upside'], d['eps'], pe_med, d['is_fin'], override_score=final_sc)
                    ind_results.append(res)

                if ind_results:
                    df_ind = pd.DataFrame(ind_results).sort_values(by='Total_Score', ascending=False).head(6)
                    with results_container:
                        st.markdown(f"### 🏆 {ind} (強勢排名 Top 6)")
                        st.dataframe(df_ind[cols_display], use_container_width=True)
                pb.progress((idx + 1) / total_inds)
            status_text.text("✅ 全市場產業掃描完成！")

# --- Tab 2: 單股查詢 ---
with tab2:
    col_input, col_