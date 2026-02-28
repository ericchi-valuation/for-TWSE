import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import warnings

st.set_page_config(page_title="V6.9 Eric Chi估值模型", page_icon="📈", layout="wide")
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 0. 基礎函數 & 強大容錯提取器
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_industry_list_v6():
    try: return pd.read_csv('tw_stock_list.csv')
    except: return pd.DataFrame() 

def get_growth_data(stock, symbol):
    try:
        url = f"https://tw.stock.yahoo.com/quote/{symbol.split('.')[0]}.TW/revenue"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        for row in soup.select('li.List\(n\)'):
            lbl = row.select_one('div > span')
            if lbl and '累計營收年增率' in lbl.text:
                return float(row.select('div > span')[-1].text.replace('%', '').replace(',', '')) / 100.0
    except: pass
    rev_g = stock.info.get('revenueGrowth', 0.0)
    return float(rev_g) if rev_g is not None else 0.0

def safe_get(df, idx, keys, default=0):
    """無敵資料萃取器：自動找尋同義欄位，並處理重複日期造成的 Series 錯誤"""
    if idx not in df.index: return default
    for k in keys:
        if k in df.columns:
            val = df.loc[idx, k]
            if isinstance(val, pd.Series): val = val.iloc[0] # 解決 yfinance 日期重複的 Bug
            if pd.notna(val): return float(val)
    return default

# ==========================================
# 1. 歷史區間計算
# ==========================================
def get_historical_metrics(stock, hist_data):
    try:
        if hist_data.empty: return ["-"]*4, 0, 0, 0
        hist_data.index = hist_data.index.tz_localize(None) if hist_data.index.tz else hist_data.index
        
        fin = stock.quarterly_financials.T
        bs = stock.quarterly_balance_sheet.T
        if fin.empty or bs.empty: return ["-"]*4, 0, 0, 0
        
        fin.index = pd.to_datetime(fin.index).tz_localize(None) if fin.index.tz else pd.to_datetime(fin.index)
        bs.index = pd.to_datetime(bs.index).tz_localize(None) if bs.index.tz else pd.to_datetime(bs.index)
            
        pe_vals, pb_vals, ps_vals, evebitda_vals = [], [], [], []
        sh = stock.info.get('sharesOutstanding', 1)
        shares = float(sh) if sh is not None and sh > 0 else 1.0
        
        for r_date in fin.index:
            try:
                if r_date not in hist_data.index:
                    nearest = hist_data.index.get_indexer([r_date], method='nearest')[0]
                    if nearest == -1: continue
                    p = float(hist_data.iloc[nearest]['Close'])
                else: p = float(hist_data.loc[r_date]['Close'])
                
                if r_date in bs.index:
                    debt = safe_get(bs, r_date, ['Total Debt'])
                    cash = safe_get(bs, r_date, ['Cash And Cash Equivalents', 'Cash'])
                    ev = (p * shares) + debt - cash
                    ebitda = safe_get(fin, r_date, ['EBITDA', 'EBIT', 'Operating Income'])
                    if ebitda > 0 and 0 < (ev / (ebitda * 4)) < 100: evebitda_vals.append(ev / (ebitda * 4))
                
                eps = safe_get(fin, r_date, ['Basic EPS', 'Diluted EPS'])
                if eps > 0: pe_vals.append(p / (eps * 4))
                
                rev = safe_get(fin, r_date, ['Total Revenue', 'Operating Revenue'])
                if rev > 0: ps_vals.append(p / ((rev/shares) * 4))
                    
                bv = safe_get(bs, r_date, ['Stockholders Equity', 'Total Equity Gross Minority Interest'])
                if bv > 0: pb_vals.append(p / (bv/shares))
            except: continue
                
        def fmt_rng(v): return f"{min(v):.1f}-{max(v):.1f}" if v else "-"
        c_pe = [v for v in pe_vals if 0<v<150]
        c_pb = [v for v in pb_vals if 0<v<150]
        
        avg_pe = np.mean(c_pe) if c_pe else 0
        min_pb = min(c_pb) if c_pb else 0
        avg_pb = np.mean(c_pb) if c_pb else 0
        
        return [fmt_rng(c_pe), fmt_rng(c_pb), fmt_rng([v for v in ps_vals if 0<v<150]), fmt_rng(evebitda_vals)], avg_pe, min_pb, avg_pb
    except: return ["-"]*4, 0, 0, 0

# ==========================================
# 2. 估值核心 
# ==========================================
def get_3_stage_valuation(stock, is_fin, real_g):
    try:
        bs, fin = stock.balance_sheet.fillna(0), stock.financials.fillna(0)
        if bs.empty or fin.empty: return 0, 0, 0.1, 0
        eq = bs.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in bs.index else 1
        debt = bs.loc['Total Debt'].iloc[0] if 'Total Debt' in bs.index else 0
        cash = bs.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs.index else 0
        ebit = fin.loc['EBIT'].iloc[0] if 'EBIT' in fin.index else 0
        
        b = stock.info.get('beta', 1.0)
        beta = float(b) if b is not None else 1.0
        wacc = max((eq/(eq+debt))*max(0.035+(beta*0.06),0.07) + (debt/(eq+debt))*0.025, 0.08 if is_fin else 0.0)
        roic = (ebit * 0.8 / (eq + debt - cash)) if (eq + debt - cash) > 0 else 0.05
        
        g1, g_term = min(max(real_g * 0.8, 0.02), 0.25), 0.025
        base_cf = (stock.info.get('netIncomeToCommon', 0) * 0.6) if is_fin else (ebit * 0.56)
        if base_cf <= 0: return 0, g1, wacc, roic
            
        dcf = sum([base_cf*((1+g1)**i)/((1+wacc)**i) for i in range(1,4)]) + sum([(base_cf*((1+g1)**3))*((1+(g1+g_term)/2)**(i-3))/((1+wacc)**i) for i in range(4,6)])
        dcf += ((base_cf*((1+g1)**3)*((1+(g1+g_term)/2)**2))*(1+g_term)/(wacc-g_term)) / ((1+wacc)**5)
        
        sh = stock.info.get('sharesOutstanding', 1)
        shares = float(sh) if sh is not None and sh > 0 else 1.0
        return max((dcf - (debt if not is_fin else 0) + (cash if not is_fin else 0)) / shares, 0), g1, wacc, roic
    except: return 0, 0, 0.1, 0

# ==========================================
# 3. 評分整合
# ==========================================
def calculate_scores(info, real_g, qoq_g, upside, cur_pe, cur_ev, avg_pe, med_pe, cur_pb, min_pb, avg_pb, wacc, roic, debt_ebitda, op_m, ind):
    s = {'Q': 0, 'V': 0, 'G': 0, 'Total': 0, 'Msg': []}
    wq, wv, wg = (0.2, 0.3, 0.5) if real_g > 0.15 else ((0.5, 0.4, 0.1) if real_g < 0.05 else (0.3, 0.4, 0.3))
    s['Lifecycle'] = "Growth" if real_g > 0.15 else ("Mature" if real_g < 0.05 else "Stable")

    cyclical_industries = ["航運業", "鋼鐵工業", "塑膠工業", "玻璃陶瓷", "造紙工業", "橡膠工業", "水泥工業", "建材營造", "光電業", "油電燃氣業"]
    is_cyclical = ind in cyclical_industries

    # --- Quality ---
    if debt_ebitda > 0:
        if debt_ebitda < 4.0: s['Q'] += 3
        elif debt_ebitda > 4.0: s['Q'] -= 5; s['Msg'].append("高財務風險")
    if roic > wacc: s['Q'] += 4
    else: s['Q'] -= 2; s['Msg'].append("ROIC<WACC")
    if len(op_m) >= 4 and all(op_m[i] > op_m[i+1] for i in range(3)): s['Q'] += 3
    elif len(op_m) >= 2 and op_m[0] > op_m[1]: s['Q'] += 2
    elif len(op_m) >= 2 and op_m[0] < op_m[1]: s['Q'] -= 1; s['Msg'].append("營益率下滑")

    # --- Value ---
    if is_cyclical:
        if min_pb > 0 and 0 < cur_pb < (min_pb * 1.1): s['V'] += 4
        if avg_pb > 0 and 0 < cur_pb < (avg_pb * 0.8): s['V'] += 3
        if 0 < cur_ev < 7: s['V'] += 3
    else:
        if upside > 0.30: s['V'] += 4
        elif upside > 0.0: s['V'] += 2
        if avg_pe > 0 and 0 < cur_pe < (avg_pe * 1.1): s['V'] += 2
        if med_pe > 0 and 0 < cur_pe < med_pe: s['V'] += 1
        if 0 < cur_ev < 15: s['V'] += 1
        if min_pb > 0 and 0 < cur_pb < (min_pb * 1.1): s['V'] += 2

    # --- Growth ---
    if real_g > 0.10 and roic < wacc: s['G'] -= 5; s['Msg'].append("無效成長")
    else:
        if real_g > 0.25: s['G'] += 5
        elif real_g > 0.10: s['G'] += 3
    if qoq_g > 0.05: s['G'] += 3
    elif qoq_g < -0.05: s['G'] -= 3; s['Msg'].append("動能轉弱")
    
    peg = info.get('pegRatio')
    if peg is not None and 0 < float(peg) < 1.5: s['G'] += 2

    s['Total'] = (s['Q']*wq*10) + (s['V']*wv*10) + (s['G']*wg*10)
    if is_cyclical: s['Msg'].append(f"🔄循環股估值")
    return s

def compile_stock_data(symbol, ind, stock, info, price, real_g, qoq_g, wacc, roic, ranges, avg_pe, min_pb, avg_pb, cur_pe, cur_ev, intrinsic, upside, eps, is_fin, med_pe):
    q_fin = stock.quarterly_financials.T if not stock.quarterly_financials.empty else pd.DataFrame()
    q_bs = stock.quarterly_balance_sheet.T if not stock.quarterly_balance_sheet.empty else pd.DataFrame()
    
    op_margins = []
    if not q_fin.empty:
        for d in q_fin.index[:4]:
            r = safe_get(q_fin, d, ['Total Revenue', 'Operating Revenue'])
            o = safe_get(q_fin, d, ['Operating Income', 'EBIT'])
            op_margins.append(o/r if r > 0 else 0)
            
    debt = safe_get(q_bs, q_bs.index[0] if not q_bs.empty else None, ['Total Debt'])
    ttm_ebitda = sum([safe_get(q_fin, d, ['EBITDA', 'EBIT']) for d in q_fin.index[:4]]) if not q_fin.empty else 0
    
    cur_pb = float(info.get('priceToBook', 0) or 0)
    cur_ps = float(info.get('priceToSalesTrailing12Months', 0) or 0)
    
    scores = calculate_scores(info, real_g, qoq_g, upside, cur_pe, cur_ev, avg_pe, med_pe, cur_pb, min_pb, avg_pb, wacc, roic, debt/ttm_ebitda if ttm_ebitda > 0 else 0, op_margins, ind)
    
    status = f"{scores['Lifecycle']} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}" + (f" | {' '.join(scores['Msg'])}" if scores['Msg'] else "")
    logic = f"Score: {int(scores['Total'])}" + (" (首選)" if scores['Total'] >= 70 else "")
    
    op_m_val = float(info.get('operatingMargins', 0) or 0)
    pm_val = float(info.get('profitMargins', 0) or 0)
    
    return {
        '股票代碼': symbol, 
        '名稱': info.get('shortName', symbol), 
        '現價': price,
        '營收成長率': f"{real_g*100:.1f}%", 
        '預估EPS': round(eps * (1 + min(real_g, 0.1)), 2),
        '營業利益率': f"{op_m_val*100:.1f}%", 
        '淨利率': f"{pm_val*100:.1f}%",
        'P/E (TTM)': round(cur_pe, 1) if cur_pe else "-", 
        'P/B (Lag)': round(cur_pb, 2),
        'P/S (Lag)': round(cur_ps, 2),
        'EV/EBITDA': f"{cur_ev:.1f}" if cur_ev > 0 else "-",
        '預估範圍P/E': ranges[0], 
        '預估範圍P/B': ranges[1], 
        '預估範圍P/S': ranges[2], 
        '預估範圍EV/EBITDA': ranges[3],
        'DCF/DDM合理價': round(intrinsic, 1), 
        '狀態': status, 
        'vs產業PE': "低於同業" if cur_pe < med_pe else "高於同業",
        '選股邏輯': logic, 
        'Total_Score': scores['Total'],
        '產業別': ind
    }

# ==========================================
# 4. 時點回測引擎
# ==========================================
def run_pit_backtest(sym, stock, target_date, is_finance, industry_name):
    try:
        target_dt = pd.to_datetime(target_date).tz_localize(None)
        hist = stock.history(start=target_dt - pd.Timedelta(days=3650), end=datetime.today())
        if hist.empty: raise ValueError("無股價資料")
        if hist.index.tz: hist.index = hist.index.tz_localize(None)
        if hist[hist.index >= target_dt].empty: raise ValueError("無目標日後股價")

        ep = float(hist[hist.index >= target_dt]['Close'].iloc[0])
        cp = float(hist['Close'].iloc[-1])
        
        q_fin, q_bs = stock.quarterly_financials.T, stock.quarterly_balance_sheet.T
        if q_fin.empty or q_bs.empty: raise ValueError("YF未提供歷史季報")
        
        q_fin.index = pd.to_datetime(q_fin.index).tz_localize(None) if q_fin.index.tz else pd.to_datetime(q_fin.index)
        q_bs.index = pd.to_datetime(q_bs.index).tz_localize(None) if q_bs.index.tz else pd.to_datetime(q_bs.index)
        
        valid_dates = q_fin.index[q_fin.index + pd.Timedelta(days=45) <= target_dt]
        if len(valid_dates) < 1: raise ValueError("無足夠歷史財報")

        ld = valid_dates[0]
        eps_ttm = sum([safe_get(q_fin, d, ['Basic EPS', 'Diluted EPS']) for d in valid_dates[:4]])
        rev_ttm = sum([safe_get(q_fin, d, ['Total Revenue', 'Operating Revenue']) for d in valid_dates[:4]])
        prev_rev = sum([safe_get(q_fin, d, ['Total Revenue', 'Operating Revenue']) for d in valid_dates[4:8]])
        
        real_growth = (rev_ttm - prev_rev) / prev_rev if prev_rev > 0 else 0.05
        qoq_growth = 0
        if len(valid_dates) > 1:
            cr, pr = safe_get(q_fin, valid_dates[0], ['Total Revenue']), safe_get(q_fin, valid_dates[1], ['Total Revenue'])
            qoq_growth = (cr - pr) / pr if pr > 0 else 0

        op_margins = []
        for d in valid_dates[:4]:
            r, o = safe_get(q_fin, d, ['Total Revenue']), safe_get(q_fin, d, ['Operating Income', 'EBIT'])
            op_margins.append(o/r if r > 0 else 0)

        ebit = safe_get(q_fin, ld, ['EBIT', 'Operating Income'])
        ebitda = safe_get(q_fin, ld, ['EBITDA', 'EBIT'])
        equity = safe_get(q_bs, ld, ['Stockholders Equity', 'Total Equity Gross Minority Interest'], default=1)
        debt = safe_get(q_bs, ld, ['Total Debt'])
        cash = safe_get(q_bs, ld, ['Cash And Cash Equivalents', 'Cash'])
        
        sh = stock.info.get('sharesOutstanding', 1)
        shares = float(sh) if sh is not None and sh > 0 else 1.0

        ttm_ebitda = sum([safe_get(q_fin, d, ['EBITDA', 'EBIT']) for d in valid_dates[:4]])
        debt_to_ebitda = debt / ttm_ebitda if ttm_ebitda > 0 else 0
        cur_pb = ep / (equity / shares) if equity > 0 else 0
        cur_pe = ep / eps_ttm if eps_ttm > 0 else 0
        cur_ev = ((ep * shares) + debt - cash) / (ebitda * 4) if ebitda > 0 else 0

        b = stock.info.get('beta', 1.0)
        beta = float(b) if b is not None else 1.0
        ke = max(0.035 + beta*0.06, 0.07)
        ic = equity + debt - cash
        roic = (ebit * 0.8 * 4 / ic) if ic > 0 else 0.05
        wacc = max((equity/(equity+debt))*ke + (debt/(equity+debt))*0.025, 0.08 if is_finance else 0.025)

        g1, g_term = min(max(real_growth * 0.8, 0.02), 0.25), 0.025
        base_cf = (safe_get(q_fin, ld, ['Net Income', 'Net Income Common Stockholders']) * 4 * 0.6) if is_finance else (ebit * 4 * 0.56)
        
        intrinsic = 0
        if base_cf > 0:
            dcf = sum([base_cf*((1+g1)**i)/((1+wacc)**i) for i in range(1,4)]) + sum([(base_cf*((1+g1)**3))*((1+(g1+g_term)/2)**(i-3))/((1+wacc)**i) for i in range(4,6)])
            dcf += ((base_cf*((1+g1)**3)*((1+(g1+g_term)/2)**2))*(1+g_term)/(wacc-g_term)) / ((1+wacc)**5)
            intrinsic = max((dcf - (debt if not is_finance else 0) + (cash if not is_finance else 0)) / shares, 0)

        upside = (intrinsic - ep) / ep if intrinsic > 0 else -1
        pb_vals = [ep / (safe_get(q_bs, d, ['Stockholders Equity'])/shares) for d in valid_dates[:20] if safe_get(q_bs, d, ['Stockholders Equity']) > 0]
        pe_vals = [ep / (safe_get(q_fin, d, ['Basic EPS'])*4) for d in valid_dates[:20] if safe_get(q_fin, d, ['Basic EPS']) > 0]

        avg_pe = np.mean([v for v in pe_vals if 0<v<150]) if pe_vals else 0
        min_pb = min([v for v in pb_vals if 0<v<150]) if pb_vals else 0
        avg_pb = np.mean([v for v in pb_vals if 0<v<150]) if pb_vals else 0

        scores = calculate_scores(stock.info, real_growth, qoq_growth, upside, cur_pe, cur_ev, avg_pe, 22.0, cur_pb, min_pb, avg_pb, wacc, roic, debt_to_ebitda, op_margins, industry_name)

        dts = hist[hist.index >= target_dt].index
        def ret(days): 
            idx = dts.searchsorted(dts[0]+pd.Timedelta(days=days))
            return (hist['Close'].iloc[idx] - ep)/ep if idx < len(dts) else None

        return {
            '代碼': sym, '名稱': stock.info.get('shortName', sym), '進場日': target_dt.strftime('%Y-%m-%d'),
            '進場價': round(ep, 1), '現價': round(cp, 1),
            '當時總分': int(scores['Total']), '當時狀態': f"{scores['Lifecycle']} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}" + (f" | {' '.join(scores['Msg'])}" if scores['Msg'] else ""),
            '當時合理價': round(intrinsic, 1), '當時PE': round(cur_pe, 1),
            '3個月': f"{ret(90)*100:.1f}%" if ret(90) else "-", '6個月': f"{ret(180)*100:.1f}%" if ret(180) else "-",
            '12個月': f"{ret(365)*100:.1f}%" if ret(365) else "-", '至今報酬': f"{(cp - ep)/ep*100:.1f}%", 'Raw': (cp - ep)/ep
        }
    except Exception as e:
        return {'代碼': sym, '名稱': '-', '進場日': target_date, '進場價': 0, '現價': 0, '當時總分': 0, '當時狀態': f"⚠️ 無法計算 ({str(e)[:10]})", '當時合理價': 0, '當時PE': 0, '3個月': "-", '6個月': "-", '12個月': "-", '至今報酬': "-", 'Raw': 0}

# ==========================================
# UI 介面
# ==========================================
st.title("V6.9 Eric Chi估值模型")
tab1, tab2, tab3 = st.tabs(["全產業掃描", "單股查詢", "真·時光機回測"])

cols_display = ['股票代碼', '名稱', '現價', '營收成長率', '預估EPS', '營業利益率', '淨利率', 'P/E (TTM)', 'P/B (Lag)', 'P/S (Lag)', 'EV/EBITDA', '預估範圍P/E', '預估範圍P/B', '預估範圍P/S', '預估範圍EV/EBITDA', 'DCF/DDM合理價', '狀態', 'vs產業PE', '選股邏輯']

df_all = fetch_industry_list_v6()

with tab1:
    if df_all.empty: st.error("❌ 找不到 tw_stock_list.csv。")
    else:
        selected_inds = st.multiselect("選擇掃描產業 (可多選):", sorted([i for i in df_all['Industry'].unique()]), default=["半導體業"])
        if st.button("執行產業掃描", type="primary") and selected_inds:
            pb, status_text, results_container, all_data = st.progress(0), st.empty(), st.container(), []
            for idx, ind in enumerate(selected_inds):
                status_text.text(f"精算 [{ind}]...")
                caps = []
                # V6.9: 安全處理市值為 None 的狀況
                for t in df_all[df_all["Industry"] == ind]["Ticker"].tolist():
                    try:
                        mc = yf.Ticker(t).fast_info.get('market_cap', 0)
                        caps.append((t, float(mc) if mc is not None else 0.0))
                    except:
                        caps.append((t, 0.0))
                # 市值排序後，只取前 50% 進行運算以節省時間
                targets = [x[0] for x in sorted(caps, key=lambda x: x[1], reverse=True)[:max(len(caps)//2, 1)]]
                
                raw_data, ind_pes = [], []
                for sym in targets:
                    try:
                        stock = yf.Ticker(sym); info = stock.info
                        p = info.get('currentPrice') or info.get('previousClose')
                        if not p: continue
                        real_g = get_growth_data(stock, sym)
                        q_fin = stock.quarterly_financials
                        qoq_g = (q_fin.iloc[0,0]-q_fin.iloc[0,1])/q_fin.iloc[0,1] if not q_fin.empty and len(q_fin.columns)>1 else 0
                        rng, avg_pe, min_pb, avg_pb = get_historical_metrics(stock, stock.history(period="10y"))
                        
                        eps = info.get('trailingEps')
                        eps = float(eps) if eps is not None else 0
                        c_pe = p/eps if eps>0 else 0
                        if 0<c_pe<120: ind_pes.append(c_pe)
                        
                        c_ev = info.get('enterpriseToEbitda')
                        if c_ev is None:
                            sh = info.get('sharesOutstanding', 1)
                            shares = float(sh) if sh is not None else 1.0
                            debt = float(info.get('totalDebt') or 0)
                            cash = float(info.get('totalCash') or 0)
                            ebitda = float(info.get('ebitda') or 1)
                            if ebitda == 0: ebitda = 1
                            c_ev = ((p * shares) + debt - cash) / ebitda
                        else:
                            c_ev = float(c_ev)
                            
                        is_fin = any(x in ind for x in ["金融", "保險"])
                        intrin, g, wacc, roic = get_3_stage_valuation(stock, is_fin, real_g)
                        raw_data.append((sym, ind, stock, info, float(p), real_g, qoq_g, wacc, roic, rng, avg_pe, min_pb, avg_pb, c_pe, c_ev, intrin, (intrin-p)/p if intrin>0 else -1, eps, is_fin))
                    except: pass
                
                med_pe = np.median(ind_pes) if ind_pes else 22.0
                res = [compile_stock_data(*d, med_pe) for d in raw_data]
                if res:
                    all_data.extend(res)
                    with results_container:
                        st.markdown(f"### 🏆 {ind}")
                        st.dataframe(pd.DataFrame(res).sort_values('Total_Score', ascending=False).head(6)[cols_display], use_container_width=True)
                pb.progress((idx + 1) / len(selected_inds))
            status_text.text("✅ 完成！")
            if all_data: st.download_button("📥 下載完整名單", pd.DataFrame(all_data).sort_values('Total_Score', ascending=False)[cols_display].to_csv(index=False).encode('utf-8-sig'), f"V6.9_Scan.csv", "text/csv")

with tab2:
    c_in, c_out = st.columns([1, 2])
    with c_in:
        sym_input = st.text_input("輸入代碼:", value="2330")
        if st.button("查詢", type="primary"):
            # V6.9: 自動補全 .TW 邏輯
            sym = sym_input.strip().upper()
            if not sym.endswith('.TW') and not sym.endswith('.TWO'):
                if not df_all.empty:
                    match = df_all[df_all['Code'].astype(str) == str(sym)]
                    sym = match.iloc[0]['Ticker'] if not match.empty else f"{sym}.TW"
                else:
                    sym = f"{sym}.TW"

            with st.spinner(f"查詢中 ({sym})..."):
                try:
                    ind_lookup = df_all[df_all['Ticker'] == sym]['Industry'] if not df_all.empty else pd.Series()
                    real_industry = ind_lookup.iloc[0] if not ind_lookup.empty else "未知"

                    stock = yf.Ticker(sym); info = stock.info
                    p = info.get('currentPrice') or info.get('previousClose')
                    real_g = get_growth_data(stock, sym)
                    q_fin = stock.quarterly_financials
                    qoq_g = (q_fin.iloc[0,0]-q_fin.iloc[0,1])/q_fin.iloc[0,1] if not q_fin.empty and len(q_fin.columns)>1 else 0
                    rng, avg_pe, min_pb, avg_pb = get_historical_metrics(stock, stock.history(period="10y"))
                    
                    eps = info.get('trailingEps')
                    eps = float(eps) if eps is not None else 0
                    c_pe = p/eps if eps>0 else 0
                    
                    c_ev = info.get('enterpriseToEbitda')
                    c_ev = float(c_ev) if c_ev is not None else 0
                    
                    is_fin = any(x in real_industry for x in ["金融", "保險"])
                    intrin, g, wacc, roic = get_3_stage_valuation(stock, is_fin, real_g)
                    
                    data = compile_stock_data(sym, real_industry, stock, info, p, real_g, qoq_g, wacc, roic, rng, avg_pe, min_pb, avg_pb, c_pe, c_ev, intrin, (intrin-p)/p if intrin>0 else -1, eps, is_fin, 22.0)
                    st.metric("合理價", f"{intrin:.1f}", f"{(intrin-p)/p if intrin>0 else -1:.1%} 空間")
                    st.success(data['狀態'])
                    with c_out: st.dataframe(pd.DataFrame([{k: data[k] for k in cols_display if k in data}]).T, use_container_width=True)
                except Exception as e: st.error("查無資料，請確認代碼是否正確。")

with tab3:
    c1, c2 = st.columns(2)
    with c1: t_input = st.text_area("代碼:", "2603, 2002, 2330") 
    with c2: s_date = st.date_input("日期:", datetime(2023, 11, 27)); run_bt = st.button("執行", type="primary")
    if run_bt:
        # V6.9: 智慧處理輸入的代碼
        t_list_raw = [t.strip().upper() for t in t_input.split(',')]
        t_list = []
        for sym in t_list_raw:
            if not sym.endswith('.TW') and not sym.endswith('.TWO'):
                if not df_all.empty:
                    match = df_all[df_all['Code'].astype(str) == str(sym)]
                    t_list.append(match.iloc[0]['Ticker'] if not match.empty else f"{sym}.TW")
                else: t_list.append(f"{sym}.TW")
            else: t_list.append(sym)

        res_bt, pb = [], st.progress(0)
        for i, sym in enumerate(t_list):
            ind_lookup = df_all[df_all['Ticker'] == sym]['Industry'] if not df_all.empty else pd.Series()
            real_industry = ind_lookup.iloc[0] if not ind_lookup.empty else "未知"
            is_fin = any(x in real_industry for x in ["金融", "保險"])
            
            res_bt.append(run_pit_backtest(sym, yf.Ticker(sym), s_date.strftime('%Y-%m-%d'), is_fin, real_industry))
            pb.progress((i+1)/len(t_list))
        if res_bt:
            df_bt = pd.DataFrame([r for r in res_bt if r])
            st.metric("平均報酬", f"{df_bt['Raw'].mean()*100:.1f}%")
            st.dataframe(df_bt.drop(columns=['Raw']), use_container_width=True)