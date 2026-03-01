import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings

st.set_page_config(page_title="V7.0 Eric Chi 估值模型 (本地金庫版)", page_icon="🏦", layout="wide")
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 0. 讀取本地三大金庫 (快取機制，秒速載入)
# ==========================================
@st.cache_data(show_spinner=False)
def load_local_databases():
    df_list = pd.read_csv('tw_stock_list.csv') if os.path.exists('tw_stock_list.csv') else pd.DataFrame()
    
    # 讀取 FinMind 財報資料庫
    df_is = pd.read_csv('tw_historical_is.csv') if os.path.exists('tw_historical_is.csv') else pd.DataFrame()
    df_bs = pd.read_csv('tw_historical_bs.csv') if os.path.exists('tw_historical_bs.csv') else pd.DataFrame()
    df_cf = pd.read_csv('tw_historical_cf.csv') if os.path.exists('tw_historical_cf.csv') else pd.DataFrame()
    
    if not df_is.empty: df_is['date'] = pd.to_datetime(df_is['date'])
    if not df_bs.empty: df_bs['date'] = pd.to_datetime(df_bs['date'])
    if not df_cf.empty: df_cf['date'] = pd.to_datetime(df_cf['date'])
        
    return df_list, df_is, df_bs, df_cf

df_all, DB_IS, DB_BS, DB_CF = load_local_databases()

# ==========================================
# 1. 核心資料萃取器 (將長表格轉為寬表格)
# ==========================================
def get_stock_financials(ticker):
    clean_ticker = str(ticker).replace('.TW', '').replace('.TWO', '')
    
    s_is = DB_IS[DB_IS['stock_id'].astype(str) == clean_ticker] if not DB_IS.empty else pd.DataFrame()
    s_bs = DB_BS[DB_BS['stock_id'].astype(str) == clean_ticker] if not DB_BS.empty else pd.DataFrame()
    s_cf = DB_CF[DB_CF['stock_id'].astype(str) == clean_ticker] if not DB_CF.empty else pd.DataFrame()
    
    p_is = s_is.pivot_table(index='date', columns='type', values='value').sort_index(ascending=False) if not s_is.empty else pd.DataFrame()
    p_bs = s_bs.pivot_table(index='date', columns='type', values='value').sort_index(ascending=False) if not s_bs.empty else pd.DataFrame()
    p_cf = s_cf.pivot_table(index='date', columns='type', values='value').sort_index(ascending=False) if not s_cf.empty else pd.DataFrame()
    
    return p_is, p_bs, p_cf

def safe_val(df, idx_date, keys, default=0):
    if df.empty or idx_date not in df.index: return default
    for k in keys:
        if k in df.columns and pd.notna(df.loc[idx_date, k]): return float(df.loc[idx_date, k])
    return default

# ==========================================
# 2. 歷史區間計算 (V7.0 直接讀取本地資料)
# ==========================================
def get_historical_metrics_local(p_is, p_bs, p_cf, hist_price, shares):
    try:
        if p_is.empty or hist_price.empty: return ["-"]*4, 0, 0, 0
        hist_price.index = hist_price.index.tz_localize(None) if hist_price.index.tz else hist_price.index
        
        pe_vals, pb_vals, ps_vals, evebitda_vals = [], [], [], []
        
        for r_date in p_is.index:
            nearest = hist_price.index.get_indexer([r_date], method='nearest')[0]
            if nearest == -1: continue
            p = float(hist_price.iloc[nearest]['Close'])
            
            # 負債與現金
            debt = safe_val(p_bs, r_date, ['CurrentLiabilities', 'NoncurrentLiabilities']) # 簡化估算總負債
            cash = safe_val(p_bs, r_date, ['CashAndCashEquivalents', 'CashAndCashEquivalents_per'])
            
            # EBITDA (營業利益 + 折舊 + 攤銷)
            op_inc = safe_val(p_is, r_date, ['OperatingIncome'])
            dep = safe_val(p_cf, r_date, ['Depreciation'])
            ebitda = op_inc + dep
            if ebitda <= 0: ebitda = op_inc * 1.2 # 若無折舊資料則推估
            
            ev = (p * shares) + debt - cash
            if ebitda > 0 and 0 < (ev / (ebitda * 4)) < 100: evebitda_vals.append(ev / (ebitda * 4))
            
            eps = safe_val(p_is, r_date, ['EPS'])
            if eps > 0: pe_vals.append(p / (eps * 4))
            
            rev = safe_val(p_is, r_date, ['Revenue'])
            if rev > 0: ps_vals.append(p / ((rev/shares) * 4))
                
            bv = safe_val(p_bs, r_date, ['EquityAttributableToOwnersOfParent', 'TotalEquity'])
            if bv > 0: pb_vals.append(p / (bv/shares))
                
        def fmt_rng(v): return f"{min(v):.1f}-{max(v):.1f}" if v else "-"
        c_pe = [v for v in pe_vals if 0<v<150]
        c_pb = [v for v in pb_vals if 0<v<150]
        
        return [fmt_rng(c_pe), fmt_rng(c_pb), fmt_rng([v for v in ps_vals if 0<v<150]), fmt_rng(evebitda_vals)], np.mean(c_pe) if c_pe else 0, min(c_pb) if c_pb else 0, np.mean(c_pb) if c_pb else 0
    except: return ["-"]*4, 0, 0, 0

# ==========================================
# 3. 終極 DCF 現金流估值 (V7.0 真實自由現金流)
# ==========================================
def get_3_stage_valuation_local(p_is, p_bs, p_cf, shares, is_fin, real_g, beta):
    try:
        if p_is.empty: return 0, 0, 0.1, 0
        ld = p_is.index[0]
        
        eq = safe_val(p_bs, ld, ['EquityAttributableToOwnersOfParent', 'TotalEquity'], 1)
        debt = safe_val(p_bs, ld, ['CurrentLiabilities'])
        cash = safe_val(p_bs, ld, ['CashAndCashEquivalents'])
        op_inc = safe_val(p_is, ld, ['OperatingIncome'])
        
        # 自由現金流 (FCF = 營業現金流 + 投資現金流)
        op_cf = safe_val(p_cf, ld, ['CashFlowsFromOperatingActivities', 'NetCashInflowFromOperatingActivities'])
        inv_cf = safe_val(p_cf, ld, ['CashProvidedByInvestingActivities'])
        fcf = op_cf + inv_cf
        if fcf <= 0: fcf = op_inc * 0.7 # 備用推算法
        
        wacc = max((eq/(eq+debt))*max(0.035+(beta*0.06),0.07) + (debt/(eq+debt))*0.025, 0.08 if is_fin else 0.0)
        ic = eq + debt - cash
        roic = (op_inc * 0.8 / ic) if ic > 0 else 0.05
        
        g1, g_term = min(max(real_g * 0.8, 0.02), 0.25), 0.025
        base_cf = safe_val(p_is, ld, ['NetIncome']) if is_fin else fcf
        if base_cf <= 0: return 0, g1, wacc, roic
            
        dcf = sum([base_cf*((1+g1)**i)/((1+wacc)**i) for i in range(1,4)]) + sum([(base_cf*((1+g1)**3))*((1+(g1+g_term)/2)**(i-3))/((1+wacc)**i) for i in range(4,6)])
        dcf += ((base_cf*((1+g1)**3)*((1+(g1+g_term)/2)**2))*(1+g_term)/(wacc-g_term)) / ((1+wacc)**5)
        
        return max((dcf - (debt if not is_fin else 0) + (cash if not is_fin else 0)) / (shares if shares > 0 else 1), 0), g1, wacc, roic
    except: return 0, 0, 0.1, 0

# ==========================================
# 4. Q-V-G 評分與狀態編譯
# ==========================================
def calculate_scores(info, real_g, qoq_g, upside, cur_pe, cur_ev, avg_pe, med_pe, cur_pb, min_pb, avg_pb, wacc, roic, debt_ebitda, op_m, ind):
    s = {'Q': 0, 'V': 0, 'G': 0, 'Total': 0, 'Msg': []}
    wq, wv, wg = (0.2, 0.3, 0.5) if real_g > 0.15 else ((0.5, 0.4, 0.1) if real_g < 0.05 else (0.3, 0.4, 0.3))
    s['Lifecycle'] = "Growth" if real_g > 0.15 else ("Mature" if real_g < 0.05 else "Stable")

    cyclical_industries = ["航運業", "鋼鐵工業", "塑膠工業", "玻璃陶瓷", "造紙工業", "橡膠工業", "水泥工業", "建材營造", "光電業", "油電燃氣業"]
    is_cyclical = ind in cyclical_industries

    if debt_ebitda > 0:
        if debt_ebitda < 4.0: s['Q'] += 3
        elif debt_ebitda > 4.0: s['Q'] -= 5; s['Msg'].append("高財務風險")
    if roic > wacc: s['Q'] += 4
    else: s['Q'] -= 2; s['Msg'].append("ROIC<WACC")
    if len(op_m) >= 4 and all(op_m[i] > op_m[i+1] for i in range(3)): s['Q'] += 3
    elif len(op_m) >= 2 and op_m[0] > op_m[1]: s['Q'] += 2
    elif len(op_m) >= 2 and op_m[0] < op_m[1]: s['Q'] -= 1; s['Msg'].append("營益率下滑")

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

# ==========================================
# 5. 真·時點回測引擎 (V7.0 本地光速版)
# ==========================================
def run_pit_backtest_local(sym, stock, target_date, is_finance, industry_name):
    try:
        target_dt = pd.to_datetime(target_date).tz_localize(None)
        hist = stock.history(start=target_dt - pd.Timedelta(days=3650), end=datetime.today())
        if hist.empty: raise ValueError("無股價資料")
        if hist.index.tz: hist.index = hist.index.tz_localize(None)
        if hist[hist.index >= target_dt].empty: raise ValueError("無目標日後股價")

        ep = float(hist[hist.index >= target_dt]['Close'].iloc[0])
        cp = float(hist['Close'].iloc[-1])
        
        # 讀取本地三大表
        p_is, p_bs, p_cf = get_stock_financials(sym)
        if p_is.empty: raise ValueError("本地庫無財報")
        
        # 嚴格過濾未來資訊 (延遲 45 天發布)
        valid_dates = p_is.index[p_is.index + pd.Timedelta(days=45) <= target_dt]
        if len(valid_dates) < 1: raise ValueError("無歷史財報")

        ld = valid_dates[0]
        eps_list = [safe_val(p_is, d, ['EPS']) for d in valid_dates[:4]]
        eps_ttm = np.mean(eps_list) * 4 if eps_list else 0
        
        rev_list = [safe_val(p_is, d, ['Revenue']) for d in valid_dates[:4]]
        rev_ttm = np.mean(rev_list) * 4 if rev_list else 0
        prev_rev = np.mean([safe_val(p_is, d, ['Revenue']) for d in valid_dates[4:8]]) * 4 if len(valid_dates) >= 8 else 0
        
        real_growth = (rev_ttm - prev_rev) / prev_rev if prev_rev > 0 else 0.05
        qoq_growth = (safe_val(p_is, valid_dates[0], ['Revenue']) - safe_val(p_is, valid_dates[1], ['Revenue'])) / safe_val(p_is, valid_dates[1], ['Revenue']) if len(valid_dates) > 1 and safe_val(p_is, valid_dates[1], ['Revenue'])>0 else 0

        op_margins = [safe_val(p_is, d, ['OperatingIncome']) / safe_val(p_is, d, ['Revenue']) for d in valid_dates[:4] if safe_val(p_is, d, ['Revenue']) > 0]

        sh = stock.info.get('sharesOutstanding', 1)
        shares = float(sh) if sh is not None and sh > 0 else 1.0

        equity = safe_val(p_bs, ld, ['EquityAttributableToOwnersOfParent'], 1)
        debt = safe_val(p_bs, ld, ['CurrentLiabilities'])
        cash = safe_val(p_bs, ld, ['CashAndCashEquivalents'])
        
        ebitda_list = [(safe_val(p_is, d, ['OperatingIncome']) + safe_val(p_cf, d, ['Depreciation'])) for d in valid_dates[:4]]
        ttm_ebitda = np.mean(ebitda_list) * 4 if ebitda_list else 0
        
        cur_pb = ep / (equity / shares) if equity > 0 else 0
        cur_pe = ep / eps_ttm if eps_ttm > 0 else 0
        cur_ev = ((ep * shares) + debt - cash) / (safe_val(p_is, ld, ['OperatingIncome'])*4) if safe_val(p_is, ld, ['OperatingIncome']) > 0 else 0

        rng, avg_pe, min_pb, avg_pb = get_historical_metrics_local(p_is, p_bs, p_cf, hist, shares)
        intrin, g, wacc, roic = get_3_stage_valuation_local(p_is, p_bs, p_cf, shares, is_finance, real_growth, stock.info.get('beta', 1.0))

        upside = (intrin - ep) / ep if intrin > 0 else -1
        scores = calculate_scores(stock.info, real_growth, qoq_growth, upside, cur_pe, cur_ev, avg_pe, 22.0, cur_pb, min_pb, avg_pb, wacc, roic, debt/ttm_ebitda if ttm_ebitda > 0 else 0, op_margins, industry_name)

        dts = hist[hist.index >= target_dt].index
        def ret(days): 
            idx = dts.searchsorted(dts[0]+pd.Timedelta(days=days))
            return (hist['Close'].iloc[idx] - ep)/ep if idx < len(dts) else None

        status_msg = f"{scores['Lifecycle']} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}"
        if scores['Msg']: status_msg += f" | {' '.join(scores['Msg'])}"

        return {
            '代碼': sym, '名稱': stock.info.get('shortName', sym), '進場日': target_dt.strftime('%Y-%m-%d'),
            '進場價': round(ep, 1), '現價': round(cp, 1),
            '當時總分': int(scores['Total']), '當時狀態': status_msg,
            '當時合理價': round(intrin, 1), '當時PE': round(cur_pe, 1),
            '3個月': f"{ret(90)*100:.1f}%" if ret(90) else "-", '6個月': f"{ret(180)*100:.1f}%" if ret(180) else "-",
            '12個月': f"{ret(365)*100:.1f}%" if ret(365) else "-", '至今報酬': f"{(cp - ep)/ep*100:.1f}%", 'Raw': (cp - ep)/ep
        }
    except Exception as e:
        return {'代碼': sym, '名稱': '-', '進場日': target_date, '進場價': 0, '現價': 0, '當時總分': 0, '當時狀態': f"⚠️ 無法計算 ({str(e)[:10]})", '當時合理價': 0, '當時PE': 0, '3個月': "-", '6個月': "-", '12個月': "-", '至今報酬': "-", 'Raw': 0}

# ==========================================
# UI 介面
# ==========================================
st.title("V7.0 Eric Chi估值模型 (本地金庫版)")
tab1, tab2, tab3 = st.tabs(["全產業掃描", "單股查詢", "真·時光機回測"])

cols_display = ['股票代碼', '名稱', '現價', '營收成長率', '預估EPS', '營業利益率', '淨利率', 'P/E (TTM)', 'P/B (Lag)', 'P/S (Lag)', 'EV/EBITDA', '預估範圍P/E', '預估範圍P/B', '預估範圍P/S', '預估範圍EV/EBITDA', 'DCF/DDM合理價', '狀態', 'vs產業PE', '選股邏輯']

with tab2:
    st.info("💡 溫馨提示：V7.0 單股查詢已全面升級！所有財報皆秒讀本地 CSV 金庫，股價由 Yahoo 即時更新。")
    c_in, c_out = st.columns([1, 2])
    with c_in:
        sym_input = st.text_input("輸入代碼:", value="2330")
        if st.button("查詢", type="primary"):
            sym = sym_input.strip().upper()
            if not sym.endswith('.TW') and not sym.endswith('.TWO'):
                if not df_all.empty:
                    match = df_all[df_all['Code'].astype(str) == str(sym)]
                    sym = match.iloc[0]['Ticker'] if not match.empty else f"{sym}.TW"
                else:
                    sym = f"{sym}.TW"

            with st.spinner(f"正在從本地資料庫萃取 ({sym})..."):
                try:
                    ind_lookup = df_all[df_all['Ticker'] == sym]['Industry'] if not df_all.empty else pd.Series()
                    real_industry = ind_lookup.iloc[0] if not ind_lookup.empty else "未知"
                    is_fin = any(x in real_industry for x in ["金融", "保險"])

                    stock = yf.Ticker(sym); info = stock.info
                    p = info.get('currentPrice') or info.get('previousClose')
                    
                    # 讀取本地三大表
                    p_is, p_bs, p_cf = get_stock_financials(sym)
                    
                    if p_is.empty:
                        st.error("❌ 本地資料庫中找不到這檔股票的財報！請確認建庫時是否包含此代碼。")
                    else:
                        ld = p_is.index[0]
                        eps = safe_val(p_is, ld, ['EPS']) * 4
                        real_g = (safe_val(p_is, p_is.index[0], ['Revenue']) - safe_val(p_is, p_is.index[4], ['Revenue'])) / safe_val(p_is, p_is.index[4], ['Revenue']) if len(p_is) >= 5 and safe_val(p_is, p_is.index[4], ['Revenue']) > 0 else 0
                        qoq_g = (safe_val(p_is, p_is.index[0], ['Revenue']) - safe_val(p_is, p_is.index[1], ['Revenue'])) / safe_val(p_is, p_is.index[1], ['Revenue']) if len(p_is) > 1 and safe_val(p_is, p_is.index[1], ['Revenue']) > 0 else 0
                        
                        shares = float(info.get('sharesOutstanding', 1) or 1)
                        rng, avg_pe, min_pb, avg_pb = get_historical_metrics_local(p_is, p_bs, p_cf, stock.history(period="10y"), shares)
                        
                        c_pe = p/eps if eps>0 else 0
                        c_pb = p / (safe_val(p_bs, ld, ['EquityAttributableToOwnersOfParent'])/shares) if safe_val(p_bs, ld, ['EquityAttributableToOwnersOfParent']) > 0 else 0
                        
                        debt = safe_val(p_bs, ld, ['CurrentLiabilities'])
                        cash = safe_val(p_bs, ld, ['CashAndCashEquivalents'])
                        ebitda = safe_val(p_is, ld, ['OperatingIncome']) + safe_val(p_cf, ld, ['Depreciation'])
                        c_ev = ((p * shares) + debt - cash) / (ebitda*4) if ebitda > 0 else 0
                        
                        intrin, g, wacc, roic = get_3_stage_valuation_local(p_is, p_bs, p_cf, shares, is_fin, real_g, info.get('beta', 1.0))
                        upside = (intrin - p) / p if intr