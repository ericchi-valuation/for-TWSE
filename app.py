import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import io
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
import warnings

st.set_page_config(page_title="V7.3 Eric Chi 估值模型 (真實市值極速版)", page_icon="🏦", layout="wide")
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 0. 建立「反封鎖」連線 Session
# ==========================================
@st.cache_resource(show_spinner=False)
def get_yf_session():
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    })
    return session

yf_session = get_yf_session()

# ==========================================
# 1. 讀取本地三大金庫 (Parquet 極速版)
# ==========================================
@st.cache_data(show_spinner=False)
def load_local_databases():
    df_list = pd.read_csv('tw_stock_list.csv') if os.path.exists('tw_stock_list.csv') else pd.DataFrame()
    
    df_is = pd.read_parquet('tw_is_lite.parquet') if os.path.exists('tw_is_lite.parquet') else pd.DataFrame()
    df_bs = pd.read_parquet('tw_bs_lite.parquet') if os.path.exists('tw_bs_lite.parquet') else pd.DataFrame()
    df_cf = pd.read_parquet('tw_cf_lite.parquet') if os.path.exists('tw_cf_lite.parquet') else pd.DataFrame()
    
    if not df_is.empty: df_is['date'] = pd.to_datetime(df_is['date'])
    if not df_bs.empty: df_bs['date'] = pd.to_datetime(df_bs['date'])
    if not df_cf.empty: df_cf['date'] = pd.to_datetime(df_cf['date'])
        
    return df_list, df_is, df_bs, df_cf

df_all, DB_IS, DB_BS, DB_CF = load_local_databases()

IND_PE_DEFAULT = {
    "半導體業": 25.0, "金融業": 14.0, "航運業": 10.0,
    "生技醫療業": 35.0, "鋼鐵工業": 12.0, "電子零組件業": 20.0,
    "光電業": 18.0, "電腦及週邊設備業": 18.0, "通信網路業": 20.0,
    "電機機械": 22.0
}

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

def get_historical_shares(p_bs, date, fallback_shares):
    cap = safe_val(p_bs, date, ['OrdinaryShare', 'CapitalStock', 'OrdinaryShare_per', 'CapitalStock_per'])
    return cap / 10.0 if cap > 0 else fallback_shares

# ==========================================
# 2. 估值核心引擎
# ==========================================
def get_historical_metrics_local(p_is, p_bs, p_cf, hist_price, current_shares):
    try:
        if p_is.empty or hist_price.empty: return ["-"]*4, 0, 0, 0
        hist_price.index = hist_price.index.tz_localize(None) if hist_price.index.tz else hist_price.index
        
        pe_vals, pb_vals, ps_vals, evebitda_vals = [], [], [], []
        
        for r_date in p_is.index:
            nearest = hist_price.index.get_indexer([r_date], method='nearest')[0]
            if nearest == -1: continue
            p = float(hist_price.iloc[nearest]['Close']) 
            
            hist_shares = get_historical_shares(p_bs, r_date, current_shares)
            debt = safe_val(p_bs, r_date, ['CurrentLiabilities']) + safe_val(p_bs, r_date, ['NoncurrentLiabilities'])
            cash = safe_val(p_bs, r_date, ['CashAndCashEquivalents'])
            
            op_inc = safe_val(p_is, r_date, ['OperatingIncome'])
            dep = safe_val(p_cf, r_date, ['Depreciation'])
            ebitda = op_inc + dep if (op_inc + dep) > 0 else op_inc * 1.2
            
            ev = (p * hist_shares) + debt - cash
            if ebitda > 0 and 0 < (ev / (ebitda * 4)) < 100: evebitda_vals.append(ev / (ebitda * 4))
            
            eps = safe_val(p_is, r_date, ['EPS'])
            if eps > 0: pe_vals.append(p / (eps * 4))
            
            rev = safe_val(p_is, r_date, ['Revenue'])
            if rev > 0: ps_vals.append(p / ((rev/hist_shares) * 4))
                
            bv = safe_val(p_bs, r_date, ['EquityAttributableToOwnersOfParent', 'TotalEquity'])
            if bv > 0: pb_vals.append(p / (bv/hist_shares))
                
        def fmt_rng(v): return f"{min(v):.1f}-{max(v):.1f}" if v else "-"
        c_pe = [v for v in pe_vals if 0<v<150]
        c_pb = [v for v in pb_vals if 0<v<150]
        
        return [fmt_rng(c_pe), fmt_rng(c_pb), fmt_rng([v for v in ps_vals if 0<v<150]), fmt_rng(evebitda_vals)], np.mean(c_pe) if c_pe else 0, min(c_pb) if c_pb else 0, np.mean(c_pb) if c_pb else 0
    except: return ["-"]*4, 0, 0, 0

def get_3_stage_valuation_local(p_is, p_bs, p_cf, shares, is_fin, real_g, beta):
    try:
        if p_is.empty: return (0, 0, 0), 0, 0.1, 0
        ld = p_is.index[0]
        
        eq = safe_val(p_bs, ld, ['EquityAttributableToOwnersOfParent', 'TotalEquity'], 1)
        debt = safe_val(p_bs, ld, ['CurrentLiabilities']) + safe_val(p_bs, ld, ['NoncurrentLiabilities'])
        cash = safe_val(p_bs, ld, ['CashAndCashEquivalents'])
        op_inc = safe_val(p_is, ld, ['OperatingIncome'])
        
        op_cf = safe_val(p_cf, ld, ['CashFlowsFromOperatingActivities', 'NetCashInflowFromOperatingActivities'])
        inv_cf = safe_val(p_cf, ld, ['CashProvidedByInvestingActivities'])
        fcf = op_cf + inv_cf if (op_cf + inv_cf) > 0 else op_inc * 0.7 
        
        int_exp = abs(safe_val(p_cf, ld, ['InterestExpense', 'PayTheInterest']) or safe_val(p_is, ld, ['TotalNonoperatingIncomeAndExpense']))
        kd = int_exp / debt if debt > 0 else 0.025
        kd = max(min(kd, 0.12), 0.015) 
        ke = max(0.035 + (beta * 0.06), 0.07)
        wacc = max((eq/(eq+debt))*ke + (debt/(eq+debt))*kd*(1-0.20), 0.08 if is_fin else 0.04)
        
        ic = eq + debt - cash
        roic = (op_inc * 0.8 / ic) if ic > 0 else 0.05
        
        g1_base, g_term = min(max(real_g * 0.8, 0.02), 0.25), 0.025
        base_cf = safe_val(p_is, ld, ['NetIncome']) if is_fin else fcf
        if base_cf <= 0: return (0, 0, 0), g1_base, wacc, roic
            
        def calc_dcf(g, w):
            dcf = sum([base_cf*((1+g)**i)/((1+w)**i) for i in range(1,4)]) + sum([(base_cf*((1+g)**3))*((1+(g+g_term)/2)**(i-3))/((1+w)**i) for i in range(4,6)])
            dcf += ((base_cf*((1+g)**3)*((1+(g+g_term)/2)**2))*(1+g_term)/(w-g_term)) / ((1+w)**5)
            return max((dcf - (debt if not is_fin else 0) + cash) / (shares if shares > 0 else 1), 0)

        return (calc_dcf(g1_base, wacc), calc_dcf(g1_base * 0.5, wacc * 1.1), calc_dcf(g1_base * 1.2, wacc * 0.95)), g1_base, wacc, roic
    except: return (0, 0, 0), 0, 0.1, 0

def calculate_scores(info, real_g, qoq_g, upside, cur_pe, cur_ev, avg_pe, med_pe, cur_pb, min_pb, avg_pb, wacc, roic, debt_ebitda, op_m, ind):
    s = {'Q': 0, 'V': 0, 'G': 0, 'Total': 0, 'Msg': []}
    wq, wv, wg = (0.2, 0.3, 0.5) if real_g > 0.15 else ((0.5, 0.4, 0.1) if real_g < 0.05 else (0.3, 0.4, 0.3))
    s['Lifecycle'] = "Growth" if real_g > 0.15 else ("Mature" if real_g < 0.05 else "Stable")

    cyclical_industries = ["航運", "鋼鐵", "塑膠", "玻璃陶瓷", "造紙", "橡膠", "水泥", "建材營造", "光電", "油電燃氣"]
    is_cyclical = any(c in ind for c in cyclical_industries)

    if debt_ebitda > 0:
        if debt_ebitda < 4.0: s['Q'] += 3
        elif debt_ebitda > 4.0: s['Q'] -= 5; s['Msg'].append("高風險")
    if roic > wacc: s['Q'] += 4
    else: s['Q'] -= 2; s['Msg'].append("ROIC<WACC")
    if len(op_m) >= 4 and all(op_m[i] > op_m[i+1] for i in range(3)): s['Q'] += 3
    elif len(op_m) >= 2 and op_m[0] > op_m[1]: s['Q'] += 2
    elif len(op_m) >= 2 and op_m[0] < op_m[1]: s['Q'] -= 1; s['Msg'].append("營益率降")
    
    dy = float(info.get('dividendYield', 0) or 0)
    if dy > 0.06: s['Q'] += 2
    elif dy > 0.03: s['Q'] += 1

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
    elif qoq_g < -0.05: s['G'] -= 3; s['Msg'].append("動能弱")
    
    if float(info.get('pegRatio') or 99) < 1.5: s['G'] += 2

    s['Total'] = (s['Q']*wq*10) + (s['V']*wv*10) + (s['G']*wg*10)
    return s

# ==========================================
# 3. 回測引擎
# ==========================================
def run_pit_backtest_local(sym, target_date, is_finance, industry_name):
    try:
        target_dt = pd.to_datetime(target_date).tz_localize(None)
        stock = yf.Ticker(sym, session=yf_session)
        hist = stock.history(start=target_dt - pd.Timedelta(days=3650), end=datetime.today())
        if hist.empty: raise ValueError("無股價資料")
        if hist.index.tz: hist.index = hist.index.tz_localize(None)
        if hist[hist.index >= target_dt].empty: raise ValueError("無目標日股價")

        ep = float(hist[hist.index >= target_dt]['Close'].iloc[0])
        cp = float(hist['Close'].iloc[-1])
        
        p_is, p_bs, p_cf = get_stock_financials(sym)
        if p_is.empty: raise ValueError("無財報")
        valid_dates = p_is.index[p_is.index + pd.Timedelta(days=45) <= target_dt]
        if len(valid_dates) < 1: raise ValueError("無歷史財報")

        ld = valid_dates[0]
        eps_ttm = np.mean([safe_val(p_is, d, ['EPS']) for d in valid_dates[:4]]) * 4
        rev_ttm = np.mean([safe_val(p_is, d, ['Revenue']) for d in valid_dates[:4]]) * 4
        prev_rev = np.mean([safe_val(p_is, d, ['Revenue']) for d in valid_dates[4:8]]) * 4 if len(valid_dates) >= 8 else 0
        real_growth = (rev_ttm - prev_rev) / prev_rev if prev_rev > 0 else 0.05
        
        r_now = safe_val(p_is, valid_dates[0], ['Revenue'])
        r_prev = safe_val(p_is, valid_dates[1], ['Revenue']) if len(valid_dates) > 1 else 0
        qoq_growth = (r_now - r_prev) / r_prev if r_prev > 0 else 0
        op_margins = [safe_val(p_is, d, ['OperatingIncome']) / safe_val(p_is, d, ['Revenue']) for d in valid_dates[:4] if safe_val(p_is, d, ['Revenue']) > 0]

        try: info = stock.info
        except: info = {}

        current_shares = float(info.get('sharesOutstanding', 1) or 1)
        hist_shares = get_historical_shares(p_bs, ld, current_shares)

        equity = safe_val(p_bs, ld, ['EquityAttributableToOwnersOfParent'], 1)
        debt = safe_val(p_bs, ld, ['CurrentLiabilities']) + safe_val(p_bs, ld, ['NoncurrentLiabilities'])
        cash = safe_val(p_bs, ld, ['CashAndCashEquivalents'])
        ttm_ebitda = np.mean([(safe_val(p_is, d, ['OperatingIncome']) + safe_val(p_cf, d, ['Depreciation'])) for d in valid_dates[:4]]) * 4
        
        cur_pb = ep / (equity / hist_shares) if equity > 0 else 0
        cur_pe = ep / eps_ttm if eps_ttm > 0 else 0
        cur_ev = ((ep * hist_shares) + debt - cash) / (safe_val(p_is, ld, ['OperatingIncome'])*4) if safe_val(p_is, ld, ['OperatingIncome']) > 0 else 0

        rng, avg_pe, min_pb, avg_pb = get_historical_metrics_local(p_is, p_bs, p_cf, hist[hist.index <= target_dt], current_shares)
        vals, g, wacc, roic = get_3_stage_valuation_local(p_is, p_bs, p_cf, hist_shares, is_finance, real_growth, info.get('beta', 1.0))
        base_intrin = vals[0]

        upside = (base_intrin - ep) / ep if base_intrin > 0 else -1
        med_pe = IND_PE_DEFAULT.get(industry_name, 22.0)
        scores = calculate_scores(info, real_growth, qoq_growth, upside, cur_pe, cur_ev, avg_pe, med_pe, cur_pb, min_pb, avg_pb, wacc, roic, debt/ttm_ebitda if ttm_ebitda > 0 else 0, op_margins, industry_name)

        dts = hist[hist.index >= target_dt].index
        def ret(days): 
            idx = dts.searchsorted(dts[0]+pd.Timedelta(days=days))
            return (hist['Close'].iloc[idx] - ep)/ep if idx < len(dts) else None

        status_msg = f"{scores['Lifecycle']} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}"
        if scores['Msg']: status_msg += f" | {' '.join(scores['Msg'])}"

        return {
            '代碼': sym, '名稱': info.get('shortName', sym), '進場日': target_dt.strftime('%Y-%m-%d'),
            '進場價': round(ep, 1), '現價': round(cp, 1), '當時總分': int(scores['Total']), '當時狀態': status_msg,
            '當時合理價(Base)': round(base_intrin, 1), '當時PE': round(cur_pe, 1),
            '3個月': f"{ret(90)*100:.1f}%" if ret(90) else "-", '6個月': f"{ret(180)*100:.1f}%" if ret(180) else "-",
            '12個月': f"{ret(365)*100:.1f}%" if ret(365) else "-", '至今報酬': f"{(cp - ep)/ep*100:.1f}%", 'Raw': (cp - ep)/ep
        }
    except Exception as e:
        return {'代碼': sym, '名稱': '-', '進場日': target_date, '進場價': 0, '現價': 0, '當時總分': 0, '當時狀態': f"⚠️ {str(e)[:15]}", '當時合理價(Base)': 0, '當時PE': 0, '3個月': "-", '6個月': "-", '12個月': "-", '至今報酬': "-", 'Raw': 0}

# ==========================================
# UI 介面
# ==========================================
st.title("V7.3 Eric Chi 估值模型 (真實市值極速版)")
tab1, tab2, tab3 = st.tabs(["全產業掃描", "單股深度查詢", "真·時光機回測"])
cols_display = ['股票代碼', '名稱', '現價', '營收成長率', '預估EPS', '營業利益率', '淨利率', 'P/E (TTM)', 'P/B (Lag)', 'EV/EBITDA', 'DCF合理價區間', '狀態', 'vs產業PE', '選股邏輯']

with tab1:
    st.info("⚡ V7.3 突破：採用『批量向量下載技術 (Bulk Download)』，1 次 API 呼叫獲取全產業股價，恢復100%真實市值排序且防封鎖！")
    if df_all.empty: st.error("❌ 找不到本地資料庫。")
    else:
        selected_inds = st.multiselect("選擇掃描產業 (可多選):", sorted([i for i in df_all['Industry'].unique()]), default=["半導體業"])
        if st.button("執行產業掃描", type="primary") and selected_inds:
            pb, status_text, results_container, all_data = st.progress(0), st.empty(), st.container(), []
            
            for idx, ind in enumerate(selected_inds):
                status_text.text(f"正在批量獲取 [{ind}] 真實市值...")
                tickers_list = df_all[df_all["Industry"] == ind]["Ticker"].tolist()
                
                # 🌟 V7.3 核心突破：批量下載整個產業的最新收盤價 (只耗費 1 次 API！)
                try:
                    bulk_data = yf.download(tickers_list, period="5d", progress=False, session=yf_session)
                    if isinstance(bulk_data.columns, pd.MultiIndex):
                        latest_prices = bulk_data['Close'].ffill().iloc[-1]
                    else:
                        latest_prices = pd.Series({tickers_list[0]: bulk_data['Close'].ffill().iloc[-1]})
                except:
                    latest_prices = pd.Series()

                # 精算每一檔的真實市值：最新收盤價 * 本地股本
                caps = []
                for t in tickers_list:
                    clean_ticker = str(t).replace('.TW', '').replace('.TWO', '')
                    s_bs = DB_BS[(DB_BS['stock_id'].astype(str) == clean_ticker) & (DB_BS['type'].isin(['OrdinaryShare', 'CapitalStock', 'OrdinaryShare_per', 'CapitalStock_per']))]
                    shares = float(s_bs['value'].iloc[0]) / 10.0 if not s_bs.empty else 1.0
                    
                    p = float(latest_prices.get(t, 0))
                    if pd.isna(p): p = 0
                    caps.append((t, p * shares))
                
                # 依真實市值排序，精準篩選前 50% 的領頭羊
                targets = [x[0] for x in sorted(caps, key=lambda x: x[1], reverse=True)[:max(len(caps)//2, 1)]]
                raw_data = []
                
                status_text.text(f"精算 [{ind}] 估值模型...")
                for sym in targets:
                    try:
                        stock = yf.Ticker(sym, session=yf_session)
                        p = float(latest_prices.get(sym, 0)) # 直接使用剛剛批次抓下來的價格，0 呼叫！
                        if p == 0: continue 
                        
                        try: info = stock.info
                        except: info = {} 
                        
                        p_is, p_bs, p_cf = get_stock_financials(sym)
                        if p_is.empty: continue
                        ld = p_is.index[0]
                        
                        eps = safe_val(p_is, ld, ['EPS']) * 4
                        r_now = safe_val(p_is, p_is.index[0], ['Revenue'])
                        r_prev = safe_val(p_is, p_is.index[4], ['Revenue']) if len(p_is) >= 5 else 0
                        r_prev_qoq = safe_val(p_is, p_is.index[1], ['Revenue']) if len(p_is) > 1 else 0
                        
                        real_g = (r_now - r_prev) / r_prev if r_prev > 0 else 0
                        qoq_g = (r_now - r_prev_qoq) / r_prev_qoq if r_prev_qoq > 0 else 0
                        
                        shares = float(info.get('sharesOutstanding', 1) or 1)
                        rng, avg_pe, min_pb, avg_pb = get_historical_metrics_local(p_is, p_bs, p_cf, stock.history(period="10y"), shares)
                        
                        c_pe = p/eps if eps>0 else 0
                        c_pb = p / (safe_val(p_bs, ld, ['EquityAttributableToOwnersOfParent'])/shares) if safe_val(p_bs, ld, ['EquityAttributableToOwnersOfParent']) > 0 else 0
                        
                        debt = safe_val(p_bs, ld, ['CurrentLiabilities']) + safe_val(p_bs, ld, ['NoncurrentLiabilities'])
                        cash = safe_val(p_bs, ld, ['CashAndCashEquivalents'])
                        ebitda = safe_val(p_is, ld, ['OperatingIncome']) + safe_val(p_cf, ld, ['Depreciation'])
                        c_ev = ((p * shares) + debt - cash) / (ebitda*4) if ebitda > 0 else 0
                        
                        is_fin = any(x in ind for x in ["金融", "保險"])
                        vals, g, wacc, roic = get_3_stage_valuation_local(p_is, p_bs, p_cf, shares, is_fin, real_g, info.get('beta', 1.0))
                        
                        upside = (vals[0] - p) / p if vals[0] > 0 else -1
                        op_margins = [safe_val(p_is, d, ['OperatingIncome']) / safe_val(p_is, d, ['Revenue']) for d in p_is.index[:4] if safe_val(p_is, d, ['Revenue']) > 0]
                        
                        med_pe = IND_PE_DEFAULT.get(ind, 22.0)
                        scores = calculate_scores(info, real_g, qoq_g, upside, c_pe, c_ev, avg_pe, med_pe, c_pb, min_pb, avg_pb, wacc, roic, debt/(ebitda*4) if ebitda > 0 else 0, op_margins, ind)
                        
                        raw_data.append({
                            '股票代碼': sym, '名稱': info.get('shortName', sym), '現價': float(p),
                            '營收成長率': f"{real_g*100:.1f}%", '預估EPS': round(eps * (1 + min(real_g, 0.1)), 2),
                            '營業利益率': f"{(safe_val(p_is, ld, ['OperatingIncome'])/safe_val(p_is, ld, ['Revenue']))*100:.1f}%" if safe_val(p_is, ld, ['Revenue']) > 0 else "-", 
                            '淨利率': f"{(safe_val(p_is, ld, ['NetIncome'])/safe_val(p_is, ld, ['Revenue']))*100:.1f}%" if safe_val(p_is, ld, ['Revenue']) > 0 else "-",
                            'P/E (TTM)': round(c_pe, 1) if c_pe else "-", 'P/B (Lag)': round(c_pb, 2),
                            'EV/EBITDA': f"{c_ev:.1f}" if c_ev > 0 else "-",
                            'DCF合理價區間': f"{vals[0]:.1f} ({vals[1]:.1f}-{vals[2]:.1f})", 
                            '狀態': f"{scores['Lifecycle']} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}" + (f" | {' '.join(scores['Msg'])}" if scores['Msg'] else ""), 
                            'vs產業PE': "低於同業" if c_pe < med_pe else "高於同業",
                            '選股邏輯': f"Score: {int(scores['Total'])}" + (" (首選)" if scores['Total'] >= 70 else ""),
                            'Total_Score': scores['Total']
                        })
                    except: pass
                
                if raw_data:
                    df_display = pd.DataFrame(raw_data).sort_values('Total_Score', ascending=False)
                    all_data.extend(raw_data)
                    with results_container:
                        st.markdown(f"### 🏆 {ind}")
                        st.dataframe(df_display.head(6)[cols_display], use_container_width=True)
                pb.progress((idx + 1) / len(selected_inds))
            
            status_text.text("✅ 完成！")
            if all_data: 
                df_export = pd.DataFrame(all_data).sort_values('Total_Score', ascending=False)[cols_display]
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_export.to_excel(writer, index=False, sheet_name='掃描結果')
                st.download_button("📥 下載 Excel 名單", data=buffer.getvalue(), file_name=f"V7.3_Scan_{datetime.today().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

with tab2:
    c_in, c_out = st.columns([1, 2])
    with c_in:
        sym_input = st.text_input("輸入代碼:", value="2330")
        if st.button("查詢", type="primary"):
            sym = sym_input.strip().upper()
            if not sym.endswith('.TW') and not sym.endswith('.TWO'):
                if not df_all.empty:
                    match = df_all[df_all['Code'].astype(str) == str(sym)]
                    sym = match.iloc[0]['Ticker'] if not match.empty else f"{sym}.TW"
                else: sym = f"{sym}.TW"

            with st.spinner(f"正在穿透防火牆解析 ({sym})..."):
                try:
                    ind_lookup = df_all[df_all['Ticker'] == sym]['Industry'] if not df_all.empty else pd.Series()
                    real_industry = ind_lookup.iloc[0] if not ind_lookup.empty else "未知"
                    is_fin = any(x in real_industry for x in ["金融", "保險"])

                    stock = yf.Ticker(sym, session=yf_session)
                    try:
                        p = stock.fast_info.get('lastPrice', 0)
                        if p == 0: p = float(stock.history(period="1d")['Close'].iloc[-1])
                    except: st.error("⚠️ 無法獲取最新股價，可能遭遇嚴格封鎖"); st.stop()

                    try: info = stock.info
                    except: info = {}

                    p_is, p_bs, p_cf = get_stock_financials(sym)
                    if p_is.empty: st.error("❌ 本地資料庫中找不到這檔股票的財報！")
                    else:
                        ld = p_is.index[0]
                        eps = safe_val(p_is, ld, ['EPS']) * 4
                        r_now = safe_val(p_is, p_is.index[0], ['Revenue'])
                        r_prev = safe_val(p_is, p_is.index[4], ['Revenue']) if len(p_is) >= 5 else 0
                        r_prev_qoq = safe_val(p_is, p_is.index[1], ['Revenue']) if len(p_is) > 1 else 0
                        
                        real_g = (r_now - r_prev) / r_prev if r_prev > 0 else 0
                        qoq_g = (r_now - r_prev_qoq) / r_prev_qoq if r_prev_qoq > 0 else 0
                        
                        shares = float(info.get('sharesOutstanding', 1) or 1)
                        rng, avg_pe, min_pb, avg_pb = get_historical_metrics_local(p_is, p_bs, p_cf, stock.history(period="10y"), shares)
                        
                        c_pe = p/eps if eps>0 else 0
                        c_pb = p / (safe_val(p_bs, ld, ['EquityAttributableToOwnersOfParent'])/shares) if safe_val(p_bs, ld, ['EquityAttributableToOwnersOfParent']) > 0 else 0
                        
                        debt = safe_val(p_bs, ld, ['CurrentLiabilities']) + safe_val(p_bs, ld, ['NoncurrentLiabilities'])
                        cash = safe_val(p_bs, ld, ['CashAndCashEquivalents'])
                        ebitda = safe_val(p_is, ld, ['OperatingIncome']) + safe_val(p_cf, ld, ['Depreciation'])
                        c_ev = ((p * shares) + debt - cash) / (ebitda*4) if ebitda > 0 else 0
                        
                        vals, g, wacc, roic = get_3_stage_valuation_local(p_is, p_bs, p_cf, shares, is_fin, real_g, info.get('beta', 1.0))
                        upside = (vals[0] - p) / p if vals[0] > 0 else -1
                        op_margins = [safe_val(p_is, d, ['OperatingIncome']) / safe_val(p_is, d, ['Revenue']) for d in p_is.index[:4] if safe_val(p_is, d, ['Revenue']) > 0]
                        med_pe = IND_PE_DEFAULT.get(real_industry, 22.0)
                        
                        scores = calculate_scores(info, real_g, qoq_g, upside, c_pe, c_ev, avg_pe, med_pe, c_pb, min_pb, avg_pb, wacc, roic, debt/(ebitda*4) if ebitda > 0 else 0, op_margins, real_industry)
                        status = f"{scores['Lifecycle']} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}" + (f" | {' '.join(scores['Msg'])}" if scores['Msg'] else "")
                        
                        data = {
                            '股票代碼': sym, '名稱': info.get('shortName', sym), '現價': float(p),
                            '營收成長率': f"{real_g*100:.1f}%", '預估EPS': round(eps * (1 + min(real_g, 0.1)), 2),
                            '營業利益率': f"{(safe_val(p_is, ld, ['OperatingIncome'])/safe_val(p_is, ld, ['Revenue']))*100:.1f}%" if safe_val(p_is, ld, ['Revenue']) > 0 else "-", 
                            '淨利率': f"{(safe_val(p_is, ld, ['NetIncome'])/safe_val(p_is, ld, ['Revenue']))*100:.1f}%" if safe_val(p_is, ld, ['Revenue']) > 0 else "-",
                            'P/E (TTM)': round(c_pe, 1) if c_pe else "-", 'P/B (Lag)': round(c_pb, 2),
                            'EV/EBITDA': f"{c_ev:.1f}" if c_ev > 0 else "-",
                            'DCF合理價區間': f"{vals[0]:.1f} ({vals[1]:.1f}-{vals[2]:.1f})", '狀態': status, 
                            'vs產業PE': "低於同業" if c_pe < med_pe else "高於同業",
                            '選股邏輯': f"Score: {int(scores['Total'])}" + (" (首選)" if scores['Total'] >= 70 else "")
                        }
                        st.metric("基準合理價", f"{vals[0]:.1f}", f"{upside:.1%} 空間")
                        st.caption(f"🛡️ 悲觀情境: {vals[1]:.1f} | 🚀 樂觀情境: {vals[2]:.1f}")
                        st.success(data['狀態'])
                        with c_out: st.dataframe(pd.DataFrame([{k: data[k] for k in cols_display if k in data}]).T, use_container_width=True)
                except Exception as e: st.error(f"查詢報錯: {e}")

with tab3:
    c1, c2 = st.columns(2)
    with c1: t_input = st.text_area("代碼:", "2603, 2002, 2330") 
    with c2: s_date = st.date_input("日期:", datetime(2022, 10, 25)); run_bt = st.button("執行", type="primary") 
    
    if run_bt:
        t_list_raw = [t.strip().upper() for t in t_input.split(',')]
        t_list = []
        for sym in t_list_raw:
            if not sym.endswith('.TW') and not sym.endswith('.TWO'):
                if not df_all.empty:
                    match = df_all[df_all['Code'].astype(str) == str(sym)]
                    t_list.append(match.iloc[0]['Ticker'] if not match.empty else f"{sym}.TW")
                else: t_list.append(f"{sym}.TW")
            else: t_list.append(sym)

        res_bt, pb = st.empty(), st.progress(0)
        res_list = []
        for i, sym in enumerate(t_list):
            ind_lookup = df_all[df_all['Ticker'] == sym]['Industry'] if not df_all.empty else pd.Series()
            real_industry = ind_lookup.iloc[0] if not ind_lookup.empty else "未知"
            is_fin = any(x in real_industry for x in ["金融", "保險"])
            
            res_list.append(run_pit_backtest_local(sym, s_date.strftime('%Y-%m-%d'), is_fin, real_industry))
            pb.progress((i+1)/len(t_list))
        
        pb.empty()
        if res_list:
            df_bt = pd.DataFrame([r for r in res_list if r])
            st.metric("平均報酬", f"{df_bt['Raw'].mean()*100:.1f}%")
            st.dataframe(df_bt.drop(columns=['Raw']), use_container_width=True)