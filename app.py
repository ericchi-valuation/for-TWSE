import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import warnings

# 設定頁面配置 (必須是第一行)
st.set_page_config(
    page_title="V5.2 企業理財估值模型",
    page_icon="🚀",
    layout="wide"
)

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 0. 基礎建設與爬蟲邏輯 (V5.2 強化版)
# ==========================================
@st.cache_data(ttl=86400) # 快取一天，避免證交所封鎖
def fetch_twse_isin():
    headers = {'User-Agent': 'Mozilla/5.0'}
    data = []
    for mode in [2, 4]: # 2:上市, 4:上櫃
        try:
            url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
            res = requests.get(url, headers=headers, timeout=15)
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
        except Exception as e:
            pass
    return pd.DataFrame(data)

def get_tw_yahoo_cum_growth(symbol):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        clean_code = symbol.split('.')[0]
        url = f"https://tw.stock.yahoo.com/quote/{clean_code}.TW/revenue"
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        rows = soup.select('li.List\(n\)')
        for row in rows:
            label = row.select_one('div > span')
            if label and '累計營收年增率' in label.text:
                val_text = row.select('div > span')[-1].text.replace('%', '').replace(',', '').strip()
                return float(val_text) / 100.0
        return None
    except: return None

# ==========================================
# 1. 歷史區間計算
# ==========================================
def get_historical_metrics_v4_9(stock, hist_data):
    try:
        if hist_data.empty: return ["-"]*4 + [0]
        hist_data.index = hist_data.index.tz_localize(None)
        fin = stock.quarterly_financials.T
        bs = stock.quarterly_balance_sheet.T
        if fin.empty or bs.empty: return ["-"]*4 + [0]
        fin.index = pd.to_datetime(fin.index).tz_localize(None)
        bs.index = pd.to_datetime(bs.index).tz_localize(None)
        
        pe_vals, pb_vals, ps_vals, evebitda_vals = [], [], [], []
        shares = stock.info.get('sharesOutstanding', 1)
        
        for rpt_date in fin.index:
            if rpt_date not in hist_data.index:
                nearest_idx = hist_data.index.get_indexer([rpt_date], method='nearest')[0]
                if nearest_idx == -1: continue
                price = hist_data.iloc[nearest_idx]['Close']
            else: price = hist_data.loc[rpt_date]['Close']
            
            if rpt_date in bs.index:
                total_debt = bs.loc[rpt_date, 'Total Debt'] if 'Total Debt' in bs.columns else 0
                cash = bs.loc[rpt_date, 'Cash And Cash Equivalents'] if 'Cash And Cash Equivalents' in bs.columns else 0
                market_cap = price * shares
                ev = market_cap + total_debt - cash
                ebitda = 0
                if 'EBITDA' in fin.columns: ebitda = fin.loc[rpt_date, 'EBITDA']
                elif 'EBIT' in fin.columns: ebitda = fin.loc[rpt_date, 'EBIT']
                if ebitda > 0:
                    ratio = ev / (ebitda * 4) 
                    if 0 < ratio < 100: evebitda_vals.append(ratio)
            
            if 'Basic EPS' in fin.columns:
                eps = fin.loc[rpt_date, 'Basic EPS']
                if eps > 0: pe_vals.append(price / (eps * 4))
            if 'Total Revenue' in fin.columns:
                rev = fin.loc[rpt_date, 'Total Revenue']
                if rev > 0: ps_vals.append(price / ((rev/shares) * 4))
            if rpt_date in bs.index and 'Stockholders Equity' in bs.columns:
                bv = bs.loc[rpt_date, 'Stockholders Equity']
                if bv > 0: pb_vals.append(price / (bv/shares))
                
        def fmt_rng(vals):
            if not vals: return "-"
            clean = [v for v in vals if 0 < v < 150]
            if not clean: return "-"
            return f"{min(clean):.1f}-{max(clean):.1f}"
            
        return fmt_rng(pe_vals), fmt_rng(pb_vals), fmt_rng(ps_vals), fmt_rng(evebitda_vals), (np.mean(pe_vals) if pe_vals else 0)
    except: return "-", "-", "-", "-", 0

# ==========================================
# 2. 估值核心 (3-Stage DCF)
# ==========================================
def get_3_stage_valuation(stock, is_finance, real_growth):
    RISK_FREE = 0.035; ERP = 0.06
    try:
        info = stock.info; shares = info.get('sharesOutstanding', 1)
        bs = stock.balance_sheet.fillna(0); fin = stock.financials.fillna(0)
        if bs.empty or fin.empty: return 0, 0, 0.1, 0
        
        beta = info.get('beta', 1.0) or 1.0
        ke = max(RISK_FREE + beta * ERP, 0.07)
        equity = bs.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in bs.index else 1
        total_debt = bs.loc['Total Debt'].iloc[0] if 'Total Debt' in bs.index else 0
        cash = bs.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs.index else 0
        ebit = fin.loc['EBIT'].iloc[0] if 'EBIT' in fin.index else 0
        
        tax = 0.20
        invested_capital = equity + total_debt - cash
        roic = (ebit * (1-tax) / invested_capital) if invested_capital > 0 else 0.05

        wacc = (equity/(equity+total_debt))*ke + (total_debt/(equity+total_debt))*0.025
        if is_finance: wacc = max(ke, 0.08)
        
        g1 = min(max(real_growth * 0.8, 0.02), 0.25)
        g_term = 0.025; g2 = (g1 + g_term) / 2
        
        if is_finance: base_cf = (info.get('netIncomeToCommon', 0) * 0.6)
        else:
            nopat = ebit * (1-tax)
            if nopat <= 0: return 0, g1, wacc, roic
            base_cf = nopat * 0.7 
            
        dcf_sum = 0; cf = base_cf
        for i in range(1, 4): cf *= (1 + g1); dcf_sum += cf / ((1 + wacc)**i)
        for i in range(4, 6): cf *= (1 + g2); dcf_sum += cf / ((1 + wacc)**i)
        tv = (cf * (1 + g_term)) / (wacc - g_term)
        dcf_sum += tv / ((1 + wacc)**5)
        equity_val = dcf_sum - (total_debt if not is_finance else 0) + (cash if not is_finance else 0)
        
        return max(equity_val / shares, 0), g1, wacc, roic
    except: return 0, 0, 0.1, 0

# ==========================================
# 3. 企業理財評分系統
# ==========================================
def calculate_corp_finance_scores(info, financials, growth_rate, qoq_growth, valuation_upside, 
                                  current_pe, current_ev_ebitda, hist_avg_pe, industry_pe_median, 
                                  wacc, roic):
    scores = {'Q': 0, 'V': 0, 'G': 0, 'Total': 0, 'Msg': []}
    
    if growth_rate > 0.15: w_q, w_v, w_g = 0.2, 0.3, 0.5; scores['Lifecycle'] = "Growth"
    elif growth_rate < 0.05: w_q, w_v, w_g = 0.5, 0.4, 0.1; scores['Lifecycle'] = "Mature"
    else: w_q, w_v, w_g = 0.3, 0.4, 0.3; scores['Lifecycle'] = "Stable"

    # Quality
    try:
        ebit = financials.loc['EBIT'].iloc[0]
        interest = abs(financials.loc['Interest Expense'].iloc[0])
        icr = ebit / interest if interest > 0 else 100
    except: icr = 10
    if icr > 5: scores['Q'] += 4
    elif icr < 1.5: scores['Q'] -= 5; scores['Msg'].append("高財務風險(ICR<1.5)")
    else: scores['Q'] += 1
    
    if roic > wacc: scores['Q'] += 4
    else: scores['Q'] -= 2; scores['Msg'].append("ROIC<WACC")

    try:
        op_now = financials.loc['Operating Income'].iloc[0] / financials.loc['Total Revenue'].iloc[0]
        op_prev = financials.loc['Operating Income'].iloc[1] / financials.loc['Total Revenue'].iloc[1]
        if op_now < op_prev * 0.9: scores['Q'] -= 2; scores['Msg'].append("利潤率下滑")
        else: scores['Q'] += 2
    except: pass

    # Value
    if valuation_upside > 0.3: scores['V'] += 4
    elif valuation_upside > 0.1: scores['V'] += 2
    if hist_avg_pe > 0 and current_pe < hist_avg_pe: scores['V'] += 3
    if industry_pe_median > 0 and current_pe < industry_pe_median: scores['V'] += 3
    if current_ev_ebitda > 0 and current_ev_ebitda < 10: scores['V'] += 3

    # Growth
    if growth_rate > 0.10 and roic < wacc: scores['G'] -= 5; scores['Msg'].append("無效高成長")
    else:
        if growth_rate > 0.20: scores['G'] += 5
        elif growth_rate > 0.10: scores['G'] += 3
    if qoq_growth > 0.05: scores['G'] += 3
    elif qoq_growth < -0.05: scores['G'] -= 3; scores['Msg'].append("動能轉弱")
    
    peg = info.get('pegRatio', 0)
    if peg and 0 < peg < 1.2: scores['G'] += 2

    scores['Total'] = (scores['Q'] * w_q * 10) + (scores['V'] * w_v * 10) + (scores['G'] * w_g * 10)
    return scores

# ==========================================
# 4. 資料整理輔助函數
# ==========================================
def compile_stock_data(s, industry_name, stock, info, price, real_growth, qoq_growth, wacc, roic, 
                       pe_rng, pb_rng, ps_rng, ev_rng, avg_pe, cur_pe, cur_ev_ebitda, 
                       intrinsic, upside, eps, g_used, median_pe, is_finance):
    
    fin_annual = stock.financials.fillna(0)
    scores = calculate_corp_finance_scores(info, fin_annual, real_growth, qoq_growth, upside, 
                                           cur_pe, cur_ev_ebitda, avg_pe, median_pe, wacc, roic)
    
    warnings_str = " ".join(scores['Msg'])
    status = f"{scores['Lifecycle']} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}"
    if warnings_str: status += f" | ⚠️{warnings_str}"
    
    logic = f"Score: {int(scores['Total'])}"
    if scores['Total'] >= 70: logic += " (首選)"
    
    est_eps = eps * (1 + min(real_growth, 0.1))
    ratios = {'op': info.get('operatingMargins', 0), 'net': info.get('profitMargins', 0)}
    ev_str = f"{cur_ev_ebitda:.1f}" if cur_ev_ebitda > 0 else "-"
    
    return {
        '產業別': industry_name, # 用於分組
        '股票代碼': s,
        '名稱': info.get('shortName', s),
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
# UI 介面與分頁 (Streamlit App)
# ==========================================
st.title("🚀 V5.2 企業理財估值模型")
st.caption("Damodaran 體質檢查 + IB 分析師動能模型 | 支援全產業多維度掃描")

tab1, tab2, tab3 = st.tabs(["🔍 產業龍頭掃描", "🏢 單股深度查詢", "⏳ 時光機回測"])

# ------------------------------------------
# Tab 1: 產業掃描 (V5.2: 綜合排名 + 產業排名)
# ------------------------------------------
with tab1:
    st.header("產業龍頭批量掃描")
    
    with st.spinner("載入證交所最新產業清單中..."):
        df_all = fetch_twse_isin()
        
    if df_all.empty:
        st.error("無法連線至證交所，請稍後再試。")
    else:
        valid_industries = sorted([i for i in df_all['Industry'].unique() if i and "ETF" not in i])
        
        # 允許多選產業
        selected_industries = st.multiselect(
            "選擇要掃描的產業 (可多選，選越多掃描時間越長):", 
            options=valid_industries,
            default=["半導體業"]
        )
        
        if st.button("開始批量掃描", type="primary"):
            if not selected_industries:
                st.warning("請至少選擇一個產業！")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 收集所有目標股票並篩選市值前50%
                target_tuples = [] # (ticker, industry)
                total_industries = len(selected_industries)
                
                for idx, ind in enumerate(selected_industries):
                    status_text.text(f"正在篩選 [{ind}] 市值前 50% 的公司...")
                    tickers = df_all[df_all["Industry"] == ind]["Ticker"].tolist()
                    
                    if len(tickers) > 4:
                        caps = []
                        for t in tickers:
                            try:
                                m = yf.Ticker(t).fast_info['market_cap']
                                if m > 0: caps.append((t, m))
                            except: pass
                        caps.sort(key=lambda x: x[1], reverse=True)
                        ind_targets = [x[0] for x in caps[:max(len(caps)//2, 1)]]
                    else:
                        ind_targets = tickers
                    
                    for t in ind_targets:
                        target_tuples.append((t, ind))
                        
                    progress_bar.progress((idx + 1) / total_industries * 0.2) # 前20%進度分配給篩選

                results = []
                industry_pes_dict = {ind: [] for ind in selected_industries}
                raw_data_list = []
                
                status_text.text(f"正在精算 {len(target_tuples)} 檔股票的財務模型...")
                for i, (s, ind) in enumerate(target_tuples):
                    try:
                        stock = yf.Ticker(s)
                        info = stock.info
                        price = info.get('currentPrice') or info.get('previousClose')
                        if not price: continue
                        
                        real_growth = get_tw_yahoo_cum_growth(s)
                        if real_growth is None: real_growth = info.get('revenueGrowth', 0.0)
                        
                        qoq_growth = 0
                        q_fin = stock.quarterly_financials
                        if not q_fin.empty and 'Total Revenue' in q_fin.index and len(q_fin.columns) >= 2:
                            rev_now = q_fin.loc['Total Revenue'].iloc[0]
                            rev_prev = q_fin.loc['Total Revenue'].iloc[1]
                            if rev_prev > 0: qoq_growth = (rev_now - rev_prev) / rev_prev
                        
                        hist = stock.history(period="10y")
                        pe_rng, pb_rng, ps_rng, ev_rng, avg_pe = get_historical_metrics_v4_9(stock, hist)
                        
                        eps = info.get('trailingEps', 0)
                        cur_pe = price / eps if eps > 0 else 0
                        if cur_pe > 0 and cur_pe < 120: industry_pes_dict[ind].append(cur_pe)
                        
                        cur_ev_ebitda = info.get('enterpriseToEbitda', 0)
                        if not cur_ev_ebitda:
                            mcap = price * info.get('sharesOutstanding', 1)
                            debt = info.get('totalDebt', 0); cash = info.get('totalCash', 0)
                            ebitda = info.get('ebitda', 0)
                            if ebitda > 0: cur_ev_ebitda = (mcap + debt - cash) / ebitda
                        
                        is_finance = any(x in ind for x in ["金融", "保險", "證券"])
                        intrinsic, g_used, wacc, roic = get_3_stage_valuation(stock, is_finance, real_growth)
                        upside = (intrinsic - price) / price if intrinsic > 0 else -1
                        
                        raw_data_list.append({
                            'symbol': s, 'industry': ind, 'stock': stock, 'info': info, 'price': price,
                            'real_growth': real_growth, 'qoq_growth': qoq_growth,
                            'wacc': wacc, 'roic': roic, 'pe_rng': pe_rng, 'pb_rng': pb_rng,
                            'ps_rng': ps_rng, 'ev_rng': ev_rng, 'avg_pe': avg_pe,
                            'cur_pe': cur_pe, 'cur_ev_ebitda': cur_ev_ebitda,
                            'intrinsic': intrinsic, 'upside': upside, 'eps': eps, 'g_used': g_used,
                            'is_finance': is_finance
                        })
                    except: pass
                    progress_bar.progress(0.2 + ((i + 1) / len(target_tuples) * 0.8)) # 後80%進度分配給精算

                # 計算每個產業的中位數PE
                pe_medians = {ind: (np.median(pes) if pes else 22.0) for ind, pes in industry_pes_dict.items()}
                
                for d in raw_data_list:
                    stock_dict = compile_stock_data(
                        d['symbol'], d['industry'], d['stock'], d['info'], d['price'], d['real_growth'], d['qoq_growth'], 
                        d['wacc'], d['roic'], d['pe_rng'], d['pb_rng'], d['ps_rng'], d['ev_rng'], 
                        d['avg_pe'], d['cur_pe'], d['cur_ev_ebitda'], d['intrinsic'], d['upside'], 
                        d['eps'], d['g_used'], pe_medians[d['industry']], d['is_finance']
                    )
                    results.append(stock_dict)
                
                progress_bar.progress(1.0)
                status_text.text("分析完成！")
                
                if results:
                    df_res = pd.DataFrame(results)
                    cols = ['股票代碼', '名稱', '現價', '營收成長率', '營業利益率', '淨利率', 
                            '預估EPS', 'P/E (TTM)', 'P/B (Lag)', 'P/S (Lag)', 'EV/EBITDA',
                            '預估範圍P/E', '預估範圍P/B', '預估範圍P/S', '預估範圍EV/EBITDA',
                            'DCF/DDM合理價', '狀態', 'vs產業PE', '選股邏輯']
                    
                    # 1. 跨產業綜合高分榜
                    st.subheader("🏆 跨產業綜合高分榜 (Total Score Top 10)")
                    df_top_all = df_res.sort_values(by='Total_Score', ascending=False).head(10)
                    st.dataframe(df_top_all[['產業別'] + cols], use_container_width=True)
                    
                    st.divider()
                    
                    # 2. 各產業內部排名
                    st.subheader("🏢 各產業內部評分排名")
                    for ind in selected_industries:
                        df_ind = df_res[df_res['產業別'] == ind].sort_values(by='Total_Score', ascending=False)
                        if not df_ind.empty:
                            st.markdown(f"#### 🔹 {ind} (共 {len(df_ind)} 檔符合市值門檻)")
                            st.dataframe(df_ind[cols], use_container_width=True)
                else:
                    st.warning("查無合適標的或資料不足。")

# ------------------------------------------
# Tab 2: 單股深度查詢 (維持 V5.1 邏輯)
# ------------------------------------------
with tab2:
    st.header("公司體質 360 度分析")
    col_input, col_info = st.columns([1, 2])
    
    with col_input:
        stock_code = st.text_input("輸入股票代碼 (例如: 2330):", value="2330")
        if st.button("查看單股分析報告", type="primary"):
            symbol = f"{stock_code}.TW"
            with st.spinner(f"正在診斷 {symbol}..."):
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    if not info.get('currentPrice') and not info.get('previousClose'):
                        st.error("找不到該股票資料，請確認代碼是否正確。")
                    else:
                        price = info.get('currentPrice') or info.get('previousClose')
                        real_growth = get_tw_yahoo_cum_growth(symbol)
                        if real_growth is None: real_growth = info.get('revenueGrowth', 0.0)
                        
                        qoq_growth = 0
                        q_fin = stock.quarterly_financials
                        if not q_fin.empty and 'Total Revenue' in q_fin.index and len(q_fin.columns) >= 2:
                            rev_now = q_fin.loc['Total Revenue'].iloc[0]
                            rev_prev = q_fin.loc['Total Revenue'].iloc[1]
                            if rev_prev > 0: qoq_growth = (rev_now - rev_prev) / rev_prev
                            
                        hist = stock.history(period="10y")
                        pe_rng, pb_rng, ps_rng, ev_rng, avg_pe = get_historical_metrics_v4_9(stock, hist)
                        
                        eps = info.get('trailingEps', 0)
                        cur_pe = price / eps if eps > 0 else 0
                        
                        cur_ev_ebitda = info.get('enterpriseToEbitda', 0)
                        if not cur_ev_ebitda:
                            mcap = price * info.get('sharesOutstanding', 1)
                            debt = info.get('totalDebt', 0); cash = info.get('totalCash', 0)
                            ebitda = info.get('ebitda', 0)
                            if ebitda > 0: cur_ev_ebitda = (mcap + debt - cash) / ebitda
                            
                        is_finance = "Financial" in info.get('sector', '')
                        intrinsic, g_used, wacc, roic = get_3_stage_valuation(stock, is_finance, real_growth)
                        upside = (intrinsic - price) / price if intrinsic > 0 else -1
                        
                        stock_data = compile_stock_data(
                            symbol, info.get('industry', 'N/A'), stock, info, price, real_growth, qoq_growth, wacc, roic, 
                            pe_rng, pb_rng, ps_rng, ev_rng, avg_pe, cur_pe, cur_ev_ebitda, 
                            intrinsic, upside, eps, g_used, 22.0, is_finance
                        )
                        
                        st.metric("當前股價", f"{price} TWD", f"{upside:.1%} (與合理價差距)")
                        st.metric("V5.2 保守合理價", f"{intrinsic:.1f} TWD")
                        
                        st.subheader("因子評分卡")
                        st.progress(stock_data['Total_Score']/100, text=f"綜合總分: {int(stock_data['Total_Score'])} 分")
                        st.success(f"📌 {stock_data['狀態']}")
                        
                        with col_info:
                            st.subheader("完整估值與體質表")
                            df_single = pd.DataFrame([stock_data])
                            df_single = df_single.drop(columns=['Total_Score', '產業別'])
                            st.dataframe(df_single.T, use_container_width=True)
                            
                except Exception as e:
                    st.error(f"分析發生錯誤: {e}")

# ------------------------------------------
# Tab 3: 時光機回測 (V5.2: 新增名稱欄位)
# ------------------------------------------
with tab3:
    st.header("時光機回測")
    st.markdown("輸入股票代碼與進場日期，模擬若當時買進持有的真實報酬率。")
    
    c1, c2 = st.columns(2)
    with c1:
        default_tickers = "1519.TW, 3017.TW, 2330.TW, 2382.TW, 2454.TW, 2881.TW"
        tickers_input = st.text_area("輸入股票代碼 (逗號分隔):", value=default_tickers, height=100)
    with c2:
        start_date = st.date_input("進場日期:", value=datetime(2023, 11, 27))
        run_backtest = st.button("執行回測", type="primary")
        
    if run_backtest:
        ticker_list = [t.strip() for t in tickers_input.split(',')]
        results_bt = []
        pb = st.progress(0)
        
        for i, symbol in enumerate(ticker_list):
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(start=start_date)
                
                # 新增：抓取名稱
                name = stock.info.get('shortName', symbol)
                
                if hist.empty:
                    st.warning(f"{name} ({symbol}): 無歷史數據")
                    continue
                    
                entry_price = hist['Close'].iloc[0]
                curr_price = hist['Close'].iloc[-1]
                dates = hist.index
                
                def get_ret(days):
                    target_date = dates[0] + timedelta(days=days)
                    idx = dates.searchsorted(target_date)
                    if idx < len(dates):
                        p = hist['Close'].iloc[idx]
                        return (p - entry_price) / entry_price
                    return None

                ret_3m = get_ret(90)
                ret_6m = get_ret(180)
                ret_12m = get_ret(365)
                total_ret = (curr_price - entry_price) / entry_price
                
                results_bt.append({
                    '代碼': symbol,
                    '名稱': name, # <--- 新增的名稱欄位
                    '進場價': round(entry_price, 1),
                    '現價': round(curr_price, 1),
                    '3個月': f"{ret_3m*100:.1f}%" if ret_3m else "-",
                    '6個月': f"{ret_6m*100:.1f}%" if ret_6m else "-",
                    '12個月': f"{ret_12m*100:.1f}%" if ret_12m else "-",
                    '至今報酬': f"{total_ret*100:.1f}%",
                    'Raw_Ret': total_ret
                })
            except: pass
            pb.progress((i + 1) / len(ticker_list))
            
        if results_bt:
            df_bt = pd.DataFrame(results_bt)
            avg_ret = df_bt['Raw_Ret'].mean()
            
            st.metric("投資組合平均至今報酬率", f"{avg_ret*100:.1f}%")
            st.dataframe(df_bt.drop(columns=['Raw_Ret']), use_container_width=True)
            st.bar_chart(df_bt.set_index('名稱')['Raw_Ret']) # 圖表改用名稱顯示