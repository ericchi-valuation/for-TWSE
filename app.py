import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import io
import glob
import json
import re

from datetime import datetime, timedelta
import warnings

st.set_page_config(page_title="V7.5 Eric Chi 估值模型 (真實市值極速版)", page_icon="🏦", layout="wide")
warnings.simplefilter(action='ignore', category=FutureWarning)

# ✅ V7.5: 新版 yfinance (>=0.2.38) 已內建 curl_cffi 防封鎖
# 不再接受 requests.Session，由 yfinance 自動管理連線

# ==========================================
# ✅ My-TW-Coverage 質化資料讀取
# ==========================================
# ⚠️ 不加 @st.cache_data：Markdown 報告會在本地頻繁修改，
#    讀取幾 KB 文字不到 1ms，不快取才能確保每次 F5 看到最新版本
def get_qualitative_report(base_dir, ticker):
    """自動搜尋 My-TW-Coverage（含解壓縮後多層資料夾），讀取個股質化報告。"""
    clean_ticker = str(ticker).replace('.TW', '').replace('.TWO', '').replace('*', '')
    # ✅ 改用 app.py 所在的絕對路徑，不受 Streamlit 工作目錄影響
    app_dir = os.path.dirname(os.path.abspath(__file__))
    abs_base_dir = os.path.join(app_dir, base_dir)
    search_path = os.path.join(abs_base_dir, "**", f"{clean_ticker}_*.md")
    matched_files = glob.glob(search_path, recursive=True)
    if not matched_files:
        return None
    
    file_path = matched_files[0]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 擷取 "## 業務簡介" 到 "## 財務概況" 之間（排除已過時的財務表格，app.py 算得更準）
        start_marker = "## 業務簡介"
        end_marker = "## 財務概況"
        
        if start_marker in content:
            start_idx = content.find(start_marker)
            end_idx = content.find(end_marker, start_idx)
            if end_idx != -1:
                return content[start_idx:end_idx].strip()
            else:
                return content[start_idx:].strip()
        else:
            return content.strip()
    except Exception as e:
        return f"讀取質化資料時發生錯誤: {e}"
@st.cache_data(show_spinner=False)
def load_all_themes(base_dir):
    """解析 My-TW-Coverage/themes/*.md，建立 主題名稱 → [股票代碼] 映射表。"""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(app_dir, base_dir, "**", "themes", "*.md"), recursive=True)
    results = {}
    for f in files:
        name = os.path.basename(f).replace(".md", "")
        if name == "README":
            continue
        try:
            content = open(f, 'r', encoding='utf-8').read()
            codes = re.findall(r'\*\*(\d{4,5})\s', content)
            if codes:
                results[name] = codes
        except:
            pass
    return results

@st.cache_data(show_spinner=False)
def load_graph_data(base_dir):
    """載入 My-TW-Coverage/network/graph_data.json。"""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(app_dir, base_dir, "**", "graph_data.json"), recursive=True)
    if not files:
        return None
    with open(files[0], 'r', encoding='utf-8') as f:
        return json.load(f)

def build_stock_network_html(graph_data, wikilinks):
    """依公司報告中的 [[wikilink]]，過濾 graph_data.json，生成 D3.js 互動式網路圖 HTML。"""
    if not graph_data or not wikilinks:
        return None
    found_ids = {w for w in wikilinks if any(n['id'] == w for n in graph_data['nodes'])}
    if not found_ids:
        return None
    connected_ids = set(found_ids)
    for lnk in graph_data['links']:
        if lnk['source'] in found_ids or lnk['target'] in found_ids:
            connected_ids.add(lnk['source'])
            connected_ids.add(lnk['target'])
    nodes_f = [n for n in graph_data['nodes'] if n['id'] in connected_ids]
    links_f = [lnk for lnk in graph_data['links'] if lnk['source'] in connected_ids and lnk['target'] in connected_ids]
    nodes_j = json.dumps(nodes_f, ensure_ascii=False)
    links_j = json.dumps(links_f, ensure_ascii=False)
    return f"""<!DOCTYPE html><html><head>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
body{{margin:0;background:#0f1117;font-family:sans-serif;overflow:hidden;}}
.link{{stroke:rgba(255,255,255,0.18);stroke-width:1px;}}
.node text{{font-size:11px;fill:#eee;text-anchor:middle;pointer-events:none;text-shadow:0 1px 3px rgba(0,0,0,0.95);}}
.tooltip{{position:fixed;padding:8px 12px;background:rgba(20,20,30,0.92);color:#fff;border-radius:6px;font-size:12px;pointer-events:none;border:1px solid #555;}}
</style></head><body>
<div id="tip" class="tooltip" style="opacity:0;top:0;left:0;"></div>
<script>
const nodes={nodes_j};
const links={links_j};
const w=window.innerWidth,h=window.innerHeight;
const svg=d3.select('body').append('svg').attr('width',w).attr('height',h);
const sim=d3.forceSimulation(nodes)
  .force('link',d3.forceLink(links).id(d=>d.id).distance(d=>Math.max(55,110-((d.weight||1)*0.4))))
  .force('charge',d3.forceManyBody().strength(-220))
  .force('center',d3.forceCenter(w/2,h/2))
  .force('collide',d3.forceCollide(32));
const link=svg.append('g').selectAll('line').data(links).join('line').attr('class','link')
  .attr('stroke-width',d=>Math.max(1,Math.sqrt(d.weight||1)*0.35));
const node=svg.append('g').selectAll('g').data(nodes).join('g').attr('class','node')
  .call(d3.drag()
    .on('start',(e,d)=>{{if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y;}})
    .on('drag',(e,d)=>{{d.fx=e.x;d.fy=e.y;}})
    .on('end',(e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}}));
node.append('circle').attr('r',d=>Math.max(9,Math.min(26,Math.sqrt(d.count||10)*1.9)))
  .attr('fill',d=>d.color||'#888').attr('opacity',0.88);
node.append('text').attr('dy','0.32em')
  .attr('y',d=>Math.max(9,Math.min(26,Math.sqrt(d.count||10)*1.9))+13).text(d=>d.id);
const tip=document.getElementById('tip');
const catMap={{'taiwan_company':'台灣企業','international_company':'國際企業','technology':'技術','application':'應用','material':'材料'}};
node.on('mouseover',(e,d)=>{{tip.style.opacity=1;tip.innerHTML=`<b>${{d.id}}</b><br>類別: ${{catMap[d.category]||d.category}}<br>關聯度: ${{d.count||'-'}}`;tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-8)+'px';}})
  .on('mousemove',(e)=>{{tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-8)+'px';}})
  .on('mouseout',()=>{{tip.style.opacity=0;}});
sim.on('tick',()=>{{
  link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
  node.attr('transform',d=>`translate(${{d.x}},${{d.y}})`);
}});
</script></body></html>"""

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

@st.cache_data(show_spinner=False)
def load_monthly_rev_db():
    """載入月營收 Parquet，回傳以 stock_id 為 key 的 DataFrame。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tw_monthly_rev.parquet')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

DB_MR = load_monthly_rev_db()

def get_monthly_rev_growth(ticker):
    """從月營收 DB 計算最新累積年增率（YTD YoY）。
    優先取最近 2 個月的 YTD 做比較（避免單月波動太大）。
    回傳 float 或 None（無資料時）。"""
    if DB_MR.empty:
        return None
    clean = str(ticker).replace('.TW', '').replace('.TWO', '').replace('*', '')
    df = DB_MR[DB_MR['stock_id'].astype(str) == clean].copy()
    if df.empty:
        return None
    df = df.sort_values('date')
    ly = int(df.iloc[-1]['date'].year)
    lm = int(df.iloc[-1]['date'].month)
    # 年初資料不穩定保護：至少需要 3 個月才計算，避免單月雜訊太大
    months_this_yr = int((df['date'].dt.year == ly).sum())
    if months_this_yr < 3:
        return None
    # YTD current year (1月 ~ 最新月)
    rev_ytd = float(df[(df['date'].dt.year == ly) & (df['date'].dt.month <= lm)]['revenue'].sum())
    # YTD same period last year
    rev_ytd_py = float(df[(df['date'].dt.year == (ly - 1)) & (df['date'].dt.month <= lm)]['revenue'].sum())
    if rev_ytd_py > 0 and rev_ytd > 0:
        return (rev_ytd - rev_ytd_py) / rev_ytd_py
    return None

IND_PE_DEFAULT = {
    "半導體業": 25.0, "金融業": 14.0, "航運業": 10.0,
    "生技醫療業": 35.0, "鋼鐵工業": 12.0, "電子零組件業": 20.0,
    "光電業": 18.0, "電腦及週邊設備業": 18.0, "通信網路業": 20.0,
    "電機機械": 22.0
}

def get_stock_financials(ticker):
    clean_ticker = str(ticker).replace('.TW', '').replace('.TWO', '').replace('*', '')
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

def get_single_quarter_cf(p_cf_df, dates, keys):
    """將累計現金流量與折舊轉為單季值後加總(TTM)，避免跨年或例外值干擾。"""
    total = 0.0
    for i, d in enumerate(dates[:4]):
        month = d.month if hasattr(d, 'month') else pd.to_datetime(d).month
        val = safe_val(p_cf_df, d, keys)
        if month == 3:
            total += val
        else:
            prev_date = dates[i + 1] if (i + 1) < len(dates) else None
            # 確保 prev_date 是同一年，否則減去錯誤基準會導致負值
            prev_year = prev_date.year if (prev_date is not None and hasattr(prev_date, 'year')) else None
            curr_year = d.year if hasattr(d, 'year') else pd.to_datetime(d).year
            prev_val = safe_val(p_cf_df, prev_date, keys) if prev_year == curr_year else 0
            single_q = val - prev_val
            # 跨年邊界保護：若相減後為負，代表跨年、缺漏或異常，根據建議改用當季原值
            if single_q < 0:
                single_q = val
            total += single_q
    return total

def get_single_quarter_is(p_is_df, date, keys):
    """將台灣累計損益表數據精準剝離為單季值。
    台灣損益表採年度累計制：Q3 報表 = Q1+Q2+Q3 累計。
    需減去前一期（Q2 累計）才能得到真正的單季 Q3 值。
    Q1（3月）報表本身即為單季值，直接回傳。
    若相減結果為負（跨年邊界或資料缺漏），保守地回傳原值。
    """
    val = safe_val(p_is_df, date, keys)
    if val == 0:
        return 0.0
    month = date.month if hasattr(date, 'month') else pd.to_datetime(date).month
    if month == 3:  # Q1：本身即為單季，無需剝離
        return val
    # 找同年中、比 date 早的最近一期（即前一個累計期）
    date_ts = pd.Timestamp(date)
    same_year_prev = [
        d for d in p_is_df.index
        if hasattr(d, 'year') and d.year == date_ts.year and d < date_ts
    ]
    if not same_year_prev:
        return val  # 找不到前期（可能是年報），保守回傳原值
    prev_date = max(same_year_prev)  # 取最近的前期
    prev_val = safe_val(p_is_df, prev_date, keys)
    single_q = val - prev_val
    # 若相減為負（資料異常），保守地回傳原值
    return single_q if single_q >= 0 else val

def build_annual_financials_table(p_is, p_bs, shares):
    """
    從季報 Parquet 抽取近 3 個完整年度（12月 Q4）的財報摘要。
    - IS: 營業收入、營業利益、EPS、營業利益率、淨利率
    - BS: 現金、股東權益、流動/非流動負債、每股淨值
    自動依規模選擇億元 / 百萬元 / 千元單位。
    回傳 (is_df, bs_df, unit_label)；資料不足時回傳 (None, None, '')。
    """
    # --- 找近 3 個 12 月份日期（完整年度結算） ---
    annual_is_dates = sorted(
        [d for d in p_is.index if hasattr(d, 'month') and d.month == 12],
        reverse=True
    )[:3]
    annual_bs_dates = sorted(
        [d for d in p_bs.index if hasattr(d, 'month') and d.month == 12],
        reverse=True
    )[:3]

    if not annual_is_dates:
        return None, None, ''

    # --- 自動選擇顯示單位（依最新年度營業收入規模） ---
    sample_rev = safe_val(p_is, annual_is_dates[0], ['Revenue'])
    if sample_rev > 50_000_000:     # > 約 500 億（大型股，如台積電）
        divisor, unit = 100_000, "億元"
    elif sample_rev > 500_000:      # > 約 5 億（中型股）
        divisor, unit = 1_000, "百萬元"
    else:
        divisor, unit = 1, "千元"

    # --- 損益表 IS ---
    is_rows = []
    for d in annual_is_dates:
        rev    = safe_val(p_is, d, ['Revenue'])
        op_inc = safe_val(p_is, d, ['OperatingIncome'])
        net_inc= safe_val(p_is, d, ['NetIncome'])
        eps    = safe_val(p_is, d, ['EPS'])
        op_m   = f"{op_inc/rev*100:.1f}%" if rev > 0 else "-"
        net_m  = f"{net_inc/rev*100:.1f}%" if (rev > 0 and net_inc != 0) else "-"
        is_rows.append({
            '年度': str(d.year),
            f'營業收入 ({unit})': round(rev / divisor, 1),
            f'營業利益 ({unit})': round(op_inc / divisor, 1),
            'EPS (元)': round(eps, 2),
            '營業利益率': op_m,
            '淨利率': net_m,
        })
    is_df = pd.DataFrame(is_rows).set_index('年度') if is_rows else pd.DataFrame()

    # --- 資產負債表 BS ---
    bs_rows = []
    for d in annual_bs_dates:
        cash     = safe_val(p_bs, d, ['CashAndCashEquivalents'])
        equity   = safe_val(p_bs, d, ['EquityAttributableToOwnersOfParent', 'TotalEquity'])
        cur_liab = safe_val(p_bs, d, ['CurrentLiabilities'])
        non_cur  = safe_val(p_bs, d, ['NoncurrentLiabilities'])
        bps      = round(equity / shares, 2) if (shares > 0 and equity > 0) else 0
        bs_rows.append({
            '年度': str(d.year),
            f'現金 ({unit})': round(cash / divisor, 1),
            f'股東權益 ({unit})': round(equity / divisor, 1),
            f'流動負債 ({unit})': round(cur_liab / divisor, 1),
            f'非流動負債 ({unit})': round(non_cur / divisor, 1),
            '每股淨值 (元)': bps,
        })
    bs_df = pd.DataFrame(bs_rows).set_index('年度') if bs_rows else pd.DataFrame()

    return is_df, bs_df, unit

# ==========================================
# ✅ FIX A: yf.download 最新價格統一解析器
# 相容新舊版 yfinance MultiIndex 格式
# ==========================================
def parse_bulk_close(bulk_data, tickers_list):
    """
    相容 yfinance 新舊版 MultiIndex 格式，安全取得每檔最新收盤價。
    回傳 pd.Series，index = ticker symbol
    """
    if bulk_data.empty:
        return pd.Series(dtype=float)

    try:
        cols = bulk_data.columns
        if isinstance(cols, pd.MultiIndex):
            # 判斷層級：新版 yfinance (>=0.2.31) 第0層是 Price 類型, 第1層是 Ticker
            # 舊版第0層是 Price 類型 (Close/Open...), 第1層是 Ticker  — 相同
            # 但更新的版本可能有 3 層，需要先壓縮
            if cols.nlevels == 3:
                # 取 'Close' 的第二層
                close_df = bulk_data.xs('Close', axis=1, level=1).ffill()
            else:
                # 標準 2 層 MultiIndex
                if 'Close' in cols.get_level_values(0):
                    close_df = bulk_data['Close'].ffill()
                elif 'Close' in cols.get_level_values(1):
                    close_df = bulk_data.xs('Close', axis=1, level=1).ffill()
                else:
                    close_df = bulk_data.ffill()
            latest = close_df.iloc[-1] if not close_df.empty else pd.Series(dtype=float)
        else:
            # 只有一檔時，bulk_data 退化成普通 DataFrame
            latest = pd.Series({tickers_list[0]: float(bulk_data['Close'].ffill().iloc[-1])})
        return latest
    except Exception:
        return pd.Series(dtype=float)


def get_stock_name(sym, info):
    if not df_all.empty:
        match = df_all[df_all['Ticker'] == sym]
        if not match.empty:
            return match.iloc[0]['Name']
    return info.get('shortName', sym)

# ==========================================
# ✅ FIX B: 安全取得即時股價 (相容新版 yfinance)
# fast_info 是物件，不能用 .get()
# ==========================================
def get_current_price(stock_obj):
    """
    嘗試多種方式取得最新股價，防止 AttributeError。
    """
    try:
        fi = stock_obj.fast_info
        # 新版 yfinance fast_info 屬性直接存取
        for attr in ('last_price', 'lastPrice', 'regularMarketPrice'):
            val = getattr(fi, attr, None)
            if val and val > 0:
                return float(val)
    except Exception:
        pass
    try:
        hist = stock_obj.history(period="2d")
        if not hist.empty:
            return float(hist['Close'].ffill().iloc[-1])
    except Exception:
        pass
    return 0.0


# ==========================================
# 2. 估值核心引擎
# ==========================================
def get_historical_metrics_local(p_is, p_bs, p_cf, hist_price, current_shares):
    try:
        if p_is.empty or hist_price.empty: return ["-"]*4, 0, 0, 0
        hist_price = hist_price.copy()
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

def get_3_stage_valuation_local(p_is, p_bs, p_cf, shares, is_fin, real_g, beta, div_per_share=0):
    try:
        if p_is.empty: return (0, 0, 0), 0, 0.1, 0
        ld = p_is.index[0]

        eq = safe_val(p_bs, ld, ['EquityAttributableToOwnersOfParent', 'TotalEquity'], 1)
        debt = safe_val(p_bs, ld, ['CurrentLiabilities']) + safe_val(p_bs, ld, ['NoncurrentLiabilities'])
        cash = safe_val(p_bs, ld, ['CashAndCashEquivalents'])
        op_inc = safe_val(p_is, ld, ['OperatingIncome'])
        ke = max(0.035 + (beta * 0.06), 0.07)
        g_term = 0.025

        ic = eq + debt - cash
        roic = (op_inc * 0.8 / ic) if ic > 0 else 0.05

        # ==========================================
        # ✅ 金融保險業：三階段 DDM（股利折現模型）
        # ==========================================
        if is_fin:
            # 取得每股股利：優先用 yfinance 回報的年化股利，其次用 EPS × 標準配息率估算
            eps_ttm = sum(safe_val(p_is, d, ['EPS']) for d in p_is.index[:4])
            if div_per_share > 0:
                d0 = float(div_per_share)
            elif eps_ttm > 0:
                d0 = eps_ttm * 0.65  # 金融業平均配息率約 60-70%
            else:
                return (0, 0, 0), 0, ke, roic

            g1 = min(max(real_g * 0.8, 0.01), 0.12)  # 金融業成長上限 12%

            def calc_ddm(g, r):
                if r <= g_term: r = g_term + 0.01
                d, pv = d0, 0.0
                # 第一階段：高速成長 3 年
                for i in range(1, 4):
                    d *= (1 + g)
                    pv += d / ((1 + r) ** i)
                # 第二階段：過渡 2 年
                g_mid = (g + g_term) / 2
                for i in range(4, 6):
                    d *= (1 + g_mid)
                    pv += d / ((1 + r) ** i)
                # 終端價值（Gordon Growth）
                tv = d * (1 + g_term) / (r - g_term) / ((1 + r) ** 5)
                return max(pv + tv, 0)

            return (
                calc_ddm(g1, ke),
                calc_ddm(g1 * 0.5, ke * 1.05),
                calc_ddm(g1 * 1.2, ke * 0.95)
            ), g1, ke, roic

        # ==========================================
        # 非金融業：三階段 DCF
        # ==========================================
        # ✅ FCF TTM 精確計算（去累計化）


        all_dates = p_is.index.tolist()
        ocf_ttm = get_single_quarter_cf(p_cf, all_dates, ['CashFlowsFromOperatingActivities', 'NetCashInflowFromOperatingActivities'])
        icf_ttm = get_single_quarter_cf(p_cf, all_dates, ['CashProvidedByInvestingActivities'])
        fcf_ttm = ocf_ttm + icf_ttm

        # 層 2 備援：TTM 營業利益 × 70%
        op_inc_ttm = sum(safe_val(p_is, d, ['OperatingIncome']) for d in p_is.index[:4])

        if fcf_ttm > 0:
            fcf = fcf_ttm
        elif op_inc_ttm > 0:
            fcf = op_inc_ttm * 0.7
        else:
            fcf = 0

        int_exp = abs(safe_val(p_cf, ld, ['InterestExpense', 'PayTheInterest']) or safe_val(p_is, ld, ['TotalNonoperatingIncomeAndExpense']))
        kd = int_exp / debt if debt > 0 else 0.025
        kd = max(min(kd, 0.12), 0.015)
        wacc = max((eq/(eq+debt))*ke + (debt/(eq+debt))*kd*(1-0.20), 0.04)

        g1_base = min(max(real_g * 0.8, 0.02), 0.25)
        base_cf = fcf
        if base_cf <= 0: return (0, 0, 0), g1_base, wacc, roic

        def calc_dcf(g, w):
            dcf = sum([base_cf*((1+g)**i)/((1+w)**i) for i in range(1,4)])
            dcf += sum([(base_cf*((1+g)**3))*((1+(g+g_term)/2)**(i-3))/((1+w)**i) for i in range(4,6)])
            dcf += ((base_cf*((1+g)**3)*((1+(g+g_term)/2)**2))*(1+g_term)/(w-g_term)) / ((1+w)**5)
            return max((dcf - debt + cash) / (shares if shares > 0 else 1), 0)

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
    cur_pe_safe = cur_pe if cur_pe and cur_pe > 0 else 0
    # 配息率防護：若配息率 > 100%（EPS 撐不住配息）或 EPS < 0，視為財務警訊而非加分
    payout = dy * cur_pe_safe if cur_pe_safe > 0 else 999
    if payout > 1.2 or cur_pe_safe == 0:
        if dy > 0:
            s['Msg'].append("配息率異常")
    elif dy > 0.06:
        s['Q'] += 2
    elif dy > 0.03:
        s['Q'] += 1

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
        target_dt = pd.to_datetime(target_date)
        # ✅ FIX C: 統一為 tz-naive，避免 tz 比對問題
        if target_dt.tzinfo is not None:
            target_dt = target_dt.tz_localize(None)

        stock = yf.Ticker(sym)
        # ✅ FIX C: end 使用 datetime.today() 確保 tz-naive
        hist = stock.history(
            start=(target_dt - pd.Timedelta(days=3650)).strftime('%Y-%m-%d'),
            end=datetime.today().strftime('%Y-%m-%d')
        )
        if hist.empty: raise ValueError("無股價資料")
        # 統一去除 timezone
        if hist.index.tz:
            hist.index = hist.index.tz_localize(None)

        future_prices = hist[hist.index >= target_dt]
        if future_prices.empty: raise ValueError("無目標日股價")

        ep = float(future_prices['Close'].iloc[0])
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
        
        # ✅ QoQ 動能: 精確單季營收（去累計） vs 去年同季精確單季
        r_now_sq_bt   = get_single_quarter_is(p_is, valid_dates[0], ['Revenue'])
        r_prev_sq_bt  = get_single_quarter_is(p_is, valid_dates[4], ['Revenue']) if len(valid_dates) >= 5 else 0
        qoq_growth = (r_now_sq_bt - r_prev_sq_bt) / r_prev_sq_bt if r_prev_sq_bt > 0 else 0
        op_margins = [safe_val(p_is, d, ['OperatingIncome']) / safe_val(p_is, d, ['Revenue']) for d in valid_dates[:4] if safe_val(p_is, d, ['Revenue']) > 0]

        try: info = stock.info
        except: info = {}

        current_shares = float(info.get('sharesOutstanding', 1) or 1)
        hist_shares = get_historical_shares(p_bs, ld, current_shares)

        equity = safe_val(p_bs, ld, ['EquityAttributableToOwnersOfParent'], 1)
        debt = safe_val(p_bs, ld, ['CurrentLiabilities']) + safe_val(p_bs, ld, ['NoncurrentLiabilities'])
        cash = safe_val(p_bs, ld, ['CashAndCashEquivalents'])
        op_ttm = sum(safe_val(p_is, d, ['OperatingIncome']) for d in valid_dates[:4])
        dep_ttm = get_single_quarter_cf(p_cf, valid_dates, ['Depreciation'])
        ttm_ebitda = op_ttm + abs(dep_ttm)
        
        cur_pb = ep / (equity / hist_shares) if equity > 0 else 0
        cur_pe = ep / eps_ttm if eps_ttm > 0 else 0
        cur_ev = ((ep * hist_shares) + debt - cash) / (safe_val(p_is, ld, ['OperatingIncome'])*4) if safe_val(p_is, ld, ['OperatingIncome']) > 0 else 0

        # 歷史區段的股價（不超過目標日）
        hist_for_metrics = hist[hist.index <= target_dt]
        rng, avg_pe, min_pb, avg_pb = get_historical_metrics_local(p_is, p_bs, p_cf, hist_for_metrics, current_shares)
        vals, g, wacc, roic = get_3_stage_valuation_local(p_is, p_bs, p_cf, hist_shares, is_finance, real_growth, info.get('beta', 1.0), float(info.get('dividendRate', 0) or 0))
        base_intrin = vals[0]

        upside = (base_intrin - ep) / ep if base_intrin > 0 else -1
        med_pe = IND_PE_DEFAULT.get(industry_name, 22.0)
        scores = calculate_scores(info, real_growth, qoq_growth, upside, cur_pe, cur_ev, avg_pe, med_pe, cur_pb, min_pb, avg_pb, wacc, roic, debt/ttm_ebitda if ttm_ebitda > 0 else 0, op_margins, industry_name)

        dts = hist[hist.index >= target_dt].index
        def ret(days): 
            idx = dts.searchsorted(dts[0] + pd.Timedelta(days=days))
            return (float(hist['Close'].iloc[idx]) - ep) / ep if idx < len(dts) else None

        status_msg = f"{scores['Lifecycle']} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}"
        if scores['Msg']: status_msg += f" | {' '.join(scores['Msg'])}"

        return {
            '代碼': sym, '名稱': get_stock_name(sym, info), '進場日': target_dt.strftime('%Y-%m-%d'),
            '進場價': round(ep, 1), '現價': round(cp, 1), '當時總分': int(scores['Total']), '當時狀態': status_msg,
            '當時合理價(Base)': round(base_intrin, 1), '當時PE': round(cur_pe, 1),
            '3個月': f"{ret(90)*100:.1f}%" if ret(90) is not None else "-",
            '6個月': f"{ret(180)*100:.1f}%" if ret(180) is not None else "-",
            '12個月': f"{ret(365)*100:.1f}%" if ret(365) is not None else "-",
            '至今報酬': f"{(cp - ep)/ep*100:.1f}%", 'Raw': (cp - ep)/ep
        }
    except Exception as e:
        return {
            '代碼': sym, '名稱': '-', '進場日': target_date, '進場價': 0, '現價': 0,
            '當時總分': 0, '當時狀態': f"⚠️ {str(e)[:30]}",
            '當時合理價(Base)': 0, '當時PE': 0,
            '3個月': "-", '6個月': "-", '12個月': "-", '至今報酬': "-", 'Raw': 0
        }

# ==========================================
# UI 介面
# ==========================================
st.title("V7.4 Eric Chi 估值模型 (真實市值極速版)")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["全產業掃描", "單股深度查詢", "真·時光機回測", "⏳ 時點回測×全產業掃描", "🔥 主題概念掃描"])
cols_display = ['股票代碼', '名稱', '現價', '營收成長率', '預估EPS', '營業利益率', '淨利率',
                'P/E (TTM)', 'P/B (Lag)', 'P/S (Lag)', 'EV/EBITDA', 'DCF/DDM合理價區間',
                '歷史P/E區間', '歷史P/B區間', '歷史P/S區間', '狀態', 'vs產業PE', '選股邏輯']

# ==========================================
# Tab 1. 全產業掃描
# ==========================================
with tab1:
    st.info("⚡ V7.4 修正：採用相容新舊版 yfinance 的『安全批量下載解析器』，確保批次股價正確讀取！")
    if df_all.empty:
        st.error("❌ 找不到本地資料庫 (tw_stock_list.csv)。")
    else:
        selected_inds = st.multiselect(
            "選擇掃描產業 (可多選):",
            sorted([i for i in df_all['Industry'].unique()]),
            default=["半導體業"]
        )
        if st.button("執行產業掃描", type="primary") and selected_inds:
            pb = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            all_data = []
            
            for idx, ind in enumerate(selected_inds):
                status_text.text(f"正在批量獲取 [{ind}] 真實市值...")
                tickers_list = df_all[df_all["Industry"] == ind]["Ticker"].tolist()
                
                # ✅ FIX A: 使用相容新舊版的安全解析器取得批量收盤價
                try:
                    bulk_data = yf.download(tickers_list, period="5d", progress=False)
                    latest_prices = parse_bulk_close(bulk_data, tickers_list)
                except Exception as e:
                    st.warning(f"批量下載 [{ind}] 失敗：{e}")
                    latest_prices = pd.Series(dtype=float)

                # 精算每一檔的真實市值：最新收盤價 * 本地股本
                caps = []
                for t in tickers_list:
                    clean_ticker = str(t).replace('.TW', '').replace('.TWO', '').replace('*', '')
                    s_bs = DB_BS[
                        (DB_BS['stock_id'].astype(str) == clean_ticker) &
                        (DB_BS['type'].isin(['OrdinaryShare', 'CapitalStock', 'OrdinaryShare_per', 'CapitalStock_per']))
                    ]
                    shares = float(s_bs['value'].iloc[0]) / 10.0 if not s_bs.empty else 1.0
                    p = float(latest_prices.get(t, 0) or 0)
                    if pd.isna(p): p = 0.0
                    caps.append((t, p * shares))
                
                # 依真實市值排序，精準篩選前 50% 的領頭羊
                targets = [x[0] for x in sorted(caps, key=lambda x: x[1], reverse=True)[:max(len(caps)//2, 1)]]
                raw_data = []
                
                status_text.text(f"精算 [{ind}] 估值模型...")
                for sym in targets:
                    try:
                        stock = yf.Ticker(sym)
                        p = float(latest_prices.get(sym, 0) or 0)
                        if pd.isna(p) or p == 0:
                            continue
                        
                        try:
                            info = stock.info
                        except:
                            info = {}
                        
                        p_is, p_bs, p_cf = get_stock_financials(sym)
                        if p_is.empty:
                            continue
                        ld = p_is.index[0]
                        
                        # ✅ TTM EPS: 近四季加總
                        eps = sum(safe_val(p_is, d, ['EPS']) for d in p_is.index[:4])
                        # ✅ TTM YoY: 近四季累積營收 vs 去年同期四季累積
                        rev_ttm  = sum(safe_val(p_is, d, ['Revenue']) for d in p_is.index[:4])
                        rev_prev = sum(safe_val(p_is, d, ['Revenue']) for d in p_is.index[4:8]) if len(p_is) >= 8 else 0
                        real_g_q = (rev_ttm - rev_prev) / rev_prev if rev_prev > 0 else 0
                        # ✅ 月營收累計年增率覆蓋（比季報更準確，且能偵測 KY 股混用年/季資料的失真）
                        real_g_m = get_monthly_rev_growth(sym)
                        real_g   = real_g_m if real_g_m is not None else real_g_q
                        # ✅ QoQ 動能: 精確單季營收（去累計） vs 去年同季精確單季
                        r_now_sq     = get_single_quarter_is(p_is, p_is.index[0], ['Revenue'])
                        r_prev_sq    = get_single_quarter_is(p_is, p_is.index[4], ['Revenue']) if len(p_is) >= 5 else 0
                        qoq_g = (r_now_sq - r_prev_sq) / r_prev_sq if r_prev_sq > 0 else 0
                        
                        # ✅ shares: 優先 yfinance，失敗時用本地 BS 資料（避免預設 1 導致 DCF 飆高）
                        shares = float(info.get('sharesOutstanding', 0) or 0)
                        if shares <= 0:
                            shares = get_historical_shares(p_bs, ld, 0)
                        if shares <= 0:
                            continue

                        # 使用批次抓取的股价做歷史比對（避免逐筆呼叫 history API）
                        hist_10y = stock.history(period="10y")
                        if hist_10y.index.tz:
                            hist_10y.index = hist_10y.index.tz_localize(None)
                        rng, avg_pe, min_pb, avg_pb = get_historical_metrics_local(p_is, p_bs, p_cf, hist_10y, shares)

                        c_pe = p / eps if eps > 0 else 0
                        eq_val = safe_val(p_bs, ld, ['EquityAttributableToOwnersOfParent'])
                        c_pb = p / (eq_val / shares) if eq_val > 0 else 0

                        debt = safe_val(p_bs, ld, ['CurrentLiabilities']) + safe_val(p_bs, ld, ['NoncurrentLiabilities'])
                        cash = safe_val(p_bs, ld, ['CashAndCashEquivalents'])
                        # ✅ EV/EBITDA: 用 TTM + 準確去累積折舊 避免負値抗消與累積疊加
                        op_ttm = sum(safe_val(p_is, d, ['OperatingIncome']) for d in p_is.index[:4])
                        dep_ttm = get_single_quarter_cf(p_cf, p_is.index, ['Depreciation'])
                        ebitda_ttm = op_ttm + abs(dep_ttm)
                        c_ev = ((p * shares) + debt - cash) / ebitda_ttm if ebitda_ttm > 0 else 0
                        
                        is_fin = any(x in ind for x in ["金融", "保險"])
                        vals, g, wacc, roic = get_3_stage_valuation_local(p_is, p_bs, p_cf, shares, is_fin, real_g, info.get('beta', 1.0), float(info.get('dividendRate', 0) or 0))
                        
                        upside = (vals[0] - p) / p if vals[0] > 0 else -1
                        op_margins = [
                            safe_val(p_is, d, ['OperatingIncome']) / safe_val(p_is, d, ['Revenue'])
                            for d in p_is.index[:4] if safe_val(p_is, d, ['Revenue']) > 0
                        ]
                        
                        med_pe = IND_PE_DEFAULT.get(ind, 22.0)
                        scores = calculate_scores(
                            info, real_g, qoq_g, upside, c_pe, c_ev, avg_pe, med_pe,
                            c_pb, min_pb, avg_pb, wacc, roic,
                            debt / ebitda_ttm if ebitda_ttm > 0 else 0, op_margins, ind
                        )
                        
                        op_rev = safe_val(p_is, ld, ['Revenue'])
                        # ✅ 淨利率：用 EPS_TTM × 股數 / Revenue_TTM 反推（NetIncome資料較舊）
                        _ni_ttm = eps * shares
                        _rev_ttm = rev_ttm if rev_ttm > 0 else (op_rev * 4)
                        raw_data.append({
                            '股票代碼': sym,
                            '名稱': get_stock_name(sym, info),
                            '現價': float(p),
                            '營收成長率': f"{real_g*100:.1f}%",
                            '預估EPS': round(eps * (1 + min(real_g, 0.1)), 2),
                            '營業利益率': f"{(safe_val(p_is, ld, ['OperatingIncome'])/op_rev)*100:.1f}%" if op_rev > 0 else "-",
                            '淨利率': f"{(_ni_ttm / _rev_ttm * 100):.1f}%" if _rev_ttm > 0 else "-",
                            'P/E (TTM)': round(c_pe, 1) if c_pe else "-",
                            'P/B (Lag)': round(c_pb, 2),
                            'P/S (Lag)': round(p / (op_rev * 4 / shares), 2) if op_rev > 0 else "-",
                            'EV/EBITDA': f"{c_ev:.1f}" if c_ev > 0 else "-",
                            '歷史P/E區間': rng[0],
                            '歷史P/B區間': rng[1],
                            '歷史P/S區間': rng[2],
                            'DCF/DDM合理價區間': f"{vals[0]:.1f} ({vals[1]:.1f}-{vals[2]:.1f})",
                            '狀態': (
                                f"{scores['Lifecycle']} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}"
                                + (f" | {' '.join(scores['Msg'])}" if scores['Msg'] else "")
                            ),
                            'vs產業PE': "低於同業" if c_pe < med_pe else "高於同業",
                            '選股邏輯': f"Score: {int(scores['Total'])}" + (" (首選)" if scores['Total'] >= 70 else ""),
                            'Total_Score': scores['Total']
                        })
                    except Exception:
                        pass
                
                if raw_data:
                    df_display = pd.DataFrame(raw_data).sort_values('Total_Score', ascending=False)
                    all_data.extend(raw_data)
                    with results_container:
                        st.markdown(f"### 🏆 {ind}")
                        st.dataframe(df_display.head(6)[cols_display], use_container_width=True)
                else:
                    with results_container:
                        st.warning(f"⚠️ [{ind}] 未能取得任何有效估值結果，請確認財報與股價資料完整。")

                pb.progress((idx + 1) / len(selected_inds))
            
            status_text.text("✅ 完成！")
            if all_data:
                df_export = pd.DataFrame(all_data).sort_values('Total_Score', ascending=False)[cols_display]
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_export.to_excel(writer, index=False, sheet_name='掃描結果')
                st.download_button(
                    "📥 下載 Excel 名單",
                    data=buffer.getvalue(),
                    file_name=f"V7.4_Scan_{datetime.today().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )

# ==========================================
# Tab 2. 單股深度查詢
# ==========================================
def resolve_ticker(sym_str, df_all):
    sym = sym_str.strip().upper()
    if sym.endswith('.TW') or sym.endswith('.TWO'):
        return sym
    
    clean_sym = sym.replace('*', '')
    if not df_all.empty:
        match = df_all[df_all['Code'].astype(str).str.replace('*', '', regex=False) == clean_sym]
        if not match.empty:
            return match.iloc[0]['Ticker']
            
    # 如果本地清單沒抓到，智能判斷 .TW 或 .TWO
    tw_sym = f"{clean_sym}.TW"
    try:
        # 用 history 測一下上市代碼存不存在
        if not yf.Ticker(tw_sym).history(period="1d").empty:
            return tw_sym
    except:
        pass
    
    return f"{clean_sym}.TWO"

with tab2:
    c_in, c_out = st.columns([1, 2])
    with c_in:
        sym_input = st.text_input("輸入代碼:", value="2330")
        if st.button("查詢", type="primary"):
            sym = resolve_ticker(sym_input, df_all)

            with st.spinner(f"正在穿透防火牆解析 ({sym})..."):
                try:
                    ind_lookup = df_all[df_all['Ticker'] == sym]['Industry'] if not df_all.empty else pd.Series()
                    real_industry = ind_lookup.iloc[0] if not ind_lookup.empty else "未知"
                    is_fin = any(x in real_industry for x in ["金融", "保險"])

                    stock = yf.Ticker(sym)

                    # ✅ FIX B: 使用安全的多重回退方式取得股價
                    p = get_current_price(stock)
                    if p == 0:
                        st.error("⚠️ 無法獲取最新股價，請稍後再試或確認代碼正確。")
                        st.stop()

                    try:
                        info = stock.info
                    except:
                        info = {}

                    p_is, p_bs, p_cf = get_stock_financials(sym)
                    if p_is.empty:
                        st.error("❌ 本地資料庫中找不到這檔股票的財報！")
                    else:
                        ld = p_is.index[0]
                        # ✅ TTM EPS: 近四季加總
                        eps = sum(safe_val(p_is, d, ['EPS']) for d in p_is.index[:4])
                        # ✅ TTM YoY: 近四季累積營收 vs 去年同期四季累積
                        rev_ttm  = sum(safe_val(p_is, d, ['Revenue']) for d in p_is.index[:4])
                        rev_prev = sum(safe_val(p_is, d, ['Revenue']) for d in p_is.index[4:8]) if len(p_is) >= 8 else 0
                        real_g_q = (rev_ttm - rev_prev) / rev_prev if rev_prev > 0 else 0
                        # ✅ 月營收累計年增率覆蓋
                        real_g_m = get_monthly_rev_growth(sym)
                        real_g   = real_g_m if real_g_m is not None else real_g_q
                        # ✅ QoQ 動能: 精確單季營收（去累計） vs 去年同季精確單季
                        r_now_sq     = get_single_quarter_is(p_is, p_is.index[0], ['Revenue'])
                        r_prev_sq    = get_single_quarter_is(p_is, p_is.index[4], ['Revenue']) if len(p_is) >= 5 else 0
                        qoq_g = (r_now_sq - r_prev_sq) / r_prev_sq if r_prev_sq > 0 else 0
                        
                        # ✅ shares: 優先 yfinance，失敗時用本地 BS 資料（避免預設 1 導致 DCF 飆高）
                        shares = float(info.get('sharesOutstanding', 0) or 0)
                        if shares <= 0:
                            shares = get_historical_shares(p_bs, ld, 0)
                        if shares <= 0:
                            shares = 1  # 最後備消
                        hist_10y = stock.history(period="10y")
                        if hist_10y.index.tz:
                            hist_10y.index = hist_10y.index.tz_localize(None)
                        rng, avg_pe, min_pb, avg_pb = get_historical_metrics_local(p_is, p_bs, p_cf, hist_10y, shares)

                        c_pe = p / eps if eps > 0 else 0
                        eq_val = safe_val(p_bs, ld, ['EquityAttributableToOwnersOfParent'])
                        c_pb = p / (eq_val / shares) if eq_val > 0 else 0

                        debt = safe_val(p_bs, ld, ['CurrentLiabilities']) + safe_val(p_bs, ld, ['NoncurrentLiabilities'])
                        cash = safe_val(p_bs, ld, ['CashAndCashEquivalents'])
                        # ✅ EV/EBITDA: 用 TTM + 準確去累積折舊 避免負値抗消與累積疊加
                        op_ttm = sum(safe_val(p_is, d, ['OperatingIncome']) for d in p_is.index[:4])
                        dep_ttm = get_single_quarter_cf(p_cf, p_is.index, ['Depreciation'])
                        ebitda_ttm = op_ttm + abs(dep_ttm)
                        c_ev = ((p * shares) + debt - cash) / ebitda_ttm if ebitda_ttm > 0 else 0
                        
                        vals, g, wacc, roic = get_3_stage_valuation_local(p_is, p_bs, p_cf, shares, is_fin, real_g, info.get('beta', 1.0), float(info.get('dividendRate', 0) or 0))
                        upside = (vals[0] - p) / p if vals[0] > 0 else -1
                        op_margins = [
                            safe_val(p_is, d, ['OperatingIncome']) / safe_val(p_is, d, ['Revenue'])
                            for d in p_is.index[:4] if safe_val(p_is, d, ['Revenue']) > 0
                        ]
                        med_pe = IND_PE_DEFAULT.get(real_industry, 22.0)
                        
                        scores = calculate_scores(
                            info, real_g, qoq_g, upside, c_pe, c_ev, avg_pe, med_pe,
                            c_pb, min_pb, avg_pb, wacc, roic,
                            debt / ebitda_ttm if ebitda_ttm > 0 else 0, op_margins, real_industry
                        )
                        status = (
                            f"{scores['Lifecycle']} | Q:{scores['Q']} V:{scores['V']} G:{scores['G']}"
                            + (f" | {' '.join(scores['Msg'])}" if scores['Msg'] else "")
                        )

                        op_rev = safe_val(p_is, ld, ['Revenue'])
                        # ✅ 淨利率：用 EPS_TTM × 股數 / Revenue_TTM 反推（NetIncome資料較舊）
                        _ni_ttm = eps * shares
                        _rev_ttm = rev_ttm if rev_ttm > 0 else (op_rev * 4)
                        data = {
                            '股票代碼': sym, '名稱': get_stock_name(sym, info), '現價': float(p),
                            '營收成長率': f"{real_g*100:.1f}%",
                            '預估EPS': round(eps * (1 + min(real_g, 0.1)), 2),
                            '營業利益率': f"{(safe_val(p_is, ld, ['OperatingIncome'])/op_rev)*100:.1f}%" if op_rev > 0 else "-",
                            '淨利率': f"{(_ni_ttm / _rev_ttm * 100):.1f}%" if _rev_ttm > 0 else "-",
                            'P/E (TTM)': round(c_pe, 1) if c_pe else "-",
                            'P/B (Lag)': round(c_pb, 2),
                            'P/S (Lag)': round(p / (op_rev * 4 / shares), 2) if op_rev > 0 else "-",
                            'EV/EBITDA': f"{c_ev:.1f}" if c_ev > 0 else "-",
                            '歷史P/E區間': rng[0],
                            '歷史P/B區間': rng[1],
                            '歷史P/S區間': rng[2],
                            'DCF/DDM合理價區間': f"{vals[0]:.1f} ({vals[1]:.1f}-{vals[2]:.1f})",
                            '狀態': status,
                            'vs產業PE': "低於同業" if c_pe < med_pe else "高於同業",
                            '選股邏輯': f"Score: {int(scores['Total'])}" + (" (首選)" if scores['Total'] >= 70 else "")
                        }
                        st.metric("基準合理價", f"{vals[0]:.1f}", f"{upside:.1%} 空間")
                        st.caption(f"🛡️ 悲觀情境: {vals[1]:.1f} | 🚀 樂觀情境: {vals[2]:.1f}")
                        st.success(data['狀態'])
                        with c_out:
                            st.dataframe(
                                pd.DataFrame([{k: data[k] for k in cols_display if k in data}]).T,
                                use_container_width=True
                            )
                        
                        # === 近三年財報摘要 ===
                        st.divider()
                        st.subheader("📊 近三年財報摘要 (年度累計)")
                        is_annual, bs_annual, unit_lbl = build_annual_financials_table(p_is, p_bs, shares)
                        if is_annual is not None and not is_annual.empty:
                            col_is, col_bs = st.columns(2)
                            with col_is:
                                st.caption(f"📈 損益表 (單位：{unit_lbl})")
                                st.dataframe(is_annual, use_container_width=True)
                            with col_bs:
                                st.caption(f"🏦 資產負債表 (單位：{unit_lbl})")
                                if bs_annual is not None and not bs_annual.empty:
                                    st.dataframe(bs_annual, use_container_width=True)
                                else:
                                    st.info("無年度資產負債表資料")
                        else:
                            st.info("⚠️ 財報資料中無完整年度（12月）紀錄，無法建立年度摘要。")

                        # === My-TW-Coverage 質化資料 ===
                        st.divider()
                        st.subheader("📖 企業基本面與供應鏈 (來源: My-TW-Coverage)")
                        md_content = get_qualitative_report("My-TW-Coverage", sym)
                        if md_content:
                            st.markdown(md_content)
                        else:
                            st.info("尚無此標的的質化分析資料。")

                        # === 主題標籤 ===
                        all_themes_map = load_all_themes("My-TW-Coverage")
                        clean_code = str(sym).replace('.TW', '').replace('.TWO', '').replace('*', '')
                        stock_in_themes = {t: codes for t, codes in all_themes_map.items() if clean_code in codes}
                        if stock_in_themes:
                            st.divider()
                            st.subheader("🏷️ 概念主題標籤")
                            theme_cols = st.columns(min(len(stock_in_themes), 5))
                            for ti, (tname, tcodes) in enumerate(stock_in_themes.items()):
                                theme_cols[ti % 5].success(f"**{tname}**\n\n{len(tcodes)} 檔概念股")

                        # === 供應鏈網路圖 ===
                        if md_content:
                            # [[CompanyName|DisplayText]] 格式支援：只取 | 前的純名稱
                            raw_links = re.findall(r'\[\[([^\]]+)\]\]', md_content)
                            wikilinks = list({w.split('|')[0].strip() for w in raw_links})
                            graph_data_raw = load_graph_data("My-TW-Coverage")
                            net_html = build_stock_network_html(graph_data_raw, wikilinks)
                            if net_html:
                                st.divider()
                                st.subheader("🕸️ 供應鏈關係網路圖")
                                st.caption("節點大小代表關聯度，可拖曳互動。🔴 台灣企業 ｜ 🔵 國際企業 ｜ 🟢 技術 ｜ 🟣 應用")
                                st.components.v1.html(net_html, height=480, scrolling=False)
                except Exception as e:
                    st.error(f"查詢報錯: {e}")

# ==========================================
# Tab 3. 真·時光機回測
# ==========================================
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        t_input = st.text_area("代碼 (逗號分隔):", "2603, 2002, 2330")
    with c2:
        s_date = st.date_input("回溯進場日:", datetime(2022, 10, 25))
        run_bt = st.button("執行", type="primary")
    
    if run_bt:
        t_list_raw = [t.strip().upper() for t in t_input.split(',')]
        t_list = []
        for sym in t_list_raw:
            if not sym:
                continue
            t_list.append(resolve_ticker(sym, df_all))

        pb = st.progress(0)
        res_list = []
        for i, sym in enumerate(t_list):
            ind_lookup = df_all[df_all['Ticker'] == sym]['Industry'] if not df_all.empty else pd.Series()
            real_industry = ind_lookup.iloc[0] if not ind_lookup.empty else "未知"
            is_fin = any(x in real_industry for x in ["金融", "保險"])
            
            res_list.append(
                run_pit_backtest_local(sym, s_date.strftime('%Y-%m-%d'), is_fin, real_industry)
            )
            pb.progress((i + 1) / len(t_list))
        
        pb.empty()
        if res_list:
            df_bt = pd.DataFrame([r for r in res_list if r])
            avg_raw = df_bt['Raw'].mean()
            # ✅ 改進: 顯示平均報酬及詳細結果
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("平均報酬", f"{avg_raw*100:.1f}%")
            col_m2.metric("採樣股票數", f"{len(df_bt)} 檔")
            st.dataframe(df_bt.drop(columns=['Raw']), use_container_width=True)

# ==========================================
# Tab 4. 時點回測 × 全產業掃描
# ==========================================
with tab4:
    st.info(
        "⏳ **時點回測 × 全產業掃描**：選定一個過去的時間點，掃描全產業（依真實市值篩選前 50% 領頭羊），"
        "以當時財報與股價計算估值分數，並回看進場後的實際報酬表現。"
    )
    if df_all.empty:
        st.error("❌ 找不到本地資料庫 (tw_stock_list.csv)。")
    else:
        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            all_inds_sorted = sorted(df_all['Industry'].unique().tolist())
            scan_all_inds = st.checkbox("掃描全部產業（速度較慢）", value=False)
            if not scan_all_inds:
                pit_selected_inds = st.multiselect(
                    "選擇掃描產業 (可多選):",
                    all_inds_sorted,
                    default=["半導體業"],
                    key="tab4_inds"
                )
            else:
                pit_selected_inds = all_inds_sorted
                st.caption(f"共 {len(all_inds_sorted)} 個產業將被掃描")
        with col_b:
            pit_date = st.date_input(
                "回測進場時間點:",
                datetime(2022, 10, 25),
                key="tab4_date"
            )
        with col_c:
            min_score = st.number_input("最低總分門檻:", min_value=0, max_value=100, value=50, step=5, key="tab4_minscore")
            top_n = st.number_input("每產業顯示前 N 名:", min_value=1, max_value=20, value=5, step=1, key="tab4_topn")

        run_pit_scan = st.button("🚀 執行時點回測全產業掃描", type="primary", key="tab4_run")

        if run_pit_scan and pit_selected_inds:
            pit_date_str = pit_date.strftime('%Y-%m-%d')
            pit_dt = pd.to_datetime(pit_date_str)

            pb4 = st.progress(0)
            status4 = st.empty()
            results4_container = st.container()
            all_pit_data = []

            for idx4, ind4 in enumerate(pit_selected_inds):
                status4.text(f"[{idx4+1}/{len(pit_selected_inds)}] 批量預取 [{ind4}] 市值排序...")
                tickers4 = df_all[df_all["Industry"] == ind4]["Ticker"].tolist()

                # 批次取當時收盤的市值排序依據（用完全不穿越時空的市值粗篩領頭羊）
                try:
                    pit_dt_start = (pit_dt - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
                    pit_dt_end = (pit_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    bulk4 = yf.download(tickers4, start=pit_dt_start, end=pit_dt_end, progress=False)
                    latest4 = parse_bulk_close(bulk4, tickers4)
                except Exception:
                    latest4 = pd.Series(dtype=float)

                caps4 = []
                for t4 in tickers4:
                    clean4 = str(t4).replace('.TW', '').replace('.TWO', '')
                    s_bs4 = DB_BS[
                        (DB_BS['stock_id'].astype(str) == clean4) &
                        (DB_BS['type'].isin(['OrdinaryShare', 'CapitalStock', 'OrdinaryShare_per', 'CapitalStock_per']))
                    ]
                    sh4 = float(s_bs4['value'].iloc[0]) / 10.0 if not s_bs4.empty else 1.0
                    pr4 = float(latest4.get(t4, 0) or 0)
                    if pd.isna(pr4): pr4 = 0.0
                    caps4.append((t4, pr4 * sh4))

                targets4 = [x[0] for x in sorted(caps4, key=lambda x: x[1], reverse=True)[:max(len(caps4)//2, 1)]]

                status4.text(f"[{idx4+1}/{len(pit_selected_inds)}] 執行 [{ind4}] 時點回測 ({len(targets4)} 檔)...")
                ind_pit_results = []
                for sym4 in targets4:
                    is_fin4 = any(x in ind4 for x in ["金融", "保險"])
                    r4 = run_pit_backtest_local(sym4, pit_date_str, is_fin4, ind4)
                    if r4 and r4.get('當時總分', 0) >= min_score:
                        r4['產業'] = ind4
                        ind_pit_results.append(r4)

                if ind_pit_results:
                    df_ind4 = (
                        pd.DataFrame(ind_pit_results)
                        .sort_values('當時總分', ascending=False)
                        .head(int(top_n))
                    )
                    all_pit_data.extend(df_ind4.to_dict('records'))
                    with results4_container:
                        st.markdown(f"### 🏆 {ind4}")
                        show_cols4 = ['代碼', '名稱', '產業', '進場日', '進場價', '現價',
                                      '當時總分', '當時狀態', '當時合理價(Base)', '當時PE',
                                      '3個月', '6個月', '12個月', '至今報酬']
                        display_cols4 = [c for c in show_cols4 if c in df_ind4.columns]
                        st.dataframe(df_ind4[display_cols4], use_container_width=True)
                else:
                    with results4_container:
                        st.warning(f"⚠️ [{ind4}] 在此時間點無達標股票（門檻 {min_score} 分）")

                pb4.progress((idx4 + 1) / len(pit_selected_inds))

            status4.text("✅ 完成！")

            if all_pit_data:
                df_all_pit = pd.DataFrame(all_pit_data).sort_values('當時總分', ascending=False)
                raw_vals = pd.to_numeric(df_all_pit['Raw'], errors='coerce').dropna()

                st.divider()
                st.subheader("📊 全產業彙整排行")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("掃描股票數", f"{len(df_all_pit)} 檔")
                m2.metric("平均至今報酬", f"{raw_vals.mean()*100:.1f}%" if not raw_vals.empty else "-")
                m3.metric("正報酬比例", f"{(raw_vals > 0).sum()}/{len(raw_vals)}" if not raw_vals.empty else "-")
                m4.metric("最高報酬", f"{raw_vals.max()*100:.1f}%" if not raw_vals.empty else "-")

                show_cols_all = ['代碼', '名稱', '產業', '進場日', '進場價', '現價',
                                 '當時總分', '當時狀態', '當時合理價(Base)', '當時PE',
                                 '3個月', '6個月', '12個月', '至今報酬']
                display_cols_all = [c for c in show_cols_all if c in df_all_pit.columns]
                st.dataframe(df_all_pit[display_cols_all].reset_index(drop=True), use_container_width=True)

                # Excel 下載
                buf4 = io.BytesIO()
                export4 = df_all_pit[display_cols_all]
                with pd.ExcelWriter(buf4, engine='xlsxwriter') as writer4:
                    export4.to_excel(writer4, index=False, sheet_name='時點回測全產業')
                st.download_button(
                    "📥 下載 Excel（全產業時點回測）",
                    data=buf4.getvalue(),
                    file_name=f"PIT_Scan_{pit_date_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    key="tab4_dl"
                )

# ==========================================
# Tab 5. 🔥 主題概念掃描
# ==========================================
with tab5:
    st.info(
        "🔥 **主題概念掃描**：選擇特定概念主題（如 CoWoS、AI_伺服器、電動車），"
        "系統自動撈出相關概念股並執行完整 DCF/DDM 估值分析，快速找出**分數最高且最被低估**的領頭羊！"
    )

    all_themes_t5 = load_all_themes("My-TW-Coverage")

    if not all_themes_t5:
        st.warning("⚠️ 找不到主題資料庫。請確認 My-TW-Coverage 已正確上傳至 GitHub。")
    else:
        col_t5a, col_t5b = st.columns([3, 1])
        with col_t5a:
            theme_opts = sorted(all_themes_t5.keys())
            default_t5 = [t for t in ["AI_伺服器", "CoWoS"] if t in theme_opts]
            selected_themes_t5 = st.multiselect(
                "🎯 選擇概念主題 (可多選):",
                theme_opts,
                default=default_t5 or theme_opts[:1],
                key="t5_themes"
            )
        with col_t5b:
            min_score_t5 = st.number_input("最低總分門檻", 0, 100, 45, 5, key="t5_minscore")

        if selected_themes_t5:
            # 預覽股票數
            preview_codes = set()
            theme_code_map = {}
            for t in selected_themes_t5:
                for c in all_themes_t5[t]:
                    preview_codes.add(c)
                    theme_code_map.setdefault(c, []).append(t)
            st.caption(f"📊 共涵蓋 **{len(preview_codes)}** 檔概念股 ({' + '.join(f'{t}({len(all_themes_t5[t])})' for t in selected_themes_t5)})")

        run_t5 = st.button("⚡ 執行主題估值掃描", type="primary", key="t5_run")

        if run_t5 and selected_themes_t5:
            # ---- 整合所有主題的股票 ----
            all_codes_t5 = set()
            theme_map_t5 = {}
            for t in selected_themes_t5:
                for code in all_themes_t5[t]:
                    all_codes_t5.add(code)
                    theme_map_t5.setdefault(code, []).append(t)

            # 將代碼對應至 Ticker
            tickers_t5 = []
            for code in all_codes_t5:
                match = df_all[df_all['Code'].astype(str) == str(code)]
                tickers_t5.append(match.iloc[0]['Ticker'] if not match.empty else f"{code}.TW")

            pb_t5 = st.progress(0)
            status_t5 = st.empty()

            # 批次股價下載
            status_t5.text(f"📡 批量取得 {len(tickers_t5)} 檔概念股股價...")
            try:
                bulk_t5 = yf.download(tickers_t5, period="5d", progress=False)
                latest_t5 = parse_bulk_close(bulk_t5, tickers_t5)
            except:
                latest_t5 = pd.Series(dtype=float)

            raw_t5 = []
            for i5, sym5 in enumerate(tickers_t5):
                pb_t5.progress((i5 + 1) / len(tickers_t5))
                try:
                    p5 = float(latest_t5.get(sym5, 0) or 0)
                    if pd.isna(p5) or p5 == 0:
                        continue

                    stock5 = yf.Ticker(sym5)
                    try:
                        info5 = stock5.info
                    except:
                        info5 = {}

                    p_is5, p_bs5, p_cf5 = get_stock_financials(sym5)
                    if p_is5.empty:
                        continue
                    ld5 = p_is5.index[0]

                    eps5        = sum(safe_val(p_is5, d, ['EPS']) for d in p_is5.index[:4])
                    rev_ttm5    = sum(safe_val(p_is5, d, ['Revenue']) for d in p_is5.index[:4])
                    rev_prev5   = sum(safe_val(p_is5, d, ['Revenue']) for d in p_is5.index[4:8]) if len(p_is5) >= 8 else 0
                    real_g_q5   = (rev_ttm5 - rev_prev5) / rev_prev5 if rev_prev5 > 0 else 0
                    real_g_m5   = get_monthly_rev_growth(sym5)
                    real_g5     = real_g_m5 if real_g_m5 is not None else real_g_q5

                    shares5 = float(info5.get('sharesOutstanding', 0) or 0)
                    if shares5 <= 0:
                        shares5 = get_historical_shares(p_bs5, ld5, 0)
                    if shares5 <= 0:
                        continue

                    ind_m5  = df_all[df_all['Ticker'] == sym5]
                    ind5    = ind_m5.iloc[0]['Industry'] if not ind_m5.empty else "未知"
                    is_fin5 = any(x in ind5 for x in ["金融", "保險"])
                    med_pe5 = IND_PE_DEFAULT.get(ind5, 22.0)

                    hist5 = stock5.history(period="10y")
                    if hist5.index.tz:
                        hist5.index = hist5.index.tz_localize(None)
                    rng5, avg_pe5, min_pb5, avg_pb5 = get_historical_metrics_local(p_is5, p_bs5, p_cf5, hist5, shares5)

                    c_pe5   = p5 / eps5 if eps5 > 0 else 0
                    eq_val5 = safe_val(p_bs5, ld5, ['EquityAttributableToOwnersOfParent'])
                    c_pb5   = p5 / (eq_val5 / shares5) if eq_val5 > 0 else 0
                    debt5   = safe_val(p_bs5, ld5, ['CurrentLiabilities']) + safe_val(p_bs5, ld5, ['NoncurrentLiabilities'])
                    cash5   = safe_val(p_bs5, ld5, ['CashAndCashEquivalents'])
                    
                    op_ttm5 = sum(safe_val(p_is5, d, ['OperatingIncome']) for d in p_is5.index[:4])
                    dep_ttm5 = get_single_quarter_cf(p_cf5, p_is5.index, ['Depreciation'])
                    ebitda5 = op_ttm5 + abs(dep_ttm5)
                    
                    c_ev5 = ((p5 * shares5) + debt5 - cash5) / ebitda5 if ebitda5 > 0 else 0

                    vals5, g5, wacc5, roic5 = get_3_stage_valuation_local(
                        p_is5, p_bs5, p_cf5, shares5, is_fin5, real_g5,
                        info5.get('beta', 1.0), float(info5.get('dividendRate', 0) or 0)
                    )
                    upside5 = (vals5[0] - p5) / p5 if vals5[0] > 0 else -1
                    op_margins5 = [
                        safe_val(p_is5, d, ['OperatingIncome']) / safe_val(p_is5, d, ['Revenue'])
                        for d in p_is5.index[:4] if safe_val(p_is5, d, ['Revenue']) > 0
                    ]
                    de5 = debt5 / ebitda5 if ebitda5 > 0 else 0
                    
                    # ✅ QoQ 動能: 精確單季營收（去累計） vs 去年同季精確單季
                    r_now_sq5    = get_single_quarter_is(p_is5, p_is5.index[0], ['Revenue'])
                    r_prev_sq5   = get_single_quarter_is(p_is5, p_is5.index[4], ['Revenue']) if len(p_is5) >= 5 else 0
                    qoq_g5 = (r_now_sq5 - r_prev_sq5) / r_prev_sq5 if r_prev_sq5 > 0 else 0
                    
                    scores5 = calculate_scores(
                        info5, real_g5, qoq_g5, upside5, c_pe5, c_ev5, avg_pe5, med_pe5,
                        c_pb5, min_pb5, avg_pb5, wacc5, roic5, de5, op_margins5, ind5
                    )
                    if scores5['Total'] < min_score_t5:
                        continue

                    clean5  = str(sym5).replace('.TW', '').replace('.TWO', '')
                    op_rev5 = safe_val(p_is5, ld5, ['Revenue'])
                    raw_t5.append({
                        '股票代碼' : sym5,
                        '名稱'     : get_stock_name(sym5, info5),
                        '概念主題' : ' ｜ '.join(theme_map_t5.get(clean5, [])),
                        '現價'     : float(p5),
                        '營收成長率': f"{real_g5*100:.1f}%",
                        '預估EPS'  : round(eps5 * (1 + min(real_g5, 0.1)), 2),
                        '營業利益率': f"{(safe_val(p_is5, ld5, ['OperatingIncome'])/op_rev5)*100:.1f}%" if op_rev5 > 0 else "-",
                        'P/E(TTM)' : round(c_pe5, 1) if c_pe5 else "-",
                        'P/B(Lag)' : round(c_pb5, 2),
                        'EV/EBITDA': f"{c_ev5:.1f}" if c_ev5 > 0 else "-",
                        'DCF/DDM合理價(Base)': round(vals5[0], 1),
                        'DCF/DDM區間': f"{vals5[1]:.1f} ~ {vals5[2]:.1f}",
                        '低估幅度' : f"{upside5*100:.1f}%" if upside5 > -1 else "-",
                        '狀態'     : f"{scores5['Lifecycle']} | Q:{scores5['Q']} V:{scores5['V']} G:{scores5['G']}" + (f" | {' '.join(scores5['Msg'])}" if scores5['Msg'] else ""),
                        '總分'     : int(scores5['Total']),
                    })
                except:
                    pass

            pb_t5.empty()
            status_t5.empty()

            if raw_t5:
                df_t5 = pd.DataFrame(raw_t5).sort_values('總分', ascending=False)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("掃描股票數", f"{len(df_t5)} 檔")
                m2.metric("首選 (≥70分)", f"{(df_t5['總分'] >= 70).sum()} 檔")
                avg_up = pd.to_numeric(df_t5['低估幅度'].str.replace('%', ''), errors='coerce').mean()
                m3.metric("平均低估幅度", f"{avg_up:.1f}%" if not pd.isna(avg_up) else "-")
                m4.metric("最高總分", f"{df_t5['總分'].max()} 分")

                st.subheader("📊 主題估值掃描排行")
                st.dataframe(df_t5.reset_index(drop=True), use_container_width=True)

                buf_t5 = io.BytesIO()
                with pd.ExcelWriter(buf_t5, engine='xlsxwriter') as w5:
                    df_t5.to_excel(w5, index=False, sheet_name='主題掃描')
                st.download_button(
                    "📥 下載 Excel（主題掃描結果）",
                    data=buf_t5.getvalue(),
                    file_name=f"Thematic_Scan_{'_'.join(selected_themes_t5[:2])}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    key="t5_dl"
                )
            else:
                st.warning(f"⚠️ 未找到總分 ≥ {min_score_t5} 的達標概念股，請調低門檻或更換主題。")

