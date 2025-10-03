# app.py — SwingFinder (Webull-style with Screener + Watchlists + Analysis)
# Educational tool only. Not financial advice.

import warnings, os, io, json, math
warnings.filterwarnings("ignore")

import datetime as dt
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="SwingFinder", layout="wide")
st.title("📈 SwingFinder — Screener + Watchlists + Analysis")
st.caption("Run scanners, build watchlists, and analyze individual stocks. Educational only.")

# ---------------- Session state ----------------
if "ran" not in st.session_state: st.session_state.ran = False
if "saved_watchlists" not in st.session_state: st.session_state.saved_watchlists = {}
if "active_watchlist" not in st.session_state: st.session_state.active_watchlist = []

def app_dir() -> str:
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.getcwd()

WATCHLISTS_PATH = os.path.join(app_dir(), "watchlists.json")

# ---------------- Persistence ----------------
def load_watchlists_from_disk():
    try:
        if os.path.exists(WATCHLISTS_PATH):
            with open(WATCHLISTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    st.session_state.saved_watchlists.update({
                        k: [s.strip().upper() for s in v if str(s).strip()]
                        for k,v in data.items() if isinstance(v, list)
                    })
    except: pass

def save_watchlists_to_disk():
    try:
        with open(WATCHLISTS_PATH, "w", encoding="utf-8") as f:
            json.dump(st.session_state.saved_watchlists, f, indent=2)
    except: pass

def export_watchlists_json():
    return io.BytesIO(json.dumps(st.session_state.saved_watchlists, indent=2).encode("utf-8"))

def delete_watchlist(name: str):
    if name in st.session_state.saved_watchlists:
        st.session_state.saved_watchlists.pop(name, None)
        save_watchlists_to_disk()

def save_watchlist(name: str, symbols: list[str]):
    syms = [s.strip().upper() for s in symbols if str(s).strip()]
    if not syms or not name: return False, "Need a name and at least one ticker"
    st.session_state.saved_watchlists[name] = syms
    save_watchlists_to_disk()
    return True, f"✅ Saved {len(syms)} symbols to watchlist '{name}'"

# auto-load once
if "watchlists_loaded_once" not in st.session_state:
    load_watchlists_from_disk()
    st.session_state.watchlists_loaded_once = True

# ---------------- Sidebar Controls ----------------
with st.sidebar:
    st.header("⚙️ Screener Settings")
    uni_choice = st.selectbox("Universe", ["All (Combined)","S&P 500","NASDAQ-100","DOW 30"], index=0)
    years_back   = st.slider("History (years)", 1, 5, 2)
    price_min    = st.number_input("Min price ($)", 0.0, value=5.0, step=0.5)
    price_max    = st.number_input("Max price ($)", 0.0, value=60.0, step=1.0)
    liq_min      = st.number_input("Min Avg $ Vol (20d, M)", 0.0, value=0.5, step=0.5)
    strictness   = st.selectbox("Strictness", ["Loose","Balanced","Strict"], index=1)
    include_near = st.checkbox("Also show near-miss list", value=True)
    st.markdown("---")
    st.caption("Use All (Combined) for a broad scan of S&P + NASDAQ-100 + DOW 30.")

# ---------------- Universes ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_tickers(): return ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","TSLA","GOOG","LLY"]
@st.cache_data(ttl=3600, show_spinner=False)
def get_nasdaq100_tickers(): return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","NFLX","ADBE","AMD","TSLA"]
@st.cache_data(ttl=3600, show_spinner=False)
def get_dow30_tickers(): return ["AAPL","AMGN","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS"]
def combined_universe():
    return sorted(set(get_sp500_tickers()) | set(get_nasdaq100_tickers()) | set(get_dow30_tickers()))

if uni_choice == "All (Combined)": universe_ticks = combined_universe()
elif uni_choice == "S&P 500": universe_ticks = get_sp500_tickers()
elif uni_choice == "NASDAQ-100": universe_ticks = get_nasdaq100_tickers()
else: universe_ticks = get_dow30_tickers()

# ---------------- Indicators ----------------
def ema(s: pd.Series, span: int): return pd.Series(s, index=s.index).ewm(span=span, adjust=False).mean()
def sma(s: pd.Series, window: int): return pd.Series(s, index=s.index).rolling(window).mean()
def rsi(series, period: int = 14):
    series = pd.to_numeric(pd.Series(series).squeeze(), errors="coerce")
    d = series.diff()
    g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    g = pd.Series(g, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    l = pd.Series(l, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    return (100 - (100/(1+rs))).bfill()
def atr(df: pd.DataFrame, period: int = 14):
    pc = df["Close"].shift(1)
    tr = pd.concat([(df["High"]-df["Low"]).abs(), (df["High"]-pc).abs(), (df["Low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = sma(df["Close"], 20); df["SMA50"] = sma(df["Close"], 50)
    df["EMA20"] = ema(df["Close"], 20); df["EMA50"] = ema(df["Close"], 50)
    df["RSI14"] = rsi(df["Close"], 14); df["ATR14"] = atr(df)
    df["HH20"] = df["High"].rolling(20).max(); df["LL20"] = df["Low"].rolling(20).min()
    return df

# ---------------- Screener logic ----------------
def good_position_flags(last, prev, mode: str = "Balanced"):
    trend_up = bool(last["EMA20"] > last["EMA50"])
    rsi_v = float(last["RSI14"]); close = float(last["Close"])
    hh20 = float(last.get("HH20", np.nan))

    if mode == "Strict":
        pull_rsi_lo, pull_rsi_hi = 45, 60; brk_near = 0.995; brk_rsi_min = 50
    elif mode == "Loose":
        pull_rsi_lo, pull_rsi_hi = 35, 70; brk_near = 0.980; brk_rsi_min = 45
    else:  # Balanced
        pull_rsi_lo, pull_rsi_hi = 40, 65; brk_near = 0.990; brk_rsi_min = 48

    near_high = np.isfinite(hh20) and (close >= brk_near * hh20) and trend_up and (rsi_v >= brk_rsi_min)
    in_pull   = trend_up and (pull_rsi_lo <= rsi_v <= pull_rsi_hi)

    ok = False; score = 0; reasons = []
    if in_pull: ok = True; score += 2; reasons.append("pullback-window")
    if near_high: ok = True; score += 2; reasons.append("near-20d-high")
    if trend_up and rsi_v > 50: score += 1
    return ok, score, ",".join(reasons) if reasons else "-"

@st.cache_data(ttl=900, show_spinner=True)
def scan_list(ticks, years, price_min, price_max, min_dollar_vol_millions, strictness_label, include_near=True, max_symbols=300):
    rows, near_rows = [], []
    for t in ticks[:max_symbols]:
        try:
            df = yf.download(t, period=f"{years}y", auto_adjust=True, progress=False, threads=False)
            if df.empty or len(df) < 30: continue
            df = build_indicators(df)
            last, prev = df.iloc[-1], df.iloc[-2]
            close = float(last["Close"])
            if not (price_min <= close <= price_max): continue
            adv = (df["Close"]*df["Volume"]).rolling(20).mean().iloc[-1] / 1_000_000.0
            if adv < min_dollar_vol_millions: continue

            ok, score, why = good_position_flags(last, prev, strictness_label)
            base = {"Ticker":t,"Close":round(close,2),"Avg$Vol(20d, M)":round(float(adv),2),
                    "RSI14":round(float(last["RSI14"]),1),"TrendUp":bool(last["EMA20"]>last["EMA50"])}
            if ok: rows.append({**base,"Reason":why,"Score":score})
            elif include_near: near_rows.append({**base,"Reason":"near-miss","Score":1})
        except: continue

    df_ok, df_near = pd.DataFrame(rows), pd.DataFrame(near_rows)
    if not df_ok.empty: df_ok.insert(0,"Rank",np.arange(1,len(df_ok)+1))
    if not df_near.empty: df_near.insert(0,"Rank",np.arange(1,len(df_near)+1))
    return df_ok, df_near

# ---------------- Screener Run ----------------
st.header("📊 Screener")
if st.button("🔎 Run Screener", key="run_screener"):
    res_ok, res_near = scan_list(universe_ticks, years_back, price_min, price_max, liq_min, strictness, include_near)

    if res_ok is not None and not res_ok.empty:
        st.subheader("✅ Matches"); st.dataframe(res_ok, use_container_width=True)
    else:
        st.info("No strict matches — loosen filters or check Near-Misses.")

    if include_near and (res_near is not None) and (not res_near.empty):
        st.subheader("⚠️ Near-Miss Candidates"); st.dataframe(res_near, use_container_width=True)

    tickers_from_run = []
    if res_ok is not None and not res_ok.empty: tickers_from_run += list(res_ok["Ticker"])
    if include_near and res_near is not None and not res_near.empty: tickers_from_run += list(res_near["Ticker"])

    if tickers_from_run:
        picks = st.multiselect("Select tickers to add to Active Watchlist", tickers_from_run, key="add_from_scan")
        if picks and st.button(f"➕ Add {len(picks)} selected", key="btn_add_selected"):
            st.session_state.active_watchlist = sorted(set(st.session_state.active_watchlist) | set(picks))
            st.success(f"Added: {', '.join(picks)}")
# ---------------- Active Watchlist Manager ----------------
st.header("📋 Active Watchlist")

wl_input = st.text_area(
    "Tickers (comma separated)",
    value=", ".join(st.session_state.active_watchlist),
    height=80
)
st.session_state.active_watchlist = [t.strip().upper() for t in wl_input.split(",") if t.strip()]

col1, col2, col3 = st.columns(3)

# Save watchlist
with col1:
    new_name = st.text_input("Save active list as:", "")
    if st.button("💾 Save Watchlist") and new_name.strip():
        ok, msg = save_watchlist(new_name.strip(), st.session_state.active_watchlist)
        (st.success if ok else st.error)(msg)
        if ok: st.rerun()

# Load watchlist
with col2:
    if st.session_state.saved_watchlists:
        sel = st.selectbox("Load saved watchlist", list(st.session_state.saved_watchlists.keys()), key="load_sel")
        if st.button("📂 Load"):
            st.session_state.active_watchlist = st.session_state.saved_watchlists[sel].copy()
            st.success(f"Loaded {sel}")
            st.rerun()

# Delete watchlist
with col3:
    if st.session_state.saved_watchlists:
        sel_del = st.selectbox("Delete saved watchlist", list(st.session_state.saved_watchlists.keys()), key="del_sel")
        if st.button("🗑️ Delete"):
            delete_watchlist(sel_del)
            st.success(f"Deleted {sel_del}")
            st.rerun()

# Export JSON
if st.session_state.saved_watchlists:
    st.download_button(
        "⬇️ Download All Watchlists (JSON)",
        data=export_watchlists_json(),
        file_name="watchlists.json",
        mime="application/json"
    )

st.markdown("---")

# ---------------- Single-Ticker Analysis Entry ----------------
st.header("🔍 Analyze Stock from Watchlist")

if not st.session_state.active_watchlist:
    st.info("No tickers in active watchlist yet.")
else:
    ticker_choice = st.selectbox("Choose a ticker to analyze", st.session_state.active_watchlist)
    if st.button("Run Analysis"):
        st.session_state.ran = True
        st.session_state.analysis_ticker = ticker_choice
# =========================
# Analysis Modules
# =========================
import requests, feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------- Plot ----------
def plot_price(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"
    ))
    for name in ["SMA20","SMA50","EMA20","EMA50"]:
        if name in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[name], name=name, mode="lines"))
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    return fig

# ---------- Trade plan ----------
def plan_trade(latest, trend_up, atr_val, hh20, ll20, ema20, close,
               account_size, risk_pct, stop_atr_mult, target_rr, entry_style):
    risk_d = max(0.0, account_size*(risk_pct/100.0))
    if entry_style == "Pullback to EMA20":
        entry = float(ema20)
        stop = entry - stop_atr_mult*atr_val if trend_up else entry + stop_atr_mult*atr_val
        target = entry + target_rr*(entry - stop) if trend_up else entry - target_rr*(stop - entry)
    else:  # Breakout
        if trend_up:
            entry = float(max(close, hh20))
            stop = entry - stop_atr_mult*atr_val
            target = entry + target_rr*(entry - stop)
        else:
            entry = float(min(close, ll20))
            stop = entry + stop_atr_mult*atr_val
            target = entry - target_rr*(stop - entry)
    sd = abs(entry - stop)
    shares = int(math.floor(risk_d/sd)) if (sd > 0 and risk_d > 0) else 0
    notional = shares * entry
    rr = (abs(target - entry)/sd) if sd > 0 else np.nan
    reward_d = abs(target - entry) * shares
    return {"entry":entry,"stop":stop,"target":target,"shares":shares,
            "risk_dollars":risk_d,"reward_dollars":reward_d,"notional":notional,"rr":rr}

def make_order_ticket(ticker, side_long, entry_style, entry, stop, target, shares, rr, atr_val, risk_dollars, reward_dollars):
    bias = "LONG" if side_long else "SHORT"
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"Order Ticket - {now}",
        f"Ticker: {ticker.upper()}",
        f"Bias: {bias}",
        f"Entry style: {entry_style}",
        f"Entry: {entry:,.2f}",
        f"Stop: {stop:,.2f}",
        f"Target: {target:,.2f}",
        f"Shares: {shares:,}",
        f"Risk $: {risk_dollars:,.2f}",
        f"Est. Reward $: {reward_dollars:,.2f}",
        f"Approx R:R: {rr:.2f}" if np.isfinite(rr) else "Approx R:R: n/a",
        f"ATR(14): {atr_val:,.2f}",
        "Notes: Use a bracket order (entry + stop + limit)."
    ]
    text = "\n".join(lines)
    payload = {
        "timestamp": now, "ticker": ticker.upper(), "bias": bias,
        "entry_style": entry_style, "entry": round(entry, 4), "stop": round(stop, 4),
        "target": round(target, 4), "shares": int(shares),
        "risk_dollars": round(risk_dollars, 2), "reward_dollars": round(reward_dollars, 2),
        "rr": round(rr, 3) if np.isfinite(rr) else None, "atr14": round(atr_val, 4)
    }
    json_bytes = io.BytesIO(json.dumps(payload, indent=2).encode("utf-8"))
    csv_bytes  = io.BytesIO(pd.DataFrame([payload]).to_csv(index=False).encode("utf-8"))
    txt_bytes  = io.BytesIO(text.encode("utf-8"))
    return text, json_bytes, csv_bytes, txt_bytes

# ---------- ETA helpers ----------
def eta_trend_slope_days(df, current_price, target_price, lookback=20):
    try:
        if not (np.isfinite(current_price) and np.isfinite(target_price)): return None
        if current_price <= 0 or target_price <= 0 or target_price <= current_price: return None
        s = pd.Series(df["Close"]).dropna().tail(max(lookback, 10))
        if len(s) < 10: return None
        x = np.arange(len(s)).reshape(-1, 1)
        y = np.log(s.values)
        lr = LinearRegression().fit(x, y)
        slope = float(lr.coef_[0])
        if slope <= 0 or not np.isfinite(slope): return None
        bars_needed = math.log(target_price / current_price) / slope
        return float(min(bars_needed, 200.0)) if bars_needed > 0 else None
    except: return None

def eta_historical_window(df, pct_move, lookback_bars=250, max_horizon=60):
    try:
        if not np.isfinite(pct_move) or pct_move <= 0: return (None, None, None)
        s = pd.Series(df["Close"]).dropna().tail(lookback_bars)
        if len(s) < 30: return (None, None, None)
        times = []
        for i in range(len(s)-2):
            base = float(s.iloc[i]); tgt = base * (1.0 + pct_move)
            forward = s.iloc[i+1 : min(i+1+max_horizon, len(s))]
            hit = np.where(forward.values >= tgt, True, False)
            if hit.any(): times.append(int(np.argmax(hit) + 1))
        if not times: return (None, None, None)
        arr = np.array(times)
        return float(np.median(arr)), float(np.percentile(arr,25)), float(np.percentile(arr,75))
    except: return (None, None, None)

# =========================
# Run Analysis if ticker selected
# =========================
if st.session_state.get("ran", False) and st.session_state.get("analysis_ticker"):
    ticker = st.session_state.analysis_ticker
    st.subheader(f"Analysis — {ticker}")

    df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    if df.empty:
        st.error("No data found.")
    else:
        # Indicators
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()

        st.plotly_chart(plot_price(df), use_container_width=True)

        latest = df.iloc[-1]
        last_close = float(latest["Close"])
        sma20_val  = float(latest["SMA20"])
        ema50_val  = float(latest["EMA50"])
        atr_val    = float((df["High"]-df["Low"]).rolling(14).mean().iloc[-1])

        c1,c2,c3 = st.columns(3)
        c1.metric("Last Close", f"{last_close:.2f}")
        c2.metric("20d SMA", f"{sma20_val:.2f}")
        c3.metric("50d EMA", f"{ema50_val:.2f}")

        # ---------- Forecast ----------
        st.subheader("Tiny Forecast")
        s = df["Close"].pct_change().dropna().tail(20)
        if len(s) > 5:
            X = np.arange(len(s)).reshape(-1,1); y = s.values
            model = LinearRegression().fit(X,y)
            nxt = float(model.predict(np.array([[len(s)]]))[0])
            expc = last_close*(1+nxt)
            d1,d2 = st.columns(2)
            d1.metric("Direction", "Up" if nxt>0 else "Down")
            d2.metric("Expected Close", f"{expc:.2f}")
        else:
            st.info("Not enough data for forecast.")

        # ---------- ML Edge ----------
        st.subheader("ML Edge")
        try:
            X = pd.concat([
                pd.Series(df["Close"].pct_change(), name="RET1"),
                pd.Series(df["Close"].pct_change(5), name="RET5"),
                pd.Series(df["Close"].pct_change(10), name="RET10")
            ], axis=1).dropna()
            y = (df["Close"].shift(-1) > df["Close"]).astype(int).reindex(X.index)
            if len(X)>200 and y.notna().sum()>50:
                Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,shuffle=False)
                clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
                clf.fit(Xtr,ytr)
                acc = accuracy_score(yte, clf.predict(Xte))
                proba_up = float(clf.predict_proba(X.iloc[[-1]])[0,1])
                st.metric("Prob Up", f"{proba_up*100:.1f}%")
                st.metric("Backtest Acc", f"{acc*100:.1f}%")
        except Exception as e:
            st.warning(f"ML error: {e}")

        # ---------- News Sentiment ----------
        st.subheader("News Sentiment")
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
            d = feedparser.parse(url)
            headlines = [e.title for e in d.entries[:5]]
            if headlines:
                ana = SentimentIntensityAnalyzer()
                rows = [{"title":h, "score":ana.polarity_scores(h)["compound"]} for h in headlines]
                st.table(pd.DataFrame(rows))
            else:
                st.info("No headlines fetched.")
        except Exception as e:
            st.info(f"Sentiment skipped: {e}")

        # ---------- Trade Plan ----------
        st.subheader("Trade Plan")
        trend_up = latest["EMA20"] > latest["EMA50"]
        hh20 = float(df["High"].rolling(20).max().iloc[-1])
        ll20 = float(df["Low"].rolling(20).min().iloc[-1])
        ema20 = float(latest["EMA20"])

        cpa, cpb, cpc = st.columns(3)
        with cpa: account_size = st.number_input("Account size ($)", min_value=0.0, value=10000.0, step=100.0)
        with cpb: risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, step=0.1)
        with cpc: entry_style = st.selectbox("Entry style", ["Pullback to EMA20","Breakout 20-day"])
        cpd, cpe, cpf = st.columns(3)
        with cpd: stop_atr_mult = st.slider("Stop distance (x ATR)", 0.5, 5.0, 1.5, step=0.1)
        with cpe: target_rr = st.slider("Target multiple (R:R)", 0.5, 5.0, 2.0, step=0.5)
        with cpf: lookback_eta = st.slider("ETA slope lookback (days)", 10, 60, 20, step=5)

        plan = plan_trade(latest, trend_up, atr_val, hh20, ll20, ema20, last_close,
                          account_size, risk_pct, stop_atr_mult, target_rr, entry_style)

        colA,colB,colC = st.columns(3)
        colA.metric("Suggested Entry", f"{plan['entry']:,.2f}")
        colB.metric("Stop", f"{plan['stop']:,.2f}")
        colC.metric("Target", f"{plan['target']:,.2f}")
        colD,colE,colF = st.columns(3)
        colD.metric("Shares", f"{plan['shares']:,}")
        colE.metric("Risk ($)", f"{plan['risk_dollars']:,.2f}")
        colF.metric("Est. Reward ($)", f"{plan['reward_dollars']:,.2f}")

        # ETA
        try:
            if plan["target"] > last_close:
                eta1 = eta_trend_slope_days(df, current_price=last_close, target_price=plan["target"], lookback=lookback_eta)
                pct_move = (plan["target"] / last_close) - 1.0 if last_close > 0 else np.nan
                med, q1, q3 = eta_historical_window(df, pct_move=pct_move)
                st.subheader("Time-to-Target (ETA)")
                if eta1 is not None: st.metric("Trend slope ETA", f"{eta1:.1f} d")
                if med is not None: st.metric("Historical median", f"{med:.0f} d (IQR {q1:.0f}-{q3:.0f})")
        except: pass

        # Order ticket
        ticket_text, json_bytes, csv_bytes, txt_bytes = make_order_ticket(
            ticker=ticker, side_long=trend_up, entry_style=entry_style,
            entry=plan['entry'], stop=plan['stop'], target=plan['target'], shares=plan['shares'],
            rr=plan['rr'], atr_val=atr_val, risk_dollars=plan['risk_dollars'], reward_dollars=plan['reward_dollars']
        )
        st.subheader("Order Ticket")
        st.code(ticket_text, language="text")
        cdl1,cdl2,cdl3 = st.columns(3)
        with cdl1: st.download_button("Download TXT",  data=txt_bytes,  file_name=f"{ticker.upper()}_ticket.txt",  mime="text/plain")
        with cdl2: st.download_button("Download JSON", data=json_bytes, file_name=f"{ticker.upper()}_ticket.json", mime="application/json")
        with cdl3: st.download_button("Download CSV",  data=csv_bytes,  file_name=f"{ticker.upper()}_ticket.csv",  mime="text/csv")
