# app.py
# SwingFinder — Screener + Scanner + Plan + ML + Seasonality + News
# Run with: streamlit run app.py
# Educational tool only. Not financial advice.

import warnings, os, io, json, math, time
warnings.filterwarnings("ignore")

import datetime as dt
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import requests

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="SwingFinder", layout="wide")
st.title("SwingFinder")
st.caption("For learning only. Not financial advice.")

# ---------------- Session State ----------------
if "ran" not in st.session_state: 
    st.session_state.ran = False
if "watchlists" not in st.session_state:
    st.session_state.watchlists = {
        "Default": ["AAPL","MSFT","NVDA","TSLA","META"]
    }
if "active_watchlist" not in st.session_state:
    st.session_state.active_watchlist = "Default"

# ---------------- Helpers ----------------
def get_secret(name: str, default=None):
    v = os.environ.get(name)
    if v is not None:
        return v
    try:
        return st.secrets[name]
    except Exception:
        return default

def _as_series(x):
    if isinstance(x, pd.DataFrame): return x.iloc[:, 0]
    return pd.Series(x).squeeze()

# ---------------- Indicators ----------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return pd.Series(s, index=s.index).ewm(span=span, adjust=False).mean()

def sma(s: pd.Series, window: int) -> pd.Series:
    return pd.Series(s, index=s.index).rolling(window).mean()

def rsi(series, period: int = 14) -> pd.Series:
    if isinstance(series, pd.DataFrame): series = series.iloc[:, 0]
    series = pd.to_numeric(pd.Series(series).squeeze(), errors="coerce")
    d = series.diff()
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    g = pd.Series(g, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    l = pd.Series(l, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    return (100 - (100/(1+rs))).bfill()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    s = _as_series(series)
    fast_ = ema(s, fast); slow_ = ema(s, slow)
    line = fast_ - slow_; sig = ema(line, signal); hist = line - sig
    return line, sig, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([(df["High"]-df["Low"]).abs(), (df["High"]-pc).abs(), (df["Low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            s = df[c]
            if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
            df[c] = pd.to_numeric(pd.Series(s).squeeze(), errors="coerce")
    df["SMA20"] = sma(df["Close"], 20)
    df["SMA50"] = sma(df["Close"], 50)
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["RSI14"] = rsi(df["Close"], 14)
    m, sig, _ = macd(df["Close"]); df["MACD"] = m; df["MACDsig"] = sig
    df["ATR14"] = atr(df)
    df["HH20"] = df["High"].rolling(20).max()
    df["LL20"] = df["Low"].rolling(20).min()
    bbw = 20; bbstd = pd.Series(df["Close"], index=df.index).rolling(bbw).std()
    df["BB_MID"] = sma(df["Close"], bbw)
    df["BB_UP"]  = df["BB_MID"] + 2*bbstd
    df["BB_DN"]  = df["BB_MID"] - 2*bbstd
    cu = (df["EMA20"]>df["EMA50"]) & (df["EMA20"].shift(1)<=df["EMA50"].shift(1))
    cd = (df["EMA20"]<df["EMA50"]) & (df["EMA20"].shift(1)>=df["EMA50"].shift(1))
    df["buy_signal"]  = cu & (df["RSI14"]>50) & (df["Close"]>df["SMA50"])
    df["sell_signal"] = cd & (df["RSI14"]<50) & (df["Close"]<df["SMA50"])
    return df

# ---------------- Data ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, years: int = 3) -> pd.DataFrame:
    t = (ticker or "").strip().upper()
    if not t:
        return pd.DataFrame()
    end = date.today(); start = end - timedelta(days=int(years*365.25))
    df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    if (df is None or df.empty) and "." in t:
        alt = t.replace(".","-")
        df2 = yf.download(alt, start=start, end=end, auto_adjust=True, progress=False, threads=True)
        if df2 is not None and not df2.empty:
            df, t = df2, alt
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[keep].copy()
    for c in df.columns:
        s = df[c]
        if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
        df[c] = pd.to_numeric(np.asarray(s).reshape(-1), errors="coerce")
    return df.dropna()
# ---------------- Forecast (toy) ----------------
def quick_forecast(df: pd.DataFrame, lookback: int = 20):
    s = df["Close"].pct_change().dropna().tail(lookback)
    if len(s) < 5:
        return None
    X = np.arange(len(s)).reshape(-1,1); y = s.values
    model = LinearRegression().fit(X, y)
    nxt = float(model.predict(np.array([[len(s)]]))[0])
    last = float(df["Close"].iloc[-1]); expc = last*(1+nxt)
    conf = float(min(0.95, max(0.05, abs(model.coef_[0])/(s.std()+1e-9))))
    return {"direction":"up" if nxt>0 else "down",
            "next_ret":nxt,
            "last_close":last,
            "expected_close":float(expc),
            "confidence":conf}

# ---------------- Plot ----------------
def plot_price(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))
    buys = df[df["buy_signal"]]; sells = df[df["sell_signal"]]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", marker_symbol="triangle-up", marker_size=10, name="Buy"))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", marker_symbol="triangle-down", marker_size=10, name="Sell"))
    fig.update_layout(height=600, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                      margin=dict(l=10,r=10,t=30,b=10))
    return fig

# ---------------- Trade Plan & Ticket ----------------
def plan_trade(latest_row: pd.Series, trend_up: bool, atr_val: float, hh20: float, ll20: float,
               ema20: float, close: float, account_size: float, risk_pct: float,
               stop_atr_mult: float, target_rr: float, entry_style: str):
    risk_d = max(0.0, account_size*(risk_pct/100.0))
    if entry_style == "Pullback to EMA20":
        entry = float(ema20)
        if trend_up:
            stop = entry - stop_atr_mult*atr_val; target = entry + target_rr*(entry - stop)
        else:
            stop = entry + stop_atr_mult*atr_val; target = entry - target_rr*(stop - entry)
    else:
        if trend_up:
            entry = float(max(close, hh20)); stop = entry - stop_atr_mult*atr_val; target = entry + target_rr*(entry - stop)
        else:
            entry = float(min(close, ll20)); stop = entry + stop_atr_mult*atr_val; target = entry - target_rr*(stop - entry)
    sd = abs(entry - stop)
    shares = int(math.floor(risk_d/sd)) if (sd > 0 and risk_d > 0) else 0
    notional = shares * entry
    rr = (abs(target - entry)/sd) if sd > 0 else np.nan
    reward_d = abs(target - entry) * shares
    return {"entry":entry,"stop":stop,"target":target,"shares":shares,"risk_dollars":risk_d,
            "reward_dollars":reward_d,"notional":notional,"rr":rr}

def make_order_ticket(ticker: str, side_long: bool, entry_style: str, entry: float, stop: float, target: float,
                      shares: int, rr: float, atr_val: float, risk_dollars: float, reward_dollars: float):
    bias = "LONG" if side_long else "SHORT"
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"Webull Order Ticket - {now}",
        f"Ticker: {ticker.upper()}",
        f"Bias: {bias}",
        f"Entry style: {entry_style}",
        f"Entry: {entry:,.2f}",
        f"Stop: {stop:,.2f}",
        f"Target: {target:,.2f}",
        f"Shares: {shares:,}",
        f"Risk $: {risk_dollars:,.2f}",
        f"Est. Reward $: {reward_dollars:,.2f}",
        f"Approx R:R: {rr:.2f}",
        f"ATR(14): {atr_val:,.2f}",
        "Notes: Consider a bracket order (entry + stop + limit target)."
    ]
    text = "\n".join(lines)
    payload = {
        "timestamp": now, "broker": "Webull", "ticker": ticker.upper(), "bias": bias,
        "entry_style": entry_style, "entry": round(entry, 4), "stop": round(stop, 4),
        "target": round(target, 4), "shares": int(shares),
        "risk_dollars": round(risk_dollars, 2), "reward_dollars": round(reward_dollars, 2),
        "rr": round(rr, 3) if np.isfinite(rr) else None, "atr14": round(atr_val, 4)
    }
    json_bytes = io.BytesIO(json.dumps(payload, indent=2).encode("utf-8"))
    csv_bytes  = io.BytesIO(pd.DataFrame([payload]).to_csv(index=False).encode("utf-8"))
    txt_bytes  = io.BytesIO(text.encode("utf-8"))
    return text, json_bytes, csv_bytes, txt_bytes

# ---------------- Watchlist Manager ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("📌 Watchlist Manager")

wl_names = list(st.session_state.watchlists.keys())
chosen = st.sidebar.selectbox("Select watchlist", wl_names, 
                              index=wl_names.index(st.session_state.active_watchlist))

if st.sidebar.button("Load Watchlist"):
    st.session_state.active_watchlist = chosen

# Show current watchlist for editing
wl_input = st.sidebar.text_area(
    "Edit watchlist (comma tickers)",
    ", ".join(st.session_state.watchlists[chosen]),
    height=80
)

# Save or update
new_name = st.sidebar.text_input("Save as new watchlist name")
if st.sidebar.button("Save Watchlist") and new_name.strip():
    st.session_state.watchlists[new_name.strip()] = [
        t.strip().upper() for t in wl_input.split(",") if t.strip()
    ]
if st.sidebar.button("Update Current"):
    st.session_state.watchlists[chosen] = [
        t.strip().upper() for t in wl_input.split(",") if t.strip()
    ]

# ---------------- Watchlist Screener ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Watchlist Screener")
screen_years = st.sidebar.slider("Screener lookback (years)", 1, 5, 1)
run_screen   = st.sidebar.button("Run Screener")

def run_watchlist_screener(tickers, years):
    passes, near_rows = [], []
    for t in tickers:
        sdf = load_data(t, years=years)
        if sdf.empty: 
            continue
        sdf = build_indicators(sdf); last = sdf.iloc[-1]
        close = float(last["Close"])
        trend_up = bool(last["EMA20"] > last["EMA50"])
        buy  = bool(last["buy_signal"]); sell = bool(last["sell_signal"])
        rsi_v = float(last["RSI14"])
        bb_dn = float(last.get("BB_DN", np.nan)); bb_up = float(last.get("BB_UP", np.nan))
        band = (close - bb_dn)/(bb_up - bb_dn) if np.isfinite(bb_dn) and np.isfinite(bb_up) and (bb_up-bb_dn)>0 else np.nan

        score = 0; reason = []
        if trend_up and 43 <= rsi_v <= 62 and (0.08 <= (band if np.isfinite(band) else 0.5) <= 0.60):
            score += 2; reason.append("pullback-window")
        if buy:
            score += 2; reason.append("buy-signal")
        if trend_up and rsi_v > 50:
            score += 1

        row = {"Ticker":t,"Close":round(close,2),"RSI14":round(rsi_v,1),
               "BandPos":None if np.isnan(band) else round(band,2),
               "TrendUp":trend_up,"Score":score,
               "Reason":",".join(reason) if reason else "-"}

        if score >= 2:
            passes.append(row)
        elif score >= 1:
            near_rows.append(row)

    return passes, near_rows

if run_screen:
    tickers = [t.strip().upper() for t in wl_input.split(",") if t.strip()]
    if not tickers:
        st.warning("No tickers provided.")
    else:
        passes, near_rows = run_watchlist_screener(tickers, years=screen_years)
        if passes:
            st.subheader("✅ Screener Passes")
            st.dataframe(pd.DataFrame(passes).sort_values("Score", ascending=False).reset_index(drop=True), use_container_width=True)
        if near_rows:
            st.subheader("⚠️ Near Misses")
            st.dataframe(pd.DataFrame(near_rows).sort_values("Score", ascending=False).reset_index(drop=True), use_container_width=True)
        if not passes and not near_rows:
            st.info("No watchlist candidates met your screener filters.")
# ---------------- Scanner logic ----------------
def good_position_flags(last: pd.Series, mode: str, strictness: str = "Balanced"):
    trend_up = bool(last["EMA20"] > last["EMA50"])
    rsi_v = float(last["RSI14"])
    close = float(last["Close"])
    bb_dn = float(last.get("BB_DN", np.nan))
    bb_up = float(last.get("BB_UP", np.nan))
    hh20  = float(last.get("HH20", np.nan))
    band_pos = np.nan
    if np.isfinite(bb_dn) and np.isfinite(bb_up) and (bb_up - bb_dn) > 0:
        band_pos = (close - bb_dn) / (bb_up - bb_dn)

    if strictness == "Strict":
        pull_rsi_lo, pull_rsi_hi = 45, 60
        pull_band_lo, pull_band_hi = 0.10, 0.50
        brk_near = 0.995; brk_rsi_min = 50
    elif strictness == "Loose":
        pull_rsi_lo, pull_rsi_hi = 40, 65
        pull_band_lo, pull_band_hi = 0.05, 0.70
        brk_near = 0.985; brk_rsi_min = 47
    else:  # Balanced
        pull_rsi_lo, pull_rsi_hi = 43, 62
        pull_band_lo, pull_band_hi = 0.08, 0.60
        brk_near = 0.990; brk_rsi_min = 48

    buy_trigger = bool(last.get("buy_signal", False))
    score = 0; ok = False; reason = []

    # Pullback window (trend up, RSI and bandpos inside window)
    cond_pull = trend_up and (pull_rsi_lo <= rsi_v <= pull_rsi_hi) and \
                (pull_band_lo <= (band_pos if np.isfinite(band_pos) else 0.5) <= pull_band_hi)
    if mode in ("Pullback","Both") and cond_pull:
        ok = True; score += 2; reason.append("pullback-window")

    # Breakout: near 20d high or buy trigger
    near_high = np.isfinite(hh20) and (close >= brk_near*hh20) and trend_up and (rsi_v >= brk_rsi_min)
    if mode in ("Breakout","Both") and (buy_trigger or near_high):
        ok = True; score += 2; reason.append("buy-signal" if buy_trigger else "breakout-near-high")

    if trend_up and rsi_v > 50:
        score += 1

    return ok, score, band_pos, (",".join(reason) if reason else "-")

@st.cache_data(ttl=900, show_spinner=True)
def scan_universe(universe: str, years: int, price_min: float, price_max: float,
                  min_dollar_vol_millions: float, mode: str, strictness: str = "Balanced",
                  max_symbols: int = 100):
    if universe == "S&P 500":
        try:
            tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
            syms = tables[0]["Symbol"].astype(str).tolist()
            tickers = [s.replace(".","-").upper().strip() for s in syms if s]
        except Exception:
            tickers = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","GOOG","JPM"]
    elif universe == "NASDAQ-100":
        try:
            tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
            table = None
            for t in tables:
                cols = [str(c).lower() for c in t.columns]
                if "ticker" in cols or "symbol" in cols:
                    table = t; break
            if table is None: raise RuntimeError("No table")
            col = [c for c in table.columns if str(c).lower() in ("ticker","symbol")][0]
            syms = table[col].astype(str).tolist()
            tickers = [s.replace(".","-").upper().strip() for s in syms if s]
        except Exception:
            tickers = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","AVGO","NFLX","ADBE","AMD"]
    else:
        tickers = []
    tickers = tickers[:max_symbols]

    stats = {"universe": universe, "start": len(tickers), "after_price": 0,
             "after_liquidity": 0, "ok": 0, "near": 0, "errors": 0, "sample": tickers[:10]}

    passes, near_rows = [], []

    for t in tickers:
        try:
            df = load_data(t, years)
            if df.empty: 
                continue
            df = build_indicators(df); last = df.iloc[-1]
            close = float(last["Close"])

            if not (price_min <= close <= price_max): 
                continue
            stats["after_price"] += 1

            adv = (df["Close"]*df["Volume"]).rolling(20).mean().iloc[-1]
            adv_m = float(adv)/1_000_000.0 if pd.notna(adv) else 0.0
            if adv_m < min_dollar_vol_millions: 
                continue
            stats["after_liquidity"] += 1

            ok, score, band, why = good_position_flags(last, mode, strictness)
            base = {"Ticker":t,"Close":round(close,2),"Avg$Vol(20d, M)":round(adv_m,2),
                    "TrendUp":bool(last["EMA20"]>last["EMA50"]),
                    "RSI14":round(float(last["RSI14"]),1),
                    "BandPos(0=low,1=high)": None if np.isnan(band) else round(band,2),
                    "BUY?":bool(last.get("buy_signal",False)),"Score":int(score),"Reason":why}

            if ok and score >= 2:
                passes.append(base); stats["ok"] += 1
            else:
                # near-miss heuristics
                near_score=0
                if base["TrendUp"]: near_score += 1
                if base["RSI14"] >= 48: near_score += 1
                bp = base["BandPos(0=low,1=high)"]
                if bp is not None and bp <= 0.75: near_score += 1
                if near_score >= 2:
                    nm = dict(base); nm["Reason"]="near-miss"; nm["Score"]=max(nm["Score"], near_score)
                    near_rows.append(nm); stats["near"] += 1
        except Exception:
            stats["errors"] += 1
            continue

    def _sort(df_):
        return df_.sort_values(["Score","Avg$Vol(20d, M)","RSI14"], ascending=[False,False,False]).reset_index(drop=True)

    df_pass = _sort(pd.DataFrame(passes)) if passes else pd.DataFrame()
    df_near = _sort(pd.DataFrame(near_rows)) if near_rows else pd.DataFrame()
    if not df_pass.empty:
        df_pass.insert(0,"Rank",np.arange(1,len(df_pass)+1))
    if not df_near.empty:
        df_near.insert(0,"Rank",np.arange(1,len(df_near)+1))
    return df_pass, df_near, stats

@st.cache_data(ttl=600, show_spinner=True)
def scan_fixed_list(ticks: list[str], years: int, price_min: float, price_max: float,
                    min_dollar_vol_millions: float, mode: str, strictness: str, max_symbols: int):
    ticks = [t.strip().upper() for t in ticks if t.strip()][:max_symbols]
    passes, near_rows = [], []
    stats = {"universe":"Custom","start":len(ticks),"after_price":0,"after_liquidity":0,
             "ok":0,"near":0,"errors":0,"sample":ticks[:10]}
    for t in ticks:
        try:
            df = load_data(t, years)
            if df.empty: 
                continue
            df = build_indicators(df); last = df.iloc[-1]
            close = float(last["Close"])
            if not (price_min <= close <= price_max): 
                continue
            stats["after_price"] += 1
            adv = (df["Close"]*df["Volume"]).rolling(20).mean().iloc[-1]
            adv_m = float(adv)/1_000_000.0 if pd.notna(adv) else 0.0
            if adv_m < min_dollar_vol_millions: 
                continue
            stats["after_liquidity"] += 1
            ok, score, band, why = good_position_flags(last, mode, strictness)
            base = {"Ticker":t,"Close":round(close,2),"Avg$Vol(20d, M)":round(adv_m,2),
                    "TrendUp":bool(last["EMA20"]>last["EMA50"]),
                    "RSI14":round(float(last["RSI14"]),1),
                    "BandPos(0=low,1=high)": None if np.isnan(band) else round(band,2),
                    "BUY?":bool(last.get("buy_signal",False)),"Score":int(score),"Reason":why}
            if ok and score >= 2:
                passes.append(base); stats["ok"] += 1
            else:
                near_score=0
                if base["TrendUp"]: near_score += 1
                if base["RSI14"] >= 48: near_score += 1
                bp = base["BandPos(0=low,1=high)"]
                if bp is not None and bp <= 0.75: near_score += 1
                if near_score >= 2:
                    nm = dict(base); nm["Reason"]="near-miss"; nm["Score"]=max(nm["Score"], near_score)
                    near_rows.append(nm); stats["near"] += 1
        except Exception:
            stats["errors"] += 1
    df_pass = pd.DataFrame(passes); df_near = pd.DataFrame(near_rows)
    if not df_pass.empty:
        df_pass = df_pass.sort_values(["Score","Avg$Vol(20d, M)","RSI14"], ascending=[False,False,False]).reset_index(drop=True)
        df_pass.insert(0,"Rank",np.arange(1,len(df_pass)+1))
    if not df_near.empty:
        df_near = df_near.sort_values(["Score","Avg$Vol(20d, M)","RSI14"], ascending=[False,False,False]).reset_index(drop=True)
        df_near.insert(0,"Rank",np.arange(1,len(df_near)+1))
    return df_pass, df_near, stats

# ---------------- Seasonality & News helpers ----------------
def _prep_series_for_ts(df: pd.DataFrame) -> pd.Series:
    s = df["Close"].copy()
    s.index = pd.DatetimeIndex(s.index)
    return s.asfreq("B").ffill()

def seasonality_tables(df: pd.DataFrame):
    s = _prep_series_for_ts(df)
    ret_next = s.pct_change().shift(-1)
    dfw = pd.DataFrame({"ret_next": ret_next})
    dfw["dow"] = dfw.index.dayofweek
    dfw["mon"] = dfw.index.month

    dow_tbl = dfw.groupby("dow")["ret_next"].mean().reindex([0,1,2,3,4]).to_frame("avg_next_ret").dropna()
    dow_tbl.index = ["Mon","Tue","Wed","Thu","Fri"]

    mon_order = list(range(1,13))
    mon_tbl = dfw.groupby("mon")["ret_next"].mean().reindex(mon_order).to_frame("avg_next_ret").dropna()
    mon_tbl.index = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    return dow_tbl, mon_tbl

def get_upcoming_earnings(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        dfcal = None
        for method in ("get_earnings_dates", "earnings_dates"):
            f = getattr(tk, method, None)
            if callable(f):
                try:
                    dfcal = f(limit=6)
                    break
                except Exception:
                    pass
        if dfcal is None or len(dfcal)==0:
            return None, None
        if isinstance(dfcal.index, pd.DatetimeIndex):
            next_dt = pd.to_datetime(dfcal.index[0]).date()
        else:
            cands = [c for c in dfcal.columns if "earn" in str(c).lower() and "date" in str(c).lower()]
            if not cands:
                return None, None
            next_dt = pd.to_datetime(dfcal[cands[0]].iloc[0]).date()
        today = date.today()
        return next_dt, (next_dt - today).days
    except Exception:
        return None, None

_analyzer = None
def _get_vader():
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer

def fetch_rss_headlines(ticker: str, limit: int = 12):
    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={ticker}%20stock&hl=en-US&gl=US&ceid=US:en",
    ]
    out = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            for e in d.entries[:limit]:
                title = getattr(e, "title", "").strip()
                link  = getattr(e, "link", "")
                pub   = getattr(e, "published", getattr(e, "updated", ""))
                if title:
                    out.append({"title": title, "link": link, "published": pub})
            if out:
                break
        except Exception:
            continue
    return out[:limit]

def score_headlines_vader(headlines: list[dict]):
    ana = _get_vader()
    rows = []
    for h in headlines:
        comp = ana.polarity_scores(h["title"])["compound"]
        if comp >= 0.15: lab = "Bullish"
        elif comp <= -0.15: lab = "Bearish"
        else: lab = "Neutral"
        rows.append({**h, "compound": comp, "label": lab})
    if rows:
        df = pd.DataFrame(rows)
        return df, float(df["compound"].mean())
    return pd.DataFrame(columns=["title","link","published","compound","label"]), 0.0

# ---------------- Sidebar (Analysis & Scanner controls) ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Single-Ticker Analysis")
ticker    = st.sidebar.text_input("Stock ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")
years     = st.sidebar.slider("Years of history", 1, 10, 3)
lookback  = st.sidebar.slider("Forecast lookback (days)", 5, 60, 20)

st.sidebar.subheader("🛡️ Risk & Plan")
account_size  = st.sidebar.number_input("Account size ($)", min_value=0.0, value=10000.0, step=100.0)
risk_pct      = st.sidebar.slider("Risk per trade (%)", 0.1, 5.0, 1.0, step=0.1)
stop_atr_mult = st.sidebar.slider("Stop distance (x ATR)", 0.5, 5.0, 1.5, step=0.1)
target_rr     = st.sidebar.slider("Target multiple (R:R)", 0.5, 5.0, 2.0, step=0.5)
entry_style   = st.sidebar.selectbox("Entry style", ["Pullback to EMA20","Breakout 20-day"])

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Market Scanner")
universe     = st.sidebar.selectbox("Universe", ["S&P 500","NASDAQ-100"], index=0)
scan_years   = st.sidebar.slider("History (years)", 1, 5, 2)
price_min    = st.sidebar.number_input("Min price ($)", 0.0, value=5.0, step=0.5)
price_max    = st.sidebar.number_input("Max price ($)", 0.0, value=20.0, step=0.5)
liq          = st.sidebar.number_input("Min Avg $ Volume (20d, M)", 0.0, value=2.0, step=0.5)
setup_mode   = st.sidebar.radio("Setup type", ["Both","Pullback","Breakout"], horizontal=True)
strictness   = st.sidebar.selectbox("Scanner strictness", ["Balanced","Strict","Loose"], index=0)
max_symbols  = st.sidebar.slider("Max symbols to scan", 20, 200, 100, step=10)
custom_universe = st.sidebar.text_area("Custom universe (optional, comma tickers)", "", height=60)
debug_scan   = st.sidebar.checkbox("Show scanner debug", value=False)

# ---------------- Top Buttons ----------------
col_top_a, col_top_b, col_top_c = st.columns([1,1,3])
with col_top_a:
    if st.button("Analyze"):
        st.session_state.ran = True
with col_top_b:
    st.button("New analysis (reset)", on_click=lambda: st.session_state.update(ran=False))

# ---------------- Market Scanner output (Passes & Near Misses) ----------------
scan_clicked = st.sidebar.button("Scan Market")
if scan_clicked:
    if custom_universe.strip():
        tickers_override = [t.strip().upper() for t in custom_universe.split(",") if t.strip()]
        df_pass, df_near, stats = scan_fixed_list(
            tickers_override, scan_years, price_min, price_max, liq, setup_mode, strictness, max_symbols
        )
        universe_label = f"Custom ({len(tickers_override)})"
    else:
        df_pass, df_near, stats = scan_universe(
            universe, scan_years, price_min, price_max, liq, setup_mode, strictness, max_symbols
        )
        universe_label = universe

    st.subheader(f"Market Scanner — {universe_label} ({strictness})")
    st.caption(f"Filters: {universe_label} | ${price_min:.2f}–${price_max:.2f} | Min Avg $ Vol(20d): {liq:.1f}M | Mode: {setup_mode} | Scanned: {stats['start']}")

    if debug_scan:
        st.write({
            "universe": stats["universe"],
            "start_symbols": stats["start"],
            "after_price": stats["after_price"],
            "after_liquidity": stats["after_liquidity"],
            "ok_setups": stats["ok"],
            "near_miss": stats["near"],
            "errors": stats["errors"],
            "sample_symbols": stats["sample"],
        })

    if (df_pass is None or df_pass.empty) and (df_near is None or df_near.empty):
        st.info("No candidates met your filters. Try Strictness = Loose, lower Min Avg $ Vol, widen price, or use a Custom universe.")
    else:
        if df_pass is not None and not df_pass.empty:
            st.markdown("### ✅ Passes")
            st.dataframe(df_pass, use_container_width=True)
        if df_near is not None and not df_near.empty:
            st.markdown("### ⚠️ Near Misses")
            st.dataframe(df_near, use_container_width=True)

# ---------------- Single Ticker analysis ----------------
if st.session_state.ran:
    df = load_data(ticker, years)
    if df.empty:
        st.error("No data found. Some tickers need exchange suffixes like .L, .TO, .SA.")
        st.stop()
    df = build_indicators(df)

    st.subheader(f"{ticker.upper()} price and swing signals")
    st.plotly_chart(plot_price(df), use_container_width=True)

    latest  = df.iloc[-1]
    trend_up = latest["EMA20"] > latest["EMA50"]
    side_txt = "long bias" if trend_up else "short bias"
    atr_val = float(latest["ATR14"]); close = float(latest["Close"])
    ema20   = float(latest["EMA20"]); ema50 = float(latest["EMA50"])
    hh20    = float(latest.get("HH20", np.nan)); ll20 = float(latest.get("LL20", np.nan))

    c1,c2,c3 = st.columns(3)
    c1.metric("Trend", side_txt); c2.metric("ATR(14)", f"{atr_val:,.2f}"); c3.metric("Last close", f"{close:,.2f}")

    st.subheader("Tiny toy forecast (next session)")
    fc = quick_forecast(df, lookback=lookback)
    if fc:
        d1,d2,d3 = st.columns(3)
        d1.metric("Direction", "Up" if fc["direction"]=="up" else "Down")
        d2.metric("Expected close", f"{fc['expected_close']:,.2f}")
        d3.metric("Confidence-ish", f"{int(fc['confidence']*100)}%")
        st.caption("Simple momentum estimate. Educational only.")
    else:
        st.info("Not enough data to estimate.")

    st.subheader("ML edge (next-day direction)")
    try:
        X = pd.concat([
            pd.Series(df["Close"].pct_change(),   name="RET1"),
            pd.Series(df["Close"].pct_change(5),  name="RET5"),
            pd.Series(df["Close"].pct_change(10), name="RET10"),
            pd.Series((df["High"]-df["Low"])/df["Close"], name="HL_PCT"),
            pd.Series(df["ATR14"]/df["Close"], name="ATR_PCT"),
            df[["EMA20","EMA50","SMA20","SMA50","RSI14","MACD","MACDsig"]],
        ], axis=1).dropna()
        y = (df["Close"].shift(-1) > df["Close"]).astype(int)
        y = y.loc[X.index]
        if len(X) > 200 and y.notna().sum() > 50:
            Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,shuffle=False)
            clf = RandomForestClassifier(n_estimators=300,max_depth=5,random_state=42,n_jobs=-1)
            clf.fit(Xtr,ytr)
            acc = accuracy_score(yte, clf.predict(Xte))
            proba_up = float(clf.predict_proba(X.iloc[[-1]])[0,1])
            imps = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
            m1,m2,m3 = st.columns(3)
            m1.metric("Prob. Up (next day)", f"{proba_up*100:.1f}%")
            m2.metric("Backtest Accuracy*", f"{acc*100:.1f}%")
            m3.metric("Top feature", imps.index[0])
            with st.expander("Feature importances"):
                st.dataframe(imps.to_frame("importance"))
            st.caption("*Toy model, simple split. Educational only.")
        else:
            st.info("Need more history for ML preview (200+ rows).")
    except Exception as e:
        st.warning(f"ML section error: {e}")

    # -------- Seasonality & News Context --------
    st.subheader("Seasonality & News Context (experimental)")
    try:
        # Seasonality
        dow_tbl, mon_tbl = seasonality_tables(df)
        csa, csb = st.columns(2)
        with csa:
            st.markdown("**Avg NEXT-day return by weekday**")
            st.bar_chart(dow_tbl["avg_next_ret"]*100, use_container_width=True)
            st.caption("Percent. Positive = on average, tomorrow leans up after that weekday.")
        with csb:
            st.markdown("**Avg NEXT-day return by month**")
            st.bar_chart(mon_tbl["avg_next_ret"]*100, use_container_width=True)
            this_mon = date.today().strftime("%b")
            if this_mon in mon_tbl.index:
                st.caption(f"Current month ({this_mon}) avg next-day: {mon_tbl.loc[this_mon,'avg_next_ret']*100:.2f}%")

        # Earnings proximity banner
        nxt_dt, dleft = get_upcoming_earnings(ticker)
        if nxt_dt is not None and dleft is not None and dleft <= 14:
            st.warning(f"Earnings in {dleft} day(s): {nxt_dt}. Consider reducing size or skipping swings.")

        # Headlines + sentiment
        news = fetch_rss_headlines(ticker, limit=10)
        if news:
            df_news, overall = score_headlines_vader(news)
            colh1, colh2 = st.columns([2,1])
            with colh1:
                st.markdown("**Latest headlines**")
                for _, r in df_news.iterrows():
                    tag = "🟢" if r["label"]=="Bullish" else ("🔴" if r["label"]=="Bearish" else "⚪")
                    st.markdown(f"{tag} [{r['title']}]({r['link']})  \n"
                                f"<span style='opacity:0.7;font-size:0.85em'>{r['published']}</span>",
                                unsafe_allow_html=True)
            with colh2:
                st.metric("Headline sentiment (avg)", f"{overall:+.2f}")
                st.write(df_news["label"].value_counts())
        else:
            st.info("No headlines fetched (RSS may be blocked/limited).")
    except Exception as e:
        st.info(f"Seasonality/news module skipped: {e}")

    # -------- Trade Plan --------
    st.subheader("Trade Plan (educational)")
    plan = plan_trade(latest, trend_up, atr_val, hh20, ll20, ema20, close,
                      account_size, risk_pct, stop_atr_mult, target_rr, entry_style)

    colA,colB,colC = st.columns(3)
    colA.metric("Suggested Entry", f"{plan['entry']:,.2f}")
    colB.metric("Stop", f"{plan['stop']:,.2f}")
    colC.metric("Target", f"{plan['target']:,.2f}")

    colD,colE,colF = st.columns(3)
    colD.metric("Shares", f"{plan['shares']:,}")
    colE.metric("Risk ($)", f"{plan['risk_dollars']:,.2f}")
    colF.metric("Est. Reward ($)", f"{plan['reward_dollars']:,.2f}")

    st.caption(f"Method: {entry_style} | Stop {stop_atr_mult}x ATR | Target {target_rr}R (R = entry - stop). Approx R:R = {plan['rr']:.2f}")

    ticket_text, json_bytes, csv_bytes, txt_bytes = make_order_ticket(
        ticker=ticker, side_long=trend_up, entry_style=entry_style,
        entry=plan["entry"], stop=plan["stop"], target=plan["target"], shares=plan["shares"],
        rr=plan["rr"], atr_val=atr_val, risk_dollars=plan["risk_dollars"], reward_dollars=plan["reward_dollars"]
    )

    st.subheader("Webull Order Ticket (copy/paste)")
    st.code(ticket_text, language="text")
    cdl1,cdl2,cdl3 = st.columns(3)
    with cdl1: st.download_button("Download TXT",  data=txt_bytes,  file_name=f"{ticker.upper()}_ticket.txt",  mime="text/plain",        key="dl_txt")
    with cdl2: st.download_button("Download JSON", data=json_bytes, file_name=f"{ticker.upper()}_ticket.json", mime="application/json",    key="dl_json")
    with cdl3: st.download_button("Download CSV",  data=csv_bytes,  file_name=f"{ticker.upper()}_ticket.csv",  mime="text/csv",            key="dl_csv")
else:
    st.info("Type a ticker (e.g., AAPL) and press Analyze.")