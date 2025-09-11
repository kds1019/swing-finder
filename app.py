# app.py
# Simple Swing Trading Helper — scanner + screener + plan + ML + seasonality/news
# Run: streamlit run app.py
# NOTE: Educational tool only. Not financial advice.

import warnings, os, io, json, math, time, re
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

# sentiment/news deps
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="Swing Trader (Learn)", layout="wide")
st.title("Simple Swing Trading Helper")
st.caption("For learning only. Not financial advice.")

# Session state
if "ran" not in st.session_state: st.session_state.ran = False
if "last_ai" not in st.session_state: st.session_state.last_ai = ""
if "last_ai_ts" not in st.session_state: st.session_state.last_ai_ts = 0.0

# ---------------- Small helpers ----------------
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

# -------- build_indicators with BANDPOS --------
def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            s = df[c]
            if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
            df[c] = pd.to_numeric(pd.Series(s).squeeze(), errors="coerce")

    # MAs
    df["SMA20"] = sma(df["Close"], 20)
    df["SMA50"] = sma(df["Close"], 50)
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)

    # RSI / MACD / ATR
    df["RSI14"] = rsi(df["Close"], 14)
    m, sig, _ = macd(df["Close"]); df["MACD"] = m; df["MACDsig"] = sig
    df["ATR14"] = atr(df)

    # High/Low windows
    df["HH20"] = df["High"].rolling(20).max()
    df["LL20"] = df["Low"].rolling(20).min()

    # Bollinger + BANDPOS (0 = lower band, 1 = upper band)
    bbw = 20
    bbstd = pd.Series(df["Close"], index=df.index).rolling(bbw).std()
    df["BB_MID"] = sma(df["Close"], bbw)
    df["BB_UP"]  = df["BB_MID"] + 2*bbstd
    df["BB_DN"]  = df["BB_MID"] - 2*bbstd
    width = (df["BB_UP"] - df["BB_DN"]).replace(0, np.nan)
    df["BANDPOS"] = ((df["Close"] - df["BB_DN"]) / width).clip(lower=0, upper=1)

    # Simple buy/sell crosses
    cu = (df["EMA20"]>df["EMA50"]) & (df["EMA20"].shift(1)<=df["EMA50"].shift(1))
    cd = (df["EMA20"]<df["EMA50"]) & (df["EMA20"].shift(1)>=df["EMA50"].shift(1))
    df["buy_signal"]  = cu & (df["RSI14"]>50) & (df["Close"]>df["SMA50"])
    df["sell_signal"] = cd & (df["RSI14"]<50) & (df["Close"]<df["SMA50"])
    return df

# ---------------- Data (ROBUST) ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, years: int = 3) -> pd.DataFrame:
    """
    Robust loader:
    1) Try Yahoo three ways (threads disabled)
    2) If still empty, fall back to Stooq via pandas-datareader
    3) Normalize OHLCV & trim to recent window
    """
    t = (ticker or "").strip().upper()
    if not t:
        return pd.DataFrame()

    def _sym_fix(s: str) -> str:
        return s.replace(".", "-").upper()

    end = date.today() + timedelta(days=1)  # include last trading day
    start = end - timedelta(days=int(years * 366 + 14))

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        if not keep:
            return pd.DataFrame()

        df = df[keep].copy()
        for c in keep:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")

        tz = getattr(df.index, "tz", None)
        if tz is not None:
            df.index = df.index.tz_convert(None)

        df = df.dropna().sort_index()
        if df.empty:
            return df

        approx_rows = int(years * 252) + 40
        return df.tail(approx_rows)

    # ---- Yahoo attempts ----
    attempts = []
    try:
        dfa = yf.download(_sym_fix(t), start=start, end=end, auto_adjust=True,
                          progress=False, threads=False)
        attempts.append(("yahoo:download(start/end)", _normalize(dfa)))
    except Exception:
        attempts.append(("yahoo:download(start/end)", pd.DataFrame()))

    try:
        dfb = yf.Ticker(_sym_fix(t)).history(start=start, end=end, auto_adjust=True)
        attempts.append(("yahoo:ticker.history", _normalize(dfb)))
    except Exception:
        attempts.append(("yahoo:ticker.history", pd.DataFrame()))

    try:
        dfc = yf.download(_sym_fix(t), period="max", auto_adjust=True,
                          progress=False, threads=False)
        attempts.append(("yahoo:download(period=max)", _normalize(dfc)))
    except Exception:
        attempts.append(("yahoo:download(period=max)", pd.DataFrame()))

    for src, df in attempts:
        if df is not None and not df.empty:
            df.attrs["source"] = src
            return df

    # ---- Stooq fallback via pandas-datareader ----
    try:
        from pandas_datareader import data as pdr
        dfS = pdr.DataReader(t.replace("^", ""), "stooq").sort_index()
        if all(c in dfS.columns for c in ["Open","High","Low","Close","Volume"]):
            dfS = dfS[["Open","High","Low","Close","Volume"]]
        norm = _normalize(dfS)
        if norm is not None and not norm.empty:
            norm.attrs["source"] = "stooq"
            return norm
    except Exception:
        pass

    return pd.DataFrame()

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
    if "SMA20" in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
    if "SMA50" in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))
    buys = df[df.get("buy_signal", pd.Series(False, index=df.index))] if "buy_signal" in df.columns else df.iloc[0:0]
    sells = df[df.get("sell_signal", pd.Series(False, index=df.index))] if "sell_signal" in df.columns else df.iloc[0:0]
    if not isinstance(buys, pd.DataFrame): buys = df.iloc[0:0]
    if not isinstance(sells, pd.DataFrame): sells = df.iloc[0:0]
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", marker_symbol="triangle-up", marker_size=10, name="Buy"))
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", marker_symbol="triangle-down", marker_size=10, name="Sell"))
    fig.update_layout(height=600, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                      margin=dict(l=10,r=10,t=30,b=10))
    return fig

# ---------------- Plan & Ticket ----------------
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
        f"Approx R:R: {rr:.2f}" if np.isfinite(rr) else "Approx R:R: n/a",
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

def explain_plan(ticker, trend_up, entry_style, entry, stop, target, rsi_val, atr, ema20, ema50, hh20, ll20):
    bias = "long" if trend_up else "short"
    lines = [
        f"{ticker.upper()} - Plan rationale",
        f"- Bias: {bias} (EMA20 {'>' if trend_up else '<'} EMA50)",
        f"- Entry style: {entry_style} at ~{entry:,.2f}",
        f"- Stop uses ATR ({atr:,.2f}) -> stop ~{stop:,.2f}",
        f"- Target via R multiple -> ~{target:,.2f}",
        f"- Trend context: EMA20 {ema20:,.2f}, EMA50 {ema50:,.2f}"
    ]
    if pd.notna(hh20) and pd.notna(ll20):
        lines.append(f"- Recent 20-day range: {ll20:,.2f} -> {hh20:,.2f}")
    if pd.notna(rsi_val):
        if rsi_val < 40: lines.append("- RSI < 40 -> weak momentum; be conservative.")
        elif rsi_val > 60: lines.append("- RSI > 60 -> firm momentum; breakouts may follow through.")
        else: lines.append("- RSI ~ 40-60 -> neutral; expect chop.")
    return "\n".join(lines)

# ---------------- Robust Watchlist helper (headered or headerless) ----------------
def _candidate_sheet_urls(url: str) -> list[str]:
    u = (url or "").strip()
    u = re.sub(r"\s+", "", u)
    cands = [u]
    if "/pub" in u and "output=csv" in u and "gid=" not in u:
        sep = "&" if "?" in u else "?"
        cands.append(f"{u}{sep}single=true&gid=0")
    seen, out = set(), []
    for x in cands:
        if x and x not in seen:
            out.append(x); seen.add(x)
    return out

def _read_csv_strict(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return pd.read_csv(io.StringIO(r.text))

def load_watchlist_from_url(url: str) -> list[str]:
    last_err = None; tried = []
    for candidate in _candidate_sheet_urls(url):
        tried.append(candidate)
        try:
            df = _read_csv_strict(candidate)
            df.columns = [str(c).strip() for c in df.columns]
            ticker_col = next((c for c in df.columns if c.lower() in ("ticker","symbol")), df.columns[0])
            syms = [str(t).strip().upper() for t in df[ticker_col].tolist() if str(t).strip()]
            if syms:
                return syms
            last_err = "No tickers found"
        except Exception as e:
            last_err = str(e); continue
    st.warning(f"[watchlist] failed to read CSV from URL: {last_err}\nTried: {', '.join(tried)}")
    return []

# ---------------- Universes with fallbacks ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_tickers() -> list[str]:
    fallback = [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","TSLA","AVGO","GOOG",
        "LLY","JPM","V","XOM","JNJ","WMT","UNH","PG","MA","COST","HD","BAC",
        "MRK","ABBV","KO","PEP","CVX","NFLX","ADBE","CSCO","ACN","LIN","CRM",
        "AMD","TMO","MCD","INTC","WFC","ABT","BMY","TXN","COP","PM","IBM",
        "AMAT","NEE","GE","NOW","CAT","ORCL","PFE","QCOM","LOW","BKNG","MS",
        "HON","GS","RTX","SPGI","AXP","BLK","DE","ISRG","PLD","SCHW","UPS"
    ]
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        syms = tables[0]["Symbol"].astype(str).tolist()
        out = [s.replace(".","-").upper().strip() for s in syms if s]
        return out or fallback
    except Exception:
        return fallback

@st.cache_data(ttl=3600, show_spinner=False)
def get_nasdaq100_tickers() -> list[str]:
    fallback = [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","NFLX","ADBE",
        "AMD","TSLA","COST","PEP","INTC","CSCO","AMAT","QCOM","PDD","BKNG","LIN",
        "VRTX","TMUS","LRCX","MU","PANW","REGN","ISRG","ADI","HON","GEHC","ZM",
        "ADP","MDLZ","MAR","KDP","ABNB","FTNT","CRWD","MNST","GILD","KLAC","SNPS",
        "NXPI","CSX","MRVL","CHTR","AEP","ORLY","MELI","ODFL","EA","CDNS"
    ]
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        table = None
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if "ticker" in cols or "symbol" in cols:
                table = t; break
        if table is None: return fallback
        col = [c for c in table.columns if str(c).lower() in ("ticker","symbol")][0]
        syms = table[col].astype(str).tolist()
        out = [s.replace(".","-").upper().strip() for s in syms if s]
        return out or fallback
    except Exception:
        return fallback

@st.cache_data(ttl=3600, show_spinner=False)
def get_dow30_tickers() -> list[str]:
    fallback = [
        "AAPL","AMGN","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS",
        "DOW","GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD",
        "MMM","MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WMT"
    ]
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("symbol" in c for c in cols):
                col = [c for c in t.columns if str(c).lower().startswith("symbol")][0]
                syms = t[col].astype(str).tolist()
                out = [s.replace(".","-").upper().strip() for s in syms if s]
                return out or fallback
    except Exception:
        pass
    return fallback

# -------- Freshness-aware setup logic --------
def good_position_flags(last: pd.Series, prev: pd.Series, mode: str,
                        strictness: str = "Balanced", only_today: bool = False):
    """Return (ok, score, band_pos, reason, fresh_today)."""
    trend_up  = bool(last["EMA20"] > last["EMA50"])
    rsi_v     = float(last["RSI14"])
    rsi_prev  = float(prev["RSI14"]) if "RSI14" in prev else np.nan
    close     = float(last["Close"])
    close_p   = float(prev["Close"]) if "Close" in prev else np.nan
    hh20      = float(last.get("HH20", np.nan))
    hh20_p    = float(prev.get("HH20", np.nan)) if "HH20" in prev else np.nan
    band      = float(last.get("BANDPOS", np.nan))
    band_p    = float(prev.get("BANDPOS", np.nan)) if "BANDPOS" in prev else np.nan

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

    # Pullback window
    in_pull  = trend_up and (pull_rsi_lo <= rsi_v <= pull_rsi_hi) and \
               (pull_band_lo <= (band if np.isfinite(band) else 0.5) <= pull_band_hi)
    was_pull = (bool(prev.get("EMA20", 0) > prev.get("EMA50", 0))) and \
               (pull_rsi_lo <= rsi_prev <= pull_rsi_hi) and \
               (pull_band_lo <= (band_p if np.isfinite(band_p) else 0.5) <= pull_band_hi)
    pull_fresh = in_pull and (not was_pull)

    # Breakout near the 20-day high
    near_high     = np.isfinite(hh20)   and (close   >= brk_near * hh20)   and trend_up and (rsi_v >= brk_rsi_min)
    was_near_high = np.isfinite(hh20_p) and (close_p >= brk_near * hh20_p) and bool(prev.get("EMA20",0) > prev.get("EMA50",0))
    brk_fresh = near_high and (not was_near_high)

    ok = False; score = 0; reasons = []; fresh = False

    if mode in ("Pullback","Both") and in_pull:
        is_fresh = pull_fresh or buy_trigger
        if (not only_today) or is_fresh:
            ok = True; fresh = fresh or is_fresh; score += 2
            reasons.append("pullback-window" + ("-fresh" if is_fresh else ""))

    if mode in ("Breakout","Both") and (buy_trigger or near_high):
        is_fresh = buy_trigger or brk_fresh
        if (not only_today) or is_fresh:
            ok = True; fresh = fresh or is_fresh; score += 2
            reasons.append("buy-signal" if buy_trigger else ("breakout-near-high" + ("-fresh" if is_fresh else "")))

    if trend_up and rsi_v > 50:
        score += 1

    return ok, score, (None if not np.isfinite(band) else round(band,2)), (",".join(reasons) if reasons else "-"), fresh

# -------- Benchmark & RS slope --------
@st.cache_data(ttl=3600, show_spinner=False)
def load_benchmark(years: int = 3, symbol: str = "SPY") -> pd.DataFrame:
    return load_data(symbol, years)

def rs_slope_bps_per_day(df_sym: pd.DataFrame, df_bench: pd.DataFrame, lookback: int = 20) -> float:
    """
    Relative strength series = Close_sym / Close_bench (aligned on dates).
    Returns slope (basis points per day) of a linear regression over last `lookback` points.
    """
    try:
        if df_sym is None or df_sym.empty or df_bench is None or df_bench.empty:
            return 0.0
        s_sym = df_sym["Close"].copy(); s_b = df_bench["Close"].copy()
        idx = s_sym.index.intersection(s_b.index)
        if len(idx) < lookback + 2:
            return 0.0
        rs = (s_sym.loc[idx] / s_b.loc[idx]).dropna().tail(lookback)
        if len(rs) < lookback:
            return 0.0
        x = np.arange(len(rs)).reshape(-1,1)
        y = rs.values
        lr = LinearRegression().fit(x, y)
        last = float(rs.iloc[-1])
        if last == 0:
            return 0.0
        slope_pct_per_day = lr.coef_[0] / last
        return float(slope_pct_per_day * 10000.0)  # bps/day
    except Exception:
        return 0.0

# -------- Pro Score helpers --------
def _lin_slope_bps(series: pd.Series, lookback: int = 20) -> float:
    s = pd.Series(series).dropna().tail(lookback)
    if len(s) < lookback: return np.nan
    x = np.arange(len(s)).reshape(-1,1); y = s.values
    lr = LinearRegression().fit(x, y)
    last = float(s.iloc[-1]) if len(s) else np.nan
    if not np.isfinite(last) or last == 0: return np.nan
    return float(lr.coef_[0] / last * 10000.0)  # basis points/day

def _bb_contraction(df: pd.DataFrame, lookback: int = 120) -> float:
    # Percentile of today's BB width vs last N; return 1 - pct (higher = tighter than usual)
    try:
        w = ((df["BB_UP"] - df["BB_DN"]) / df["Close"]).dropna().tail(lookback)
        if len(w) < 20: return np.nan
        ranks = w.rank(pct=True)
        return float(1.0 - ranks.iloc[-1])
    except Exception:
        return np.nan

def _safe_pct(s: pd.Series, n: int) -> float:
    try: return float(s.pct_change(n).iloc[-1])
    except Exception: return np.nan

def _pro_metrics(df: pd.DataFrame, bench_df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    close = float(last["Close"])
    # Relative strength vs SPY
    rs_bps = rs_slope_bps_per_day(df, bench_df, lookback=20)
    # Momentum horizons
    ret20 = _safe_pct(df["Close"], 20)
    ret60 = _safe_pct(df["Close"], 60)
    # Trend quality
    ema20_slope = _lin_slope_bps(df["EMA20"], 20)
    # Volatility contraction
    bb_contr = _bb_contraction(df, 120)  # 0..1 (higher is tighter)
    # Participation today (RVOL)
    adv20 = float(df["Volume"].rolling(20, min_periods=1).mean().iloc[-1])
    rvol20 = float(last["Volume"]) / adv20 if adv20 > 0 else np.nan
    # Range expansion vs ATR
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else np.nan
    tr_today = max(
        float(last["High"] - last["Low"]),
        abs(float(last["High"]) - prev_close) if np.isfinite(prev_close) else 0.0,
        abs(float(last["Low"]) - prev_close) if np.isfinite(prev_close) else 0.0
    )
    atr = float(last.get("ATR14", np.nan))
    range_atr = tr_today / atr if atr and np.isfinite(atr) and atr > 0 else np.nan
    # Proximity to ~200d high (or available window)
    win = min(252, len(df))
    hi = float(df["Close"].rolling(win, min_periods=max(40, win//2)).max().iloc[-1]) if win >= 40 else np.nan
    prox_200h = (close / hi) if (np.isfinite(hi) and hi > 0) else np.nan
    return {
        "RS_slope(bps/day)": rs_bps,
        "RET20": ret20,
        "RET60": ret60,
        "EMA20_slope(bps/day)": ema20_slope,
        "BB_CONTR": bb_contr,
        "RVOL20": rvol20,
        "RANGE_ATR": range_atr,
        "PROX_200D_HIGH": prox_200h,
    }

def _apply_pro_score(df: pd.DataFrame) -> pd.DataFrame:
    # Convert raw metrics to 0..1 ranks (percentiles) across the candidate set
    def pr(col):
        x = df[col].astype(float)
        return x.rank(pct=True, na_option="keep")
    # guard missing columns
    for c in ["RS_slope(bps/day)","RET20","RET60","EMA20_slope(bps/day)","BB_CONTR","RVOL20","RANGE_ATR","PROX_200D_HIGH"]:
        if c not in df.columns: df[c] = np.nan

    df["rank_rs"]     = pr("RS_slope(bps/day)")
    df["rank_ret20"]  = pr("RET20")
    df["rank_ret60"]  = pr("RET60")
    df["rank_ema20s"] = pr("EMA20_slope(bps/day)")
    df["rank_bb"]     = df["BB_CONTR"].astype(float)  # already 0..1
    df["rank_rvol"]   = pr("RVOL20")
    df["rank_range"]  = pr("RANGE_ATR")
    df["rank_prox"]   = pr("PROX_200D_HIGH")

    # Weighted composite (sum to 1.0)
    df["Score100"] = 100.0 * (
        0.25*df["rank_rs"]   + 0.15*df["rank_ret20"] + 0.10*df["rank_ret60"] +
        0.15*df["rank_ema20s"] + 0.10*df["rank_bb"] + 0.15*df["rank_rvol"] +
        0.10*df["rank_range"] + 0.10*df["rank_prox"]
    )
    # Fresh setups get a small kicker
    if "Fresh" in df.columns:
        df.loc[df["Fresh"]==True, "Score100"] = df["Score100"] + 5.0
    return df

# -------- Scanners (freshness + quality filters + Pro Score + fallback) --------
@st.cache_data(ttl=900, show_spinner=True)
def scan_universe(universe: str, years: int, price_min: float, price_max: float,
                  min_dollar_vol_millions: float, mode: str, strictness: str = "Balanced",
                  only_today: bool = False, max_symbols: int = 100,
                  min_rs_slope: float = 0.0, min_vol_spike: float = 0.0, exclude_earn_days: int = 0):
    if universe == "S&P 500":
        tickers = get_sp500_tickers()
    elif universe == "NASDAQ-100":
        tickers = get_nasdaq100_tickers()
    elif universe == "DOW 30":
        tickers = get_dow30_tickers()
    else:
        tickers = []
    tickers = tickers[:max_symbols]

    bench_df = load_benchmark(years)

    stats = {"universe": universe, "start": len(tickers), "after_price": 0,
             "after_liquidity": 0, "ok": 0, "near": 0, "errors": 0, "sample": tickers[:10]}

    rows = []; near_rows = []; fallback_pool = []

    for t in tickers:
        try:
            df = load_data(t, years)
            if df.empty or len(df) < 2:
                continue
            df = build_indicators(df)
            if len(df) < 2:
                continue
            last, prev = df.iloc[-1], df.iloc[-2]
            close = float(last["Close"])

            if not (price_min <= close <= price_max):
                continue
            stats["after_price"] += 1

            adv = (df["Close"]*df["Volume"]).rolling(20).mean().iloc[-1]
            adv_m = float(adv)/1_000_000.0 if pd.notna(adv) else 0.0
            if adv_m < min_dollar_vol_millions:
                continue
            stats["after_liquidity"] += 1

            # PRO metrics (for both ranking and fallback)
            pro = _pro_metrics(df, bench_df)
            rs_bps = pro["RS_slope(bps/day)"]

            # add to fallback pool early (after basic gates)
            fallback_pool.append({
                "Ticker": t, "Close": round(close,2), "Avg$Vol(20d, M)": round(adv_m,2),
                "RSI14": round(float(last["RSI14"]),1),
                "Fresh": False,  # will be True for final matches only
                **pro
            })

            # --- Quality filters ---
            if rs_bps < min_rs_slope:
                continue

            if min_vol_spike > 0:
                vol = float(last.get("Volume", np.nan))
                adv20 = float(df["Volume"].rolling(20, min_periods=1).mean().iloc[-1])
                if not (np.isfinite(vol) and adv20 > 0 and (vol >= min_vol_spike * adv20)):
                    continue

            if exclude_earn_days > 0:
                nxt_dt, dleft = get_upcoming_earnings(t)
                if (nxt_dt is not None) and (dleft is not None) and (dleft <= exclude_earn_days):
                    continue

            ok, score, band, why, fresh = good_position_flags(last, prev, mode, strictness, only_today)
            base = {"Ticker":t,"Close":round(close,2),"Avg$Vol(20d, M)":round(adv_m,2),
                    "TrendUp":bool(last["EMA20"]>last["EMA50"]),
                    "RSI14":round(float(last["RSI14"]),1),
                    "BandPos(0=low,1=high)": band,
                    "BUY?":bool(last.get("buy_signal",False)),"Score":int(score),
                    "Fresh": bool(fresh)}
            base.update(pro)

            if ok:
                base["Reason"]=why; rows.append(base); stats["ok"] += 1
            else:
                # near-miss bucket
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

    if rows:
        out = pd.DataFrame(rows).copy()
        out = _apply_pro_score(out)
        out = out.sort_values(
            ["Fresh","Score100","RS_slope(bps/day)","Avg$Vol(20d, M)","RSI14"],
            ascending=[False,False,False,False,False]
        ).reset_index(drop=True)
        out.insert(0,"Rank",np.arange(1,len(out)+1))
        return out, stats

    if near_rows:
        out = pd.DataFrame(near_rows).copy()
        out = _apply_pro_score(out)
        out = out.sort_values(
            ["Fresh","Score100","RS_slope(bps/day)","Avg$Vol(20d, M)","RSI14"],
            ascending=[False,False,False,False,False]
        ).head(20).reset_index(drop=True)
        out.insert(0,"Rank",np.arange(1,len(out)+1))
        return out, stats

    # Fallback: top-20 by Pro Score (or RS if NaNs)
    if fallback_pool:
        out = pd.DataFrame(fallback_pool).copy()
        out = _apply_pro_score(out)
        out["Reason"] = "fallback-top-proscore"
        out = out.sort_values(
            ["Score100","RS_slope(bps/day)","Avg$Vol(20d, M)"],
            ascending=[False,False,False]
        ).head(20).reset_index(drop=True)
        out.insert(0,"Rank",np.arange(1,len(out)+1))
        return out, stats

    return pd.DataFrame(), stats


@st.cache_data(ttl=600, show_spinner=True)
def scan_fixed_list(ticks: list[str], years: int, price_min: float, price_max: float,
                    min_dollar_vol_millions: float, mode: str, strictness: str,
                    only_today: bool = False, max_symbols: int = 100,
                    min_rs_slope: float = 0.0, min_vol_spike: float = 0.0, exclude_earn_days: int = 0):
    rows = []; near_rows = []
    ticks = [t.strip().upper() for t in ticks if t.strip()][:max_symbols]
    bench_df = load_benchmark(years)
    stats = {"universe":"Custom","start":len(ticks),"after_price":0,"after_liquidity":0,
             "ok":0,"near":0,"errors":0,"sample":ticks[:10]}
    fallback_pool = []

    for t in ticks:
        try:
            df = load_data(t, years)
            if df.empty or len(df) < 2:
                continue
            df = build_indicators(df)
            if len(df) < 2:
                continue
            last, prev = df.iloc[-1], df.iloc[-2]
            close = float(last["Close"])
            if not (price_min <= close <= price_max):
                continue
            stats["after_price"] += 1

            adv = (df["Close"]*df["Volume"]).rolling(20).mean().iloc[-1]
            adv_m = float(adv)/1_000_000.0 if pd.notna(adv) else 0.0
            if adv_m < min_dollar_vol_millions:
                continue
            stats["after_liquidity"] += 1

            pro = _pro_metrics(df, bench_df)
            rs_bps = pro["RS_slope(bps/day)"]

            fallback_pool.append({
                "Ticker": t, "Close": round(close,2), "Avg$Vol(20d, M)": round(adv_m,2),
                "RSI14": round(float(last["RSI14"]),1),
                "Fresh": False,
                **pro
            })

            # --- Quality filters ---
            if rs_bps < min_rs_slope:
                continue

            if min_vol_spike > 0:
                vol = float(last.get("Volume", np.nan))
                adv20 = float(df["Volume"].rolling(20, min_periods=1).mean().iloc[-1])
                if not (np.isfinite(vol) and adv20 > 0 and (vol >= min_vol_spike * adv20)):
                    continue

            if exclude_earn_days > 0:
                nxt_dt, dleft = get_upcoming_earnings(t)
                if (nxt_dt is not None) and (dleft is not None) and (dleft <= exclude_earn_days):
                    continue

            ok, score, band, why, fresh = good_position_flags(last, prev, mode, strictness, only_today)
            base = {"Ticker":t,"Close":round(close,2),"Avg$Vol(20d, M)":round(adv_m,2),
                    "TrendUp":bool(last["EMA20"]>last["EMA50"]),
                    "RSI14":round(float(last["RSI14"]),1),
                    "BandPos(0=low,1=high)": band,
                    "BUY?":bool(last.get("buy_signal",False)),"Score":int(score),
                    "Fresh": bool(fresh)}
            base.update(pro)

            if ok:
                base["Reason"]=why; rows.append(base); stats["ok"] += 1
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

    if rows:
        out = pd.DataFrame(rows).copy()
        out = _apply_pro_score(out)
        out = out.sort_values(
            ["Fresh","Score100","RS_slope(bps/day)","Avg$Vol(20d, M)","RSI14"],
            ascending=[False,False,False,False,False]
        ).reset_index(drop=True)
        out.insert(0,"Rank",np.arange(1,len(out)+1)); return out, stats

    if near_rows:
        out = pd.DataFrame(near_rows).copy()
        out = _apply_pro_score(out)
        out = out.sort_values(
            ["Fresh","Score100","RS_slope(bps/day)","Avg$Vol(20d, M)","RSI14"],
            ascending=[False,False,False,False,False]
        ).head(20).reset_index(drop=True)
        out.insert(0,"Rank",np.arange(1,len(out)+1)); return out, stats

    if fallback_pool:
        out = pd.DataFrame(fallback_pool).copy()
        out = _apply_pro_score(out)
        out["Reason"] = "fallback-top-proscore"
        out = out.sort_values(
            ["Score100","RS_slope(bps/day)","Avg$Vol(20d, M)"],
            ascending=[False,False,False]
        ).head(20).reset_index(drop=True)
        out.insert(0,"Rank",np.arange(1,len(out)+1)); return out, stats

    return pd.DataFrame(), stats

# ---------------- Local AI helpers (Ollama) ----------------
def ollama_available():
    try: return requests.get("http://127.0.0.1:11434/api/tags", timeout=2).status_code == 200
    except Exception: return False

def list_ollama_models():
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        r.raise_for_status()
        return [m.get("name") for m in r.json().get("models", []) if m.get("name")]
    except Exception:
        return []

def ask_local_llm(model_name: str, system_text: str, user_text: str, max_retries: int = 3) -> str:
    url = "http://127.0.0.1:11434/api/chat"
    payload = {"model":model_name, "messages":[{"role":"system","content":system_text},{"role":"user","content":user_text}], "stream":False}
    fallbacks = ["llama3.2:1b","tinyllama:1.1b","qwen2.5:0.5b"]; tried=[model_name]; last=None
    for i in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=120)
            if r.status_code == 200:
                return r.json().get("message", {}).get("content", "").strip()
            txt = r.text or ""
            if r.status_code == 500 and "requires more system memory" in txt:
                for fb in fallbacks:
                    if fb in tried: continue
                    tried.append(fb); payload["model"] = fb
                    rr = requests.post(url, json=payload, timeout=120)
                    if rr.status_code == 200:
                        return f"(Used fallback {fb})\n\n" + rr.json().get("message", {}).get("content", "").strip()
                return "(Local AI chat error: model too large; use a smaller model.)"
            if r.status_code in (404, 400) and ("not found" in txt.lower() or "pull" in txt.lower()):
                return "(Model not found. In PowerShell run:  ollama pull llama3.2:1b  then select it.)"
            last = f"HTTP {r.status_code}: {txt[:200]}"
        except Exception as e:
            last = str(e)
        time.sleep(2**i)
    return f"(Local AI chat error after retries: {last})"

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

# ---------------- Sidebar ----------------
# ---------------- Sidebar (with one-click presets) ----------------
with st.sidebar:
    ticker    = st.text_input("Stock ticker (e.g., AAPL, MSFT, TSLA)", "AAPL", key="main_ticker_input")
    years     = st.slider("Years of history", 1, 10, 3, key="main_years_slider")
    lookback  = st.slider("Forecast lookback (days)", 5, 60, 20, key="main_lookback_slider")
    st.caption("Tip: some tickers need suffixes like VOD.L or SHOP.TO.")

    st.markdown("---")
    st.subheader("Risk & Plan")
    account_size  = st.number_input("Account size ($)", min_value=0.0, value=10000.0, step=100.0, key="main_account_size")
    risk_pct      = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, step=0.1, key="main_risk_pct")
    stop_atr_mult = st.slider("Stop distance (x ATR)", 0.5, 5.0, 1.5, step=0.1, key="main_stop_atr_mult")
    target_rr     = st.slider("Target multiple (R:R)", 0.5, 5.0, 2.0, step=0.5, key="main_target_rr")
    entry_style   = st.selectbox("Entry style", ["Pullback to EMA20","Breakout 20-day"], key="main_entry_style")

    # Watchlist via URL (optional)
    WATCHLIST_URL = get_secret("WATCHLIST_URL", None)
    default_wl = ["AAPL","MSFT","NVDA","TSLA","META"]
    if WATCHLIST_URL:
        wl_from_url = load_watchlist_from_url(WATCHLIST_URL)
        if wl_from_url: default_wl = wl_from_url

    st.markdown("---")
    st.subheader("Watchlist Screener")
    wl_input     = st.text_area("Comma-separated tickers", ", ".join(default_wl), height=80, key="main_wl_input")
    screen_years = st.slider("Screener lookback (years)", 1, 5, 1, key="main_screen_years")
    run_screen   = st.button("Run Screener", key="btn_run_screen")

    st.markdown("---")
    st.subheader("Market Scanner (no watchlist)")
    universe     = st.selectbox("Universe", ["S&P 500","NASDAQ-100","DOW 30"], index=0, key="scan_universe")
    scan_years   = st.slider("History (years)", 1, 5, 2, key="scan_years")
    price_min    = st.number_input("Min price ($)", 0.0, value=5.0, step=0.5, key="scan_price_min")
    price_max    = st.number_input("Max price ($)", 0.0, value=500.0, step=0.5, key="scan_price_max")
    liq          = st.number_input("Min Avg $ Volume (20d, M)", 0.0, value=1.0, step=0.5, key="scan_min_dollar_vol")
    setup_mode   = st.radio("Setup type", ["Both","Pullback","Breakout"], horizontal=True, key="scan_mode")
    strictness   = st.selectbox("Scanner strictness", ["Balanced","Strict","Loose"], index=0, key="scan_strictness")
    max_symbols  = st.slider("Max symbols to scan", 20, 200, 100, step=10, key="scan_max_symbols")

    # Freshness toggle
    only_today  = st.checkbox("Only show **new today** setups", value=False, key="scan_only_today",
                              help="Require the condition to become true today vs yesterday.")

    # Quality filters (keys added so presets can set them)
    st.markdown("**Quality filters**")
    min_rs_slope = st.slider("Min RS slope vs SPY (20d, bp/day)", -5.0, 5.0, 0.0, 0.1,
                             key="scan_min_rs_slope",
                             help="Relative strength slope of (Close/SPY). >0 means outperforming.")
    min_vol_spike = st.slider("Min volume spike × 20d avg (0 = off)", 0.0, 5.0, 0.0, 0.1,
                              key="scan_min_vol_spike",
                              help="Require Volume ≥ this multiple of 20d average.")
    exclude_earn_days = st.slider("Exclude if earnings within N days (0 = off)", 0, 30, 0,
                                  key="scan_exclude_earn_days",
                                  help="Skips symbols with upcoming earnings inside this window.")

    custom_universe = st.text_area("Custom universe (optional, comma tickers)", "", height=60, key="scan_custom_universe")
    debug_scan   = st.checkbox("Show scanner debug", value=False, key="scan_debug")
    run_market_scan = st.button("Scan Market", key="btn_scan_market")

    st.markdown("---")
    # -------- Presets --------
    st.subheader("Presets")

    def _apply_preset(vals: dict):
        for k, v in vals.items():
            st.session_state[k] = v
        st.rerun()

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("🔵 Breakout Momentum", use_container_width=True):
            _apply_preset({
                "scan_mode": "Breakout",
                "scan_strictness": "Balanced",
                "scan_only_today": True,
                "scan_price_min": 10.0,
                "scan_price_max": 60.0,
                "scan_min_dollar_vol": 2.0,
                "scan_max_symbols": 150,
                "scan_min_rs_slope": 1.0,     # ≥ +1.0 bp/day
                "scan_min_vol_spike": 1.5,    # ≥ 1.5× RVOL
                "scan_exclude_earn_days": 7,
                "main_entry_style": "Breakout 20-day",
            })
    with col_p2:
        if st.button("🟢 Pullback Uptrend", use_container_width=True):
            _apply_preset({
                "scan_mode": "Pullback",
                "scan_strictness": "Balanced",
                "scan_only_today": False,
                "scan_price_min": 5.0,
                "scan_price_max": 200.0,
                "scan_min_dollar_vol": 1.0,
                "scan_max_symbols": 200,
                "scan_min_rs_slope": 0.4,
                "scan_min_vol_spike": 0.0,
                "scan_exclude_earn_days": 5,
                "main_entry_style": "Pullback to EMA20",
            })

    col_p3, col_p4 = st.columns(2)
    with col_p3:
        if st.button("🟣 Post-Earnings Momentum", use_container_width=True):
            _apply_preset({
                "scan_mode": "Both",
                "scan_strictness": "Balanced",
                "scan_only_today": True,
                "scan_price_min": 5.0,
                "scan_price_max": 500.0,
                "scan_min_dollar_vol": 1.0,
                "scan_max_symbols": 200,
                "scan_min_rs_slope": 0.5,
                "scan_min_vol_spike": 2.0,
                "scan_exclude_earn_days": 0,   # allow earnings plays
                "main_entry_style": "Breakout 20-day",
            })
    with col_p4:
        if st.button("⚪ Baseline (relaxed)", use_container_width=True):
            _apply_preset({
                "scan_mode": "Both",
                "scan_strictness": "Loose",
                "scan_only_today": False,
                "scan_price_min": 5.0,
                "scan_price_max": 500.0,
                "scan_min_dollar_vol": 1.0,
                "scan_max_symbols": 200,
                "scan_min_rs_slope": 0.0,
                "scan_min_vol_spike": 0.0,
                "scan_exclude_earn_days": 0,
            })

    st.markdown("---")
    if st.button("Force data refresh"):
        st.cache_data.clear()
        st.rerun()

    # Local AI toggle
    default_ai = os.environ.get("ENABLE_LOCAL_AI","false").lower() == "true"
    AI_ENABLED = st.checkbox("Enable Local AI (Ollama)", value=default_ai,
                             help="Runs only on your machine with Ollama. Not for Streamlit Cloud.",
                             key="ai_enable_toggle")
    if AI_ENABLED:
        st.subheader("Local AI (Ollama)")
        defaults  = ["llama3.2:1b","tinyllama:1.1b","qwen2.5:0.5b","llama3.2:3b","phi3:3.8b-mini"]
        installed = list_ollama_models()
        choices   = [m for m in defaults if (not installed) or m in installed]
        model_name = st.selectbox("Model", choices, index=0, key="ai_model_select")
        if installed: st.caption("Installed: " + ", ".join(installed[:8]) + ("..." if len(installed)>8 else ""))
        if ollama_available(): st.success("Ollama detected.")
        else: st.info("Ollama not detected. Install from ollama.com, then: ollama pull llama3.2:1b")


# ---------------- Top Buttons ----------------
if st.button("Analyze", key="btn_analyze"):
    st.session_state.ran = True
st.button("New analysis (reset)", key="btn_reset",
          on_click=lambda: st.session_state.update(ran=False, last_ai="", last_ai_ts=0.0))

# ---------------- Watchlist Screener output ----------------
if run_screen:
    tickers = [t.strip().upper() for t in wl_input.split(",") if t.strip()]
    if not tickers:
        st.warning("No tickers provided.")
    else:
        rows = []
        for t in tickers:
            sdf = load_data(t, years=screen_years)
            if sdf.empty: continue
            sdf = build_indicators(sdf); last = sdf.iloc[-1]
            close = float(last["Close"])
            trend_up = bool(last["EMA20"] > last["EMA50"])
            buy  = bool(last["buy_signal"]); sell = bool(last["sell_signal"])
            rsi_v = float(last["RSI14"])
            bb_dn = float(last.get("BB_DN", np.nan)); bb_up = float(last.get("BB_UP", np.nan)); band = np.nan
            if np.isfinite(bb_dn) and np.isfinite(bb_up) and (bb_up-bb_dn)>0:
                band = (close - bb_dn)/(bb_up - bb_dn)
            score = (1 if buy else 0) - (1 if sell else 0) + (1 if (trend_up and rsi_v>50) else 0)
            rows.append({"Ticker":t,"Close":round(close,2),"TrendUp":trend_up,"RSI14":round(rsi_v,1),
                         "BandPos(0=low,1=high)": None if np.isnan(band) else round(band,2),
                         "BUY?":buy,"SELL?":sell,"Score":score})
        if rows:
            df_screen = pd.DataFrame(rows).sort_values(["Score","RSI14"], ascending=[False,False]).reset_index(drop=True)
            st.subheader("Watchlist Screener Results")
            st.dataframe(df_screen, use_container_width=True)
        else:
            st.info("No watchlist candidates met your screener filters.")

# ---------------- Market Scanner output ----------------
if run_market_scan:
    if custom_universe.strip():
        tickers_override = [t.strip().upper() for t in custom_universe.split(",") if t.strip()]
        res, stats = scan_fixed_list(
            tickers_override, scan_years, price_min, price_max, liq,
            setup_mode, strictness, only_today, max_symbols,
            min_rs_slope=min_rs_slope, min_vol_spike=min_vol_spike, exclude_earn_days=exclude_earn_days
        )
        universe_label = f"Custom ({len(tickers_override)})"
    else:
        res, stats = scan_universe(
            universe, scan_years, price_min, price_max, liq,
            setup_mode, strictness, only_today, max_symbols,
            min_rs_slope=min_rs_slope, min_vol_spike=min_vol_spike, exclude_earn_days=exclude_earn_days
        )
        universe_label = universe

    st.subheader(f"Market Scanner Results — {universe_label} ({strictness})")
    st.caption(f"Filters: {universe_label} | ${price_min:.2f} to ${price_max:.2f} | Min Avg $ Vol(20d): {liq:.1f}M "
               f"| Mode: {setup_mode} | Strictness: {strictness} | Scanned: {stats['start']} "
               f"| RS slope≥{min_rs_slope:.1f} bp/d | Vol spike≥{min_vol_spike:.1f}× | Excl earnings≤{exclude_earn_days}d")

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

    if res is None or res.empty:
        st.info("No candidates met your filters. Try Strictness = Loose, lower Min Avg $ Vol, widen price, "
                "reduce RS slope/vol spike thresholds, or use a Custom universe.")
    else:
        st.dataframe(res, use_container_width=True)

# ---------------- Single Ticker analysis ----------------
if st.session_state.ran:
    df = load_data(ticker, years)
    if df.empty:
        st.error("No data found. Some tickers need exchange suffixes like .L, .TO, .SA.")
        st.stop()
    df = build_indicators(df)

    # show source & last bar date
    src = df.attrs.get("source","yahoo")
    try:
        last_ts = pd.to_datetime(df.index, errors="coerce").max()
        last_dt = pd.Timestamp(last_ts).tz_localize(None).date() if pd.notna(last_ts) else None
        st.caption(f"Data source: {src}" + (f" | Last bar: {last_dt}" if last_dt else ""))
    except Exception:
        st.caption(f"Data source: {src}")

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
                st.write(imps.to_frame("importance"))
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
    latest  = df.iloc[-1]
    trend_up = latest["EMA20"] > latest["EMA50"]
    atr_val = float(latest["ATR14"]); close = float(latest["Close"])
    ema20   = float(latest["EMA20"]); ema50 = float(latest["EMA50"])
    hh20    = float(latest.get("HH20", np.nan)); ll20 = float(latest.get("LL20", np.nan))
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

    rr_val = plan.get("rr", float("nan"))
    rr_str = f"{rr_val:.2f}" if (isinstance(rr_val, (int, float)) and np.isfinite(rr_val)) else "n/a"
    st.caption(f"Method: {entry_style} | Stop {stop_atr_mult}x ATR | Target {target_rr}R (R = entry - stop). Approx R:R = {rr_str}")

    ticket_text, json_bytes, csv_bytes, txt_bytes = make_order_ticket(
        ticker=ticker, side_long=trend_up, entry_style=entry_style,
        entry=plan['entry'], stop=plan['stop'], target=plan['target'], shares=plan['shares'],
        rr=plan['rr'], atr_val=atr_val, risk_dollars=plan['risk_dollars'], reward_dollars=plan['reward_dollars']
    )

    st.subheader("Webull Order Ticket (copy/paste)")
    st.code(ticket_text, language="text")
    cdl1,cdl2,cdl3 = st.columns(3)
    with cdl1: st.download_button("Download TXT",  data=txt_bytes,  file_name=f"{ticker.upper()}_ticket.txt",  mime="text/plain",        key="dl_txt")
    with cdl2: st.download_button("Download JSON", data=json_bytes, file_name=f"{ticker.upper()}_ticket.json", mime="application/json",    key="dl_json")
    with cdl3: st.download_button("Download CSV",  data=csv_bytes,  file_name=f"{ticker.upper()}_ticket.csv",  mime="text/csv",            key="dl_csv")

    st.subheader("Plan explainer")
    st.markdown(explain_plan(ticker, trend_up, entry_style, plan["entry"], plan["stop"], plan["target"],
                             float(latest["RSI14"]), atr_val, ema20, ema50, hh20, ll20))
else:
    st.info("Type a ticker (e.g., AAPL) and press Analyze.")
