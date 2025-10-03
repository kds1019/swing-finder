# app.py — SwingFinder (Part 1/3)
# Educational tool only. Not financial advice.

import warnings, os, io, json, math
warnings.filterwarnings("ignore")

import datetime as dt
from datetime import date, timedelta
from dataclasses import dataclass

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="SwingFinder", layout="wide")
st.title("📈 SwingFinder — Screener + Watchlists")
st.caption("Simple swing helper. Webull-like compatibility + explain-why table. No broker login.")

# Session state
if "ran" not in st.session_state: 
    st.session_state.ran = False

# ---------------- Small helpers ----------------
def _as_series(x):
    if isinstance(x, pd.DataFrame): 
        return x.iloc[:, 0]
    return pd.Series(x).squeeze()

def app_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

WATCHLISTS_PATH = os.path.join(app_dir(), "watchlists.json")

# ---------------- Indicators ----------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return pd.Series(s, index=s.index).ewm(span=span, adjust=False).mean()

def sma(s: pd.Series, window: int) -> pd.Series:
    return pd.Series(s, index=s.index).rolling(window).mean()

def rsi(series, period: int = 14) -> pd.Series:
    if isinstance(series, pd.DataFrame): 
        series = series.iloc[:, 0]
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
            if isinstance(s, pd.DataFrame): 
                s = s.iloc[:, 0]
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

    # Bollinger + BANDPOS (0..1)
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

# ---------------- Data loader ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, years: int = 3) -> pd.DataFrame:
    t = (ticker or "").strip().upper()
    if not t:
        return pd.DataFrame()

    def _sym_fix(s: str) -> str:
        return s.replace(".", "-").upper()

    end = date.today() + timedelta(days=1)
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
    return pd.DataFrame()

# ---------------- Universes ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_tickers() -> list[str]:
    return [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","TSLA","GOOG","LLY","JPM","V","XOM","JNJ",
        "WMT","UNH","PG","MA","COST","HD","BAC","MRK","ABBV","KO","PEP","CVX","NFLX","ADBE","CSCO","ACN",
        "LIN","CRM","AMD","TMO","MCD","INTC","WFC","ABT","BMY","TXN","COP","PM","IBM","AMAT","NEE","GE",
        "NOW","CAT","ORCL","PFE","QCOM","LOW","MS","HON","RTX","SPGI","AXP","BLK","DE","ISRG","PLD","SCHW","UPS"
    ]

@st.cache_data(ttl=3600, show_spinner=False)
def get_nasdaq100_tickers() -> list[str]:
    return [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","NFLX","ADBE",
        "AMD","TSLA","COST","PEP","INTC","CSCO","AMAT","QCOM","PDD","BKNG","LIN",
        "VRTX","TMUS","LRCX","MU","PANW","REGN","ISRG","ADI","HON","GEHC","ZM",
        "ADP","MDLZ","MAR","KDP","ABNB","FTNT","CRWD","MNST","GILD","KLAC","SNPS",
        "NXPI","CSX","MRVL","CHTR","AEP","ORLY","MELI","ODFL","EA","CDNS"
    ]

@st.cache_data(ttl=3600, show_spinner=False)
def get_dow30_tickers() -> list[str]:
    return [
        "AAPL","AMGN","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS",
        "DOW","GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD",
        "MMM","MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WMT"
    ]

def combined_universe():
    s = set(get_sp500_tickers()) | set(get_nasdaq100_tickers()) | set(get_dow30_tickers())
    return sorted(s)

# ---------------- Compatibility (Webull-like) helpers ----------------
@dataclass
class ScreenerConfig:
    use_last_closed_bar: bool = True
    indicator_source: str = "close"
    # thresholds from strictness
    rsi_lo: int = 40
    rsi_hi: int = 65
    brk_near: float = 0.990
    brk_rsi_min: int = 48
    # meta
    near_miss_bps: int = 30
    avgvol_window: int = 20
    require_all: bool = True

def apply_webull_like_from_strictness(strictness_label: str, near_miss_bps: int) -> ScreenerConfig:
    cfg = ScreenerConfig()
    if strictness_label == "Strict":
        cfg.rsi_lo, cfg.rsi_hi = 45, 60
        cfg.brk_near, cfg.brk_rsi_min = 0.995, 50
    elif strictness_label == "Loose":
        cfg.rsi_lo, cfg.rsi_hi = 35, 70
        cfg.brk_near, cfg.brk_rsi_min = 0.980, 45
    else:  # Balanced
        cfg.rsi_lo, cfg.rsi_hi = 40, 65
        cfg.brk_near, cfg.brk_rsi_min = 0.990, 48
    cfg.near_miss_bps = int(near_miss_bps)
    return cfg

def _passes_rules_row(last: pd.Series, cfg: ScreenerConfig) -> tuple[bool, dict]:
    eps = float(last["Close"]) * (cfg.near_miss_bps / 10_000.0)  # bps -> price units

    ema20 = float(last["EMA20"]); ema50 = float(last["EMA50"])
    rsi14 = float(last["RSI14"]) if pd.notna(last["RSI14"]) else np.nan
    close = float(last["Close"])
    hh20  = float(last.get("HH20", np.nan))
    band  = float(last.get("BANDPOS", np.nan))

    trend_up = ema20 > ema50
    checks = {
        "trend_up": trend_up,
        "rsi_band": (rsi14 >= (cfg.rsi_lo - 0.5)) and (rsi14 <= (cfg.rsi_hi + 0.5)),
        "price_above_ema20": close > (ema20 - eps),
        "ema20_gt_ema50": (ema20 - ema50) > -1e-9,
        "near_20d_high": (np.isfinite(hh20) and trend_up and (close >= cfg.brk_near * hh20) and (rsi14 >= cfg.brk_rsi_min)),
        "bandpos_ok": (0.0 <= (band if np.isfinite(band) else 0.5) <= 1.0),
        "macd_hist_nonneg": (float(last["MACD"]) - float(last["MACDsig"])) >= (-eps / max(close, 1e-6)),
    }
    ok = all(checks.values()) if cfg.require_all else (sum(bool(v) for v in checks.values()) >= 5)
    return ok, checks

def _pro_metrics(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    close = float(last["Close"])
    win = min(60, len(df))
    if win >= 10:
        base = float(df["Close"].iloc[-win])
        rs_slope = (close / base) ** (1/win) - 1.0
    else:
        rs_slope = 0.0
    adv20 = float(df["Volume"].rolling(20, min_periods=1).mean().iloc[-1])
    rvol20 = float(last["Volume"]) / adv20 if adv20 > 0 else np.nan
    return {"RS_slope(bps/day)": float(rs_slope * 10000.0), "RVOL20": rvol20}

def _explain_failures(ticker: str, checks: dict, last: pd.Series,
                      price_min: float, price_max: float, adv_m: float,
                      min_dollar_vol_millions: float) -> dict:
    reasons = []
    close = float(last["Close"])
    if close < price_min: reasons.append(f"Price {close:.2f} < min {price_min:.2f}")
    if close > price_max: reasons.append(f"Price {close:.2f} > max {price_max:.2f}")
    if adv_m < min_dollar_vol_millions: reasons.append(f"Avg$Vol20 {adv_m:.2f}M < min {min_dollar_vol_millions:.2f}M")
    for k, v in checks.items():
        if v: 
            continue
        if k == "trend_up":
            reasons.append("EMA20 ≤ EMA50 (no uptrend)")
        elif k == "rsi_band":
            reasons.append("RSI14 outside band")
        elif k == "price_above_ema20":
            reasons.append("Close not above EMA20 (within band?)")
        elif k == "ema20_gt_ema50":
            reasons.append("EMA20 ≤ EMA50")
        elif k == "near_20d_high":
            reasons.append("Not near 20-day high (or RSI too low)")
        elif k == "bandpos_ok":
            reasons.append("BandPos out of 0..1 (insufficient BB window)")
        elif k == "macd_hist_nonneg":
            reasons.append("MACD histogram ≤ 0")
    if not reasons:
        reasons.append("Filtered by other criteria or data quality")
    return {"Ticker": ticker, "Why excluded": "; ".join(reasons)}

# ---------------- Screener core ----------------
@st.cache_data(ttl=900, show_spinner=True)
def scan_list(ticks: list[str], years: int, price_min: float, price_max: float,
              min_dollar_vol_millions: float, strictness_label: str,
              min_rs_slope: float = 0.0, include_near: bool = True, max_symbols: int = 400,
              near_miss_bps: int = 30, explain: bool = True):
    cfg = apply_webull_like_from_strictness(strictness_label, near_miss_bps=near_miss_bps)
    ticks = [t.strip().upper() for t in ticks if t.strip()][:max_symbols]

    rows, near_rows, explain_rows = [], [], []

    for t in ticks:
        try:
            df = load_data(t, years=years)
            if df.empty or len(df) < 30:
                if explain:
                    explain_rows.append({"Ticker": t, "Why excluded": "Too little history / load failed"})
                continue

            df = build_indicators(df)
            last = df.iloc[-1]

            # Hard filters
            close = float(last["Close"])
            adv = (df["Close"]*df["Volume"]).rolling(cfg.avgvol_window).mean().iloc[-1]
            adv_m = float(adv)/1_000_000.0 if pd.notna(adv) else 0.0

            if not (price_min <= close <= price_max) or (adv_m < min_dollar_vol_millions):
                if explain:
                    explain_rows.append(_explain_failures(t, {}, last, price_min, price_max, adv_m, min_dollar_vol_millions))
                continue

            # RS slope (bps/day)
            pro = _pro_metrics(df)
            if pro["RS_slope(bps/day)"] < (min_rs_slope * 100):
                if include_near:
                    near_rows.append({
                        "Ticker": t, "Close": round(close,2),
                        "Avg$Vol(20d, M)": round(adv_m,2),
                        "RSI14": round(float(last["RSI14"]),1),
                        "BandPos(0=low,1=high)": float(last.get("BANDPOS", np.nan)),
                        **pro, "Reason": "RS slope below min"
                    })
                else:
                    if explain:
                        explain_rows.append({"Ticker": t, "Why excluded": "RS slope below minimum"})
                continue

            # Webull-like checks
            ok, checks = _passes_rules_row(last, cfg)
            base = {
                "Ticker": t, "Close": round(close,2), "Avg$Vol(20d, M)": round(adv_m,2),
                "TrendUp": bool(last["EMA20"] > last["EMA50"]),
                "RSI14": round(float(last["RSI14"]),1),
                "BandPos(0=low,1=high)": (None if pd.isna(last.get("BANDPOS", np.nan)) else round(float(last["BANDPOS"]),2)),
                "BUY?": bool(last.get("buy_signal", False)),
                "Score": int(sum(bool(v) for v in checks.values())),
                "Fresh": False,
                **pro
            }

            if ok:
                base["Reason"] = "pass"
                rows.append(base)
            else:
                if include_near:
                    nm = dict(base)
                    nm["Reason"] = "near-miss"
                    nm["Score"] = max(nm["Score"], 1)
                    near_rows.append(nm)
                if explain:
                    explain_rows.append(_explain_failures(t, checks, last, price_min, price_max, adv_m, min_dollar_vol_millions))

        except Exception:
            if explain:
                explain_rows.append({"Ticker": t, "Why excluded": "Exception during screening"})
            continue

    df_ok   = pd.DataFrame(rows)
    df_near = pd.DataFrame(near_rows)
    df_exp  = pd.DataFrame(explain_rows)

    if not df_ok.empty:
        df_ok = df_ok.sort_values(
            ["Fresh","Score","RS_slope(bps/day)","Avg$Vol(20d, M)","RSI14"],
            ascending=[False,False,False,False,False]
        ).reset_index(drop=True)
        df_ok.insert(0,"Rank",np.arange(1,len(df_ok)+1))

    if not df_near.empty:
        df_near = df_near.sort_values(
            ["RS_slope(bps/day)","Avg$Vol(20d, M)","RSI14"],
            ascending=[False,False,False]
        ).reset_index(drop=True)
        df_near.insert(0,"Rank",np.arange(1,len(df_near)+1))

    if not df_exp.empty:
        df_exp = df_exp.sort_values(["Why excluded","Ticker"]).reset_index(drop=True)

    return df_ok, df_near, df_exp
# =========================
# Part 2/3 — Watchlists + Screener UI
# =========================

# ---------------- In-App Watchlist Manager (auto-persist) ----------------
def init_watchlists_state():
    if "saved_watchlists" not in st.session_state:
        st.session_state.saved_watchlists = {}
    if "active_watchlist" not in st.session_state:
        st.session_state.active_watchlist = []

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
    except Exception:
        pass

def save_watchlists_to_disk():
    try:
        with open(WATCHLISTS_PATH, "w", encoding="utf-8") as f:
            json.dump(st.session_state.saved_watchlists, f, indent=2)
    except Exception:
        pass

def export_watchlists_json():
    return io.BytesIO(json.dumps(st.session_state.saved_watchlists, indent=2).encode("utf-8"))

def delete_watchlist(name: str):
    if name in st.session_state.saved_watchlists:
        st.session_state.saved_watchlists.pop(name, None)
        save_watchlists_to_disk()

def save_watchlist(name: str, symbols: list[str]):
    syms = [s.strip().upper() for s in symbols if str(s).strip()]
    if not syms or not name:
        return False, "Need a name and at least one ticker"
    st.session_state.saved_watchlists[name] = syms
    save_watchlists_to_disk()
    return True, f"✅ Saved {len(syms)} symbols to watchlist '{name}'"

# init + auto-load from disk once
init_watchlists_state()
if "watchlists_loaded_once" not in st.session_state:
    load_watchlists_from_disk()
    st.session_state.watchlists_loaded_once = True

st.subheader("📋 Watchlists")

# Quick buttons for saved lists
if st.session_state.saved_watchlists:
    st.write("Click to load a saved watchlist:")
    cols = st.columns(min(len(st.session_state.saved_watchlists), 4))
    for i, (name, tickers) in enumerate(st.session_state.saved_watchlists.items()):
        col = cols[i % max(1, len(cols))]
        if col.button(name, key=f"load_{name}"):
            st.session_state.active_watchlist = tickers.copy()
            st.success(f"Loaded watchlist: {name}")
            st.experimental_rerun()
else:
    st.info("No saved watchlists yet. Create or import below.")

# Active list editor
wl_input = st.text_area(
    "Active Watchlist (comma separated tickers)",
    value=", ".join(st.session_state.active_watchlist),
    height=80,
    key="active_watchlist_input"
)
st.session_state.active_watchlist = [t.strip().upper() for t in wl_input.split(",") if t.strip()]

# Save current as named list
new_name = st.text_input("Save current list as:", "", key="save_list_name")
if st.button("💾 Save Watchlist", key="btn_save_list") and new_name.strip():
    ok, msg = save_watchlist(new_name.strip(), st.session_state.active_watchlist)
    (st.success if ok else st.error)(msg)
    if ok: st.experimental_rerun()

# Export all watchlists
if st.session_state.saved_watchlists:
    json_bytes = export_watchlists_json()
    st.download_button("⬇️ Download All Watchlists (JSON)", data=json_bytes,
                       file_name="watchlists.json", mime="application/json",
                       key="dl_all_watchlists")

# Import JSON Watchlists
with st.expander("📥 Import Watchlists from JSON"):
    uploaded = st.file_uploader("Upload watchlists.json file", type=["json"], key="upl_watchlists")
    if uploaded is not None:
        try:
            data = json.load(uploaded)
            if isinstance(data, dict):
                for k,v in data.items():
                    if isinstance(v, list):
                        st.session_state.saved_watchlists[k] = [s.strip().upper() for s in v if str(s).strip()]
                save_watchlists_to_disk()
                st.success("✅ Watchlists imported successfully")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid format. Expected a dictionary of lists.")
        except Exception as e:
            st.error(f"❌ Import failed: {e}")

# ---------------- Sidebar Controls ----------------
with st.sidebar:
    st.header("⚙️ Screener Settings")
    uni_choice = st.selectbox("Universe",
        ["All (Combined)","S&P 500","NASDAQ-100","DOW 30"], index=0, key="universe_select")

    years_back   = st.slider("History (years)", 1, 5, 2, key="hist_years")
    price_min    = st.number_input("Min price ($)", 0.0, value=5.0, step=0.5, key="min_price")
    price_max    = st.number_input("Max price ($)", 0.0, value=60.0, step=1.0, key="max_price")
    liq_min      = st.number_input("Min Avg $ Vol (20d, M)", 0.0, value=0.5, step=0.5, key="min_liq")
    strictness   = st.selectbox("Strictness", ["Loose","Balanced","Strict"], index=1, key="strictness")
    min_rs_slope = st.slider("Min RS slope (bp/day)", -5.0, 5.0, 0.0, 0.1, key="min_rs")
    include_near = st.checkbox("Also show near-miss list", value=True, key="include_near")

    st.markdown("### Compatibility (Webull-like)")
    near_miss_bps = st.slider("Near-miss band (bps)", 0, 100, 30, step=5,
                              help="Tolerance around thresholds so results align with Webull-like semantics",
                              key="near_miss_bps")
    explain_drops = st.checkbox("Explain why tickers were excluded", value=True, key="explain_drops")

    st.markdown("---")
    st.caption("Daily bars • last closed candle • Close price source • BB 20/2")

# Resolve universe
if uni_choice == "All (Combined)":
    universe_ticks = combined_universe()
elif uni_choice == "S&P 500":
    universe_ticks = get_sp500_tickers()
elif uni_choice == "NASDAQ-100":
    universe_ticks = get_nasdaq100_tickers()
else:
    universe_ticks = get_dow30_tickers()

# ---------------- Screener Run + Save to Watchlist ----------------
st.header("📊 Screener")
if st.button("🔎 Run Screener", key="btn_run_screener"):
    res_ok, res_near, res_explain = scan_list(
        universe_ticks, years_back, price_min, price_max, liq_min,
        strictness_label=strictness, min_rs_slope=min_rs_slope, include_near=include_near,
        near_miss_bps=near_miss_bps, explain=explain_drops
    )

    if res_ok is not None and not res_ok.empty:
        st.subheader("✅ Matches")
        st.dataframe(res_ok, use_container_width=True)
    else:
        st.info("No strict matches — loosen filters or check Near-Misses.")

    if include_near and (res_near is not None) and (not res_near.empty):
        st.subheader("⚠️ Near-Miss Candidates")
        st.dataframe(res_near, use_container_width=True)

    if explain_drops and (res_explain is not None) and (not res_explain.empty):
        st.subheader("🧐 Why some tickers were excluded")
        st.dataframe(res_explain, use_container_width=True)

    # Gather tickers from this run
    tickers_from_run = []
    if res_ok is not None and not res_ok.empty:
        tickers_from_run += list(res_ok["Ticker"])
    if include_near and res_near is not None and not res_near.empty:
        tickers_from_run += list(res_near["Ticker"])

    # Auto-save last run so you never lose it
    if tickers_from_run:
        save_watchlist("Last Screener Results", tickers_from_run)

        st.subheader("💾 Save Screener Results")
        wl_name = st.text_input("Name this watchlist", value="Picks " + date.today().isoformat(), key="name_save_screener")
        if st.button("Save Screener Results", key="btn_save_screener_results"):
            ok, msg = save_watchlist(wl_name.strip(), tickers_from_run)
            (st.success if ok else st.error)(msg)
            if ok: st.experimental_rerun()

# ---------------- Saved Watchlists management ----------------
if st.session_state.saved_watchlists:
    st.header("📂 Saved Watchlists")

    # --- Show Last Screener Results separately ---
    if "Last Screener Results" in st.session_state.saved_watchlists:
        syms = st.session_state.saved_watchlists["Last Screener Results"]
        st.subheader("🕒 Last Screener Results")
        st.write(", ".join(syms))
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📊 Re-scan Last Screener Results", key="rescan_last_screener"):
                res_ok, res_near, res_explain = scan_list(
                    syms, years_back, price_min, price_max, liq_min,
                    strictness_label=strictness, min_rs_slope=min_rs_slope, include_near=include_near,
                    near_miss_bps=near_miss_bps, explain=explain_drops
                )
                st.subheader("Re-scan — Last Screener Results")
                if res_ok is not None and not res_ok.empty:
                    st.dataframe(res_ok, use_container_width=True)
                if include_near and res_near is not None and not res_near.empty:
                    st.dataframe(res_near, use_container_width=True)
                if explain_drops and (res_explain is not None) and (not res_explain.empty):
                    st.dataframe(res_explain, use_container_width=True)
        with c2:
            if st.button("🗑️ Clear Last Screener Results", key="del_last_screener"):
                delete_watchlist("Last Screener Results")
                st.experimental_rerun()

        st.markdown("---")

    # --- Show all other saved watchlists ---
    for name, syms in st.session_state.saved_watchlists.items():
        if name == "Last Screener Results":
            continue
        st.markdown(f"**{name}** — {len(syms)} symbols")
        st.write(", ".join(syms))
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button(f"📊 Re-scan {name}", key=f"rescan_{name}"):
                res_ok, res_near, res_explain = scan_list(
                    syms, years_back, price_min, price_max, liq_min,
                    strictness_label=strictness, min_rs_slope=min_rs_slope, include_near=include_near,
                    near_miss_bps=near_miss_bps, explain=explain_drops
                )
                st.subheader(f"Re-scan — {name}")
                if res_ok is not None and not res_ok.empty:
                    st.dataframe(res_ok, use_container_width=True)
                if include_near and res_near is not None and not res_near.empty:
                    st.dataframe(res_near, use_container_width=True)
                if explain_drops and (res_explain is not None) and (not res_explain.empty):
                    st.dataframe(res_explain, use_container_width=True)
        with c2:
            buf = io.BytesIO(pd.DataFrame({"Ticker": syms}).to_csv(index=False).encode("utf-8"))
            st.download_button("⬇️ Export CSV", buf, file_name=f"{name}.csv", key=f"dl_{name}")
        with c3:
            if st.button("🗑️ Delete", key=f"del_{name}"):
                delete_watchlist(name)
                st.experimental_rerun()

    st.markdown("---")
    st.download_button(
        "⬇️ Download Saved Screener Results (JSON)",
        data=export_watchlists_json(),
        file_name="watchlists.json",
        mime="application/json",
        key="dl_saved_json"
    )
# =========================
# =========================
# =========================
# Part 3/3 — Analysis modules (single ticker) — DROP-IN with 1H/2H/4H/1D/1W
# =========================

# --- Safe optional imports (Analyzer keeps rendering even if these are missing) ---
HAVE_FEEDPARSER = True
HAVE_VADER = True
HAVE_SKLEARN = True

try:
    import requests  # typically present
except Exception:
    requests = None

try:
    import feedparser
except Exception:
    HAVE_FEEDPARSER = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    HAVE_VADER = False
    SentimentIntensityAnalyzer = None

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except Exception:
    HAVE_SKLEARN = False
    # Tiny fallback for quick_forecast()/ETA if sklearn isn't installed.
    class LinearRegression:
        def fit(self, X, y): 
            import numpy as _np
            X = _np.asarray(X).ravel(); y = _np.asarray(y).ravel()
            if len(X) < 2:
                self.coef_ = _np.array([0.0]); self.intercept_ = float(y[-1]) if len(y) else 0.0; return self
            A = _np.c_[X, _np.ones_like(X)]
            beta, *_ = _np.linalg.lstsq(A, y, rcond=None)
            self.coef_ = _np.array([beta[0]]); self.intercept_ = beta[1]; return self
        def predict(self, X):
            import numpy as _np
            X = _np.asarray(X).ravel(); return self.coef_[0]*X + self.intercept_
    def train_test_split(*a, **k): return ([], [], [], [])
    def accuracy_score(a, b): return 0.0
    class RandomForestClassifier:
        def __init__(self, *a, **k): pass
        def fit(self, X, y): return self
        def predict(self, X): return [0]*len(X)
        def predict_proba(self, X):
            import numpy as _np
            return _np.c_[[[0.5,0.5]]*len(X)]
        @property
        def feature_importances_(self):
            return []

# Plotting & data libs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import math
import datetime as dt
from datetime import date
import streamlit as st

# We use yfinance (already imported in Part 1). In case someone runs this file standalone:
try:
    import yfinance as yf  # noqa
except Exception:
    pass

# ---------- Range helper ----------
def slice_df_by_view(df: pd.DataFrame, view: str) -> pd.DataFrame:
    """Return a sliced DataFrame by time window (1W/1M/3M/6M/YTD/1Y/All)."""
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    end = idx.max()
    v = (view or "All").upper()

    if v == "ALL":
        return df
    if v == "YTD":
        start = pd.Timestamp(year=end.year, month=1, day=1)
    elif v == "1W":
        start = end - pd.Timedelta(days=7)
    elif v == "1M":
        start = end - pd.Timedelta(days=31)
    elif v == "3M":
        start = end - pd.Timedelta(days=92)
    elif v == "6M":
        start = end - pd.Timedelta(days=183)
    elif v == "1Y":
        start = end - pd.Timedelta(days=366)
    else:
        return df

    out = df.loc[(idx >= start) & (idx <= end)]
    # If too few rows for indicators to look reasonable, show at least last 60 bars
    return out if len(out) >= 25 else df.tail(60)

# ---------- Intraday fetch (1H) with resampling to 2H / 4H ----------
@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_intraday_1h(ticker: str, period: str = "180d") -> pd.DataFrame:
    """
    Fetch 1-hour bars via yfinance, then normalize (OHLCV, tz-naive, sorted).
    Yahoo supports 1h (and legacy 60m); period ~180d is usually safe.
    """
    t = (ticker or "").strip().upper()
    if not t:
        return pd.DataFrame()
    df = pd.DataFrame()
    try:
        df = yf.download(t, period=period, interval="1h", auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            df = yf.download(t, period=period, interval="60m", auto_adjust=True, progress=False, threads=False)
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[keep].copy().dropna()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    tz = getattr(df.index, "tz", None)
    if tz is not None:
        df.index = df.index.tz_convert(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.attrs["source"] = "yahoo:download(intraday 1h)"
    return df

def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV to a new rule (e.g., '2H','4H','W-FRI')."""
    if df.empty: 
        return df
    agg = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    out = df.resample(rule, label="right", closed="right").agg(agg).dropna()
    return out

def _build_indicators_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Use your existing build_indicators with a tiny guard for small frames."""
    if df is None or df.empty or len(df) < 5:
        return pd.DataFrame()
    try:
        return build_indicators(df)
    except Exception:
        return pd.DataFrame()

def get_data_for_timeframe(ticker: str, years_hist: int, tf: str) -> pd.DataFrame:
    """
    Return OHLCV + indicators at selected timeframe:
    - 1H: fetch 1-hour from Yahoo
    - 2H/4H: fetch 1-hour then resample
    - 1D: use your existing daily loader
    - 1W: daily -> weekly resample (W-FRI)
    """
    tf = (tf or "1D").upper()
    if tf in ("1H","2H","4H"):
        base = _fetch_intraday_1h(ticker, period="180d")
        if base.empty: 
            return pd.DataFrame()
        if tf == "2H":
            base = _resample_ohlc(base, "2H")
        elif tf == "4H":
            base = _resample_ohlc(base, "4H")
        return _build_indicators_safe(base)

    # Daily / Weekly from your cached loader
    daily = load_data(ticker, years_hist)
    if daily.empty:
        return pd.DataFrame()
    if tf == "1W":
        wk = _resample_ohlc(daily, "W-FRI")
        wk.attrs["source"] = daily.attrs.get("source","") + " -> weekly"
        return _build_indicators_safe(wk)
    # 1D default
    return _build_indicators_safe(daily)

# ---------- Plot: Candles + MACD + RSI with range tools ----------
def plot_price(df: pd.DataFrame):
    """
    Candles + EMA20/EMA50 on top, MACD (line+signal+hist) mid,
    RSI(14) with 30/50/70 guides bottom. Includes rangeselector & rangeslider.
    """
    ema20 = df.get("EMA20"); ema50 = df.get("EMA50")
    macd_line = df.get("MACD"); macd_sig = df.get("MACDsig")
    macd_hist = (macd_line - macd_sig) if (macd_line is not None and macd_sig is not None) else None
    rsi14 = df.get("RSI14")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.62, 0.19, 0.19],
        specs=[[{}], [{}], [{}]]
    )

    # Row 1: Price
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price"
    ), row=1, col=1)

    if "EMA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA20", mode="lines"), row=1, col=1)
    if "EMA50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=ema50, name="EMA50", mode="lines"), row=1, col=1)

    # Optional markers
    buys  = df[df.get("buy_signal", pd.Series(False, index=df.index))]
    sells = df[df.get("sell_signal", pd.Series(False, index=df.index))]
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], name="Buy",
                                 mode="markers", marker_symbol="triangle-up", marker_size=10), row=1, col=1)
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], name="Sell",
                                 mode="markers", marker_symbol="triangle-down", marker_size=10), row=1, col=1)

    # Row 2: MACD
    if (macd_line is not None) and (macd_sig is not None):
        if macd_hist is not None:
            colors = np.where(macd_hist >= 0, "rgba(38,166,154,0.6)", "rgba(239,83,80,0.6)")
            fig.add_trace(go.Bar(x=df.index, y=macd_hist, name="MACD hist", marker_color=colors), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD", mode="lines"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd_sig,  name="Signal", mode="lines", line=dict(dash="dot")), row=2, col=1)
        # zero line
        fig.add_shape(type="line", x0=df.index.min(), x1=df.index.max(), y0=0, y1=0,
                      line=dict(width=1, dash="dot"), xref="x2", yref="y2")

    # Row 3: RSI
    if rsi14 is not None:
        fig.add_trace(go.Scatter(x=df.index, y=rsi14, name="RSI(14)", mode="lines"), row=3, col=1)
        fig.add_shape(type="rect", xref="x3", yref="y3",
                      x0=df.index.min(), x1=df.index.max(), y0=30, y1=70,
                      fillcolor="rgba(180,180,180,0.15)", line_width=0)
        for yv, dsh in [(30, "dot"), (50, "dot"), (70, "dot")]:
            fig.add_shape(type="line", x0=df.index.min(), x1=df.index.max(), y0=yv, y1=yv,
                          line=dict(width=1, dash=dsh), xref="x3", yref="y3")

    # Range buttons + slider (top pane x-axis)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1W", step="day",   stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD",step="year",  stepmode="todate"),
                dict(count=1, label="1Y", step="year",  stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date",
        row=1, col=1
    )

    # Axes + layout
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MACD",  row=2, col=1)
    fig.update_yaxes(title_text="RSI",   row=3, col=1, range=[0, 100])
    fig.update_layout(
        height=840,
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

# ---------- Quick forecast (tiny) ----------
def quick_forecast(df: pd.DataFrame, lookback: int = 20):
    s = df["Close"].pct_change().dropna().tail(lookback)
    if len(s) < 5:
        return None
    X = np.arange(len(s)).reshape(-1,1); y = s.values
    model = LinearRegression().fit(X, y)
    nxt = float(np.asarray(model.predict(np.array([[len(s)]]))).ravel()[0])
    last = float(df["Close"].iloc[-1]); expc = last*(1+nxt)
    # confidence-ish: slope / volatility (capped)
    conf = float(min(0.95, max(0.05, abs(getattr(model, "coef_", [0])[0])/(s.std()+1e-9))))
    return {"direction":"up" if nxt>0 else "down","next_ret":nxt,"last_close":last,"expected_close":float(expc),"confidence":conf}

# ---------- ETA helpers ----------
def eta_trend_slope_days(df: pd.DataFrame, current_price: float, target_price: float, lookback: int = 20) -> float | None:
    try:
        if not (np.isfinite(current_price) and np.isfinite(target_price)): return None
        if current_price <= 0 or target_price <= 0: return None
        if target_price <= current_price: return None
        s = pd.Series(df["Close"]).dropna().tail(max(lookback, 10))
        if len(s) < 10: return None
        x = np.arange(len(s)).reshape(-1, 1)
        y = np.log(s.values)
        lr = LinearRegression().fit(x, y)
        slope = float(getattr(lr, "coef_", [0])[0])
        if slope <= 0 or not np.isfinite(slope): return None
        bars_needed = math.log(target_price / current_price) / slope
        if bars_needed <= 0 or not np.isfinite(bars_needed): return None
        return float(min(bars_needed, 200.0))
    except Exception:
        return None

def eta_historical_window(df: pd.DataFrame, pct_move: float, lookback_bars: int = 250, max_horizon: int = 60):
    try:
        if not np.isfinite(pct_move) or pct_move <= 0: return (None, None, None)
        s = pd.Series(df["Close"]).dropna()
        if len(s) < 30: return (None, None, None)
        s = s.tail(lookback_bars)
        times = []
        n = len(s)
        for i in range(n - 2):
            base = float(s.iloc[i])
            tgt = base * (1.0 + pct_move)
            forward = s.iloc[i+1 : min(i+1+max_horizon, n)]
            hit = np.where(forward.values >= tgt, True, False)
            if hit.any():
                days = int(np.argmax(hit) + 1)
                times.append(days)
        if not times:
            return (None, None, None)
        arr = np.array(times)
        med = float(np.median(arr)); q1 = float(np.percentile(arr, 25)); q3 = float(np.percentile(arr, 75))
        return (med, q1, q3)
    except Exception:
        return (None, None, None)

# ---------- Trade planning ----------
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
    return {"entry":entry,"stop":stop,"target":target,"shares":shares,"risk_dollars":risk_d,"reward_dollars":reward_d,"notional":notional,"rr":rr}

def make_order_ticket(ticker: str, side_long: bool, entry_style: str, entry: float, stop: float, target: float,
                      shares: int, rr: float, atr_val: float, risk_dollars: float, reward_dollars: float):
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
        f"Approx R:R: {rr:.2f}" if (isinstance(rr, (int,float)) and np.isfinite(rr)) else "Approx R:R: n/a",
        f"ATR(14): {atr_val:,.2f}",
        "Notes: Use a bracket order (entry + stop + limit)."
    ]
    text = "\n".join(lines)
    payload = {
        "timestamp": now, "ticker": ticker.upper(), "bias": bias,
        "entry_style": entry_style, "entry": round(entry, 4), "stop": round(stop, 4),
        "target": round(target, 4), "shares": int(shares),
        "risk_dollars": round(risk_dollars, 2), "reward_dollars": round(reward_dollars, 2),
        "rr": round(rr, 3) if (isinstance(rr, (int,float)) and np.isfinite(rr)) else None, "atr14": round(atr_val, 4)
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
        lines.append(f"- Recent 20-bar range: {ll20:,.2f} -> {hh20:,.2f}")
    if pd.notna(rsi_val):
        if rsi_val < 40: lines.append("- RSI < 40 -> weak momentum; be conservative.")
        elif rsi_val > 60: lines.append("- RSI > 60 -> firm momentum; breakouts may follow through.")
        else: lines.append("- RSI ~ 40-60 -> neutral; expect chop.")
    return "\n".join(lines)

# ---------- Seasonality ----------
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

# ---------- Headlines + Sentiment ----------
_analyzer = None
def _get_vader():
    global _analyzer
    if _analyzer is None and HAVE_VADER:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer

def fetch_rss_headlines(ticker: str, limit: int = 12):
    if not HAVE_FEEDPARSER:
        return []
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
                pub   = getattr(e, "published", getattr(e, "updated", "")) or ""
                if title:
                    out.append({"title": title, "link": link, "published": pub})
            if out:
                break
        except Exception:
            continue
    return out[:limit]

def score_headlines_vader(headlines: list[dict]):
    if not HAVE_VADER:
        return pd.DataFrame(columns=["title","link","published","compound","label"]), 0.0
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

# =========================
# Single-Ticker UI + analyses
# =========================
st.header("🔍 Single-Ticker Analysis")

col_in_a, col_in_b, col_in_c = st.columns([2,1,1])
with col_in_a:
    single_ticker = st.text_input("Ticker (e.g., AAPL, MSFT, TSLA)", value="AAPL", key="single_ticker")
with col_in_b:
    timeframe = st.selectbox("Timeframe", ["1H","2H","4H","1D","1W"], index=3, key="timeframe")
with col_in_c:
    years_hist = st.slider("Years (for 1D/1W)", 1, 10, 3, key="years_hist")

# Chart range picker
view_choice = st.radio(
    "Chart range", ["1W","1M","3M","6M","YTD","1Y","All"],
    index=2, horizontal=True, key="chart_range"
)

if timeframe in ("1H","2H","4H"):
    st.caption("ℹ️ Intraday uses up to ~180 days of 1-hour data (2H/4H are resampled).")

cbtn1, cbtn2 = st.columns([1,1])
with cbtn1:
    run_analyze = st.button("Analyze", key="btn_analyze")
with cbtn2:
    st.button("Reset", on_click=lambda: st.session_state.update(ran=False), key="btn_reset")

if run_analyze:
    st.session_state.ran = True

if st.session_state.ran:
    ticker = (single_ticker or "").strip().upper()
    df_all = get_data_for_timeframe(ticker, years_hist, timeframe)
    if df_all.empty or len(df_all) < 30:
        st.error("No data found or too little history for this timeframe. Try another ticker or a longer period.")
    else:
        df_view = slice_df_by_view(df_all, view_choice)

        # Source & last bar date
        try:
            src = df_all.attrs.get("source","yahoo")
            last_ts = pd.to_datetime(df_all.index, errors="coerce").max()
            last_dt = pd.Timestamp(last_ts).tz_localize(None).to_pydatetime()
            st.caption(f"Data source: {src} | Timeframe: {timeframe} | Last bar: {last_dt.date()} {last_dt.strftime('%H:%M')}")
        except Exception:
            pass

        # Price chart
        st.subheader(f"{ticker} — {timeframe} Candles, MACD & RSI")
        st.plotly_chart(plot_price(df_view), use_container_width=True)

        latest  = df_all.iloc[-1]
        trend_up = bool(latest["EMA20"] > latest["EMA50"])
        side_txt = "long bias" if trend_up else "short bias"
        atr_val = float(latest["ATR14"]); close = float(latest["Close"])
        ema20   = float(latest["EMA20"]); ema50 = float(latest["EMA50"])
        hh20    = float(latest.get("HH20", np.nan)); ll20 = float(latest.get("LL20", np.nan))

        c1,c2,c3 = st.columns(3)
        c1.metric("Trend", side_txt)
        c2.metric("ATR(14)", f"{atr_val:,.2f}")
        c3.metric("Last close", f"{close:,.2f}")

        # Tiny forecast
        st.subheader("Tiny toy forecast (next bar)")
        fc = quick_forecast(df_all, lookback=20)
        if fc:
            d1,d2,d3 = st.columns(3)
            d1.metric("Direction", "Up" if fc["direction"]=="up" else "Down")
            d2.metric("Expected close", f"{fc['expected_close']:,.2f}")
            d3.metric("Confidence-ish", f"{int(fc['confidence']*100)}%")
            st.caption("Simple momentum estimate. Educational only.")
        else:
            st.info("Not enough data to estimate.")

        # ML edge (next-bar direction)
        st.subheader("ML edge (next-bar direction)")
        try:
            if HAVE_SKLEARN:
                X = pd.concat([
                    pd.Series(df_all["Close"].pct_change(),   name="RET1"),
                    pd.Series(df_all["Close"].pct_change(5),  name="RET5"),
                    pd.Series(df_all["Close"].pct_change(10), name="RET10"),
                    pd.Series((df_all["High"] - df_all["Low"]) / df_all["Close"], name="HL_PCT"),
                    pd.Series(df_all["ATR14"] / df_all["Close"], name="ATR_PCT"),
                    df_all[["EMA20","EMA50","SMA20","SMA50","RSI14","MACD","MACDsig"]],
                ], axis=1)

                X = X.replace([np.inf, -np.inf], np.nan).dropna()
                y = (df_all["Close"].shift(-1) > df_all["Close"]).astype(int).reindex(X.index)

                if len(X) > 200 and y.notna().sum() > 50:
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, shuffle=False)
                    clf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)
                    clf.fit(Xtr, ytr)

                    acc = accuracy_score(yte, clf.predict(Xte))
                    proba_up = float(clf.predict_proba(X.iloc[[-1]])[0, 1])
                    imps = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Prob. Up (next bar)", f"{proba_up*100:.1f}%")
                    m2.metric("Backtest Accuracy*", f"{acc*100:.1f}%")
                    m3.metric("Top feature", imps.index[0] if len(imps) else "n/a")
                    with st.expander("Feature importances"):
                        st.write(imps.to_frame("importance"))
                    st.caption("*Toy model, simple split. Educational only.")
                else:
                    st.info("Need more history for ML preview (200+ rows and at least 50 label points).")
            else:
                st.info("scikit-learn not installed — skipping ML preview (optional).")
        except Exception as e:
            st.warning(f"ML section error: {e}")

        # Seasonality & News (note: seasonality is daily-based, still informative)
        st.subheader("Seasonality & News Context (experimental)")
        try:
            # Use daily for seasonality stats regardless of tf
            daily_for_season = get_data_for_timeframe(ticker, years_hist, "1D")
            if not daily_for_season.empty:
                dow_tbl, mon_tbl = seasonality_tables(daily_for_season)
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
            news = fetch_rss_headlines(ticker, limit=10)
            if news and HAVE_VADER:
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
            elif not HAVE_FEEDPARSER:
                st.info("feedparser not installed — skipping headlines (optional).")
            else:
                st.info("No headlines fetched (RSS may be blocked/limited).")
        except Exception as e:
            st.info(f"Seasonality/news module skipped: {e}")

        # Trade plan (based on selected timeframe bars)
        st.subheader("Trade Plan (educational)")
        latest  = df_all.iloc[-1]
        trend_up = latest["EMA20"] > latest["EMA50"]
        atr_val = float(latest["ATR14"]); close = float(latest["Close"])
        ema20   = float(latest["EMA20"]); ema50 = float(latest["EMA50"])
        hh20    = float(latest.get("HH20", np.nan)); ll20 = float(latest.get("LL20", np.nan))

        # Plan inputs
        cpa, cpb, cpc = st.columns(3)
        with cpa:
            account_size  = st.number_input("Account size ($)", min_value=0.0, value=10000.0, step=100.0, key="acct_size")
        with cpb:
            risk_pct      = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, step=0.1, key="risk_pct")
        with cpc:
            entry_style   = st.selectbox("Entry style", ["Pullback to EMA20","Breakout 20-day"], key="entry_style")

        cpd, cpe, cpf = st.columns(3)
        with cpd:
            stop_atr_mult = st.slider("Stop distance (x ATR)", 0.5, 5.0, 1.5, step=0.1, key="stop_mult")
        with cpe:
            target_rr     = st.slider("Target multiple (R:R)", 0.5, 5.0, 2.0, step=0.5, key="target_rr")
        with cpf:
            lookback_eta  = st.slider("ETA slope lookback (bars)", 10, 60, 20, step=5, key="eta_lookback")

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
        st.caption(f"Method: {entry_style} | Stop {stop_atr_mult}x ATR | Target {target_rr}R. Approx R:R = {rr_str}")

        # Time-to-Target (ETA) — computed in bars for the selected timeframe
        try:
            if plan["target"] > close:
                eta1 = eta_trend_slope_days(df_all, current_price=close, target_price=plan["target"], lookback=lookback_eta)
                pct_move = (plan["target"] / close) - 1.0 if close > 0 else np.nan
                med, q1, q3 = eta_historical_window(df_all, pct_move=pct_move, lookback_bars=250, max_horizon=60)

                st.subheader("Time-to-Target (rough)")
                ceta1, ceta2 = st.columns(2)
                with ceta1:
                    if eta1 is not None:
                        st.metric("Trend slope ETA (bars)", f"{eta1:.1f}", help="Based on recent log-price slope")
                    else:
                        st.metric("Trend slope ETA", "n/a")
                with ceta2:
                    if med is not None:
                        st.metric("Historical median (bars)", f"{med:.0f}", help="IQR = middle 50%")
                        if (q1 is not None) and (q3 is not None):
                            st.caption(f"IQR {q1:.0f}–{q3:.0f} bars")
                    else:
                        st.metric("Historical median", "n/a")
                st.caption("Educational estimate — volatility & news can change timing quickly.")
        except Exception as e:
            st.info(f"ETA module skipped: {e}")

        # Order ticket
        ticket_text, json_bytes, csv_bytes, txt_bytes = make_order_ticket(
            ticker=ticker, side_long=trend_up, entry_style=entry_style,
            entry=plan['entry'], stop=plan['stop'], target=plan['target'], shares=plan['shares'],
            rr=plan['rr'], atr_val=atr_val, risk_dollars=plan['risk_dollars'], reward_dollars=plan['reward_dollars']
        )

        st.subheader("Order Ticket (copy/paste)")
        st.code(ticket_text, language="text")
        cdl1,cdl2,cdl3 = st.columns(3)
        with cdl1: st.download_button("Download TXT",  data=txt_bytes,  file_name=f"{ticker.upper()}_ticket.txt",  mime="text/plain", key="dl_txt")
        with cdl2: st.download_button("Download JSON", data=json_bytes, file_name=f"{ticker.upper()}_ticket.json", mime="application/json", key="dl_json")
        with cdl3: st.download_button("Download CSV",  data=csv_bytes,  file_name=f"{ticker.upper()}_ticket.csv",  mime="text/csv", key="dl_csv")

        st.subheader("Plan explainer")
        st.markdown(explain_plan(
            ticker, trend_up, entry_style, plan["entry"], plan["stop"], plan["target"],
            float(latest["RSI14"]), atr_val, ema20, ema50, hh20, ll20
        ))

