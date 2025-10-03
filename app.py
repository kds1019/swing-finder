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
            st.rerun()
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
    if ok: st.rerun()

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
                st.rerun()
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
            if ok: st.rerun()

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
                st.rerun()

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
                st.rerun()

    st.markdown("---")
    st.download_button(
        "⬇️ Download Saved Screener Results (JSON)",
        data=export_watchlists_json(),
        file_name="watchlists.json",
        mime="application/json",
        key="dl_saved_json"
    )
# =========================
# Part 3 — Analysis modules
# =========================

import math, re, io, os
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import streamlit as st
import feedparser
from datetime import date, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression

# -----------------------------------------
# Helpers: Indicators (MACD, RSI, MAs, etc.)
# -----------------------------------------
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    signal_line = _ema(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist

def ensure_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common indicators to df if missing. Returns a new df."""
    out = df.copy()
    if "Close" not in out.columns:
        return out
    if "SMA20" not in out.columns:
        out["SMA20"] = _sma(out["Close"], 20)
    if "SMA50" not in out.columns:
        out["SMA50"] = _sma(out["Close"], 50)
    if "EMA20" not in out.columns:
        out["EMA20"] = _ema(out["Close"], 20)
    if "EMA50" not in out.columns:
        out["EMA50"] = _ema(out["Close"], 50)
    # MACD
    if not {"MACD","MACD_signal","MACD_hist"}.issubset(out.columns):
        macd, sig, hist = _macd(out["Close"])
        out["MACD"] = macd
        out["MACD_signal"] = sig
        out["MACD_hist"] = hist
    # RSI
    if "RSI14" not in out.columns:
        out["RSI14"] = _rsi(out["Close"], 14)
    # simple example signals (optional)
    if "buy_signal" not in out.columns:
        out["buy_signal"] = (out["EMA20"] > out["EMA50"]) & (out["EMA20"].shift(1) <= out["EMA50"].shift(1))
    if "sell_signal" not in out.columns:
        out["sell_signal"] = (out["EMA20"] < out["EMA50"]) & (out["EMA20"].shift(1) >= out["EMA50"].shift(1))
    return out

# ------------------------------------------------
# Data loader with intraday + resample fallbacks
# ------------------------------------------------
_SUPPORTED = {"1h": "60m", "1d": "1d", "1wk": "1wk"}  # native yf intervals

def _download_yf(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
    """Try start/end; if empty, fall back to period-based for intraday."""
    df = yf.download(
        ticker, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False, threads=False
    )
    if df is None or df.empty:
        # Intraday often needs 'period='; try generous periods based on interval
        period_map = {"60m": "730d", "1d": "10y", "1wk": "20y"}
        period = period_map.get(interval, "730d")
        df = yf.download(
            ticker, period=period, interval=interval,
            auto_adjust=True, progress=False, threads=False
        )
    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[keep].copy()
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().sort_index()

def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1h data to 2H / 4H OHLC + Volume sum."""
    o = df["Open"].resample(rule).first()
    h = df["High"].resample(rule).max()
    l = df["Low"].resample(rule).min()
    c = df["Close"].resample(rule).last()
    v = df["Volume"].resample(rule).sum(min_count=1)
    out = pd.concat([o.rename("Open"), h.rename("High"), l.rename("Low"),
                     c.rename("Close"), v.rename("Volume")], axis=1)
    return out.dropna(how="any")

def get_data_for_timeframe(ticker: str, years: int, timeframe: str) -> pd.DataFrame:
    """
    timeframe: one of {'1h','2h','4h','1d','1wk'}.
    For 2h/4h, fetch 1h then resample.
    """
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=int(years * 366 + 14))

    tf = timeframe.lower()
    if tf in ("1h", "1d", "1wk"):
        base_interval = _SUPPORTED[tf]
        df = _download_yf(ticker, start, end, base_interval)
    elif tf in ("2h", "4h"):
        # fetch base 1h then resample
        base_interval = _SUPPORTED["1h"]
        raw = _download_yf(ticker, start, end, base_interval)
        if raw.empty:
            df = pd.DataFrame()
        else:
            rule = "2H" if tf == "2h" else "4H"
            # ensure DatetimeIndex with timezone removed (yfinance usually tz-aware)
            raw = raw.tz_localize(None) if raw.index.tz is not None else raw
            df = _resample_ohlc(raw, rule)
    else:
        st.warning(f"Unsupported timeframe '{timeframe}'. Falling back to 1d.")
        df = _download_yf(ticker, start, end, "1d")

    # Final sanity & tail trim
    if df is None or df.empty:
        return pd.DataFrame()
    return df.tail(int(years * 252) + 120)  # keep enough bars for indicators

# -------------------------
# Forecast (quick + simple)
# -------------------------
def quick_forecast(df: pd.DataFrame, lookback: int = 20):
    try:
        s = df["Close"].pct_change().dropna().tail(lookback)
        if len(s) < 5:
            return None
        X = np.arange(len(s)).reshape(-1,1); y = s.values
        model = LinearRegression().fit(X, y)
        nxt = float(model.predict(np.array([[len(s)]]))[0])
        last = float(df["Close"].iloc[-1]); expc = last*(1+nxt)
        conf = float(min(0.95, max(0.05, abs(model.coef_[0])/(s.std()+1e-9))))
        return {"direction":"up" if nxt>0 else "down","next_ret":nxt,
                "last_close":last,"expected_close":float(expc),"confidence":conf}
    except Exception:
        return None

# -------------
# ETA helpers
# -------------
def eta_trend_slope_days(df, current_price, target_price, lookback=20):
    try:
        if not (np.isfinite(current_price) and np.isfinite(target_price)): return None
        if current_price <= 0 or target_price <= 0: return None
        if target_price <= current_price: return None
        s = pd.Series(df["Close"]).dropna().tail(max(lookback, 10))
        if len(s) < 10: return None
        x = np.arange(len(s)).reshape(-1, 1)
        y = np.log(s.values)
        lr = LinearRegression().fit(x, y)
        slope = float(lr.coef_[0])
        if slope <= 0 or not np.isfinite(slope): return None
        bars_needed = math.log(target_price / current_price) / slope
        return float(min(bars_needed, 200.0)) if bars_needed > 0 else None
    except Exception:
        return None

def eta_historical_window(df, pct_move: float, lookback_bars=250, max_horizon=60):
    try:
        if not np.isfinite(pct_move) or pct_move <= 0: return (None,None,None)
        s = pd.Series(df["Close"]).dropna()
        if len(s) < 30: return (None,None,None)
        s = s.tail(lookback_bars)
        times = []
        n = len(s)
        for i in range(n - 2):
            base = float(s.iloc[i]); tgt = base*(1.0+pct_move)
            forward = s.iloc[i+1:min(i+1+max_horizon,n)]
            hit = np.where(forward.values >= tgt, True, False)
            if hit.any():
                times.append(int(np.argmax(hit)+1))
        if not times: return (None,None,None)
        arr = np.array(times)
        return float(np.median(arr)), float(np.percentile(arr,25)), float(np.percentile(arr,75))
    except Exception:
        return (None,None,None)

# ------------------------------
# News / Sentiment (VADER)
# ------------------------------
_analyzer = None
def _get_vader():
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer

def fetch_rss_headlines(ticker: str, limit=12):
    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={ticker}%20stock&hl=en-US&gl=US&ceid=US:en",
    ]
    out = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            for e in d.entries[:limit]:
                title = getattr(e,"title","").strip()
                link  = getattr(e,"link","")
                pub   = getattr(e,"published",getattr(e,"updated","")) or ""
                if title:
                    out.append({"title":title,"link":link,"published":pub})
            if out: break
        except Exception:
            continue
    return out[:limit]

def score_headlines_vader(headlines):
    ana = _get_vader()
    rows = []
    for h in headlines:
        comp = ana.polarity_scores(h["title"])["compound"]
        if comp >= 0.15: lab="Bullish"
        elif comp <= -0.15: lab="Bearish"
        else: lab="Neutral"
        rows.append({**h,"compound":comp,"label":lab})
    if rows:
        df = pd.DataFrame(rows)
        return df, float(df["compound"].mean())
    return pd.DataFrame(columns=["title","link","published","compound","label"]), 0.0

# -------------------------
# Plot: Candles + MACD/RSI
# -------------------------
def plot_candles_macd_rsi(df: pd.DataFrame, ticker: str):
    """
    Returns a Figure with 3 stacked rows:
    (1) Price candles + MAs + Volume bars,
    (2) MACD + signal + histogram,
    (3) RSI(14) with 70/30 bands.
    """
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.02,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # --- Row 1: Price (candles) + MAs ---
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name=f"{ticker} OHLC", showlegend=False
        ),
        row=1, col=1, secondary_y=False
    )

    for name in ["SMA20","SMA50","EMA20","EMA50"]:
        if name in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[name], name=name, mode="lines"), row=1, col=1, secondary_y=False)

    # Volume bars (secondary Y on row 1)
    if "Volume" in df.columns and df["Volume"].notna().any():
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.3),
            row=1, col=1, secondary_y=True
        )

    # Buy/Sell markers (if present)
    buys  = df[df.get("buy_signal", pd.Series(False, index=df.index))]
    sells = df[df.get("sell_signal", pd.Series(False, index=df.index))]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys["Close"], mode="markers",
            marker_symbol="triangle-up", marker_size=10, name="Buy"
        ), row=1, col=1)
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells["Close"], mode="markers",
            marker_symbol="triangle-down", marker_size=10, name="Sell"
        ), row=1, col=1)

    # --- Row 2: MACD ---
    if {"MACD","MACD_signal","MACD_hist"}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", mode="lines"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", mode="lines"), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogram", opacity=0.5), row=2, col=1)

    # --- Row 3: RSI ---
    if "RSI14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI(14)", mode="lines"), row=3, col=1)
        # 70/30 bands
        fig.add_hline(y=70, line_dash="dot", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", row=3, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])

    fig.update_layout(
        height=750,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
    )
    return fig

# --------------------------
# Single-Ticker UI (Part 3)
# --------------------------
st.header("🔍 Single-Ticker Analysis")

ui_a, ui_b, ui_c = st.columns([2,1,1])
with ui_a:
    single_ticker = st.text_input("Ticker (e.g., AAPL, MSFT, TSLA)", value="AAPL", key="p3_ticker")
with ui_b:
    timeframe = st.selectbox(
        "Timeframe",
        options=["1h", "2h", "4h", "1d", "1wk"],
        index=3,  # default 1d
        key="p3_timeframe"
    )
with ui_c:
    years_hist = st.slider("Years of history (max fetch)", 1, 10, 3, key="p3_years")

# View range to avoid "entire year" clutter
vr = st.selectbox(
    "View Range",
    options=["Last 1W", "Last 1M", "Last 3M", "Last 6M", "Last 1Y", "All"],
    index=2,
    key="p3_view"
)

cbtn1, cbtn2 = st.columns([1,1])
with cbtn1:
    run_analyze = st.button("Analyze", key="p3_btn_analyze")
with cbtn2:
    reset_clicked = st.button("Reset", key="p3_btn_reset")

if reset_clicked:
    st.session_state.pop("p3_ran", None)

if run_analyze:
    st.session_state["p3_ran"] = True

if st.session_state.get("p3_ran", False):
    ticker = (single_ticker or "").strip().upper()

    df = get_data_for_timeframe(ticker, years_hist, timeframe)
    if df.empty or len(df) < 30:
        st.error("No data found or too little history for this timeframe. Try another ticker, a different timeframe, or more years.")
    else:
        # Trim to view range (approx by days)
        if vr != "All":
            days_map = {
                "Last 1W": 7,
                "Last 1M": 31,
                "Last 3M": 93,
                "Last 6M": 186,
                "Last 1Y": 366,
            }
            days = days_map.get(vr, None)
            if days is not None:
                cutoff = df.index.max() - pd.Timedelta(days=days)
                df = df[df.index >= cutoff]
                # if nothing after trimming (e.g., weekly bars), fall back gracefully
                if len(df) < 10:
                    df = get_data_for_timeframe(ticker, max(years_hist, 2), timeframe)

        # Indicators (RSI/MACD/MAs/signals)
        df = ensure_indicators(df)

        st.caption(f"Data source: Yahoo Finance | Last bar: {pd.to_datetime(df.index[-1]).strftime('%Y-%m-%d %H:%M')} | Timeframe: {timeframe}")

        # Chart
        st.subheader(f"{ticker} — Candles · MACD · RSI")
        st.plotly_chart(plot_candles_macd_rsi(df, ticker), use_container_width=True)

        # Quick Forecast (simple)
        fc = quick_forecast(df, lookback=20)
        if fc:
            colf1, colf2, colf3, colf4 = st.columns(4)
            with colf1:
                st.metric("Forecast Direction (next bar)", "▲ Up" if fc["direction"]=="up" else "▼ Down")
            with colf2:
                st.metric("Last Close", f"${fc['last_close']:.2f}")
            with colf3:
                st.metric("Expected Next Close", f"${fc['expected_close']:.2f}")
            with colf4:
                st.metric("Confidence (rough)", f"{fc['confidence']*100:.0f}%")

        # Headlines + sentiment
        st.subheader("📰 Headlines & Sentiment (VADER)")
        heads = fetch_rss_headlines(ticker, limit=12)
        hdf, avg_comp = score_headlines_vader(heads)
        st.caption(f"Average sentiment: {avg_comp:+.2f} (−1 to +1)")
        if not hdf.empty:
            for _, r in hdf.iterrows():
                st.markdown(f"- [{r['title']}]({r['link']}) — *{r['label']}*  \n  <small>{r['published']}</small>", unsafe_allow_html=True)
        else:
            st.info("No recent headlines were found for this ticker.")

        # (Optional) ETA sample usage (you can wire this into your Trade Plan UI as needed)
        with st.expander("⏱️ Time-to-Target (example calc)"):
            last_close = float(df["Close"].iloc[-1])
            hypothetical_target = last_close * 1.05  # +5%
            bars_needed = eta_trend_slope_days(df, last_close, hypothetical_target, lookback=20)
            med, p25, p75 = eta_historical_window(df, pct_move=0.05, lookback_bars=250, max_horizon=60)
            col_eta1, col_eta2 = st.columns(2)
            with col_eta1:
                st.markdown(f"**Trend-slope ETA to +5%:** {('~' + str(int(round(bars_needed)))) if bars_needed else 'n/a'} bars")
            with col_eta2:
                if med:
                    st.markdown(f"**Historical ETA for +5%:** median {int(med)} bars (IQR {int(p25)}–{int(p75)})")
                else:
                    st.markdown("**Historical ETA for +5%:** n/a")

# End Part 3

