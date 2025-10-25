import os, requests
from datetime import date, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv




import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import os, requests
import json




# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="SwingFinder", layout="wide")

# ✅ Add this line right here
years = 2  # default number of years of data to fetch

# ---------------- Initialize session state defaults (safeguard) ----------------
defaults = {
    "account_size": 10_000.0,
    "risk_per_trade": 1.0,
    "rr_ratio": 2.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("📈 Swing Finder — Tiingo Edition")
st.caption("Stock Scanner • Analyzer • Trade Planner")

load_dotenv()
TIINGO_KEY = os.getenv("TIINGO_API_KEY", "").strip()
if not TIINGO_KEY:
    st.error("⚠️ Missing Tiingo API key in .env file.")
    st.stop()
    
# ---------------- Initialize session state ----------------
if "account_size" not in st.session_state:
    st.session_state["account_size"] = 10_000.0
if "risk_per_trade" not in st.session_state:
    st.session_state["risk_per_trade"] = 1.0
if "rr_ratio" not in st.session_state:
    st.session_state["rr_ratio"] = 2.0
    
    # --------------------------------------------------------------
# Compatibility helper — Tiingo fetcher (used by Analyzer/Screener)
# --------------------------------------------------------------
import requests, datetime as dt
import pandas as pd

def fetch_tiingo(symbol: str, years: int = 2, token: str = None) -> pd.DataFrame:
    """
    Fetch historical daily data for a symbol from Tiingo.
    Keeps same format as original helper so existing analyzer code still works.
    """
    if token is None:
        from streamlit import secrets
        import os
        token = (
            os.getenv("TIINGO_TOKEN")
            or secrets.get("TIINGO_TOKEN", "")
            or os.getenv("TIINGO_API_KEY")
            or secrets.get("TIINGO_API_KEY", "")
        )
    if not token:
        raise ValueError("Tiingo token missing")

    end = dt.date.today()
    start = end - dt.timedelta(days=365 * years + 3)
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    params = {
        "token": token,
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "format": "json",
        "resampleFreq": "daily",
    }
    r = requests.get(url, params=params, timeout=15)
    if not r.ok:
        raise RuntimeError(f"{symbol}: {r.status_code} {r.reason}")
    data = r.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    

# ========================= SwingFinder — Webull-Style Tiingo Scanner (U.S.-only) =========================
# Full-universe scan with grid cards + “Trade Plan Target” on each card
# ---------------------------------------------------------------------------------------------------------
import os, time, math, random, concurrent.futures as futures
import datetime as dt
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ---------------- Env / constants ----------------
TIINGO_TOKEN = os.getenv("TIINGO_TOKEN") or st.secrets.get("TIINGO_TOKEN", "")
if not TIINGO_TOKEN:
    st.error("❌ Missing TIINGO_TOKEN. Add it to your environment or .streamlit/secrets.toml")
    st.stop()

SCAN_LOOKBACK_DAYS = 120              # history to compute EMA/RSI/ATR
MAX_WORKERS         = 48              # concurrency for full scan
BATCH_TICKER_COUNT  = 120             # pull tickers in chunks to keep UI responsive
REQUEST_PAUSE_S     = 0.02            # a light pause to be nice to Tiingo

# ---------------- Session init ----------------
def _init_state():
    for k, v in {
        "watchlist": [],
        "scanner_results": None,
        "analyze_symbol": None,
        "risk_rr": 2.0,               # Risk/Reward for target calc
        "risk_stop_atr_mult": 1.5,    # Stop = 1.5 * ATR by default
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init_state()

# ---------------- Helpers: indicators ----------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up, down = delta.clip(lower=0), (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_dn = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = (df["high"] - df["low"]).abs()
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Ensure consistent column naming ---
    # Tiingo returns capitalized columns: Open, High, Low, Close, Volume
    # This line makes sure all columns exist with consistent casing
    if "close" in df.columns:
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )

    # --- Moving averages ---
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # --- RSI(14) ---
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))

    # --- ATR(14) ---
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift(1))
    low_close = np.abs(df["Low"] - df["Close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # --- Band position (within 20-day range) ---
    df["HH20"] = df["High"].rolling(20).max()
    df["LL20"] = df["Low"].rolling(20).min()
    df["BandPos20"] = (df["Close"] - df["LL20"]) / (df["HH20"] - df["LL20"] + 1e-9)

    return df


# ---------------- Tiingo API ----------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)  # refresh every 24 hours
def tiingo_all_us_tickers(token: str) -> list[str]:
    """
    Fetch all active US tickers from Tiingo via the /utilities/search endpoint.
    Refreshes once per day (24h TTL). Randomized order for varied scan results.
    """
    import string, time, requests, random

    url = "https://api.tiingo.com/tiingo/utilities/search"
    headers = {"Content-Type": "application/json"}
    all_tickers = []

    # Loop over A–Z to get full alphabet coverage
    for ch in string.ascii_uppercase:
        params = {"token": token, "query": ch, "limit": 1000}
        r = requests.get(url, headers=headers, params=params, timeout=20)
        if not r.ok:
            continue
        data = r.json()
        for d in data:
            sym = d.get("ticker", "").upper()
            exch = d.get("exchange", "")
            if (
                sym.isalpha()
                and d.get("assetType") == "Stock"
                and exch not in ("CRYPTO", "FX")
            ):
                all_tickers.append(sym)
        time.sleep(0.2)  # polite pacing to avoid 429s

    # ✅ dedupe and randomize order
    clean = list(set(all_tickers))
    random.shuffle(clean)

    st.write(f"📈 Tiingo universe fetched: {len(clean):,} tickers (refreshed once per day)")
    return clean



@st.cache_data(show_spinner=False, ttl=60 * 30)
def tiingo_history(ticker: str, token: str, days: int) -> pd.DataFrame | None:
    """Fetch daily historical data for a US stock from Tiingo."""
    import datetime as dt
    import pandas as pd
    import requests

    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    url = f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices"

    params = {
        "token": token,
        "startDate": start,
        "resampleFreq": "daily",
        "format": "json",
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            print(f"⚠️ Tiingo fetch failed for {ticker}: {r.status_code}")
            return None  # ✅ must be indented inside the if

        data = r.json()
        if not data or not isinstance(data, list):
            print(f"⚠️ No data in Tiingo response for {ticker}")
            return None  # ✅ indented here, not global

        df = pd.DataFrame(data)
        if df.empty:
            return None  # ✅ indented inside the if

        df["date"] = pd.to_datetime(df["date"])
        df.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]].sort_values(
            "Date"
        ).reset_index(drop=True)

    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")
        return None  # ✅ inside except, properly indented
    
    # ---------------- Tiingo Sector Metadata ----------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_tiingo_sector(ticker: str, token: str) -> str:
    """Fetch sector classification for a ticker from Tiingo metadata."""
    import requests
    url = f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}"
    headers = {"Authorization": f"Token {token}"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if not r.ok:
            print(f"⚠️ Sector fetch failed for {ticker}: {r.status_code}")
            return "Unknown"
        data = r.json()
        sector = data.get("sector", "Unknown")
        if not sector:
            sector = "Unknown"
        return sector
    except Exception as e:
        print(f"❌ Error getting sector for {ticker}: {e}")
        return "Unknown"

# ---------------- Sector Snapshot (EMA20 vs EMA50) ----------------
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_sector_snapshot(token: str):
    """Return quick EMA-based trend snapshot for key sector ETFs."""
    import pandas as pd
    import requests

    sectors = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLI": "Industrials",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLB": "Materials",
        "XLRE": "Real Estate",
        "XLU": "Utilities",
        "XLC": "Communication Services",
    }

    start = (pd.Timestamp.utcnow().date() - pd.Timedelta(days=60)).isoformat()
    rows = []
    for sym, name in sectors.items():
        try:
            url = f"https://api.tiingo.com/tiingo/daily/{sym.lower()}/prices"
            headers = {"Authorization": f"Token {token}"}
            params = {"resampleFreq": "daily", "startDate": start}
            r = requests.get(url, headers=headers, params=params, timeout=10)
            if r.status_code != 200:
                continue
            df = pd.DataFrame(r.json())
            if df.empty:
                continue
            close = df["close"].astype(float)
            ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
            bias = "Uptrend" if ema20 > ema50 else "Downtrend"
            rows.append({"ETF": sym, "Sector": name, "Bias": bias})
        except Exception:
            continue

    return pd.DataFrame(rows)

# ---------------- Market Snapshot (SPY + VIX) ----------------
import datetime as dt
import pandas as pd
import requests

@st.cache_data(ttl=60 * 60 * 6)  # refresh every 6 hours
def get_market_snapshot(token: str):
    """Fetch SPY trend and volatility context for Smart Mode."""
    try:
        start = (dt.date.today() - dt.timedelta(days=60)).isoformat()
        url = f"https://api.tiingo.com/tiingo/daily/spy/prices"
        params = {"token": token, "startDate": start, "resampleFreq": "daily"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        df = pd.DataFrame(r.json())
        df["date"] = pd.to_datetime(df["date"])
        df["EMA20"] = df["close"].ewm(span=20).mean()
        df["EMA50"] = df["close"].ewm(span=50).mean()
        df["TR"] = df["close"].diff().abs()
        df["ATR20"] = df["TR"].rolling(20).mean()
        atrp = (df["ATR20"].iloc[-1] / df["close"].iloc[-1]) * 100

        bias = "Uptrend" if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1] else "Downtrend"
        vol_regime = "High Volatility" if atrp > 2.5 else "Low Volatility"

        return {
            "bias": bias,
            "vol_regime": vol_regime,
            "spy_price": round(df["close"].iloc[-1], 2),
            "atrp": round(atrp, 2)
        }
    except Exception as e:
        print(f"⚠️ Market snapshot error: {e}")
        return None





# ---------------- Trade Plan math (Target & Stop) ----------------
def trade_plan_levels(last_close: float, atr_val: float, mode: str,
                      stop_atr_mult: float, rr_mult: float) -> tuple[float, float]:
    """
    Returns (stop, target). For Pullback, assume bounce; for Breakout, assume expansion.
    """
    atr_val = float(atr_val) if pd.notna(atr_val) else 0.0
    stop = last_close - stop_atr_mult * atr_val
    if mode == "Pullback":
        target = last_close + rr_mult * (last_close - stop)        # bounce back
    elif mode == "Breakout":
        target = last_close + rr_mult * (stop_atr_mult * atr_val)  # push higher
    else:
        target = last_close + rr_mult * (stop_atr_mult * atr_val)
    return (round(max(stop, 0.01), 4), round(max(target, 0.01), 4))

# --------------------------------------------------------------
# Classify setup (Pullback / Breakout / Neutral)
# --------------------------------------------------------------
def classify_setup(last: pd.Series) -> str:
    try:
        ema20 = float(last["EMA20"])
        ema50 = float(last["EMA50"])
        rsi = float(last["RSI14"])
    except Exception:
        return "Neutral"

    ema_up = ema20 > ema50

    # Loosened rules for visibility
    if ema_up and rsi >= 50:
        return "Breakout"
    elif ema_up and rsi < 50:
        return "Pullback"
    elif not ema_up and rsi < 60:
        return "Pullback"
    else:
        return "Neutral"



# --------------------------------------------------------------
def passes_filters(last, price_min, price_max, min_volume, mode="Both") -> bool:
    """Apply meaningful filters to cut down noise, Webull-style."""
    try:
        px = float(last["Close"])
        vol = float(last["Volume"])
        rsi = float(last.get("RSI14", 50))
        ema20 = float(last.get("EMA20", np.nan))
        ema50 = float(last.get("EMA50", np.nan))
        band = float(last.get("BandPos20", 0.5))

        # --- Basic sanity filters ---
        if pd.isna(px) or pd.isna(vol):
            return False
        if px < price_min or px > price_max:
            return False
        if vol < min_volume:
            return False

        # --- Mode-based logic ---
        if mode == "Breakout":
            # Strong uptrend + momentum
            return ema20 > ema50 and rsi > 55 and band > 0.55

        elif mode == "Pullback":
            # Uptrend but short-term dip near support
            return ema20 > ema50 and rsi < 60 and band <= 0.45 and px <= ema20

        else:  # Both — include anything trending up
            return ema20 > ema50

    except Exception as e:
        print(f"⚠️ passes_filters error: {e}")
        return False


# ---------------- Setup Guidance Helper (with key level) ----------------
from typing import Optional

def setup_guidance_text(setup_type: str, key_level: Optional[float] = None) -> str:
    """
    Returns entry guidance text based on setup type.
    If key_level is provided, it appends a support (pullback) or resistance (breakout) hint.
    """
    if setup_type == "Breakout":
        msg = (
            "💥 **Breakout Setup** — Wait for price to **break and hold above resistance** "
            "on strong volume. Enter after a confirmed push/close above that zone; "
            "stop just below the breakout base or last consolidation low."
        )
        if key_level is not None:
            msg += f"\n\n🚀 **Resistance Level:** around **${key_level:.2f}** (recent swing high / breakout zone)"
        return msg

    elif setup_type == "Pullback":
        msg = (
            "📉 **Pullback Setup** — Let the dip **stabilize near support/VWAP**, then "
            "look for a **green reversal candle** on rising volume. Enter **above the reversal high**; "
            "stop below the swing low."
        )
        if key_level is not None:
            msg += f"\n\n🧭 **Support Zone:** around **${key_level:.2f}** (recent swing low / EMA20 area)"
        return msg

    elif setup_type == "Recent Close":
        msg = (
            "⏸ **Recent Close Setup** — Neutral today. Look for **next-session confirmation** "
            "above yesterday’s high with volume before entering."
        )
        if key_level is not None:
            msg += f"\n\n🔎 **Confirmation Trigger:** watch **${key_level:.2f}** (yesterday’s high) for strength."
        return msg

    return "🧭 Setup guidance unavailable."
 

# ---------------- Single ticker evaluation ----------------
def evaluate_ticker(ticker: str, mode: str, price_min: float, price_max: float, min_volume: float) -> dict | None:
    """Evaluate a single ticker and return a metrics card with trend context + near-miss detection."""
    try:
                # --- Load Smart Context if active ---
        market_bias = None
        vol_regime = None
        if st.session_state.get("smart_mode", False):
            market = get_market_snapshot(TIINGO_TOKEN)
            if market:
                market_bias = market["bias"]          # "Uptrend" or "Downtrend"
                vol_regime = market["vol_regime"]    # "High Volatility" / "Low Volatility"
                
        if market_bias:
            st.write(f"🧠 Smart bias active → {market_bias}, Volatility: {vol_regime}")


        # --- Fetch & compute indicators ---
        df = tiingo_history(ticker, TIINGO_TOKEN, SCAN_LOOKBACK_DAYS)
        if df is None or len(df) < 60:
            return None

        df = compute_indicators(df)
        last = df.iloc[-1]

        px = float(last["Close"])
        vol = float(last["Volume"])
        if pd.isna(px) or pd.isna(vol) or px < price_min or px > price_max or vol < min_volume:
            return None

        ema20 = float(last["EMA20"])
        ema50 = float(last["EMA50"])
        rsi = float(last["RSI14"])
        band = float(last["BandPos20"])
        atr = float(last.get("ATR14", np.nan))
        atr = px * 0.01 if pd.isna(atr) or atr <= 0 else atr
        support = float(last.get("LL20", np.nan))
        resistance = float(last.get("HH20", np.nan))
        
                # --- Dynamic filter tuning based on market bias ---
        rsi_buffer = 0
        band_buffer = 0

        if market_bias == "Uptrend":
            rsi_buffer = +3      # require slightly stronger RSI
            band_buffer = +0.05  # breakout band higher
        elif market_bias == "Downtrend":
            rsi_buffer = -3      # looser RSI requirement (accept early signs)
            band_buffer = -0.05  # breakout band lower

        # Example effect: RSI>55 → RSI>(55 + rsi_buffer)


        # --- Setup & Near Miss detection ---
        setup, near_miss, near_type = None, False, None

        if ema20 > ema50:
            # Confirmed setups
            if mode in ["Breakout", "Both"] and rsi > (55 + rsi_buffer) and band > (0.55 + band_buffer):
                setup = "Breakout"
            elif mode in ["Pullback", "Both"] and rsi < (60 + rsi_buffer) and band <= (0.45 + band_buffer) and px <= ema20:
                setup = "Pullback"


            # Near misses (broader tolerance window)
            if not setup:
                near_pct = 15.0       # within 15% of resistance
                near_atr_mult = 4.0   # within 4×ATR of support

                if mode in ["Breakout", "Both"] and 40 <= rsi <= 67 and 0.35 <= band <= 0.70:
                    near_miss, near_type = True, "RSI/Band breakout proximity"
                elif mode in ["Pullback", "Both"] and 40 <= rsi <= 70 and 0.20 <= band <= 0.60:
                    near_miss, near_type = True, "RSI/Band pullback proximity"

                # Check proximity to recent high/low
                if pd.notna(resistance) and resistance > 0 and (resistance - px) / resistance <= near_pct / 100:
                    near_miss, near_type = True, f"≤{near_pct:.0f}% below 20-day high"
                elif pd.notna(support) and (px - support) <= near_atr_mult * atr:
                    near_miss, near_type = True, f"≤{near_atr_mult:.1f}×ATR above 20-day low"

        # --- Skip if no signal ---
        if not setup and not near_miss:
            return None

        # --- Determine trend context ---
        if ema20 > ema50 * 1.02:
            trend_context = "Uptrend"
        elif ema20 < ema50 * 0.98:
            trend_context = "Downtrend"
        else:
            trend_context = "Sideways"

        # --- Combine setup + trend for readable context ---
        if setup:
            setup_context = f"{setup} in {trend_context}"
        elif near_miss:
            setup_context = f"Potential {near_type or 'setup'} in {trend_context}"
        else:
            setup_context = trend_context

        # --- Target / Stop (ATR-based) ---
        stop = px - atr * 1.5
        target = px + (px - stop) * 2.0
        
                # --- Smart Score calculation ---
        smart_score = 50  # neutral baseline

        # Setup strength: RSI/Band alignment
        if setup == "Breakout":
            smart_score += min((rsi - 50) * 1.2, 25)   # reward strong RSI
            smart_score += min((band - 0.5) * 50, 15)  # reward high band
        elif setup == "Pullback":
            smart_score += min((60 - rsi) * 1.2, 25)   # reward deeper dips
            smart_score += min((0.5 - band) * 50, 15)  # reward lower band

        # Trend context (Uptrend gets a bonus)
        if ema20 > ema50:
            smart_score += 10
        else:
            smart_score -= 10

        # If Smart Mode is active and favored sectors exist
        if st.session_state.get("smart_mode", False):
            favored = st.session_state.get("favored_sectors", [])
            ticker_sector = get_tiingo_sector(ticker, TIINGO_TOKEN)
            if any(s.lower() in ticker_sector.lower() for s in favored):
                smart_score += 10  # aligned with favored sector
            else:
                smart_score -= 5   # not in favored trend

        smart_score = int(np.clip(smart_score, 0, 100))
        
                # --- SmartScore Calculation (sector + market bias aware) ---
        smart_score = 50  # start neutral

        # Base on setup type
        if setup == "Breakout":
            smart_score += (rsi - 50) * 0.8 + (band - 0.5) * 40
        elif setup == "Pullback":
            smart_score += (60 - rsi) * 0.8 + (0.5 - band) * 40
        elif near_miss:
            smart_score += 5  # slight bump if it's a near setup

        # Sector alignment bonus (Smart Mode only)
        if st.session_state.get("smart_mode", False) and "smart_context" in st.session_state:
            try:
                smart_df = st.session_state["smart_context"]["sectors_df"]
                sector = get_tiingo_sector(ticker, TIINGO_TOKEN)
                bias = smart_df.loc[smart_df["Sector"] == sector, "Bias"].values[0] if sector in smart_df["Sector"].values else "Neutral"

                if bias == "Uptrend" and setup == "Breakout":
                    smart_score += 15
                elif bias == "Downtrend" and setup == "Pullback":
                    smart_score += 15
                elif bias == "Sideways":
                    smart_score -= 5
            except Exception as e:
                st.write(f"⚠️ SmartScore sector bias error: {e}")

        # Clamp SmartScore to 0–100 range
        smart_score = max(0, min(100, round(smart_score, 1)))

        # --- Build result card ---
        card = {
            "Symbol": ticker,
            "Price": round(px, 2),
            "Volume": int(vol),
            "RSI14": round(rsi, 1),
            "EMA20>EMA50": ema20 > ema50,
            "BandPos20": round(band, 2),
            "ATR14": round(atr, 2),
            "Setup": setup or "NearMiss",
            "Stop": round(stop, 2),
            "Target": round(target, 2),
            "SmartScore": smart_score,
            "NearMiss": near_miss,
            "NearType": near_type,
            "SetupContext": setup_context,   # ✅ now always included
        }

        # --- Debug (optional, visible in console/UI logs) ---
        if setup:
            st.write(f"✅ {ticker}: Confirmed {setup} | Trend={trend_context}")
        elif near_miss:
            st.write(f"🟡 {ticker}: Near Miss → {near_type} | Trend={trend_context}")
        else:
            st.write(f"⚙️ {ticker}: No match | Trend={trend_context}")

        return card

    except Exception as e:
        st.write(f"⚠️ {ticker} failed with error: {e}")
        return None


# ---------------- Scanner (full universe, concurrent) ----------------
def run_full_scan(mode: str, price_min: float, price_max: float, min_volume: float,
                  max_cards: int) -> list[dict]:
    tickers = tiingo_all_us_tickers(TIINGO_TOKEN)

    # Shuffle to diversify early results & keep UI feeling live
    random.seed()
    random.shuffle(tickers)

    results: list[dict] = []
    progress = st.progress(0, text="🔎 Scanning U.S. market…")
    total = len(tickers)
    scanned = 0

    for i in range(0, total, BATCH_TICKER_COUNT):
        if len(results) >= max_cards:
            break
        batch = tickers[i:i + BATCH_TICKER_COUNT]
        with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(evaluate_ticker, t, mode, price_min, price_max, min_volume) for t in batch]
            for f in futures.as_completed(futs):
                rec = f.result()
                if rec is not None:
                    results.append(rec)

        scanned = min(i + len(batch), total)
        progress.progress(scanned / total, text=f"🔎 Scanning… {scanned}/{total} tickers | Hits: {len(results)}")
        time.sleep(REQUEST_PAUSE_S)

    progress.empty()

    # --- Smart sorting: favor strong setups and favored sectors ---
    def sort_key(r):
        base_score = 0
        if r.get("Setup") == "Breakout":
            base_score = 100 - r.get("RSI14", 0)  # lower RSI = more potential
        elif r.get("Setup") == "Pullback":
            base_score = r.get("BandPos20", 0) * 100

        # Boost for favored sectors (Smart Mode)
        bonus = 0
        if st.session_state.get("smart_mode", False):
            smart_context = st.session_state.get("smart_context", {})
            favored = [s.lower() for s in smart_context.get("favored_sectors", [])]
            sector = str(r.get("Sector", "Unknown")).lower()
            if any(s in sector for s in favored):
                bonus += 15

        return -(base_score + bonus)

    results.sort(key=sort_key)

    # --- Separate confirmed vs near misses ---
    confirmed = [r for r in results if r.get("Setup") in ["Breakout", "Pullback"]]
    near_misses = [r for r in results if r.get("NearMiss")]

    st.session_state["scanner_results_confirmed"] = confirmed
    st.session_state["scanner_results_near"] = near_misses
    st.session_state["scanner_results"] = confirmed + near_misses   # ✅ add this line!

    st.write(f"🧪 Debug: {len(confirmed)} confirmed | {len(near_misses)} near misses collected")

    return confirmed + near_misses




# ---------------- Helper for Watchlist Screener only ----------------
def check_ticker_failure_reason(ticker, price_min, price_max, min_volume):
    """Identify why a watchlist ticker failed the filters."""
    try:
        df = fetch_tiingo(ticker, 10)
        if df.empty:
            return "no data returned from Tiingo"

        last_close = float(df["Close"].iloc[-1])
        last_volume = float(df["Volume"].iloc[-1])

        if last_close < price_min:
            return f"failed price filter (Close ${last_close:.2f} < Min ${price_min})"
        if last_close > price_max:
            return f"failed price filter (Close ${last_close:.2f} > Max ${price_max})"
        if last_volume < min_volume:
            return f"failed volume filter (Vol {last_volume:,.0f} < {min_volume:,.0f})"

        return "did not qualify based on technical setup"
    except Exception as e:
        return f"error checking failure reason: {e}"


# ---------------- Watchlist Screener (detailed reasons for failed tickers) ----------------
def run_watchlist_scan_only(watchlist, mode, price_min, price_max, min_volume):
    """Run the screener only for tickers in the user's watchlist, returning results and detailed debug info."""
    if not watchlist:
        return [], [("⚠️", "No tickers found in watchlist.")]

    results = []
    debug_log = []
    total = len(watchlist)
    progress = st.progress(0, text=f"🎯 Scanning {total} watchlist symbols...")

    for i, ticker in enumerate(watchlist, start=1):
        try:
            rec = evaluate_ticker(ticker, mode, price_min, price_max, min_volume)
            if rec:
                results.append(rec)
                debug_log.append(("✅", f"{ticker}: passed all filters"))
            else:
                reason = check_ticker_failure_reason(ticker, price_min, price_max, min_volume)
                debug_log.append(("🚫", f"{ticker}: {reason}"))
        except Exception as e:
            debug_log.append(("⚠️", f"{ticker}: error — {e}"))
        progress.progress(i / total, text=f"🎯 {i}/{total} scanned | Hits: {len(results)}")

    progress.empty()
    if not results:
        debug_log.append(("❌", "No tickers met your criteria."))
    return results, debug_log


# ---------------- UI: Controls ----------------
with st.sidebar:
    st.subheader("🔧 Market Scanner (Tiingo, U.S.)")
   # --- Setup Mode Dropdown (persistent in session_state) ---
    st.session_state["setup_mode"] = st.selectbox(
        "Setup Mode",
        ["Pullback", "Breakout", "Both"],
        index=["Pullback", "Breakout", "Both"].index(st.session_state.get("setup_mode", "Breakout"))
    )
    mode = st.session_state["setup_mode"]  # keep local reference for clarity

    c1, c2 = st.columns(2)
    with c1:
        price_min = st.number_input("Min Price ($)", value=3.0, min_value=0.0, step=0.5)
    with c2:
        price_max = st.number_input("Max Price ($)", value=200.0, min_value=0.0, step=1.0)

    min_volume = st.number_input("Min Volume (shares, latest bar)", value=500_000, step=50_000, min_value=0)
    max_cards  = st.slider("Max Cards to Show", min_value=24, max_value=500, value=120, step=12)
    # ---------------- Smart Mode toggle ----------------
    smart_mode = st.toggle("🧠 Enable Smart Mode", value=st.session_state.get("smart_mode", False))
    st.session_state["smart_mode"] = smart_mode

    # Make sure smart_context exists
    if "smart_context" not in st.session_state:
        st.session_state["smart_context"] = {}

    # ---------------- Compute favored sectors when Smart Mode is enabled ----------------
    if st.session_state["smart_mode"]:
        sector_df = get_sector_snapshot(TIINGO_TOKEN)
        favored = sector_df.loc[sector_df["Bias"] == "Uptrend", "Sector"].tolist()
        st.session_state["smart_context"]["favored_sectors"] = favored

        # Display current favored sectors in sidebar
        if favored:
            st.success("🌟 Favored sectors: " + ", ".join(favored))
        else:
            st.info("🌥️ No favored sectors right now")
    else:
        st.session_state["smart_context"]["favored_sectors"] = []

    st.markdown("---")
    st.caption("**Trade Plan Defaults** (used for Target/Stop on each card)")
    c3, c4 = st.columns(2)
    with c3:
        st.session_state["risk_stop_atr_mult"] = st.number_input("Stop = ATR ×", value=float(st.session_state["risk_stop_atr_mult"]), min_value=0.5, step=0.5)
    with c4:
        st.session_state["risk_rr"] = st.number_input("Reward Ratio (R)", value=float(st.session_state["risk_rr"]), min_value=0.5, step=0.5)

    run_scan = st.button("🚀 Run Full U.S. Scan", use_container_width=True)

# ---------------- UI: Results Grid ----------------
if st.session_state.get("smart_mode", False):
    market = get_market_snapshot(TIINGO_TOKEN)
    if market:
        st.markdown(f"""
        ### 🧭 Market Snapshot
        **SPY:** ${market['spy_price']} — *{market['bias']}*  
        Volatility: *{market['vol_regime']}* ({market['atrp']}% ATR)
        """)
    else:
        st.warning("⚠️ Could not load SPY snapshot for market bias.")

st.header("📊 Webull-Style Market Scanner — U.S. (Tiingo)")
st.caption("All active U.S. equities. Filters: price, volume, and setup mode (Pullback/Breakout/Both). Cards show your Trade Plan target & stop.")

if run_scan:
    st.session_state["scanner_running"] = True
    with st.spinner("Scanning the U.S. market... this may take 1–2 minutes"):
        current_mode = st.session_state.get("setup_mode", "Breakout")
        st.write(f"🧭 Running full scan mode: {current_mode}")
        st.session_state["scanner_results"] = run_full_scan(
            mode=current_mode,
            price_min=price_min,
            price_max=price_max,
            min_volume=min_volume,
            max_cards=max_cards
        )
    st.session_state["scanner_running"] = False

    # ---------------- Smart Mode Context (Sector Trend Awareness) ----------------
    if st.session_state.get("smart_mode", False):
        try:
            smart_context = {}
            smart_context["sectors_df"] = get_sector_snapshot(TIINGO_TOKEN)
            st.session_state["smart_context"] = smart_context
            st.caption("🧠 Smart Mode active — sector trends updated.")
        except Exception as e:
            st.warning(f"Smart Mode failed: {e}")

# safely retrieve results (don’t reset them on rerun)
results = st.session_state.get("scanner_results", [])

if not results:
    st.info("Run the full scan to see cards. Use the sidebar to set filters.")
else:
    # ✅ Separate Confirmed vs Near Miss tabs
    confirmed = st.session_state.get("scanner_results_confirmed", [])
    near_misses = st.session_state.get("scanner_results_near", [])

    tab_confirmed, tab_near = st.tabs([
        f"✅ Confirmed Setups ({len(confirmed)})",
        f"🟡 Near Misses ({len(near_misses)})"
    ])

    # --- Confirmed setups tab ---
    with tab_confirmed:
        if confirmed:
            per_row = 4
            rows = math.ceil(len(confirmed) / per_row)
            for r in range(rows):
                cols = st.columns(per_row)
                for j, col in enumerate(cols):
                    idx = r * per_row + j
                    if idx >= len(confirmed): break
                    rec = confirmed[idx]

                    # --- Smart Badge for Sector Alignment ---
                    favored_badge = ""
                    if st.session_state.get("smart_mode", False) and "smart_context" in st.session_state:
                        df = st.session_state["smart_context"]["sectors_df"]
                        sector = get_tiingo_sector(rec["Symbol"], TIINGO_TOKEN)
                        bias = df.loc[df["Sector"] == sector, "Bias"].values[0] if sector in df["Sector"].values else "Unknown"
                        if bias == "Uptrend" and rec["Setup"] == "Breakout":
                            favored_badge = "🟢 Sector Uptrend"
                        elif bias == "Downtrend" and rec["Setup"] == "Pullback":
                            favored_badge = "🔵 Sector Downtrend"

                    with col:
                        st.markdown(
                            f"""
                            <div style="
                                border:2px solid {'#22c55e' if rec['Setup']=='Breakout' else '#3b82f6'};
                                border-radius:14px;padding:10px;">
                                <div style="display:flex;justify-content:space-between;align-items:baseline;">
                                    <div style="font-weight:700;font-size:1.15rem">{rec['Symbol']}</div>
                                    <div style="font-weight:600">${rec['Price']:.2f}</div>
                                </div>
                                <span style="font-size:0.85rem;color:#facc15;">⭐ Smart Score: {rec.get('SmartScore', '—')}</span>
                                <div style="font-size:0.9rem;opacity:0.9;margin-top:4px;">
                                    Setup: <b>{rec['Setup']}</b> &nbsp;|&nbsp; RSI14: <b>{rec['RSI14']}</b> &nbsp;|&nbsp; Vol: <b>{rec['Volume']:,}</b><br/>
                                    EMA20&gt;EMA50: <b>{'✅' if rec['EMA20>EMA50'] else '❌'}</b> &nbsp;|&nbsp; BandPos20: <b>{rec['BandPos20']}</b> &nbsp;|&nbsp; ATR14: <b>{rec['ATR14']}</b>
                                </div>
                                <div style="margin-top:6px;font-size:0.95rem;line-height:1.4;">
                                    🛡️ Stop: <b>${rec['Stop']:.2f}</b><br/>
                                    🎯 <b style="color:#16a34a;">Target: ${rec['Target']:.2f}</b>
                                </div>
                                {f"<div style='margin-top:4px;font-size:0.85rem;opacity:0.85;color:#22c55e;'>{favored_badge}</div>" if favored_badge else ""}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f"🧭 Context: <b>{rec.get('SetupContext', 'N/A')}</b>",
                            unsafe_allow_html=True
                        )

                        cA, cB = st.columns(2)
                        with cA:
                            if st.button("➕ Watchlist", key=f"wl_c_{rec['Symbol']}"):
                                if rec["Symbol"] not in st.session_state.watchlist:
                                    st.session_state.watchlist.append(rec["Symbol"])
                                    st.success(f"Added {rec['Symbol']} to watchlist.")
                                else:
                                    st.info(f"{rec['Symbol']} already on watchlist.")
                        with cB:
                            if st.button("🔍 Send to Analyzer", key=f"scan_an_c_{rec['Symbol']}"):
                                st.session_state["analyze_symbol"] = rec["Symbol"]
                                st.session_state["sent_setup"] = rec.get("Setup", st.session_state.get("setup_mode", "Both"))
                                st.session_state["setup_mode"] = rec.get("Setup", "Both")
                                st.rerun()
        else:
            st.info("No confirmed setups found in this scan.")

    # --- Near Misses tab ---
    with tab_near:
        if near_misses:
            per_row = 4
            rows = math.ceil(len(near_misses) / per_row)
            for r in range(rows):
                cols = st.columns(per_row)
                for j, col in enumerate(cols):
                    idx = r * per_row + j
                    if idx >= len(near_misses): break
                    rec = near_misses[idx]
                    with col:
                        st.markdown(
                            f"""
                            <div style="
                                border:2px dashed #facc15;
                                border-radius:14px;padding:10px;">
                                <div style="display:flex;justify-content:space-between;align-items:baseline;">
                                    <div style="font-weight:700;font-size:1.15rem">{rec['Symbol']}</div>
                                    <div style="font-weight:600">${rec['Price']:.2f}</div>
                                </div>
                                <span style="font-size:0.85rem;color:#facc15;">⭐ Smart Score: {rec.get('SmartScore', '—')}</span>
                                <div style="font-size:0.9rem;opacity:0.9;margin-top:4px;">
                                    🟡 Potential <b>{rec['NearMiss']}</b> setup forming<br/>
                                    RSI14: <b>{rec['RSI14']}</b> | BandPos20: <b>{rec['BandPos20']}</b>
                                </div>
                                <div style="margin-top:6px;font-size:0.95rem;line-height:1.4;">
                                    🛡️ Stop: <b>${rec['Stop']:.2f}</b><br/>
                                    🎯 <b style="color:#16a34a;">Target: ${rec['Target']:.2f}</b>
                                </div>
                                {f"<div style='margin-top:4px;font-size:0.85rem;opacity:0.85;color:#22c55e;'>{favored_badge}</div>" if favored_badge else ""}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f"<div style='font-size:0.85rem;opacity:0.75;margin-top:-4px;'>🧭 Context: <b>{rec.get('SetupContext', 'N/A')}</b></div>",
                            unsafe_allow_html=True
                        )


                        cA, cB = st.columns(2)
                        with cA:
                            if st.button("➕ Watchlist", key=f"wl_n_{rec['Symbol']}"):
                                if rec["Symbol"] not in st.session_state.watchlist:
                                    st.session_state.watchlist.append(rec["Symbol"])
                                    st.success(f"Added {rec['Symbol']} to watchlist.")
                                else:
                                    st.info(f"{rec['Symbol']} already on watchlist.")
                        with cB:
                            if st.button("🔍 Send to Analyzer", key=f"scan_an_n_{rec['Symbol']}"):
                                st.session_state["analyze_symbol"] = rec["Symbol"]
                                st.session_state["sent_setup"] = rec.get("Setup", st.session_state.get("setup_mode","Both"))
                                st.session_state["setup_mode"] = rec.get("Setup","Both")
                                st.rerun()
        else:
            st.info("No near misses detected in this scan.")




# ---------------- Watchlist Screener Results (Main Page) ----------------
if "watchlist_results" in st.session_state and st.session_state["watchlist_results"]:
    st.markdown("---")
    with st.expander("🎯 Watchlist Screener Results", expanded=True):
        st.caption(f"{len(st.session_state['watchlist_results'])} symbols met your criteria.")

        per_row = 4
        results = st.session_state["watchlist_results"]
        rows = math.ceil(len(results) / per_row)
        for r in range(rows):
            cols = st.columns(per_row)
            for j, col in enumerate(cols):
                idx = r * per_row + j
                if idx >= len(results):
                    break
                rec = results[idx]

                with col:
                    st.markdown(
                        f"""
                        <div style="
                            border:2px solid {'#22c55e' if rec['Setup']=='Breakout' else ('#3b82f6' if rec['Setup']=='Pullback' else '#9ca3af')};
                            border-radius:14px;padding:10px;">
                            <div style="display:flex;justify-content:space-between;align-items:baseline;">
                                <div style="font-weight:700;font-size:1.15rem">{rec['Symbol']}</div>
                                <div style="font-weight:600">${rec['Price']:.2f}</div>
                            </div>
                            <div style="font-size:0.9rem;opacity:0.9;margin-top:4px;">
                                Setup: <b>{rec['Setup']}</b> &nbsp;|&nbsp; RSI14: <b>{rec['RSI14']}</b> &nbsp;|&nbsp; Vol: <b>{rec['Volume']:,}</b><br/>
                                EMA20&gt;EMA50: <b>{'✅' if rec['EMA20>EMA50'] else '❌'}</b> &nbsp;|&nbsp; BandPos20: <b>{rec['BandPos20']}</b> &nbsp;|&nbsp; ATR14: <b>{rec['ATR14']}</b>
                            </div>
                            <div style="margin-top:6px;font-size:0.95rem;line-height:1.4;">
                                🛡️ Stop: <b>${rec['Stop']:.2f}</b><br/>
                                🎯 <b style="color:#16a34a;">Target: ${rec['Target']:.2f}</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    cA, cB = st.columns(2)
                    with cA:
                        if st.button("➕ Watchlist", key=f"wl_watchlist_{rec['Symbol']}"):
                            if rec["Symbol"] not in st.session_state.watchlist:
                                st.session_state.watchlist.append(rec["Symbol"])
                                st.success(f"Added {rec['Symbol']} to watchlist.")
                            else:
                                st.info(f"{rec['Symbol']} already on watchlist.")
                    with cB:
                        if st.button("🔍 Send to Analyzer", key=f"wl_an_{rec['Symbol']}"):
                            # --- Connection bridge between Screener/Watchlist → Analyzer ---
                            st.session_state["analyze_symbol"] = rec["Symbol"]
                            st.session_state["sent_setup"] = rec.get("Setup", st.session_state.get("setup_mode", "Both"))
                            st.session_state["setup_mode"] = rec.get("Setup", "Both")  # keep global consistent

                            # If your app uses multipage navigation:
                            # st.switch_page("Analyzer")

                            # If you're running everything on one page (tabs or expanders),
                            # you can just trigger the Analyzer rerun:
                            st.rerun()


        # ✅ Move this outside the card loop (but still inside the expander)
        debug_log = st.session_state.get("watchlist_debug", [])
        if debug_log:
            st.markdown("---")
            st.markdown("### 📋 Debug Log (Watchlist Screener)")
            for icon, msg in debug_log:
                st.write(f"{icon} {msg}")



# =========================================================================================================

# ---------- Multi-Watchlist Gist System (No Default) ----------
import requests, json

# ----------------- Gist Helpers -----------------
def load_watchlists_from_gist():
    """Load all saved watchlists (dict of {name: [tickers]}) from GitHub Gist."""
    try:
        token = st.secrets.get("GITHUB_GIST_TOKEN", "")
        gist_id = st.secrets.get("GIST_ID", "")
        if not token or not gist_id:
            st.warning("⚠️ Missing Gist credentials.")
            return {}

        url = f"https://api.github.com/gists/{gist_id}"
        headers = {"Authorization": f"token {token}"}
        r = requests.get(url, headers=headers, timeout=10)
        if not r.ok:
            st.warning(f"⚠️ Gist fetch failed ({r.status_code})")
            return {}

        files = r.json().get("files", {})
        if not files:
            return {}

        content = list(files.values())[0]["content"]
        data = json.loads(content)
        if isinstance(data, list):
            # backward compatible with old single-list format
            return {"Unnamed": data}
        return data
    except Exception as e:
        st.warning(f"⚠️ Could not load from Gist: {e}")
        return {}

def save_watchlists_to_gist(watchlists_dict):
    """Save all watchlists (dict of {name: [tickers]}) to GitHub Gist."""
    try:
        token = st.secrets.get("GITHUB_GIST_TOKEN", "")
        gist_id = st.secrets.get("GIST_ID", "")
        if not token or not gist_id:
            st.warning("⚠️ Missing Gist credentials in secrets.")
            return

        url = f"https://api.github.com/gists/{gist_id}"
        headers = {"Authorization": f"token {token}"}
        payload = {
            "files": {"watchlist.json": {"content": json.dumps(watchlists_dict, indent=2)}}
        }
        r = requests.patch(url, headers=headers, json=payload, timeout=10)
        if not r.ok:
            st.warning(f"⚠️ Failed to save watchlists: {r.status_code}")
    except Exception as e:
        st.warning(f"⚠️ Could not save to Gist: {e}")


# ----------------- Session Initialization -----------------
if "watchlists" not in st.session_state:
    st.session_state.watchlists = load_watchlists_from_gist()

if "active_watchlist" not in st.session_state:
    keys = list(st.session_state.watchlists.keys())
    st.session_state.active_watchlist = keys[0] if keys else None

if "watchlist" not in st.session_state:
    if st.session_state.active_watchlist:
        st.session_state.watchlist = st.session_state.watchlists.get(
            st.session_state.active_watchlist, []
        )
    else:
        st.session_state.watchlist = []


# ----------------- Sidebar Watchlist Manager -----------------
with st.sidebar.expander("📂 Watchlist"):
    st.markdown("### Watchlist Manager")

    all_names = list(st.session_state.watchlists.keys())
    if not all_names:
        st.info("No watchlists yet — create one below 👇")
        selected_name = "➕ Create New"
    else:
        selected_name = st.selectbox(
            "Active Watchlist",
            options=all_names + ["➕ Create New"],
            index=all_names.index(st.session_state.active_watchlist)
            if st.session_state.active_watchlist in all_names
            else len(all_names),
        )
        
    # 🗑️ Delete current watchlist (with working confirmation)
    if selected_name not in ["➕ Create New"] and all_names:
        with st.expander("⚠️ Delete This Watchlist"):
            st.warning(f"Deleting '{selected_name}' will permanently remove it from cloud storage.")
            if st.button(f"🗑️ Confirm Delete '{selected_name}'", key=f"confirm_delete_{selected_name}"):
                try:
                    del st.session_state.watchlists[selected_name]
                    save_watchlists_to_gist(st.session_state.watchlists)
                    st.session_state.active_watchlist = None
                    st.session_state.watchlist = []
                    st.success(f"✅ Watchlist '{selected_name}' deleted successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete '{selected_name}': {e}")

    

    # --- Create New Watchlist ---
    if selected_name == "➕ Create New":
        new_name = st.text_input("New watchlist name:")
        if st.button("Create Watchlist"):
            if new_name.strip():
                new_name = new_name.strip()
                st.session_state.watchlists[new_name] = []
                st.session_state.active_watchlist = new_name
                st.session_state.watchlist = []
                save_watchlists_to_gist(st.session_state.watchlists)
                st.success(f"Created new watchlist: '{new_name}'")
                st.rerun()

    else:
        # Load selected watchlist
        st.session_state.active_watchlist = selected_name
        st.session_state.watchlist = st.session_state.watchlists.get(selected_name, [])

        # --- Add Symbols ---
        add_manual = st.text_input(
            "Add Symbol(s)",
            placeholder="e.g. AAPL or AAPL, MSFT, TSLA",
            key="wl_add_symbol",
        )
        if st.button("Add to Watchlist"):
            if add_manual:
                symbols = [
                    s.strip().upper()
                    for s in add_manual.replace(",", " ").split()
                    if s.strip()
                ]
                new_syms = [s for s in symbols if s not in st.session_state.watchlist]
                if new_syms:
                    st.session_state.watchlist.extend(new_syms)
                    st.session_state.watchlists[
                        st.session_state.active_watchlist
                    ] = st.session_state.watchlist
                    save_watchlists_to_gist(st.session_state.watchlists)
                    st.success(f"Added: {', '.join(new_syms)}")
                    st.rerun()
                else:
                    st.info("All entered symbols already exist.")
                            

        # --- Display Watchlist (Dropdown Layout) ---
        if st.session_state.watchlist:
            st.markdown("---")
            st.markdown(f"**Current Watchlist ({len(st.session_state.watchlist)})**")

            selected = st.selectbox(
                "Pick a ticker",
                options=st.session_state.watchlist,
                key="wl_pick_for_actions",
            )

            c1, c2, c3 = st.columns([2, 2, 3])
            if c1.button("🔍 Analyze", key="wl_analyze_selected"):
                st.session_state["analyze_symbol"] = selected
                st.toast(f"Sent {selected} to Analyzer", icon="🔍")
                st.rerun()

            if c2.button("❌ Remove", key="wl_remove_selected"):
                st.session_state.watchlist.remove(selected)
                st.session_state.watchlists[
                    st.session_state.active_watchlist
                ] = st.session_state.watchlist
                save_watchlists_to_gist(st.session_state.watchlists)
                st.toast(f"Removed {selected}", icon="❌")
                st.rerun()

            if st.button("🗑️ Clear Watchlist"):
                st.session_state.watchlist = []
                st.session_state.watchlists[
                    st.session_state.active_watchlist
                ] = []
                save_watchlists_to_gist(st.session_state.watchlists)
                st.warning("Cleared and synced watchlist.")
                
        # --- Watchlist Screener trigger ---
        if st.button("🎯 Run Watchlist Screener", use_container_width=True):
            if not st.session_state.watchlist:
                st.warning("⚠️ Your watchlist is empty.")
            else:
                with st.spinner(f"Scanning {len(st.session_state.watchlist)} symbols..."):
                    results, debug_log = run_watchlist_scan_only(
                        st.session_state.watchlist,
                        mode,
                        price_min,
                        price_max,
                        min_volume,
                    )
                st.session_state["watchlist_results"] = results
                st.session_state["watchlist_debug"] = debug_log
                st.session_state["show_watchlist_results"] = True
                st.rerun()





# ---------------- Analyzer ----------------
st.subheader("🔍 Analyzer — with RSI, MACD, ATR")

# Use symbol from session if sent, otherwise default to AAPL
default_symbol = st.session_state.get("analyze_symbol", "AAPL")
symbol = st.text_input("Symbol", default_symbol or "AAPL").upper()

# If a new symbol is entered manually, update the session state
if symbol != st.session_state.get("analyze_symbol"):
    st.session_state["analyze_symbol"] = symbol

# Auto-run analyzer when ticker was sent from Watchlist or Scanner
auto_trigger = st.session_state.get("analyze_symbol") == symbol
run_analysis = st.button("Analyze") or auto_trigger

if run_analysis:
    df = fetch_tiingo(symbol, years)
    if not df.empty:
        # --- Indicators ---
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

        # RSI (14)
        delta = df["Close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(gain).rolling(14).mean()
        roll_down = pd.Series(loss).rolling(14).mean()
        rs = roll_up / (roll_down + 1e-9)
        df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))

        # MACD (12,26,9)
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

        # ATR (14)
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift(1))
        low_close = np.abs(df["Low"] - df["Close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR14"] = tr.rolling(14).mean()

        # --- Plot main chart + subplots ---
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"{symbol} Price & EMAs", "MACD", "RSI(14)")
        )

        # Candlestick + EMAs
        fig.add_trace(go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA20"], mode="lines", name="EMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], mode="lines", name="EMA50"), row=1, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], mode="lines", name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_SIGNAL"], mode="lines", name="Signal"), row=2, col=1)
        fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_HIST"], name="Hist"), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI14"], mode="lines", name="RSI(14)"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- Metrics ---
        st.metric("Last Close", f"${df.iloc[-1]['Close']:.2f}")
        st.metric("ATR(14)", f"{df.iloc[-1]['ATR14']:.2f}")
        st.metric("RSI(14)", f"{df.iloc[-1]['RSI14']:.1f}")
        st.metric("MACD", f"{df.iloc[-1]['MACD']:.2f}")
        
                # --- Forecast, ML Edge, Seasonality, and Sentiment ---
        st.divider()
        st.subheader("🧠 Forecasts, ML Edge, Seasonality & Sentiment")

        try:
            # ===================== 1️⃣ TinyToy Forecast =====================
            from sklearn.linear_model import LinearRegression

            lookback = 20
            df_recent = df.tail(lookback).reset_index(drop=True)
            df_recent["Index"] = np.arange(len(df_recent))

            model = LinearRegression().fit(df_recent[["Index"]], df_recent["Close"])
            predicted_price = float(model.predict([[len(df_recent)]])[0])
            last_price = float(df_recent["Close"].iloc[-1])
            forecast_change = predicted_price - last_price
            forecast_pct = (forecast_change / last_price) * 100
            direction = "⬆️ Up" if forecast_change > 0 else "⬇️ Down"

            # ===================== 2️⃣ ML Edge =====================
            ema_trend = (df_recent["EMA20"].iloc[-1] - df_recent["EMA50"].iloc[-1]) / df_recent["Close"].iloc[-1]
            rsi_val = df_recent["RSI14"].iloc[-1]
            rsi_edge = (50 - abs(50 - rsi_val)) / 50
            atr_vol = df_recent["ATR14"].iloc[-1] / df_recent["Close"].iloc[-1]
            ml_edge_score = max(0.0, min(1.0, (ema_trend * 5 + rsi_edge - atr_vol * 2)))

            # ===================== 3️⃣ Seasonality =====================
            df["Month"] = pd.to_datetime(df["Date"]).dt.month
            monthly_returns = df.groupby("Month")["Close"].apply(lambda x: x.pct_change().mean() * 100)
            this_month = pd.Timestamp.now().month
            seasonality_avg = float(monthly_returns.get(this_month, 0.0))

            # ===================== 4️⃣ Sentiment =====================
            import requests
            from textblob import TextBlob

            sentiment_score = 0.0
            try:
                api_key = os.getenv("TIINGO_API_KEY")
                news_url = f"https://api.tiingo.com/tiingo/news?tickers={symbol}&token={api_key}"
                r = requests.get(news_url, timeout=5)
                if r.ok:
                    articles = r.json()[:5]
                    sentiments = []
                    for art in articles:
                        title = art.get("title", "")
                        polarity = TextBlob(title).sentiment.polarity
                        sentiments.append(polarity)
                    if sentiments:
                        sentiment_score = np.mean(sentiments)
            except Exception:
                pass

            sentiment_label = (
                "😊 Positive" if sentiment_score > 0.05 else
                "😐 Neutral" if sentiment_score >= -0.05 else
                "😟 Negative"
            )

            # ===================== 5️⃣ Visual Dashboard =====================
            c1, c2, c3, c4 = st.columns(4)

            # 🧭 Forecast Color
            forecast_color = "green" if forecast_change > 0 else "red"
            forecast_arrow = "⬆️" if forecast_change > 0 else "⬇️"

            # 🧠 ML Edge Color
            if ml_edge_score >= 0.7:
                edge_color = "green"
                edge_label = "Strong Edge"
            elif ml_edge_score >= 0.4:
                edge_color = "orange"
                edge_label = "Moderate Edge"
            else:
                edge_color = "red"
                edge_label = "Weak Edge"

            # 🌤️ Seasonality Color
            season_color = "green" if seasonality_avg > 0 else "red"

            # 📰 Sentiment Color
            if sentiment_score > 0.05:
                senti_color = "green"
            elif sentiment_score < -0.05:
                senti_color = "red"
            else:
                senti_color = "gray"

            with c1:
                st.markdown(
                    f"<div style='background-color:{forecast_color};padding:10px;border-radius:10px;color:white;text-align:center;'>"
                    f"<b>TinyToy Forecast</b><br>{forecast_arrow} ${predicted_price:.2f}<br>"
                    f"{forecast_pct:+.2f}%</div>",
                    unsafe_allow_html=True
                )

            with c2:
                st.markdown(
                    f"<div style='background-color:{edge_color};padding:10px;border-radius:10px;color:white;text-align:center;'>"
                    f"<b>ML Edge</b><br>{ml_edge_score*100:.1f}%<br>{edge_label}</div>",
                    unsafe_allow_html=True
                )

            with c3:
                st.markdown(
                    f"<div style='background-color:{season_color};padding:10px;border-radius:10px;color:white;text-align:center;'>"
                    f"<b>Seasonality</b><br>{seasonality_avg:+.2f}%</div>",
                    unsafe_allow_html=True
                )

            with c4:
                st.markdown(
                    f"<div style='background-color:{senti_color};padding:10px;border-radius:10px;color:white;text-align:center;'>"
                    f"<b>Sentiment</b><br>{sentiment_label}</div>",
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.warning(f"⚠️ Forecast/Edge/Seasonality/Sentiment failed: {e}")


        
                        # --- Trade Planning (smart-aggressive, recency & sanity guards) ---
        st.divider()
        st.subheader("🧾 Trade Plan")

        # --- Risk settings ---
        account = float(st.session_state.get("account_size", 10_000.0))
        risk_pct = float(st.session_state.get("risk_per_trade", 1.0))
        rr_ratio = float(st.session_state.get("rr_ratio", 2.0))

        # Prefer setup sent from Screener/Watchlist; else fall back
        setup_mode = st.session_state.get("sent_setup") or st.session_state.get("setup_mode", "Both")

        # --- Signal definitions ---
        df = df.copy()
        ema20 = df["EMA20"]
        atr14 = df["ATR14"]
        last_close = float(df["Close"].iloc[-1])

        # Breakout = cross up through EMA20
        df["BreakoutSig"] = (df["Close"] > ema20) & (df["Close"].shift(1) <= ema20.shift(1))

        # Smart-aggressive Pullback = EMA20 rising, prior bar below EMA20, now close back above, and within 3% of EMA20
        recent_up = ema20 > ema20.shift(5)
        df["PullbackSig"] = (
            recent_up &
            (df["Close"].shift(1) < ema20.shift(1)) &
            (df["Close"] > ema20) &
            ((df["Close"] - ema20).abs() / ema20 <= 0.03)
        )

        # --- Recency guard: only consider signals from the last N bars ---
        RECENT_BARS = 50
        recent_idx = set(df.index[-RECENT_BARS:].tolist())
        breakout_idxs = [i for i in df.index[df["BreakoutSig"]].tolist() if i in recent_idx]
        pullback_idxs = [i for i in df.index[df["PullbackSig"]].tolist() if i in recent_idx]

        # --- Choose entry candle by setup ---
        entry_row, entry_signal = None, "Recent Close"
        if setup_mode == "Breakout" and breakout_idxs:
            entry_row = df.loc[breakout_idxs[-1]]
            entry_signal = "Breakout"
        elif setup_mode == "Pullback" and pullback_idxs:
            entry_row = df.loc[pullback_idxs[-1]]
            entry_signal = "Pullback"
        elif setup_mode == "Both":
            if pullback_idxs:
                entry_row = df.loc[pullback_idxs[-1]]; entry_signal = "Pullback"
            elif breakout_idxs:
                entry_row = df.loc[breakout_idxs[-1]]; entry_signal = "Breakout"

        if entry_row is None:
            entry_row = df.iloc[-1]  # graceful fallback

        entry_price = float(entry_row["Close"])
        atr = float(entry_row.get("ATR14", np.nan))
        ema20_now = float(ema20.iloc[-1])

        # --- Key level for guidance (support/resistance/confirm) ---
        LEVEL_LOOKBACK = 10  # tweak if desired (5–20)
        key_level = None

        if entry_signal == "Pullback":
            # recent local support: lowest low of last N bars
            key_level = float(df["Low"].tail(LEVEL_LOOKBACK).min())

        elif entry_signal == "Breakout":
            # recent resistance: highest high of last N bars
            key_level = float(df["High"].tail(LEVEL_LOOKBACK).max())

        elif entry_signal.startswith("Recent Close"):
            # neutral setup: use yesterday's high as confirmation trigger
            if len(df) >= 2:
                key_level = float(df["High"].iloc[-2])

        # --- Sanity guard: if entry is >15% away from current price, use recent close instead ---
        if abs(entry_price - last_close) / max(1e-6, last_close) > 0.15:
            entry_price = last_close
            entry_signal = f"{entry_signal} (sanity fallback)"

        # --- Stops: swing-low or (EMA20 - 1.3*ATR), whichever is lower; with fallback if ATR missing ---
        swing_low = float(df["Low"].tail(10).min())
        if not np.isnan(atr) and atr > 0:
            atr_stop = ema20_now - 1.3 * atr
            proposed_stop = min(swing_low, atr_stop)
            if proposed_stop >= entry_price:
                proposed_stop = entry_price - 1.2 * atr  # rare edge case
        else:
            proposed_stop = entry_price * 0.97
        stop = max(0.01, proposed_stop)

        # --- Target: R-multiple vs recent swing-high (take farther if above entry) ---
        risk_per_share = max(1e-6, entry_price - stop)
        rr_target = entry_price + rr_ratio * risk_per_share
        prior_high = float(df["High"].tail(10).max())
        target = max(rr_target, prior_high) if prior_high > entry_price else rr_target

        # --- Sizing & reward ---
        risk_amt = account * (risk_pct / 100.0)
        shares = int(risk_amt // risk_per_share) if risk_per_share > 0 else 0
        reward = (target - entry_price) * shares

        # --- ETA: more realistic pace ---
        avg_abs_diff = df["Close"].diff().abs().tail(20).mean()
        if atr and atr > 0:
            pace = float(max(avg_abs_diff if pd.notna(avg_abs_diff) else 0.0, 0.7 * atr))
        else:
            pace = float(avg_abs_diff) if pd.notna(avg_abs_diff) else np.nan
        eta_days = (target - entry_price) / pace if (pace and pace > 0) else np.nan

        # --- Display (same layout you had) ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Setup Type", entry_signal)
            st.metric("Entry Price", f"${entry_price:.2f}")
        with c2:
            st.metric("Stop Loss", f"${stop:.2f}")
            st.metric("Target Price", f"${target:.2f}")
        with c3:
            st.metric("R:R Ratio", f"{rr_ratio:.2f}")
            st.metric("Shares to Buy", f"{shares}")
        with c4:
            st.metric("ETA to Target", f"{eta_days:.1f} days" if not np.isnan(eta_days) else "—")
            st.metric("Potential Reward ($)", f"{reward:,.2f}")

        st.caption(
            "Smart-aggressive: use only fresh signals (≤50 bars), pullback reclaims ≤3% above EMA20, "
            "and sanity-fallback if entry drifts >15% from market. ETA uses max(20-bar avg move, 0.7×ATR)."
        )


        # 💡 Setup guidance expander
        with st.expander("💡 How to Trade This Setup", expanded=True):
            st.markdown(setup_guidance_text(entry_signal, key_level))






# Use session state only — no duplicate defaults
st.sidebar.number_input("Account Size ($)", min_value=0.0, key="account_size")
st.sidebar.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, key="risk_per_trade")
st.sidebar.number_input("Target R:R Ratio", min_value=0.5, max_value=10.0, step=0.1, key="rr_ratio")
