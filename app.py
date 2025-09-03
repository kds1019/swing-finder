# alerts.py
# Scans a watchlist OR a whole universe (S&P500 / NASDAQ100) and sends Discord alerts.
# Educational only. Not financial advice.

import os, re, io, datetime as dt
import pandas as pd, numpy as np
import yfinance as yf
import requests

# ---------- env / secrets ----------
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# Your published Google Sheets CSV (env WATCHLIST_URL can override)
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSkYP_WYicxul3vA__28bHE_HyTPGcr3fLgIeRo3Ki9DevytIYjm3REeB1AhNUDrtG2I4pNYmlO8ikf/pub?output=csv"
WATCHLIST_URL = os.environ.get("WATCHLIST_URL", DEFAULT_SHEET_URL).strip()

# Fallback list if the sheet can’t be read
WATCHLIST = os.environ.get("WATCHLIST", "AAPL,MSFT,NVDA,TSLA,META")

# Optional dry-run (print instead of posting)
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() in ("1","true","yes","on")

# Universe mode (optional)
USE_UNIVERSE = os.environ.get("USE_UNIVERSE","false").lower() in ("1","true","yes","on")
UNIVERSE = os.environ.get("UNIVERSE","SP500")  # SP500 or NASDAQ100
YEARS = float(os.environ.get("YEARS","2"))

# Filters when USE_UNIVERSE=true
PRICE_MIN = float(os.environ.get("PRICE_MIN","5"))
PRICE_MAX = float(os.environ.get("PRICE_MAX","20"))
MIN_DOLLAR_VOL_M = float(os.environ.get("MIN_DOLLAR_VOL_M","2.0"))
SETUP_MODE = os.environ.get("SETUP_MODE","Both")  # Both, Pullback, Breakout

# Trade plan params (for suggested levels)
ACCOUNT_SIZE = float(os.environ.get("ACCOUNT_SIZE","10000"))
RISK_PCT = float(os.environ.get("RISK_PCT","1.0"))
STOP_ATR_MULT = float(os.environ.get("STOP_ATR_MULT","1.5"))
TARGET_RR = float(os.environ.get("TARGET_RR","2.0"))
ENTRY_STYLE = os.environ.get("ENTRY_STYLE","Pullback to EMA20")

# Manual test toggle (from workflow input)
TEST_ALERT = os.environ.get("TEST_ALERT","false").lower() in ("1","true","yes","on")

# ---------- discord ----------
def send_discord(url: str, text: str):
    if DRY_RUN or not url:
        print(("DRY_RUN active; " if DRY_RUN else "No DISCORD_WEBHOOK_URL set; ") + "printing instead:\n", text)
        return
    url = url.strip()
    if not re.match(r"^https://discord(?:app)?\.com/api/webhooks/\d+/.+$", url):
        print("Discord webhook URL looks invalid. Expected /webhooks/<id>/<token>.")
        return
    payload = {"content": text[:1900] + ("… (truncated)" if len(text) > 1900 else "")}
    try:
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code not in (200, 204):
            print(f"Discord webhook error: {r.status_code} {r.text}")
            if r.status_code == 404: print("Hint: Unknown webhook (deleted or wrong URL). Recreate & recopy.")
            elif r.status_code == 401: print("Hint: Unauthorized. Token part likely wrong.")
            elif r.status_code == 405: print("Hint: Method Not Allowed. URL is not a webhook endpoint.")
    except Exception as e:
        print("Discord webhook request failed:", e)

# ---------- indicators ----------
def ema(s, span): return pd.Series(s, index=s.index).ewm(span=span, adjust=False).mean()
def sma(s, window): return pd.Series(s, index=s.index).rolling(window).mean()

def rsi(series, period=14):
    if isinstance(series, pd.DataFrame): series = series.iloc[:,0]
    series = pd.to_numeric(pd.Series(series).squeeze(), errors="coerce")
    d = series.diff()
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    g = pd.Series(gain, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    l = pd.Series(loss, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.bfill()  # fix FutureWarning

def macd(series, fast=12, slow=26, signal=9):
    fast_ = ema(series, fast); slow_ = ema(series, slow)
    line = fast_ - slow_; sig = ema(line, signal); hist = line - sig
    return line, sig, hist

def atr(df, period=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([(df["High"]-df["Low"]).abs(), (df["High"]-pc).abs(), (df["Low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def build_indicators(df):
    df = df.copy()
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            s=df[c]
            if isinstance(s,pd.DataFrame): s=s.iloc[:,0]
            df[c]=pd.to_numeric(pd.Series(s).squeeze(), errors="coerce")
    df["SMA20"]=sma(df["Close"],20); df["SMA50"]=sma(df["Close"],50)
    df["EMA20"]=ema(df["Close"],20); df["EMA50"]=ema(df["Close"],50)
    df["RSI14"]=rsi(df["Close"],14)
    m,sig,_=macd(df["Close"]); df["MACD"]=m; df["MACDsig"]=sig
    df["ATR14"]=atr(df); df["HH20"]=df["High"].rolling(20).max(); df["LL20"]=df["Low"].rolling(20).min()
    bbw=20; bbstd=pd.Series(df["Close"], index=df.index).rolling(bbw).std()
    df["BB_MID"]=sma(df["Close"],bbw); df["BB_UP"]=df["BB_MID"]+2*bbstd; df["BB_DN"]=df["BB_MID"]-2*bbstd
    cu=(df["EMA20"]>df["EMA50"]) & (df["EMA20"].shift(1)<=df["EMA50"].shift(1))
    cd=(df["EMA20"]<df["EMA50"]) & (df["EMA20"].shift(1)>=df["EMA50"].shift(1))
    df["buy_signal"]=cu & (df["RSI14"]>50) & (df["Close"]>df["SMA50"])
    df["sell_signal"]=cd & (df["RSI14"]<50) & (df["Close"]<df["SMA50"])
    return df.dropna()

# ---------- robust data loader (no custom session) ----------
def load_data(ticker: str, years: float=2.0) -> pd.DataFrame:
    t = ticker.strip().upper()
    end = dt.date.today()
    start = end - dt.timedelta(days=int(years*365.25))

    # Preferred: Ticker.history (more stable)
    try:
        tk = yf.Ticker(t)
        df = tk.history(start=start, end=end, auto_adjust=True)
        if df is not None and not df.empty:
            df = df.rename(columns={c: c.capitalize() for c in df.columns})
            keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
            out = df[keep].dropna()
            if not out.empty: return out
    except Exception as e:
        print(f"[history] {t} failed: {e}")

    # Fallback: download()
    try:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
            out = df[keep].dropna()
            if not out.empty: return out
    except Exception as e:
        print(f"[download] {t} failed: {e}")

    return pd.DataFrame()

# ---------- watchlist / universe ----------
def _candidate_sheet_urls(url: str) -> list[str]:
    """Generate reasonable variants that often fix 404s on published CSV links."""
    u = (url or "").strip()
    u = re.sub(r"\s+", "", u)  # strip invisible whitespace
    cands = [u]
    # If it's a published link without a gid, try forcing the first tab
    # .../pub?output=csv -> .../pub?output=csv&single=true&gid=0
    if "/pub" in u and "output=csv" in u and "gid=" not in u:
        sep = "&" if "?" in u else "?"
        cands.append(f"{u}{sep}single=true&gid=0")
    # De-duplicate while preserving order
    seen, out = set(), []
    for x in cands:
        if x and x not in seen:
            out.append(x); seen.add(x)
    return out

def _read_csv_strict(url: str) -> pd.DataFrame:
    """Fetch with requests first so we can surface HTTP codes, then let pandas parse."""
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return pd.read_csv(io.StringIO(r.text))

def load_watchlist_from_url(url: str) -> list[str]:
    """
    Reads tickers from your Google Sheet (published CSV).
    Accepts both headerless and headered sheets; prefers 'Ticker'/'Symbol' column if present.
    """
    last_err = None
    tried = []
    for candidate in _candidate_sheet_urls(url):
        tried.append(candidate)
        try:
            df = _read_csv_strict(candidate)
            # Prefer a labeled column if present; otherwise use first column
            cols = [c for c in df.columns if str(c).strip().lower() in ("ticker","symbol")]
            series = df[cols[0]] if cols else df.iloc[:, 0]
            syms = [str(t).strip().upper() for t in series.tolist() if str(t).strip()]
            if syms:
                print(f"[watchlist] Loaded {len(syms)} tickers from: {candidate}")
                return syms
            last_err = "No tickers found"
        except Exception as e:
            last_err = str(e)
            continue
    print(f"[watchlist] failed to read CSV from URL. Last error: {last_err}\n"
          f"  Tried: {', '.join(tried)}\n"
          f"  Hints:\n"
          f"   • Publish the exact tab to the web (File → Share → Publish to web), format CSV.\n"
          f"   • If multiple tabs, add &single=true&gid=<TAB_GID>.\n"
          f"   • Ensure the sheet is public: Anyone with link → Viewer.\n"
          f"   • Remove any trailing spaces/newlines in the URL.")
    return []

def get_watchlist() -> list[str]:
    if WATCHLIST_URL:
        wl = load_watchlist_from_url(WATCHLIST_URL)
        if wl:
            return wl
    # fallback env WATCHLIST
    fallback = [t.strip().upper() for t in WATCHLIST.split(",") if t.strip()]
    print(f"[watchlist] Using fallback list ({len(fallback)}): {', '.join(fallback[:10])}...")
    return fallback

def get_sp500_tickers() -> list[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        for t in tables:
            cand = [c for c in t.columns if str(c).strip().lower() in ("symbol","ticker")]
            if cand:
                col = cand[0]
                syms = t[col].astype(str).tolist()
                return [s.replace(".","-").upper().strip() for s in syms if str(s).strip()]
        return []
    except Exception as e:
        print("[sp500] failed:", e)
        return []

def get_nasdaq100_tickers() -> list[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            cand = [c for c in t.columns if str(c).strip().lower() in ("symbol","ticker")]
            if cand:
                col = cand[0]
                syms = t[col].astype(str).tolist()
                return [s.replace(".","-").upper().strip() for s in syms if str(s).strip()]
        return []
    except Exception as e:
        print("[nasdaq100] failed:", e)
        return []

def good_position_flags(last: pd.Series, mode: str):
    trend_up = bool(last["EMA20"]>last["EMA50"])
    rsi_v = float(last["RSI14"])
    close = float(last["Close"])
    bb_dn = float(last.get("BB_DN", np.nan)); bb_up = float(last.get("BB_UP", np.nan))
    hh20 = float(last.get("HH20", np.nan))
    band = np.nan
    if np.isfinite(bb_dn) and np.isfinite(bb_up) and (bb_up-bb_dn)>0:
        band = (close-bb_dn)/(bb_up-bb_dn)
    buy = bool(last.get("buy_signal",False)); ok=False
    if mode in ("Pullback","Both"):
        if trend_up and (45<=rsi_v<=60) and (0.1 <= (band if np.isfinite(band) else 0.5) <= 0.5):
            ok=True
    if mode in ("Breakout","Both"):
        near_high = np.isfinite(hh20) and (close>=0.995*hh20) and trend_up and (rsi_v>=50)
        if buy or near_high: ok=True
    return ok, 0, band

# ---------- scanning / alerting ----------
def plan_trade(latest, trend_up, atr_val, hh20, ll20, ema20, close,
               account_size, risk_pct, stop_atr_mult, target_rr, entry_style):
    risk_d = max(0.0, account_size*(risk_pct/100.0))
    if entry_style=="Pullback to EMA20":
        entry=float(ema20)
        if trend_up:
            stop=entry - stop_atr_mult*atr_val; target=entry + target_rr*(entry-stop)
        else:
            stop=entry + stop_atr_mult*atr_val; target=entry - target_rr*(stop-entry)
    else:
        if trend_up:
            entry=float(max(close, hh20)); stop=entry - stop_atr_mult*atr_val; target=entry + target_rr*(entry-stop)
        else:
            entry=float(min(close, ll20)); stop=entry + stop_atr_mult*atr_val; target=entry - target_rr*(stop-entry)
    sd=abs(entry-stop); shares=int((risk_d/sd)) if sd>0 and risk_d>0 else 0
    rr=(abs(target-entry)/sd) if sd>0 else float("nan")
    return entry,stop,target,shares,rr

def format_alert(t: str, last: pd.Series, entry_style: str):
    trend_up = bool(last["EMA20"]>last["EMA50"])
    close=float(last["Close"]); rsi=float(last["RSI14"]); atr=float(last["ATR14"])
    hh20=float(last.get("HH20", close)); ll20=float(last.get("LL20", close)); ema20=float(last["EMA20"])
    entry,stop,target,shares,rr = plan_trade(last, trend_up, atr, hh20, ll20, ema20, close,
                                            ACCOUNT_SIZE, RISK_PCT, STOP_ATR_MULT, TARGET_RR, entry_style)
    side = "BUY" if bool(last.get("buy_signal", False)) else ("SELL" if bool(last.get("sell_signal", False)) else "INFO")
    when = last.name.strftime("%Y-%m-%d")
    return (
        f"**Swing Alert — {t} ({when})**\n"
        f"Signal: **{side}**  |  Bias: **{'LONG' if trend_up else 'SHORT'}**\n"
        f"Close: {close:,.2f}  |  RSI14: {rsi:.1f}  |  ATR14: {atr:.2f}\n"
        f"Plan ({entry_style}): Entry ~ **{entry:,.2f}** | Stop **{stop:,.2f}** | Target **{target:,.2f}**\n"
        f"Shares: {shares:,}  |  Approx R:R: {rr:.2f}\n"
        f"_Educational only — not financial advice._"
    )

def scan_watchlist_and_alert():
    msgs=[]
    tickers = get_watchlist()
    for t in tickers:
        try:
            df = load_data(t, YEARS)
            if df.empty:
                print(f"[skip] No data for {t} (Yahoo blocked? bad symbol? network?)")
                continue
            df = build_indicators(df); last=df.iloc[-1]
            if not (bool(last["buy_signal"]) or bool(last["sell_signal"])): 
                continue
            msgs.append(format_alert(t, last, ENTRY_STYLE))
        except Exception as e:
            print(f"[warn] {t}: {e}")
    return msgs

def scan_universe_and_alert():
    tickers = get_sp500_tickers() if UNIVERSE.upper() in ("SP500","S&P500","S&P 500") else get_nasdaq100_tickers()
    msgs=[]
    for t in tickers:
        try:
            df = load_data(t, YEARS)
            if df.empty:
                continue
            df = build_indicators(df); last=df.iloc[-1]
            close=float(last["Close"])
            if not (PRICE_MIN <= close <= PRICE_MAX): continue
            adv=(df["Close"]*df["Volume"]).rolling(20).mean().iloc[-1]
            adv_m=float(adv)/1_000_000.0 if pd.notna(adv) else 0.0
            if adv_m < MIN_DOLLAR_VOL_M: continue
            ok,_,_ = good_position_flags(last, SETUP_MODE)
            if not ok: continue
            msgs.append(format_alert(t, last, ENTRY_STYLE))
        except Exception as e:
            print(f"[warn] {t}: {e}")
    return msgs

def main():
    if TEST_ALERT:
        send_discord(DISCORD_WEBHOOK_URL, f"✅ Test alert {dt.datetime.now():%Y-%m-%d %H:%M} — pipeline OK.")
        print("Sent test alert."); return
    msgs = scan_universe_and_alert() if USE_UNIVERSE else scan_watchlist_and_alert()
    if msgs:
        send_discord(DISCORD_WEBHOOK_URL, "\n\n".join(msgs))
        print(f"Sent {len(msgs)} alert(s).")
    else:
        print("No alerts this run.")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            send_discord(DISCORD_WEBHOOK_URL, f"❌ Alerts error: {e}")
        except Exception:
            pass
        raise SystemExit(0)
