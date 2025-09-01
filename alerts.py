# alerts.py
# Scans a watchlist OR a whole universe (S&P500 / NASDAQ100) and sends Discord alerts.
# Educational only. Not financial advice.

import os, datetime as dt
import pandas as pd, numpy as np, yfinance as yf, requests

# ---------- env / secrets ----------
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
WATCHLIST_URL = os.environ.get("WATCHLIST_URL", "")
WATCHLIST = os.environ.get("WATCHLIST", "AAPL,MSFT,NVDA,TSLA,META")

# Universe mode (optional)
USE_UNIVERSE = str(os.environ.get("USE_UNIVERSE","false")).lower() in ("1","true","yes","on")
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
TEST_ALERT = str(os.environ.get("TEST_ALERT","false")).lower() in ("1","true","yes","on")

# ---------- indicator logic ----------
def ema(s, span): return pd.Series(s, index=s.index).ewm(span=span, adjust=False).mean()
def sma(s, window): return pd.Series(s, index=s.index).rolling(window).mean()

def rsi(series, period=14):
    if isinstance(series, pd.DataFrame): series = series.iloc[:,0]
    series = pd.to_numeric(pd.Series(series).squeeze(), errors="coerce")
    d = series.diff(); gain = np.where(d>0, d, 0.0); loss = np.where(d<0, -d, 0.0)
    g = pd.Series(gain, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    l = pd.Series(loss, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = g / l.replace(0, np.nan); out = 100 - (100/(1+rs))
    return out.fillna(method="bfill")

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

def load_data(ticker: str, years: float=2.0) -> pd.DataFrame:
    t=ticker.strip().upper()
    end=dt.date.today(); start=end - dt.timedelta(days=int(years*365.25))
    df=yf.download(t, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    keep=[c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]; df=df[keep].copy()
    for c in df.columns:
        s=df[c]
        if isinstance(s,pd.DataFrame): s=s.iloc[:,0]
        df[c]=pd.to_numeric(np.asarray(s).reshape(-1), errors="coerce")
    return df.dropna()

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

def send_discord(url: str, text: str):
    if not url:
        print("No DISCORD_WEBHOOK_URL set; printing instead:\n", text); return
    payload={"content": text}
    if len(text)>1800: payload["content"]=text[:1800]+"\n...(truncated)"
    r=requests.post(url,json=payload,timeout=20)
    if r.status_code>=300: print("Discord webhook error:", r.status_code, r.text)

# ----- watchlist + universe sources -----
def load_watchlist_from_url(url: str) -> list[str]:
    try:
        df=pd.read_csv(url, header=None)
        return [str(t).strip().upper() for t in df.iloc[:,0].tolist() if str(t).strip()]
    except Exception:
        return []

def get_watchlist() -> list[str]:
    if WATCHLIST_URL:
        wl=load_watchlist_from_url(WATCHLIST_URL)
        if wl: return wl
    return [t.strip().upper() for t in WATCHLIST.split(",") if t.strip()]

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
    except Exception:
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
    except Exception:
        return []

def good_position_flags(last: pd.Series, mode: str):
    trend_up=bool(last["EMA20"]>last["EMA50"]); rsi_v=float(last["RSI14"]); close=float(last["Close"])
    bb_dn=float(last.get("BB_DN", np.nan)); bb_up=float(last.get("BB_UP", np.nan))
    hh20=float(last.get("HH20", np.nan))
    band=np.nan
    if np.isfinite(bb_dn) and np.isfinite(bb_up) and (bb_up-bb_dn)>0:
        band=(close-bb_dn)/(bb_up-bb_dn)
    buy=bool(last.get("buy_signal",False)); score=0; ok=False
    if mode in ("Pullback","Both"):
        if trend_up and (45<=rsi_v<=60) and (0.1 <= (band if np.isfinite(band) else 0.5) <= 0.5):
            ok=True; score+=2
    if mode in ("Breakout","Both"):
        near_high=np.isfinite(hh20) and (close>=0.995*hh20) and trend_up and (rsi_v>=50)
        if buy or near_high: ok=True; score+=2
    if trend_up and rsi_v>50: score+=1
    return ok,score,band

# ---------- scanning / alerting ----------
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
    for t in get_watchlist():
        try:
            df = load_data(t, YEARS)
            if df.empty: continue
            df = build_indicators(df); last=df.iloc[-1]
            if not (bool(last["buy_signal"]) or bool(last["sell_signal"])): continue
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
            if df.empty: continue
            df = build_indicators(df); last=df.iloc[-1]
            close=float(last["Close"])
            if not (PRICE_MIN <= close <= PRICE_MAX): continue
            adv=(df["Close"]*df["Volume"]).rolling(20).mean().iloc[-1]
            adv_m=float(adv)/1_000_000.0 if pd.notna(adv) else 0.0
            if adv_m < MIN_DOLLAR_VOL_M: continue
            ok,score,band = good_position_flags(last, SETUP_MODE)
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
        # Log and notify; keep the workflow green while you stabilize
        import traceback
        traceback.print_exc()
        try:
            send_discord(DISCORD_WEBHOOK_URL, f"❌ Alerts error: {e}")
        except Exception:
            pass
        raise SystemExit(0)


