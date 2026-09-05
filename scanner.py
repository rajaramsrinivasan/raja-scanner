"""
Raja Swing Scanner
==================
Finds NSE stocks positioned to move +8% quickly — catching entries at the
start of a move rather than after it.

Replaces the old summed-score approach, which counted one event (a recent
spike) four times over and so fired on almost everything. This version uses
four independent gates that must EACH pass, plus a hard extension filter.

Every threshold lives in the TUNING block below. When the backtest finishes,
change the numbers there and nothing else.

Usage:
  pip install yfinance pandas numpy requests
  python scanner.py           # full scan
  python scanner.py --test    # 40 stocks
"""

import datetime
import json
import os
import sys
import time
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Secrets ───────────────────────────────────────────────────────────────────
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
EMAIL_FROM     = os.environ.get("EMAIL_FROM", "briefing@rajaportfolio.com")
EMAIL_TO       = os.environ.get("EMAIL_TO", "pss.rajaram@gmail.com")
GIST_ID        = os.environ.get("GIST_ID", "")
GIST_TOKEN     = os.environ.get("GIST_TOKEN", "")

TEST_MODE  = "--test" in sys.argv
BATCH_SIZE = 50
GIST_FILE  = "raja_scan_latest.json"

# ══ TUNING ════════════════════════════════════════════════════════════════════
# Universe filters
MIN_PRICE      = 50
MAX_PRICE      = 50000
MIN_AVG_VOLUME = 100000      # shares/day — below this you can't exit cleanly

# Gate 1 — Trend: is this stock in an uptrend worth joining?
MIN_RS60          = 2.0      # % outperformance vs Nifty over 60 days
MA50_SLOPE_DAYS   = 10       # MA50 must be higher than it was N days ago

# Gate 2 — Setup: is it coiled or resting, rather than extended?
BBW_RANK_MAX      = 0.35     # Bollinger bandwidth in bottom 35% of 6-month range
PULLBACK_EXT_MAX  = 3.0      # ...or price was within 3% of MA20 yesterday

# Gate 3 — Trigger: did the move start in the last day or two?
VOL_RATIO_MIN     = 1.5      # vs the PRIOR 20-day average, not including today
BREAKOUT_LOOKBACK = 20       # close must exceed this many days of prior highs

# Gate 4 — Risk: is the stop tight enough for +8% to be worth taking?
ATR_PCT_MIN       = 1.2      # too quiet = won't reach target in time
ATR_PCT_MAX       = 5.5      # too wild = stop gets hit on noise
MAX_EXT_MA20      = 6.0      # THE KEY FILTER — reject anything already extended
MAX_ROC5          = 12.0     # hasn't already run this week

# Trade parameters written into every recommendation
TARGET_PCT = 8.0
STOP_PCT   = 4.0
MAX_HOLD_D = 20
# ══════════════════════════════════════════════════════════════════════════════

FALLBACK = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","SBIN","ITC","LT","AXISBANK",
    "BHARTIARTL","TATAMOTORS","TATASTEEL","SUNPHARMA","MARUTI","TITAN","BAJFINANCE",
    "HCLTECH","WIPRO","ULTRACEMCO","ASIANPAINT","NTPC","POWERGRID","ONGC","COALINDIA",
    "JSWSTEEL","HINDALCO","CIPLA","DRREDDY","TECHM","GRASIM","TRENT","DLF","HAL","BEL",
    "IRCTC","PERSISTENT","POLYCAB","CUMMINSIND","DIXON","KPITTECH","TATAELXSI",
    "APLAPOLLO","SUPREMEIND","ASTRAL","NAVINFLUOR","DEEPAKNTR","COFORGE","MPHASIS",
    "LTIM","OBEROIRLTY","INDIGO","VBL","ZOMATO","JUBLFOOD","PIIND","SRF","BALKRISIND",
    "ESCORTS","MFSL","MUTHOOTFIN","CANBK","BANKBARODA","PNB","IDFCFIRSTB","FEDERALBNK",
]


# ── Indicators ────────────────────────────────────────────────────────────────
def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ── Universe ──────────────────────────────────────────────────────────────────
def get_universe():
    if TEST_MODE:
        print(f"Test mode — {len(FALLBACK[:40])} symbols")
        return FALLBACK[:40]

    syms = set()
    sources = {
        "Nifty 500":  "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "Midcap 150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
        "Smallcap":   "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    }
    hdr = {"User-Agent": "Mozilla/5.0"}
    for name, url in sources.items():
        try:
            r = requests.get(url, headers=hdr, timeout=20)
            df = pd.read_csv(StringIO(r.text))
            got = df["Symbol"].dropna().astype(str).str.strip().tolist()
            syms.update(got)
            print(f"  {name}: {len(got)}")
        except Exception as e:
            print(f"  {name} failed: {e}")

    if not syms:
        print(f"All NSE lists failed — using {len(FALLBACK)} fallback symbols")
        return FALLBACK
    out = sorted(syms)
    print(f"Universe: {len(out)} symbols")
    return out


def get_benchmark():
    try:
        d = yf.download("^NSEI", period="1y", interval="1d",
                        auto_adjust=True, progress=False)
        c = d["Close"].squeeze()
        return float((c.iloc[-1] / c.iloc[-61] - 1) * 100)
    except Exception as e:
        print(f"Benchmark failed ({e}) — relative strength gate disabled")
        return None


def get_vix():
    try:
        d = yf.download("^INDIAVIX", period="5d", interval="1d",
                        auto_adjust=True, progress=False)
        return round(float(d["Close"].squeeze().iloc[-1]), 1)
    except Exception:
        return None


# ── Analysis ──────────────────────────────────────────────────────────────────
def analyse(sym, df, bench_roc60):
    if df is None or len(df) < 210:
        return None

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    price = float(c.iloc[-1])
    if not (MIN_PRICE <= price <= MAX_PRICE):
        return None

    avgvol = float(v.rolling(20).mean().shift(1).iloc[-1])
    if not np.isfinite(avgvol) or avgvol < MIN_AVG_VOLUME:
        return None

    ma20  = float(c.rolling(20).mean().iloc[-1])
    ma50  = c.rolling(50).mean()
    ma200 = float(c.rolling(200).mean().iloc[-1])
    rsi_v = float(rsi(c).iloc[-1])

    roc5  = float((c.iloc[-1] / c.iloc[-6] - 1) * 100)
    roc20 = float((c.iloc[-1] / c.iloc[-21] - 1) * 100)
    roc60 = float((c.iloc[-1] / c.iloc[-61] - 1) * 100)

    volratio = float(v.iloc[-1] / avgvol)
    atr_v    = float(atr(h, l, c).iloc[-1])
    atr_pct  = atr_v / price * 100

    hi20 = float(h.rolling(BREAKOUT_LOOKBACK).max().shift(1).iloc[-1])
    ext  = (price - ma20) / ma20 * 100
    ext_prev = float(((c.iloc[-2] - c.rolling(20).mean().iloc[-2])
                      / c.rolling(20).mean().iloc[-2]) * 100)

    sd  = c.rolling(20).std()
    bbw = (2 * sd) / c.rolling(20).mean() * 100
    bbw_rank = float(bbw.rolling(120).rank(pct=True).iloc[-2])

    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd, sigl = e12 - e26, (e12 - e26).ewm(span=9, adjust=False).mean()
    macd_cross = bool(macd.iloc[-1] > sigl.iloc[-1] and macd.iloc[-2] <= sigl.iloc[-2])

    rs60 = roc60 - bench_roc60 if bench_roc60 is not None else 999
    ma50_rising = bool(float(ma50.iloc[-1]) > float(ma50.iloc[-1 - MA50_SLOPE_DAYS]))

    # ── The four gates ────────────────────────────────────────────────────────
    trend   = price > ma200 and ma50_rising and rs60 > MIN_RS60
    setup   = bbw_rank < BBW_RANK_MAX or ext_prev < PULLBACK_EXT_MAX
    trigger = (price > hi20 or macd_cross) and volratio >= VOL_RATIO_MIN
    risk    = (ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX
               and ext < MAX_EXT_MA20 and roc5 < MAX_ROC5)

    if not (trend and setup and trigger and risk):
        return None

    # Describe what actually fired, in plain terms
    reasons = []
    if price > hi20:
        reasons.append(f"Broke above its {BREAKOUT_LOOKBACK}-day high today")
    if macd_cross:
        reasons.append("Momentum turned up (MACD crossover)")
    if bbw_rank < 0.20:
        reasons.append("Was tightly coiled before this move")
    elif bbw_rank < BBW_RANK_MAX:
        reasons.append("Volatility had compressed")
    if ext_prev < 0:
        reasons.append("Entering off a pullback, not a spike")
    reasons.append(f"Volume {volratio:.1f}x its 20-day average")
    reasons.append(f"Outperforming Nifty by {rs60:.0f}% over 60 days")

    setup_type = ("Squeeze breakout" if bbw_rank < 0.20 and price > hi20 else
                  "Base breakout"    if price > hi20 else
                  "Pullback reversal")

    stop   = round(min(price * (1 - STOP_PCT / 100), price - 1.5 * atr_v), 2)
    target = round(price * (1 + TARGET_PCT / 100), 2)
    rr     = (target - price) / max(price - stop, 0.01)

    if rr < 1.5:
        return None   # not enough reward for the risk being taken

    return {
        "symbol":     sym,
        "exchange":   "NSE",
        "price":      round(price, 2),
        "setup":      setup_type,
        "entry_below": round(price * 1.01, 2),
        "stop":       stop,
        "target":     target,
        "rr":         round(rr, 2),
        "risk_pct":   round((price - stop) / price * 100, 2),
        "rsi":        round(rsi_v),
        "vol_ratio":  round(volratio, 1),
        "roc_20":     round(roc20, 1),
        "roc_5":      round(roc5, 1),
        "ext_ma20":   round(ext, 1),
        "atr_pct":    round(atr_pct, 1),
        "rs60":       round(rs60, 1),
        "est_days":   int(max(3, min(MAX_HOLD_D, round(TARGET_PCT / max(atr_pct, 0.5))))),
        "reasons":    reasons,
    }


# ── Scan ──────────────────────────────────────────────────────────────────────
def run_scan(symbols, bench):
    results, scanned = [], 0
    tickers = [s + ".NS" for s in symbols]

    for k in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[k:k + BATCH_SIZE]
        print(f"  {k + 1}-{k + len(batch)} of {len(tickers)}")
        try:
            data = yf.download(batch, period="1y", interval="1d", group_by="ticker",
                               auto_adjust=True, progress=False, threads=True)
        except Exception as e:
            print(f"    batch failed: {e}")
            continue

        for t in batch:
            try:
                sub = (data[t] if len(batch) > 1 else data).dropna()
                scanned += 1
                r = analyse(t.replace(".NS", ""), sub, bench)
                if r:
                    results.append(r)
                    print(f"    {r['symbol']:12s} {r['setup']:18s} "
                          f"RR {r['rr']}  est {r['est_days']}d")
            except Exception:
                pass
        time.sleep(1.5)

    results.sort(key=lambda r: (-r["rr"], r["est_days"]))
    return results, scanned


# ── Publish ───────────────────────────────────────────────────────────────────
def push_gist(payload):
    if not (GIST_ID and GIST_TOKEN):
        print("GIST_ID or GIST_TOKEN missing — dashboard will not update")
        sys.exit(1)
    r = requests.patch(
        f"https://api.github.com/gists/{GIST_ID}",
        headers={"Authorization": f"Bearer {GIST_TOKEN}",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "raja-scanner"},
        json={"files": {GIST_FILE: {"content": json.dumps(payload)}}},
        timeout=25,
    )
    if r.status_code != 200:
        print(f"Gist update FAILED {r.status_code}: {r.text[:300]}")
        sys.exit(1)
    print(f"Gist updated — {len(payload['buy'])} candidates published")


def build_email(payload, vix):
    buy, d = payload["buy"], payload["date"]
    if vix and vix > 20:
        head, tone = (f"India VIX at {vix}. Hold cash — no new positions today.", "#B03A2E")
    elif buy:
        t = buy[0]
        head, tone = (f"Best setup: {t['symbol']} at ₹{t['price']} — "
                      f"stop ₹{t['stop']}, target ₹{t['target']}", "#1D6F4C")
    else:
        head, tone = ("No setups passed the filters today. Sit out.", "#B08D2E")

    rows = "".join(
        f"""<tr>
          <td style="padding:10px 8px;border-bottom:1px solid #D3DAE1">
            <div style="font-weight:600;font-size:15px">{r['symbol']}</div>
            <div style="font-size:12px;color:#5A6B7A">{r['setup']}</div></td>
          <td style="padding:10px 8px;border-bottom:1px solid #D3DAE1;
              text-align:right;font-family:monospace">₹{r['price']}</td>
          <td style="padding:10px 8px;border-bottom:1px solid #D3DAE1;
              text-align:right;font-family:monospace;color:#B03A2E">₹{r['stop']}</td>
          <td style="padding:10px 8px;border-bottom:1px solid #D3DAE1;
              text-align:right;font-family:monospace;color:#1D6F4C">₹{r['target']}</td>
          <td style="padding:10px 8px;border-bottom:1px solid #D3DAE1;
              text-align:right;font-family:monospace">{r['rr']}</td>
          <td style="padding:10px 8px;border-bottom:1px solid #D3DAE1;
              text-align:right;font-family:monospace">~{r['est_days']}d</td>
        </tr>""" for r in buy[:10])

    return f"""<div style="font-family:-apple-system,Segoe UI,sans-serif;
      max-width:720px;margin:0 auto;background:#EEF1F4;padding:28px;color:#16222E">
      <div style="font-size:13px;color:#5A6B7A;margin-bottom:4px">{d}</div>
      <h1 style="font-size:22px;margin:0 0 20px">Today's setups</h1>
      <div style="border-left:3px solid {tone};padding:14px 16px;background:#fff;
        font-size:16px;font-weight:600;color:{tone};margin-bottom:24px">{head}</div>
      {'<table style="width:100%;border-collapse:collapse;background:#fff">'
       '<tr><th align="left" style="padding:8px;font-size:12px;color:#5A6B7A">Stock</th>'
       '<th align="right" style="padding:8px;font-size:12px;color:#5A6B7A">Price</th>'
       '<th align="right" style="padding:8px;font-size:12px;color:#5A6B7A">Stop</th>'
       '<th align="right" style="padding:8px;font-size:12px;color:#5A6B7A">Target</th>'
       '<th align="right" style="padding:8px;font-size:12px;color:#5A6B7A">R:R</th>'
       '<th align="right" style="padding:8px;font-size:12px;color:#5A6B7A">Est</th></tr>'
       + rows + '</table>' if buy else ''}
      <p style="font-size:13px;color:#5A6B7A;margin-top:24px">
        Scanned {payload['total_scanned']} stocks.
        <a href="https://rajaportfolio.com" style="color:#B08D2E">Open dashboard</a>
      </p></div>"""


def send_email(subject, html):
    if not RESEND_API_KEY:
        print("No RESEND_API_KEY — skipping email")
        return
    r = requests.post("https://api.resend.com/emails",
                      headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                      json={"from": EMAIL_FROM, "to": EMAIL_TO,
                            "subject": subject, "html": html}, timeout=20)
    print("Email sent" if r.status_code in (200, 201) else f"Email failed: {r.text[:200]}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    start = time.time()
    today = datetime.date.today()
    print(f"Raja Swing Scanner — {today}\n")

    vix   = get_vix()
    bench = get_benchmark()
    print(f"India VIX: {vix}   Nifty 60d: "
          f"{f'{bench:+.1f}%' if bench is not None else 'n/a'}\n")

    symbols = get_universe()
    results, scanned = run_scan(symbols, bench)

    payload = {
        "date":          today.strftime("%d %b %Y"),
        "ts":            datetime.datetime.utcnow().isoformat() + "Z",
        "vix":           vix,
        "total_scanned": scanned,
        "target_pct":    TARGET_PCT,
        "stop_pct":      STOP_PCT,
        "buy":           results[:20],
        "watch":         [],
    }

    os.makedirs("results", exist_ok=True)
    with open(f"results/scan_{today}.json", "w") as f:
        json.dump(payload, f, indent=2)

    push_gist(payload)
    send_email(f"{len(results)} setups — {today.strftime('%d %b')}",
               build_email(payload, vix))

    print(f"\nDone in {time.time() - start:.0f}s — "
          f"{scanned} scanned, {len(results)} passed all four gates")


if __name__ == "__main__":
    main()
