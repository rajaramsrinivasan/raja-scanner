"""
Raja's NSE+BSE Full Market Scanner
====================================
Scans ALL listed NSE stocks (~1,800) for momentum setups.
Runs daily at 9:15 AM IST via GitHub Actions.
Sends actionable email with top BUY signals before market open.

Setup:
  pip install yfinance pandas pandas_ta requests
  Set env vars: RESEND_API_KEY, EMAIL_TO, ANTHROPIC_API_KEY (optional)

Usage:
  python scanner.py               # full scan
  python scanner.py --test        # test with 20 stocks only
  python scanner.py --email-only  # re-send last results
"""

import yfinance as yf
import pandas as pd
import requests
import json
import os
import sys
import time
import datetime
from io import StringIO

# ── Config ────────────────────────────────────────────────────────────────────
RESEND_API_KEY  = os.environ.get("RESEND_API_KEY", "")
EMAIL_FROM      = os.environ.get("EMAIL_FROM", "briefing@rajaportfolio.com")
EMAIL_TO        = os.environ.get("EMAIL_TO", "pss.rajaram@gmail.com")
ANTHROPIC_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")

TEST_MODE       = "--test" in sys.argv
BATCH_SIZE      = 50     # fetch 50 stocks at once (safe for Yahoo rate limits)
SLEEP_BETWEEN   = 2      # seconds between batches
MIN_PRICE       = 50     # ignore penny stocks below ₹50
MAX_PRICE       = 50000  # ignore outliers above ₹50,000
MIN_AVG_VOLUME  = 50000  # ignore illiquid stocks (avg daily volume < 50K shares)

# Signal thresholds
ROC_20_MIN      = 8      # min % gain in last 20 trading days to flag
VOL_SPIKE_MIN   = 2.0    # volume must be 2x+ 20-day average
RSI_MIN         = 45     # RSI lower bound (not oversold)
RSI_MAX         = 72     # RSI upper bound (not overbought)
SCORE_THRESHOLD = 3      # minimum score to appear in BUY signals


# ── Step 1: Build full NSE + BSE universe ─────────────────────────────────────
def get_nse_symbols():
    """Fetch all NSE-listed stocks from multiple indices."""
    print("📥 Fetching NSE universe...")
    all_symbols = set()

    # NSE indices to pull from
    index_urls = {
        "Nifty 500":    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "Nifty Midcap": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
        "Nifty Smallcap":"https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
        "Nifty Microcap":"https://archives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.nseindia.com/",
    }

    session = requests.Session()
    # Prime NSE session cookie
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
    except Exception:
        pass

    for name, url in index_urls.items():
        try:
            resp = session.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                df = pd.read_csv(StringIO(resp.text))
                # Column is usually 'Symbol'
                sym_col = next((c for c in df.columns if 'symbol' in c.lower()), None)
                if sym_col:
                    syms = df[sym_col].str.strip().tolist()
                    all_symbols.update(syms)
                    print(f"  ✅ {name}: {len(syms)} stocks")
                else:
                    print(f"  ⚠️  {name}: could not find Symbol column")
            else:
                print(f"  ❌ {name}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    # Fallback hardcoded list if NSE is blocking
    if len(all_symbols) < 100:
        print("  Using fallback hardcoded Nifty 500 list...")
        all_symbols.update(FALLBACK_SYMBOLS)

    symbols_ns = [s + ".NS" for s in sorted(all_symbols)]
    print(f"✅ Total NSE universe: {len(symbols_ns)} stocks\n")
    return symbols_ns


def get_bse_symbols():
    """
    BSE has 5000+ stocks. We use a curated BSE 500 list.
    Full BSE scrape via bhavcopy for production use.
    """
    print("📥 Fetching BSE symbols via bhavcopy...")
    try:
        # BSE bhavcopy — daily equity file
        today = datetime.date.today()
        # Try last 5 days in case of holiday
        for days_back in range(1, 6):
            dt = today - datetime.timedelta(days=days_back)
            url = f"https://www.bseindia.com/download/BhavCopy/Equity/EQ{dt.strftime('%d%m%y')}_CSV.ZIP"
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                import zipfile, io
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        df = pd.read_csv(f)
                        # BSE bhavcopy has SC_CODE and SC_NAME
                        if 'SC_CODE' in df.columns:
                            codes = df['SC_CODE'].astype(str).tolist()
                            syms = [c + ".BO" for c in codes]
                            print(f"✅ BSE bhavcopy: {len(syms)} stocks\n")
                            return syms[:2000]  # Top 2000 by liquidity
                break
    except Exception as e:
        print(f"  BSE fetch failed: {e} — skipping BSE\n")

    return []


# ── Step 2: Scan in batches ───────────────────────────────────────────────────
def scan_batch(symbols, period="2mo"):
    """Download OHLCV for a batch and return cleaned DataFrame."""
    try:
        data = yf.download(
            symbols,
            period=period,
            interval="1d",
            progress=False,
            threads=True,
            group_by="ticker",
        )
        return data
    except Exception as e:
        print(f"    Batch download error: {e}")
        return None


def analyse_symbol(sym, data):
    """
    Run all signals on one symbol's OHLCV data.
    Returns a dict of signal results or None if data insufficient.
    """
    try:
        # Extract OHLCV
        if isinstance(data.columns, pd.MultiIndex):
            if sym not in data.columns.get_level_values(0):
                return None
            df = data[sym].copy()
        else:
            df = data.copy()

        df = df.dropna(subset=["Close"])
        if len(df) < 22:
            return None

        close  = df["Close"]
        volume = df["Volume"]
        high   = df["High"]
        low    = df["Low"]

        cur_price  = float(close.iloc[-1])
        cur_volume = float(volume.iloc[-1])

        # Price filter
        if cur_price < MIN_PRICE or cur_price > MAX_PRICE:
            return None

        # Liquidity filter
        avg_vol_20 = float(volume.tail(20).mean())
        if avg_vol_20 < MIN_AVG_VOLUME:
            return None

        # ── Indicators (pure pandas — no external TA library needed) ───────

        # Rate of Change — % gain in last 20 trading days
        price_20d_ago = float(close.iloc[-21])
        roc_20 = ((cur_price - price_20d_ago) / price_20d_ago) * 100

        # RSI (14-period) — Wilder smoothing
        def calc_rsi(series, period=14):
            delta = series.diff()
            gain  = delta.clip(lower=0)
            loss  = -delta.clip(upper=0)
            avg_g = gain.ewm(alpha=1/period, min_periods=period).mean()
            avg_l = loss.ewm(alpha=1/period, min_periods=period).mean()
            rs    = avg_g / avg_l.replace(0, 1e-10)
            return float((100 - 100 / (1 + rs)).iloc[-1])

        rsi_val = calc_rsi(close)

        # Moving averages
        ma20 = float(close.tail(20).mean())
        ma50 = float(close.tail(50).mean()) if len(close) >= 50 else ma20

        # Volume spike
        vol_ratio = cur_volume / avg_vol_20 if avg_vol_20 > 0 else 1.0

        # 52-week high/low
        w52_high = float(high.tail(252).max()) if len(high) >= 252 else float(high.max())
        w52_low  = float(low.tail(252).min())  if len(low)  >= 252 else float(low.min())
        pct_from_52h = ((cur_price - w52_high) / w52_high) * 100
        pct_from_52l = ((cur_price - w52_low)  / w52_low)  * 100

        # MACD (12, 26, 9) — pure pandas
        ema12      = close.ewm(span=12, adjust=False).mean()
        ema26      = close.ewm(span=26, adjust=False).mean()
        macd_line  = ema12 - ema26
        signal_line= macd_line.ewm(span=9, adjust=False).mean()
        macd_bull  = float(macd_line.iloc[-1]) > float(signal_line.iloc[-1])

        # ATR (14-period)
        tr     = pd.concat([high - low,
                             (high - close.shift()).abs(),
                             (low  - close.shift()).abs()], axis=1).max(axis=1)
        atr_val = float(tr.tail(14).mean())
        atr_pct = (atr_val / cur_price) * 100

        # ── Scoring ─────────────────────────────────────────────────────────
        # 8 possible signals — need 3+ for BUY, 2 for WATCH
        signals = []
        score   = 0

        if roc_20 >= ROC_20_MIN:
            score += 2  # strongest signal — already moving
            signals.append(f"ROC +{roc_20:.1f}% in 20 days")

        if RSI_MIN <= rsi_val <= RSI_MAX:
            score += 1
            signals.append(f"RSI {rsi_val:.0f} (momentum zone)")
        elif rsi_val < RSI_MIN:
            signals.append(f"RSI {rsi_val:.0f} (oversold — wait)")

        if vol_ratio >= VOL_SPIKE_MIN:
            score += 1
            signals.append(f"Volume {vol_ratio:.1f}x avg (institutional)")

        if cur_price > ma20:
            score += 1
            signals.append("Above MA20")

        if cur_price > ma50:
            score += 1
            signals.append("Above MA50")

        if macd_bull:
            score += 1
            signals.append("MACD bullish crossover")

        if -5 <= pct_from_52h <= 0:
            score += 1
            signals.append(f"Near 52W high ({pct_from_52h:.1f}%)")

        # Penalty: overbought or extreme move
        if rsi_val > 80:
            score -= 1
            signals.append("⚠ RSI overbought")
        if roc_20 > 40:
            score -= 1
            signals.append("⚠ Already up 40%+ — may be late")

        # Action
        if score >= SCORE_THRESHOLD:
            action       = "BUY"
            action_color = "#00A651"
        elif score == 2:
            action       = "WATCH"
            action_color = "#E86A20"
        elif roc_20 < -10:
            action       = "AVOID"
            action_color = "#E83820"
        else:
            return None  # not interesting enough to report

        return {
            "symbol":       sym.replace(".NS", "").replace(".BO", ""),
            "exchange":     "BSE" if ".BO" in sym else "NSE",
            "price":        round(cur_price, 2),
            "roc_20":       round(roc_20, 1),
            "rsi":          round(rsi_val, 1),
            "vol_ratio":    round(vol_ratio, 1),
            "ma20":         round(ma20, 2),
            "ma50":         round(ma50, 2),
            "w52_high":     round(w52_high, 2),
            "w52_low":      round(w52_low, 2),
            "pct_from_52h": round(pct_from_52h, 1),
            "atr_pct":      round(atr_pct, 2),
            "avg_vol":      int(avg_vol_20),
            "signals":      signals,
            "score":        score,
            "action":       action,
            "action_color": action_color,
        }

    except Exception as e:
        return None


def run_full_scan(symbols):
    """Scan all symbols in batches and collect results."""
    results   = []
    failed    = []
    total     = len(symbols)
    batches   = [symbols[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    print(f"🔍 Scanning {total} stocks in {len(batches)} batches of {BATCH_SIZE}...\n")
    t0 = time.time()

    for idx, batch in enumerate(batches):
        pct = (idx / len(batches)) * 100
        print(f"  Batch {idx+1}/{len(batches)} ({pct:.0f}%) — {batch[0].split('.')[0]}..{batch[-1].split('.')[0]}", end="\r")

        data = scan_batch(batch)
        if data is None or data.empty:
            failed.extend(batch)
            time.sleep(SLEEP_BETWEEN)
            continue

        for sym in batch:
            result = analyse_symbol(sym, data)
            if result:
                results.append(result)

        time.sleep(SLEEP_BETWEEN)  # be polite to Yahoo

    elapsed = time.time() - t0
    print(f"\n\n✅ Scan complete in {elapsed:.0f}s")
    print(f"   Stocks scanned: {total - len(failed)}")
    print(f"   Signals found:  {len(results)}")
    print(f"   Failed:         {len(failed)}")

    return sorted(results, key=lambda x: x["score"], reverse=True)


# ── Step 3: Build email ───────────────────────────────────────────────────────
def build_email(results, vix=None):
    today    = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    date_str = today.strftime("%A, %d %b %Y")
    time_str = today.strftime("%I:%M %p IST")

    buy    = [r for r in results if r["action"] == "BUY"]
    watch  = [r for r in results if r["action"] == "WATCH"]

    vix_color  = "#C41400" if vix and vix > 20 else "#1A7F37"
    vix_bg     = "#FFF0EE" if vix and vix > 20 else "#EAFBEE"
    vix_note   = f"VIX {vix:.1f} — {'HIGH: Pause new buys' if vix > 20 else 'Normal: OK to trade'}" if vix else "VIX unavailable"

    action_text = ""
    if vix and vix > 20:
        action_text = f"🚨 India VIX is {vix:.1f} — Above 20. Your rules say HOLD CASH. Do not deploy new capital today regardless of signals."
        action_bg   = "#FFF0EE"
        action_color= "#C41400"
    elif buy:
        top = buy[0]
        action_text = f"✅ Deploy ₹10,000 into {top['symbol']} — {top['roc_20']:+.1f}% in 20 days, RSI {top['rsi']}, score {top['score']}/8"
        action_bg   = "#EAFBEE"
        action_color= "#1A7F37"
    else:
        action_text = "⚠️ No strong BUY signals today — hold existing positions, do not force a trade."
        action_bg   = "#FFF8E6"
        action_color= "#8A6000"

    def stock_row(r, highlight=False):
        bg = "#EAFBEE" if highlight else ("transparent")
        signal_str = " · ".join(r["signals"][:3])
        return f"""
        <tr style="background:{bg};border-bottom:1px solid #eee">
          <td style="padding:8px 12px;font-family:monospace;font-weight:700;font-size:13px">{r['symbol']}</td>
          <td style="padding:8px 12px;font-family:monospace;font-size:12px;color:#666">{r['exchange']}</td>
          <td style="padding:8px 12px;font-family:monospace;font-size:13px">₹{r['price']:,}</td>
          <td style="padding:8px 12px;font-family:monospace;font-size:13px;color:{"#00A651" if r['roc_20']>=0 else "#C41400"};font-weight:700">{r['roc_20']:+.1f}%</td>
          <td style="padding:8px 12px;font-size:12px;color:#555">{r['rsi']}</td>
          <td style="padding:8px 12px;font-size:12px;color:#555">{r['vol_ratio']:.1f}x</td>
          <td style="padding:8px 12px;font-size:11px;color:#666;max-width:300px">{signal_str}</td>
          <td style="padding:8px 12px;text-align:center">
            <span style="font-size:10px;font-weight:700;font-family:monospace;letter-spacing:.04em;
              padding:3px 8px;border:1.5px solid {r['action_color']};
              color:{r['action_color']};background:{r['action_color']}18">{r['action']}</span>
          </td>
        </tr>"""

    buy_rows   = "".join(stock_row(r, i == 0) for i, r in enumerate(buy[:10]))
    watch_rows = "".join(stock_row(r) for r in watch[:5])
    no_buy_msg = '<tr><td colspan="8" style="padding:14px;text-align:center;color:#999;font-family:monospace">No strong buy signals found today</td></tr>'

    html = f"""
    <div style="max-width:720px;margin:0 auto;font-family:Arial,sans-serif;font-size:14px;color:#333">

      <div style="border-bottom:3px solid #0A0A0A;padding-bottom:12px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:flex-end">
        <div>
          <h1 style="font-size:16px;font-family:monospace;margin:0;letter-spacing:.1em;text-transform:uppercase">NSE + BSE Full Market Scan</h1>
          <p style="font-size:12px;color:#666;margin:4px 0 0;font-family:monospace">{date_str} &nbsp;|&nbsp; {time_str}</p>
        </div>
        <span style="font-size:11px;font-weight:700;font-family:monospace;padding:4px 12px;border:1.5px solid {vix_color};color:{vix_color};background:{vix_bg}">{vix_note}</span>
      </div>

      <div style="background:{action_bg};border:1.5px solid {action_color};padding:16px 20px;margin-bottom:24px">
        <div style="font-size:9px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:{action_color};margin-bottom:5px">Today's action</div>
        <div style="font-size:15px;font-weight:700;color:{action_color};line-height:1.4">{action_text}</div>
      </div>

      <h3 style="font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#666;margin:0 0 10px">
        Buy signals — {len(buy)} stocks found across NSE+BSE
      </h3>
      <table style="width:100%;border-collapse:collapse;margin-bottom:24px;font-size:12px;border:1px solid #eee">
        <thead>
          <tr style="background:#0A0A0A;color:#fff">
            <th style="padding:8px 12px;text-align:left;font-family:monospace;font-size:10px;letter-spacing:.06em">Symbol</th>
            <th style="padding:8px 12px;text-align:left;font-size:10px">Exch</th>
            <th style="padding:8px 12px;text-align:left;font-size:10px">Price</th>
            <th style="padding:8px 12px;text-align:left;font-size:10px">20D ROC</th>
            <th style="padding:8px 12px;text-align:left;font-size:10px">RSI</th>
            <th style="padding:8px 12px;text-align:left;font-size:10px">Volume</th>
            <th style="padding:8px 12px;text-align:left;font-size:10px">Signals</th>
            <th style="padding:8px 12px;text-align:left;font-size:10px">Action</th>
          </tr>
        </thead>
        <tbody>
          {"".join(stock_row(r, i == 0) for i, r in enumerate(buy[:10])) if buy else no_buy_msg}
        </tbody>
      </table>

      {'<h3 style="font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#666;margin:0 0 10px">Watch list — approaching breakout</h3><table style="width:100%;border-collapse:collapse;margin-bottom:24px;font-size:12px;border:1px solid #eee"><thead><tr style="background:#0A0A0A;color:#fff"><th style="padding:8px 12px;text-align:left;font-family:monospace;font-size:10px">Symbol</th><th style="padding:8px 12px;text-align:left;font-size:10px">Exch</th><th style="padding:8px 12px;text-align:left;font-size:10px">Price</th><th style="padding:8px 12px;text-align:left;font-size:10px">20D ROC</th><th style="padding:8px 12px;text-align:left;font-size:10px">RSI</th><th style="padding:8px 12px;text-align:left;font-size:10px">Volume</th><th style="padding:8px 12px;text-align:left;font-size:10px">Signals</th><th style="padding:8px 12px;text-align:left;font-size:10px">Action</th></tr></thead><tbody>' + "".join(stock_row(r) for r in watch[:5]) + '</tbody></table>' if watch else ''}

      <div style="background:#F7F7F5;border:1px solid #eee;padding:14px 18px;margin-bottom:20px;font-size:12px;color:#555;line-height:1.8">
        <strong style="font-family:monospace;font-size:10px;letter-spacing:.08em;text-transform:uppercase">Scan stats</strong><br>
        Stocks scanned: <strong>{len(results) + (0)}</strong> &nbsp;|&nbsp;
        Buy signals: <strong>{len(buy)}</strong> &nbsp;|&nbsp;
        Watch: <strong>{len(watch)}</strong> &nbsp;|&nbsp;
        Top ROC: <strong>{f"{buy[0]['roc_20']:+.1f}%" if buy else "N/A"}</strong>
      </div>

      <p style="font-size:11px;color:#aaa;font-family:monospace;border-top:1px solid #eee;padding-top:12px">
        rajaportfolio.com &nbsp;|&nbsp; Rules: +10-15% exit / 3 red days stop loss / 15d max hold / VIX&gt;20 hold cash<br>
        Screened: ROC≥{ROC_20_MIN}%, RSI {RSI_MIN}-{RSI_MAX}, Volume≥{VOL_SPIKE_MIN}x, Score≥{SCORE_THRESHOLD}/8
      </p>
    </div>
    """
    return html, action_text


# ── Step 4: Optional AI summary ───────────────────────────────────────────────
def get_ai_summary(results, vix):
    """Use Claude Haiku to write a 3-line market summary."""
    if not ANTHROPIC_KEY:
        return ""
    try:
        top5 = results[:5]
        prompt = (
            f"India VIX: {vix}. Top 5 NSE/BSE momentum stocks today: "
            + ", ".join(f"{r['symbol']} ({r['roc_20']:+.1f}% ROC, RSI {r['rsi']}, signals: {', '.join(r['signals'][:2])})" for r in top5)
            + ". Write a 2-sentence market context for a swing trader. Be direct, no fluff."
        )
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
            json={"model": "claude-haiku-4-5-20251001", "max_tokens": 150, "messages": [{"role": "user", "content": prompt}]},
            timeout=15,
        )
        return resp.json()["content"][0]["text"]
    except Exception:
        return ""


# ── Step 5: Send email ────────────────────────────────────────────────────────
def send_email(subject, html_body):
    if not RESEND_API_KEY:
        print("⚠️  No RESEND_API_KEY set — printing email to stdout only")
        return False
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
        json={"from": EMAIL_FROM, "to": EMAIL_TO, "subject": subject, "html": html_body},
        timeout=15,
    )
    if resp.status_code in (200, 201):
        print(f"✅ Email sent to {EMAIL_TO}")
        return True
    else:
        print(f"❌ Email failed: {resp.text}")
        return False


# ── Step 6: Save results to JSON ──────────────────────────────────────────────
def save_results(results):
    os.makedirs("results", exist_ok=True)
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    path = "results/scan_" + date_str + ".json"

    dashboard_data = {
        "date":          date_str,
        "ts":            datetime.datetime.utcnow().isoformat() + "Z",
        "buy":           [r for r in results if r["action"] == "BUY"][:20],
        "watch":         [r for r in results if r["action"] == "WATCH"][:10],
        "total_scanned": len(results),
    }

    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    with open("results/latest.json", "w") as f:
        json.dump(dashboard_data, f)
    print("Saved to " + path)

    # Push to GitHub Gist so Cloudflare dashboard can fetch it
    # NOTE: must be GIST_TOKEN, not GITHUB_TOKEN. GitHub rejects user-defined
    # secrets starting with GITHUB_, and its built-in token has no gist scope.
    gist_id    = os.environ.get("GIST_ID", "")
    gist_token = os.environ.get("GIST_TOKEN", "")

    if not gist_id or not gist_token:
        print("ERROR: GIST_ID or GIST_TOKEN not set - dashboard will not update")
        sys.exit(1)

    try:
        resp = requests.patch(
            "https://api.github.com/gists/" + gist_id,
            headers={"Authorization": "Bearer " + gist_token,
                     "Accept": "application/vnd.github+json",
                     "User-Agent": "raja-scanner"},
            json={"files": {"raja_scan_latest.json": {"content": json.dumps(dashboard_data)}}},
            timeout=25,
        )
        if resp.status_code == 200:
            print("Gist updated OK - buy=%d watch=%d scanned=%d" % (
                len(dashboard_data["buy"]), len(dashboard_data["watch"]),
                dashboard_data["total_scanned"]))
        else:
            # Fail loudly so the workflow goes red instead of silently green
            print("ERROR: Gist update failed %s: %s" % (resp.status_code, resp.text[:300]))
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        print("ERROR: Gist push failed: " + str(e))
        sys.exit(1)

    return path


# ── Fallback symbol list (if NSE website blocks) ──────────────────────────────
FALLBACK_SYMBOLS = [
    # Nifty 50
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJAJFINSV",
    "BAJFINANCE","BHARTIARTL","BPCL","BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY",
    "EICHERMOT","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE","HEROMOTOCO","HINDALCO","HINDUNILVR",
    "ICICIBANK","INDUSINDBK","INFY","ITC","JSWSTEEL","KOTAKBANK","LT","M&M","MARUTI",
    "NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBILIFE","SBIN","SHRIRAMFIN",
    "SUNPHARMA","TATACONSUM","TATAMOTORS","TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO",
    "WIPRO","INDIGO",
    # Nifty Next 50
    "ABB","ADANIGREEN","ADANIPOWER","AMBUJACEM","BAJAJHLDNG","BANKBARODA","BERGEPAINT",
    "BOSCHLTD","CANBK","CHOLAFIN","COLPAL","DABUR","DLF","DMART","GAIL","GODREJCP",
    "GODREJPROP","HAVELLS","HAL","ICICIGI","ICICIPRULI","INDHOTEL","IOC","IRCTC","JINDALSTEL",
    "JUBLFOOD","LICHSGFIN","LICI","LUPIN","MARICO","MFSL","MUTHOOTFIN","NAUKRI","NHPC",
    "NMDC","OFSS","PAGEIND","PIDILITIND","PIIND","PNB","RECLTD","SAIL","SIEMENS",
    "SRF","TATAPOWER","TORNTPHARM","TRENT","TVSMOTOR","UBL","VEDL","ZYDUSLIFE",
    # Midcap momentum stocks
    "ABCAPITAL","APLAPOLLO","ASTRAL","AUBANK","BALRAMCHIN","BATAINDIA","BEL","BHARATFORG",
    "BIKAJI","BLUEDART","CAMS","CANFINHOME","CASTROLIND","CDSL","CESC","CGPOWER",
    "CLEAN","COCHINSHIP","CONCOR","COROMANDEL","CROMPTON","CUMMINSIND","CYIENT",
    "DATAPATTNS","DEEPAKNTR","DIXON","EDELWEISS","EMAMILTD","ESCORTS","EXIDEIND",
    "FEDERALBNK","FINCABLES","FLUOROCHEM","GICRE","GLAXO","GMRINFRA","GNFC",
    "GODREJAGRO","GRANULES","GRINDWELL","GUJGASLTD","HAPPSTMNDS","HSCL","IBREALEST",
    "IDBI","IDFCFIRSTB","IIFL","INDIGOPNTS","IRB","IREDA","ISGEC","IEX",
    "JKCEMENT","JKTYRE","JMFINANCIL","JUBLINGREA","KALYANKJIL","KANSAINER","KEI",
    "KFINTECH","KINETIC","KIOCL","KIRLOSENG","KNRCON","KPITTECH","KRBL","LAURUSLABS",
    "LICHSGFIN","LODHA","LXCHEM","MAHABANK","MAHINDCIE","MANAPPURAM","MARICO",
    "MASTEK","MAXHEALTH","MCDOWELL-N","MEDANTA","METROBRAND","MINDTREE","MGL",
    "MOTHERSON","MPHASIS","MRPL","NATCOPHARM","NBCC","NIACL","NLCINDIA","NOCIL",
    "NUVAMA","OLECTRA","ORIENTCEM","PAYTM","PERSISTENT","PETRONET","PFIZER",
    "PHOENIXLTD","POLYMED","POONAWALLA","PRESTIGE","PRINCEPIPE","PRIVISCL","PSPPROJECT",
    "RAILTEL","RAJESHEXPO","RAMCOCEM","RBLBANK","RITES","ROUTE","SAFARI",
    "SAPPHIRE","SCHAEFFLER","SEQUENT","SHYAMMETL","SOBHA","SPARC","SPICEJET",
    "STAR","STLTECH","SUMICHEM","SUNDARMFIN","SUNDRMFAST","SUPREMEIND","SUVENPHAR",
    "TANLA","TATACHEM","TATACOMM","TATAINVEST","TATATECH","TCPL","TECHNOE",
    "THYROCARE","TIINDIA","TIMKEN","TITAGARH","TORNTPOWER","TRIVENI","TRIDENT",
    "UCOBANK","UJJIVANSFB","UNION","UNIONBANK","UTIAMC","V2RETAIL","VAIBHAVGBL",
    "VARDHMANTEXT","VGUARD","VINATIORGA","VIPIND","VOLTAMP","VSTIND","WELCORP",
    "WELSPUNLIV","WHIRLPOOL","WINDLAS","WOCKPHARMA","XPRO","YATHARTH","ZENSARTECH",
    "ZENTEC","ZOMATO","ZUARI",
]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Raja's NSE+BSE Full Market Scanner")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 60 + "\n")

    # Get VIX (optional - fallback to None)
    vix = None
    try:
        vix_data = yf.download("^INDIAVIX", period="2d", interval="1d", progress=False)
        vix = float(vix_data["Close"].iloc[-1])
        print(f"📊 India VIX: {vix:.1f} ({'⚠ HIGH' if vix > 20 else '✅ Normal'})\n")
    except Exception:
        print("📊 VIX fetch failed — continuing without\n")

    # Build symbol universe
    if TEST_MODE:
        print("🧪 TEST MODE — using 30 stocks only\n")
        symbols = [s + ".NS" for s in FALLBACK_SYMBOLS[:30]]
    else:
        nse_syms = get_nse_symbols()
        bse_syms = get_bse_symbols()
        # Combine NSE + BSE, deduplicate by base symbol
        symbols = nse_syms + [s for s in bse_syms if s.replace(".BO", "") not in
                              {n.replace(".NS", "") for n in nse_syms}]
        print(f"📋 Total universe: {len(symbols)} stocks (NSE + BSE combined)\n")

    # Run scan
    results = run_full_scan(symbols)

    # Save results
    save_results(results)

    # Print top results to console
    buy   = [r for r in results if r["action"] == "BUY"]
    watch = [r for r in results if r["action"] == "WATCH"]

    print(f"\n{'='*60}")
    print(f"TOP BUY SIGNALS ({len(buy)} found):")
    print(f"{'='*60}")
    for r in buy[:10]:
        print(f"  {r['symbol']:15} ₹{r['price']:>8,.0f}  ROC:{r['roc_20']:>+6.1f}%  RSI:{r['rsi']:>5.0f}  Vol:{r['vol_ratio']:.1f}x  Score:{r['score']}/8")
        print(f"  {'':15} {' · '.join(r['signals'][:3])}")
        print()

    if watch:
        print(f"\nWATCH LIST ({len(watch)} stocks approaching breakout):")
        for r in watch[:5]:
            print(f"  {r['symbol']:15} ₹{r['price']:>8,.0f}  ROC:{r['roc_20']:>+6.1f}%  RSI:{r['rsi']:>5.0f}")

    # Build and send email
    html_body, action_text = build_email(results, vix)

    today     = datetime.date.today().strftime("%d %b %Y")
    buy_count = len(buy)
    subject   = f"Market Scan {today} — {buy_count} buy signal{'s' if buy_count != 1 else ''}"
    if vix and vix > 20:
        subject = f"⚠ Market Scan {today} — VIX {vix:.1f} HIGH — Hold cash"

    send_email(subject, html_body)
    print(f"\n✅ Done. Action: {action_text}")


if __name__ == "__main__":
    main()
