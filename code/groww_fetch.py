"""
Groww API historical data fetcher  ---  RUN THIS ON YOUR OWN MACHINE.

It pulls DAILY OHLC candles for the Indian ETF universe via the Groww Python
SDK and writes one CSV per symbol that the backtest engine can read.

SECURITY: your API token is read from the environment variable GROWW_API_TOKEN
or a local file `groww_token.txt` next to this script. NEVER hardcode it and
NEVER paste it into a chat. Add groww_token.txt to .gitignore.

INSTALL:  pip install growwapi pandas
TOKEN:    export GROWW_API_TOKEN="...."     (or put it in groww_token.txt)
RUN:      python groww_fetch.py             (writes CSVs into ./groww_data/)

CAVEATS (honest):
  * Groww daily history effectively starts ~2020 and candles are RAW close
    (NOT dividend/split adjusted). Fine for a daily-granularity check from 2020;
    for long history keep the longer adjusted series. Adjust before trusting
    total-return numbers.
  * Verify the exact trading symbols against groww.get_all_instruments() if any
    symbol fails -- ETF symbols occasionally differ.
"""
import os, sys, time, csv, datetime as dt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("groww_fetch")

SYMBOLS = ["NIFTYBEES", "JUNIORBEES", "BANKBEES", "GOLDBEES", "MON100"]
START_DATE = "2018-01-01"           # Groww may only return from ~2020; that's fine
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "groww_data")
CHUNK_DAYS = 1000                   # daily candles: <= 1080 days per request
MAX_RETRIES = 3
SLEEP_BETWEEN = 1.0

os.makedirs(OUT_DIR, exist_ok=True)


def load_token() -> str:
    tok = os.environ.get("GROWW_API_TOKEN")
    if not tok:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "groww_token.txt")
        if os.path.exists(path):
            tok = open(path).read().strip()
    if not tok:
        raise RuntimeError(
            "No Groww token found. Set env GROWW_API_TOKEN or create groww_token.txt. "
            "Do NOT hardcode the token."
        )
    return tok


def make_client(token: str):
    try:
        from growwapi import GrowwAPI
    except ImportError as e:
        raise RuntimeError("growwapi not installed. Run: pip install growwapi") from e
    return GrowwAPI(token)


def fetch_chunk(groww, symbol: str, start: str, end: str):
    """One request with retries-with-warning-then-raise (per project error rules)."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = groww.get_historical_candle_data(
                trading_symbol=symbol,
                exchange=groww.EXCHANGE_NSE,
                segment=groww.SEGMENT_CASH,
                start_time=f"{start} 09:15:00",
                end_time=f"{end} 15:30:00",
                interval_in_minutes=1440,   # daily
            )
            return resp.get("candles", []) or []
        except Exception as e:  # noqa: BLE001 -- we re-raise with context below
            last_err = e
            log.warning("fetch failed", extra={}, exc_info=False)
            log.warning("retry", extra={"symbol": symbol, "attempt": attempt,
                                        "start": start, "end": end, "error": repr(e)})
            time.sleep(SLEEP_BETWEEN * attempt)
    raise RuntimeError(
        f"Groww historical fetch failed for symbol={symbol} start={start} end={end} "
        f"after {MAX_RETRIES} attempts. Last error: {last_err!r}"
    )


def fetch_symbol(groww, symbol: str):
    start = dt.date.fromisoformat(START_DATE)
    today = dt.date.today()
    rows = {}
    cur = start
    while cur < today:
        chunk_end = min(cur + dt.timedelta(days=CHUNK_DAYS), today)
        candles = fetch_chunk(groww, symbol, cur.isoformat(), chunk_end.isoformat())
        for c in candles:
            ts, o, h, l, close, vol = c[0], c[1], c[2], c[3], c[4], c[5]
            d = dt.datetime.utcfromtimestamp(ts).date().isoformat()
            rows[d] = (round(float(close), 4), int(vol))
        log.info("fetched window", extra={"symbol": symbol, "from": cur.isoformat(),
                                          "to": chunk_end.isoformat(), "rows_so_far": len(rows)})
        cur = chunk_end + dt.timedelta(days=1)
        time.sleep(SLEEP_BETWEEN)
    if not rows:
        raise RuntimeError(f"No candles returned for {symbol}. Check the trading symbol is correct.")
    out_path = os.path.join(OUT_DIR, f"{symbol}.NS.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "adjclose", "volume"])   # NB: 'adjclose' here is RAW close (see caveats)
        for d in sorted(rows):
            w.writerow([d, rows[d][0], rows[d][1]])
    log.info("wrote csv", extra={"symbol": symbol, "path": out_path, "rows": len(rows)})
    return out_path


def main():
    token = load_token()
    groww = make_client(token)
    log.info("starting fetch", extra={"symbols": SYMBOLS, "start": START_DATE})
    written = []
    for sym in SYMBOLS:
        written.append(fetch_symbol(groww, sym))
    log.info("done", extra={"files": written})
    print("\nWrote:")
    for p in written:
        print("  ", p)
    print("\nSaved under code/groww_data/.")


if __name__ == "__main__":
    main()
