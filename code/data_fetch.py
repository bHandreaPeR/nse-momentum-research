"""Resilient Yahoo chart-API downloader for Indian ETFs.
Cookie session + spacing to dodge 429; caches each ticker to data/<TICKER>.csv
so progress survives timeouts. Uses adjusted close when available.
Raises explicitly on persistent failure (no silent fallback)."""
import urllib.request, http.cookiejar, json, time, os, csv, datetime as dt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

UNIVERSE = ["NIFTYBEES.NS", "JUNIORBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "MON100.NS"]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121 Safari/537.36"

_cj = http.cookiejar.CookieJar()
_op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_cj))
_op.addheaders = [("User-Agent", UA)]
try:
    _op.open("https://fc.yahoo.com", timeout=8)
except Exception:
    pass

def fetch_one(sym, retries=3):
    path = os.path.join(DATA_DIR, sym + ".csv")
    if os.path.exists(path) and os.path.getsize(path) > 200:
        with open(path) as f:
            n = sum(1 for _ in f) - 1
        return f"CACHED {sym} ({n} rows)"
    hosts = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
    last_err = None
    for attempt in range(retries):
        host = hosts[attempt % 2]
        url = (f"https://{host}/v8/finance/chart/{sym}"
               f"?range=max&interval=1d&events=div%2Csplit&includeAdjustedClose=true")
        try:
            raw = _op.open(url, timeout=10).read().decode()
            j = json.loads(raw)
            res = j["chart"]["result"][0]
            ts = res["timestamp"]
            ind = res["indicators"]
            adj = ind.get("adjclose", [{}])[0].get("adjclose")
            close = ind["quote"][0]["close"]
            series = adj if adj else close
            rows = []
            for t, c in zip(ts, series):
                if c is None:
                    continue
                rows.append((dt.date.fromtimestamp(t).isoformat(), round(c, 4)))
            with open(path, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["date", "adjclose"]); w.writerows(rows)
            return f"OK {sym} ({len(rows)} rows, {rows[0][0]} -> {rows[-1][0]})"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:90]}"
            time.sleep(1.5)
    raise RuntimeError(f"FAILED {sym} after {retries} attempts. Last: {last_err}")

if __name__ == "__main__":
    for i, s in enumerate(UNIVERSE):
        try:
            print(fetch_one(s), flush=True)
        except RuntimeError as e:
            print("ERROR", e, flush=True)
        time.sleep(3.5)
