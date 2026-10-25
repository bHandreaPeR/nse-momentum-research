"""
Run this on a machine that can reach nseindia.com (NSE blocks many cloud IP ranges).

Downloads the NSE 'sec_bhavdata_full' month-end files (all stocks, one file per day)
for the months the free GitHub archive is missing, so the momentum signal can be
brought current. One file per month is enough for the monthly/quarterly strategy.

Saves into code/nse_recent/<DDMMMYYYY>.csv; refresh_panel.py reads that folder.

USAGE:
    pip install requests
    python fetch_nse_recent.py
"""
import os, sys, time, datetime as dt, calendar
import requests

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nse_recent")
os.makedirs(OUT, exist_ok=True)
START = (2025, 1)                       # first month after the free GitHub archive ends (Dec 2024)
END   = (dt.date.today().year, dt.date.today().month)

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept": "text/csv,*/*", "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
})

def warm():
    try: S.get("https://www.nseindia.com", timeout=12)
    except Exception as e: print("cookie warm warning:", repr(e))

def url(d): return f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv"

def try_download(d, retries=3):
    last=None
    for a in range(retries):
        try:
            r = S.get(url(d), timeout=15)
            if r.status_code == 200 and "SYMBOL" in r.text[:200]:
                p = os.path.join(OUT, f"{d:%d%b%Y}".upper()+".csv")
                open(p, "w").write(r.text)
                return p
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = repr(e); time.sleep(1.5)
    return None

def month_iter(start, end):
    y,m = start
    while (y,m) <= end:
        yield y,m
        m += 1
        if m > 12: m = 1; y += 1

if __name__ == "__main__":
    warm()
    got=0
    for y,m in month_iter(START, END):
        last_day = calendar.monthrange(y,m)[1]
        saved=None
        for day in range(last_day, last_day-6, -1):     # try last ~6 days for the last trading day
            d = dt.date(y,m,day)
            if d.weekday() >= 5: continue
            if d > dt.date.today(): continue
            saved = try_download(d)
            if saved: break
            time.sleep(0.8)
        print(f"{y}-{m:02d}: {'saved '+os.path.basename(saved) if saved else 'NOT FOUND (holiday/not yet published?)'}")
        if saved: got+=1
        time.sleep(1.0)
    print(f"\nDone: {got} month-end files in {OUT}")
    print("Next: python3 refresh_panel.py && python3 live_signal.py")
