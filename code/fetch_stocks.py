"""Fetch monthly price history for a large-cap NSE stock universe (cached/resumable).
SURVIVORSHIP CAVEAT: these are *current* large caps, so the backtest will be
biased upward (we only see survivors). A proper test needs point-in-time
constituents incl. delisted names -- a paid dataset."""
import os, time
import data_fetch as dfetch

dfetch.DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "stocks")
os.makedirs(dfetch.DATA_DIR, exist_ok=True)

STOCKS = [
 "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","HINDUNILVR.NS",
 "ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","BAJFINANCE.NS",
 "AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","TITAN.NS","SUNPHARMA.NS","ULTRACEMCO.NS",
 "NESTLEIND.NS","WIPRO.NS","ONGC.NS","NTPC.NS","POWERGRID.NS","TATAMOTORS.NS",
 "TATASTEEL.NS","HCLTECH.NS","JSWSTEEL.NS","COALINDIA.NS","BAJAJFINSV.NS","GRASIM.NS",
 "BRITANNIA.NS","DRREDDY.NS","CIPLA.NS","EICHERMOT.NS","HEROMOTOCO.NS","HINDALCO.NS",
 "INDUSINDBK.NS","BPCL.NS","TECHM.NS","TATACONSUM.NS","UPL.NS","SBILIFE.NS",
 "HDFCLIFE.NS","DIVISLAB.NS","ADANIPORTS.NS",
]
done = 0
for s in STOCKS:
    try:
        msg = dfetch.fetch_one(s); print(msg, flush=True)
        if msg.startswith("OK") or msg.startswith("CACHED"):
            done += 1
        if msg.startswith("OK"):
            time.sleep(1.5)   # only pause after a real network fetch
    except RuntimeError as e:
        print("ERR", s, str(e)[:80], flush=True)
        time.sleep(1.0)
print(f"\n{done}/{len(STOCKS)} available")
