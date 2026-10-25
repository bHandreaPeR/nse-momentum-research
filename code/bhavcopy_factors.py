"""
SURVIVORSHIP-FREE cross-sectional factor test from NSE bhavcopy month-end files.

Why this is better than the earlier single-name test: the universe each month is
the set of stocks that ACTUALLY TRADED that month (incl. names later delisted),
picked by liquidity (turnover) -- a true point-in-time universe, not today's
survivors run backwards.

Factors (price-only): cross-sectional momentum (12-1) and low-volatility.
Honest benchmark = equal-weight of the same point-in-time universe.

CAVEATS (loud): prices are RAW (not split/bonus adjusted). A split makes a winner
look like a big loser, so it adds NOISE that biases AGAINST finding momentum -- a
positive result is therefore meaningful, a weak one inconclusive. We apply a crude
corporate-action guard (drop stock-months whose lookback contains an extreme move).
Month-end sampling only; ~2010-2024.
"""
import os, glob, csv, datetime as dt
import numpy as np, pandas as pd

NSE_DIR = os.environ.get("NSE_BHAVCOPY_DIR",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "bhavcopy_nse_monthly"))
CASH_ANNUAL = 0.055
COST = 0.0020
TOP_UNIVERSE = 200      # most liquid N stocks each month (point-in-time)
EXTREME_LO, EXTREME_HI = -0.55, 1.50   # monthly moves beyond this => likely corp action

# header: SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN
def parse_file(path):
    rows = {}
    with open(path) as f:
        rdr = list(csv.reader(f))
    if not rdr: return rows
    header = [h.strip() for h in rdr[0]]
    # NSE switched to the UDiFF bhavcopy format in 2024 (15 cols, CLOSE_PRICE at idx 8,
    # SERIES has a leading space, TURNOVER in lakhs). Old format: CLOSE idx 5, TTVAL idx 9.
    # Detect by header name, not column count: old-format files from late 2021 onward
    # carry two extra delivery columns (16 cols) but keep CLOSE at idx 5.
    new_fmt = "CLOSE_PRICE" in header
    for x in rdr[1:]:
        try:
            if new_fmt:
                if len(x) < 12 or x[1].strip() != "EQ": continue
                close = float(x[8]); ttval = float(x[11])     # turnover in lakhs (intra-month ranking unaffected)
            else:
                if len(x) < 10 or x[1].strip() != "EQ": continue
                close = float(x[5]); ttval = float(x[9])
        except Exception:
            continue
        sym = x[0].strip()
        if sym: rows[sym] = (close, ttval, sym)
    return rows

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bhavcopy_panels_refreshed.pkl")
def build_panels():
    if os.path.exists(_CACHE):
        import pickle
        with open(_CACHE, "rb") as f:
            return pickle.load(f)
    files = sorted(glob.glob(os.path.join(NSE_DIR, "*.csv")),
                   key=lambda p: dt.datetime.strptime(os.path.basename(p)[:-4], "%d%b%Y"))
    dates, price_rows, tov_rows, sym_map = [], [], [], {}
    for p in files:
        d = dt.datetime.strptime(os.path.basename(p)[:-4], "%d%b%Y")
        rec = parse_file(p)
        dates.append(pd.Timestamp(d.year, d.month, 1))  # normalize to month
        price_rows.append({k: v[0] for k, v in rec.items()})
        tov_rows.append({k: v[1] for k, v in rec.items()})
        for k, v in rec.items(): sym_map[k] = v[2]
    price = pd.DataFrame(price_rows, index=dates).sort_index()
    tov = pd.DataFrame(tov_rows, index=dates).sort_index()
    # collapse any duplicate months (keep last)
    price = price[~price.index.duplicated(keep="last")]
    tov = tov[~tov.index.duplicated(keep="last")]
    import pickle
    with open(_CACHE, "wb") as f:
        pickle.dump((price, tov, sym_map), f)
    return price, tov, sym_map

def perf(r, rf=CASH_ANNUAL):
    r = r.dropna();  n = len(r)
    if n < 6: return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan, n=n)
    yrs = n/12.0; cum=(1+r).prod()
    cagr = cum**(1/yrs)-1 if cum>0 else np.nan
    vol = r.std()*np.sqrt(12); rf_m=(1+rf)**(1/12)-1
    sharpe=(r.mean()-rf_m)/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); maxdd=(eq/eq.cummax()-1).min()
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, maxdd=maxdd, n=n)

def run():
    price, tov, sym = build_panels()
    rets = price.pct_change()
    dates = price.index
    isins = list(price.columns)

    ret_mom = pd.Series(0.0, index=dates); ret_lv = pd.Series(0.0, index=dates)
    ret_ew = pd.Series(0.0, index=dates); ret_bot = pd.Series(0.0, index=dates)
    cost_mom = pd.Series(0.0, index=dates); cost_lv = pd.Series(0.0, index=dates)
    nuniv = pd.Series(0, index=dates)
    prev_mom=set(); prev_lv=set()

    for i in range(13, len(dates)-1):
        px=price.iloc[i]; px1=price.iloc[i-1]; px13=price.iloc[i-13]
        # point-in-time universe: traded this month, has 13m history, top by turnover
        valid = px.notna() & px1.notna() & px13.notna()
        cand = [c for c in isins if valid[c]]
        if len(cand) < 50: continue
        tov_now = tov.iloc[i][cand].fillna(0)
        univ = list(tov_now.sort_values(ascending=False).head(TOP_UNIVERSE).index)
        # corporate-action guard: drop names with extreme move in lookback or next month
        look = rets.iloc[i-12:i][univ]
        clean = [c for c in univ if look[c].dropna().between(EXTREME_LO, EXTREME_HI).all()]
        if len(clean) < 30: continue
        nuniv.iloc[i]=len(clean)
        q = max(1, len(clean)//5)

        mom = (px1[clean]/px13[clean]-1)
        top = list(mom.nlargest(q).index); bot=list(mom.nsmallest(q).index)
        vol = rets.iloc[i-11:i+1][clean].std()
        lv = list(vol.nsmallest(q).index)

        nxt = rets.iloc[i+1]
        # guard next-month returns too (exclude corp-action months from realized return)
        def avg(names):
            v = nxt[names]
            v = v[v.between(EXTREME_LO, EXTREME_HI)]
            return v.mean()
        ret_mom.iloc[i+1]=avg(top); ret_bot.iloc[i+1]=avg(bot)
        ret_lv.iloc[i+1]=avg(lv);   ret_ew.iloc[i+1]=avg(clean)
        cost_mom.iloc[i+1]=len(set(top)^prev_mom)/max(1,len(top))*COST
        cost_lv.iloc[i+1]=len(set(lv)^prev_lv)/max(1,len(lv))*COST
        prev_mom=set(top); prev_lv=set(lv)

    s=14
    mom=(ret_mom-cost_mom).iloc[s:]; lv=(ret_lv-cost_lv).iloc[s:]
    ew=ret_ew.iloc[s:]; ls=(ret_mom-ret_bot).iloc[s:]
    # Nifty from existing Yahoo ETF data, aligned
    nf = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","NIFTYBEES.NS.csv"),
                     parse_dates=["date"]).set_index("date")["adjclose"]
    nf.index = nf.index.to_period("M").to_timestamp()      # month-start key to match
    nfr = nf.pct_change().reindex(mom.index)

    def line(nm,r):
        st=perf(r.dropna());
        return f"  {nm:<32} CAGR {st['cagr']*100:6.1f}%  Vol {st['vol']*100:4.1f}%  Sharpe {st['sharpe']:5.2f}  MaxDD {st['maxdd']*100:6.1f}%"
    print(f"SURVIVORSHIP-FREE bhavcopy test | {mom.dropna().index[0].date()} -> {mom.dropna().index[-1].date()}")
    print(f"Point-in-time universe: top {TOP_UNIVERSE} by turnover | avg clean names {nuniv[nuniv>0].mean():.0f} | quintile ~{int(nuniv[nuniv>0].mean()//5)}\n")
    print(line("Momentum (long top quintile, net)", mom))
    print(line("Low-vol (low-vol quintile, net)", lv))
    print(line("EQUAL-WEIGHT universe (honest bench)", ew))
    print(line("Nifty 50 (NIFTYBEES)", nfr))
    lsd=ls.dropna(); lss=lsd.mean()/lsd.std()*np.sqrt(12) if lsd.std()>0 else np.nan
    print(f"\n  Momentum long-SHORT spread (gross): ann.mean {lsd.mean()*1200:5.1f}%  Sharpe {lss:5.2f}")
    print("  (clean long-short Sharpe = the real cross-sectional momentum premium, survivorship-free)")

if __name__=="__main__":
    run()
