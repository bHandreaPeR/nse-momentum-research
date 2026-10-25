"""
Crash control for the survivorship-free momentum strategy.

Naive vol-targeting failed (de-risks right before momentum rebounds, and a
no-leverage strategy can only cut exposure). The deployable fix is a MARKET
REGIME FILTER: hold the momentum book only when the Nifty is in an uptrend
(above its 10-month SMA); otherwise sit in cash -- avoiding the bear-market /
rebound zone where momentum crashes. Modeled INSIDE the position sim so the
cost + capital-gains tax of going to cash and re-entering are captured.

Compares, all net of 25bps/side costs + 20% STCG:
  A) raw momentum quintile (always invested)
  B) regime-filtered momentum (cash when Nifty < 10m SMA)
vs Nifty buy & hold, with an out-of-sample split.
"""
import numpy as np, pandas as pd
import bhavcopy_factors as bf
import bhavcopy_strategy as bs

price, dates, rets = bs.price, bs.dates, bs.rets
safe_ret = bs.safe_ret
CASH_M = (1 + bf.CASH_ANNUAL) ** (1/12) - 1

# ---- INTERNAL market index built from bhavcopy itself (available full 2003-2024,
#      so the regime filter actually operates through the 2008 crash) ----
LO, HI = bf.EXTREME_LO, bf.EXTREME_HI
tov = bs.tov
_mkt = pd.Series(np.nan, index=dates)
for i in range(1, len(dates)):
    pxp = price.iloc[i-1]; cand = [c for c in price.columns if pd.notna(pxp[c])]
    if len(cand) < 50: continue
    univ = tov.iloc[i-1][cand].fillna(0).sort_values(ascending=False).head(200).index
    r = rets.iloc[i][univ]; r = r[r.between(LO, HI)]
    _mkt.iloc[i] = r.mean()
mkt_price = (1 + _mkt.fillna(0)).cumprod()   # equal-weight top-200 market proxy
nf = mkt_price                                # used by SMA filters below
sma10 = nf.rolling(10).mean()
risk_on = (nf > sma10)

def simulate(T, signal=None, cost_side=0.0025, stcg=0.20):
    keys = sorted(T.keys())
    def target_at(k):
        if signal is not None and not bool(signal.get(dates[k], True)):
            return []
        return T[k]
    holdings = {}; basis = {}; cash = 0.0; realized = 0.0
    nav = 1.0
    names = target_at(keys[0])
    if names:
        alloc = nav/len(names)
        for n in names:
            holdings[n] = alloc*(1-cost_side); basis[n] = holdings[n]
        nav = sum(holdings.values())
    else:
        cash = nav
    navs=[sum(holdings.values())+cash]; ndx=[dates[keys[0]]]
    for k in keys[1:]:
        for n in list(holdings): holdings[n] *= (1 + safe_ret(k, n))
        cash *= (1 + CASH_M)
        tgt = set(target_at(k)); cur = set(holdings)
        for n in (cur - tgt):
            val = holdings.pop(n); realized += val - basis.pop(n); cash += val*(1-cost_side)
        buys = tgt - cur
        if buys and cash > 0:
            each = cash/len(buys)
            for n in buys:
                holdings[n] = each*(1-cost_side); basis[n] = holdings[n]
            cash = 0.0
        nav = sum(holdings.values()) + cash
        if dates[k].month == 12 and stcg > 0 and realized > 0:
            tax = stcg*realized
            if cash >= tax: cash -= tax
            else:
                rem = tax-cash; cash = 0.0; tot = sum(holdings.values())
                if tot > 0:
                    sc = max(0.0, tot-rem)/tot
                    for n in holdings: holdings[n] *= sc
            realized = 0.0
        elif dates[k].month == 12:
            realized = 0.0
        navs.append(sum(holdings.values())+cash); ndx.append(dates[k])
    return pd.Series(navs, index=ndx).pct_change().dropna()

T = bs.compute_targets(top_universe=200, lookback=12, frac=5)
raw  = simulate(T, signal=None)
nfr  = nf.pct_change().reindex(raw.index)

print(f"Crash-control comparison | {raw.index[0].date()} -> {raw.index[-1].date()}")
print("All net of 25bps/side costs + 20% STCG:\n")
bs.stats(raw,  "A) Raw momentum (always invested)")
bs.stats(nfr,  "   Market proxy (EW top-200) buy & hold")

# ---- filter-robustness: is the ~1.2 Sharpe an artifact of the 10m SMA choice? ----
print("\nFILTER ROBUSTNESS (different regime definitions, market = EW top-200):")
signals = {
    "Mkt > 8m SMA":  nf > nf.rolling(8).mean(),
    "Mkt > 10m SMA": nf > nf.rolling(10).mean(),
    "Mkt > 12m SMA": nf > nf.rolling(12).mean(),
    "Mkt 12m return > 0": nf.pct_change(12) > 0,
}
filt10 = None
for name, sig in signals.items():
    r = simulate(T, signal=sig)
    if name == "Mkt > 10m SMA": filt10 = r
    onpct = sig.reindex(r.index).mean()*100
    bs.stats(r, f"  {name} (invested {onpct:.0f}%)")

print("\nOut-of-sample split (Nifty>10m SMA filter):")
mid = filt10.index[len(filt10)//2]
bs.stats(filt10.loc[:mid], "  First half")
bs.stats(filt10.loc[filt10.index > mid], "  Second half")
