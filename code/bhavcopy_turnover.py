"""
Turnover / tax reduction on the final momentum strategy.

Levers tested:
  * rebalance frequency (monthly vs quarterly)
  * buffer band: keep a held name as long as it stays in the top 2*quintile
    (only replace genuine drop-outs) -> less churn at the ranking edge
Tax model is HOLDING-PERIOD AWARE: gains on lots held > 12 months are taxed at
India's long-term rate (12.5%); shorter holds at the short-term rate (20%).
This is what makes lower turnover actually improve net returns.

Always includes the 8-month market-regime filter (cash when market < 8m SMA),
net of 25 bps/side costs, survivorship-free, 2004-2024 (through 2008).
"""
import numpy as np, pandas as pd
import bhavcopy_factors as bf
import bhavcopy_strategy as bs
import bhavcopy_regime_hmm as rh   # importing builds rh.mkt (market index)

price, dates, rets, tov = bs.price, bs.dates, bs.rets, bs.tov
LO, HI = bf.EXTREME_LO, bf.EXTREME_HI
CASH_M = (1 + bf.CASH_ANNUAL) ** (1/12) - 1
mkt_price = (1 + rh.mkt).cumprod()
RISK_ON = (mkt_price > mkt_price.rolling(8).mean())

def ranked(i, top_universe=200, lookback=12):
    px=price.iloc[i]; px1=price.iloc[i-1]; pxL=price.iloc[i-1-lookback]
    valid=px.notna()&px1.notna()&pxL.notna()
    cand=[c for c in price.columns if valid[c]]
    if len(cand)<50: return []
    univ=tov.iloc[i][cand].fillna(0).sort_values(ascending=False).head(top_universe).index
    look=rets.iloc[i-lookback:i][univ]
    clean=[c for c in univ if look[c].dropna().between(LO,HI).all()]
    if len(clean)<30: return []
    mom=(px1[clean]/pxL[clean]-1).sort_values(ascending=False)
    return list(mom.index)

def simulate(rebalance_every=1, buffer=False, stcg=0.20, ltcg=0.125, cost=0.0025, lookback=12):
    start=lookback+1
    holdings={}                      # name -> [value, basis, entry_i]
    cash=1.0; nav=1.0; st_real=0.0; lt_real=0.0
    navs=[]; ndx=[]; turn=0.0; lt_sold=0.0; tot_sold=0.0
    in_cash=True
    for k in range(start, len(dates)-1):
        for n in holdings: holdings[n][0]*=(1+bs.safe_ret(k,n))
        cash*=(1+CASH_M)
        ro=bool(RISK_ON.get(dates[k],True))
        target=None
        if not ro:
            target=set()                                   # liquidate to cash
        elif in_cash or (k%rebalance_every==0):
            rk=ranked(k,lookback=lookback)
            if rk:
                q=max(1,len(rk)//5); core=rk[:q]
                if buffer and holdings:
                    band=set(rk[:2*q]); keep=[n for n in holdings if n in band]
                    need=max(0,q-len(keep)); adds=[n for n in core if n not in keep][:need]
                    target=set(keep+adds)
                else:
                    target=set(core)
        if target is not None:
            cur=set(holdings); sells=cur-target; buys=target-cur
            for n in sells:
                v,basis,ei=holdings.pop(n); gain=v-basis
                if (k-ei)>12: lt_real+=gain; lt_sold+=v
                else: st_real+=gain
                tot_sold+=v; cash+=v*(1-cost); turn+=v
            if buys and cash>0:
                e=cash/len(buys)
                for n in buys: holdings[n]=[e*(1-cost), e*(1-cost), k]; turn+=e
                cash=0.0
            in_cash=(len(holdings)==0)
        nav=sum(h[0] for h in holdings.values())+cash
        if dates[k].month==12:
            tax=stcg*max(st_real,0)+ltcg*max(lt_real,0)
            if tax>0:
                if cash>=tax: cash-=tax
                else:
                    rem=tax-cash; cash=0.0; tot=sum(h[0] for h in holdings.values())
                    if tot>0:
                        sc=max(0,tot-rem)/tot
                        for n in holdings: holdings[n][0]*=sc
            st_real=0.0; lt_real=0.0; nav=sum(h[0] for h in holdings.values())+cash
        navs.append(nav); ndx.append(dates[k])
    r=pd.Series(navs,index=ndx).pct_change().dropna()
    yrs=(len(dates)-start)/12.0
    return r, turn/yrs, (lt_sold/tot_sold if tot_sold>0 else 0.0)

if __name__=="__main__":
    cfgs=[
        ("Monthly, no buffer (baseline)", dict(rebalance_every=1, buffer=False)),
        ("Monthly + buffer",              dict(rebalance_every=1, buffer=True)),
        ("Quarterly, no buffer",          dict(rebalance_every=3, buffer=False)),
        ("Quarterly + buffer",            dict(rebalance_every=3, buffer=True)),
    ]
    print("All: momentum + 8m regime filter, survivorship-free, net 25bps + holding-period tax (20% ST / 12.5% LT)\n")
    print(f"{'Config':<32}{'CAGR':>7}{'Sharpe':>8}{'MaxDD':>9}{'Turnov':>8}{'%LT-tax':>9}")
    for name,kw in cfgs:
        r,tn,ltf=simulate(**kw); st=bf.perf(r.dropna())
        print(f"{name:<32}{st['cagr']*100:6.1f}%{st['sharpe']:8.2f}{st['maxdd']*100:8.1f}%{tn:7.1f}x{ltf*100:8.0f}%")
