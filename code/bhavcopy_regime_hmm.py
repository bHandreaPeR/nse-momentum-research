"""
Faster regime detectors vs the slow SMA, tested THROUGH 2008.

Implements (dependency-free, numpy only):
  * HMM:  a 2-state Gaussian Hidden Markov Model (Baum-Welch EM, scaled
          forward-backward). Bull state = higher-mean state. Risk-ON when the
          filtered posterior favors the bull state.
  * EWMA: RiskMetrics exponentially-weighted volatility forecast (GARCH proxy).
          Risk-OFF when forecast vol exceeds its trailing (training-only) median.

Both are WALK-FORWARD: at each month they use only data available up to then
(HMM refit on the expanding window; EWMA/threshold from past data). The signal
chosen at month k governs the holding over month k+1 -> no look-ahead.

Compared against the 8m/10m SMA filters and raw momentum, net of 25bps/side
costs + 20% STCG, over 2004-2024 (includes the 2008 crash). Also reports the
drawdown specifically through the 2008 window.
"""
import numpy as np, pandas as pd
import bhavcopy_factors as bf
import bhavcopy_strategy as bs

price, dates, rets, tov = bs.price, bs.dates, bs.rets, bs.tov
LO, HI = bf.EXTREME_LO, bf.EXTREME_HI
CASH_M = (1 + bf.CASH_ANNUAL) ** (1/12) - 1

# ---------- internal market index (EW top-200 by turnover), full history ----------
_mkt = pd.Series(np.nan, index=dates)
for i in range(1, len(dates)):
    pxp = price.iloc[i-1]; cand = [c for c in price.columns if pd.notna(pxp[c])]
    if len(cand) < 50: continue
    univ = tov.iloc[i-1][cand].fillna(0).sort_values(ascending=False).head(200).index
    r = rets.iloc[i][univ]; r = r[r.between(LO, HI)]
    _mkt.iloc[i] = r.mean()
mkt = _mkt.fillna(0.0)

# ---------- cash-aware position sim with optional regime signal ----------
def simulate(T, signal=None, cost_side=0.0025, stcg=0.20):
    keys = sorted(T.keys())
    def tgt(k):
        if signal is not None and not bool(signal.get(dates[k], True)): return []
        return T[k]
    holdings={}; basis={}; cash=0.0; realized=0.0; nav=1.0
    nm = tgt(keys[0])
    if nm:
        a = nav/len(nm)
        for n in nm: holdings[n]=a*(1-cost_side); basis[n]=holdings[n]
    else: cash=nav
    navs=[sum(holdings.values())+cash]; ndx=[dates[keys[0]]]
    for k in keys[1:]:
        for n in list(holdings): holdings[n]*=(1+bs.safe_ret(k,n))
        cash*=(1+CASH_M)
        t=set(tgt(k)); cur=set(holdings)
        for n in (cur-t):
            v=holdings.pop(n); realized+=v-basis.pop(n); cash+=v*(1-cost_side)
        buys=t-cur
        if buys and cash>0:
            e=cash/len(buys)
            for n in buys: holdings[n]=e*(1-cost_side); basis[n]=holdings[n]
            cash=0.0
        if dates[k].month==12 and stcg>0 and realized>0:
            tax=stcg*realized
            if cash>=tax: cash-=tax
            else:
                rem=tax-cash; cash=0.0; tot=sum(holdings.values())
                if tot>0:
                    sc=max(0.0,tot-rem)/tot
                    for n in holdings: holdings[n]*=sc
            realized=0.0
        elif dates[k].month==12: realized=0.0
        navs.append(sum(holdings.values())+cash); ndx.append(dates[k])
    return pd.Series(navs,index=ndx).pct_change().dropna()

# ---------- 2-state Gaussian HMM (Baum-Welch, scaled) ----------
def hmm_fit(x, n_iter=30):
    x=np.asarray(x,float); T=len(x)
    mu=np.array([x.mean()-x.std(), x.mean()+x.std()]); var=np.array([x.var(),x.var()])+1e-8
    A=np.array([[0.9,0.1],[0.1,0.9]]); pi=np.array([0.5,0.5])
    for _ in range(n_iter):
        B=np.empty((T,2))
        for k in range(2):
            B[:,k]=np.exp(-0.5*(x-mu[k])**2/var[k])/np.sqrt(2*np.pi*var[k])
        B=np.clip(B,1e-300,None)
        alpha=np.empty((T,2)); c=np.empty(T)
        alpha[0]=pi*B[0]; c[0]=alpha[0].sum(); alpha[0]/=c[0]
        for t in range(1,T):
            alpha[t]=(alpha[t-1]@A)*B[t]; c[t]=alpha[t].sum(); alpha[t]/=c[t]
        beta=np.empty((T,2)); beta[-1]=1.0
        for t in range(T-2,-1,-1):
            beta[t]=A@(B[t+1]*beta[t+1]); beta[t]/=c[t+1]
        gamma=alpha*beta; gamma/=gamma.sum(1,keepdims=True)
        ab=B[1:]*beta[1:]
        num=alpha[:-1][:,:,None]*A[None,:,:]*ab[:,None,:]
        xi=(num/num.sum((1,2),keepdims=True)).sum(0)
        pi=gamma[0]
        A=xi/gamma[:-1].sum(0)[:,None]; A/=A.sum(1,keepdims=True)
        for k in range(2):
            w=gamma[:,k]; sw=w.sum()
            mu[k]=(w*x).sum()/sw; var[k]=(w*(x-mu[k])**2).sum()/sw
        var=np.clip(var,1e-8,None)
    return mu, gamma[-1]   # params, filtered posterior at last step

def hmm_signal(mkt_ret, min_train=48, refit_every=1):
    vals=mkt_ret.values; sig=pd.Series(True,index=mkt_ret.index); last=True
    for i in range(len(vals)):
        if i<min_train: continue
        if (i-min_train)%refit_every==0:
            try:
                mu,post=hmm_fit(vals[:i+1]); bull=int(np.argmax(mu))
                last = post[bull]>0.5
            except Exception: last=True
        sig.iloc[i]=last
    return sig

def ewma_signal(mkt_ret, lam=0.84, min_train=48, k=1.0):
    r=mkt_ret.values; T=len(r); var=np.empty(T); var[0]=np.var(r[:12])
    for t in range(1,T): var[t]=lam*var[t-1]+(1-lam)*r[t-1]**2
    fvol=np.sqrt(lam*var+(1-lam)*r**2)   # forecast vol using info through t
    sig=pd.Series(True,index=mkt_ret.index)
    for i in range(len(r)):
        if i<min_train: continue
        thr=np.median(fvol[12:i+1])
        sig.iloc[i]= fvol[i] <= thr*k
    return sig

def dd_2008(r):
    w=r[(r.index>=pd.Timestamp(2007,6,1))&(r.index<=pd.Timestamp(2009,12,1))]
    if len(w)<3: return np.nan
    eq=(1+w).cumprod(); return (eq/eq.cummax()-1).min()

if __name__=="__main__":
    T=bs.compute_targets(top_universe=200, lookback=12, frac=5)
    sma8 = mkt>mkt.cumsum().rolling(1).mean()  # placeholder, replaced below
    mkt_price=(1+mkt).cumprod()
    sigs={
        "Raw (always in)": None,
        "SMA 8m": mkt_price>mkt_price.rolling(8).mean(),
        "SMA 10m": mkt_price>mkt_price.rolling(10).mean(),
        "HMM 2-state": hmm_signal(mkt),
        "EWMA-vol (GARCH proxy)": ewma_signal(mkt),
        "HMM AND SMA8": None,  # filled below
    }
    sigs["HMM AND SMA8"] = sigs["HMM 2-state"] & sigs["SMA 8m"]
    sigs["SMA8 AND EWMA-calm"] = sigs["SMA 8m"] & sigs["EWMA-vol (GARCH proxy)"]
    print(f"Regime-detector comparison through 2008 | net 25bps + 20% STCG\n")
    print(f"{'Strategy':<26}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>9}{'2008 DD':>9}{'%in':>6}")
    for name,sg in sigs.items():
        r=simulate(T, signal=sg)
        st=bf.perf(r.dropna())
        inpct = 100.0 if sg is None else sg.reindex(r.index).mean()*100
        print(f"{name:<26}{st['cagr']*100:7.1f}%{st['vol']*100:6.1f}%{st['sharpe']:8.2f}{st['maxdd']*100:8.1f}%{dd_2008(r)*100:8.1f}%{inpct:5.0f}")
