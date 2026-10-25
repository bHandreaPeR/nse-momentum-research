"""Robustness scan for the survivorship-free momentum result: vary lookback,
universe size, and quintile/decile. Build panels once, reuse. Reports long-only
top-bucket net Sharpe and the clean long-short Sharpe for each setting."""
import numpy as np, pandas as pd
import bhavcopy_factors as bf

price, tov, sym = bf.build_panels()
rets = price.pct_change()
dates = price.index; isins = list(price.columns)
LO, HI = bf.EXTREME_LO, bf.EXTREME_HI

def run(lookback=12, top_universe=200, frac=5, cost=bf.COST):
    rt = pd.Series(0.0, index=dates); rb = pd.Series(0.0, index=dates)
    ct = pd.Series(0.0, index=dates); prev=set()
    L = lookback
    for i in range(L+1, len(dates)-1):
        px=price.iloc[i]; px1=price.iloc[i-1]; pxL=price.iloc[i-1-L]
        valid = px.notna()&px1.notna()&pxL.notna()
        cand=[c for c in isins if valid[c]]
        if len(cand)<50: continue
        tn=tov.iloc[i][cand].fillna(0)
        univ=list(tn.sort_values(ascending=False).head(top_universe).index)
        look=rets.iloc[i-L:i][univ]
        clean=[c for c in univ if look[c].dropna().between(LO,HI).all()]
        if len(clean)<30: continue
        q=max(1,len(clean)//frac)
        mom=(px1[clean]/pxL[clean]-1)
        top=list(mom.nlargest(q).index); bot=list(mom.nsmallest(q).index)
        nxt=rets.iloc[i+1]
        def avg(n):
            v=nxt[n]; v=v[v.between(LO,HI)]; return v.mean()
        rt.iloc[i+1]=avg(top); rb.iloc[i+1]=avg(bot)
        ct.iloc[i+1]=len(set(top)^prev)/max(1,len(top))*cost; prev=set(top)
    net=(rt-ct).iloc[L+2:]; ls=(rt-rb).iloc[L+2:]
    def sh(r):
        r=r.dropna(); rf=(1+bf.CASH_ANNUAL)**(1/12)-1
        return (r.mean()-rf)/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    def cagr(r):
        r=r.dropna(); c=(1+r).prod(); return c**(12/len(r))-1 if c>0 else np.nan
    return sh(net), cagr(net), sh(ls.dropna() if False else ls)

def main():
  print("Long-only top-bucket NET Sharpe (CAGR%) | long-short Sharpe\n")
  print(f"{'lookback':<10}{'univ100':>16}{'univ200':>16}{'univ300':>16}")
  for L in (6,9,12):
    cells=[]
    for U in (100,200,300):
        s,c,_=run(lookback=L, top_universe=U, frac=5)
        cells.append(f"{s:.2f} ({c*100:.0f}%)")
    print(f"L={L:<8}", "".join(f"{x:>16}" for x in cells))
  print("\nQuintile vs decile (L=12, univ=200):")
  for fr,nm in ((5,"quintile"),(10,"decile")):
    s,c,ls=run(lookback=12, top_universe=200, frac=fr)
    print(f"  {nm:<9} top-bucket Sharpe {s:.2f} (CAGR {c*100:.0f}%) | long-short Sharpe {ls:.2f}")

if __name__ == "__main__":
    main()
