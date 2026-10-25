"""
LIVE SIGNAL GENERATOR — the paper-trading tool.

Run this each quarter. It prints:
  1) REGIME: risk-ON or risk-OFF (market vs its 8-month average)
  2) If risk-ON: the target portfolio — the top-quintile 12-1 momentum stocks
     within the top-200-by-turnover universe, equal weighted.
  3) If risk-OFF: hold CASH.

It reads the latest month-end panel available. To trade current signals, refresh
the bhavcopy data (or plug in your Groww data) so the panel includes the latest
month, then re-run. This is the exact rule set validated in the backtest
(survivorship-free, net of costs + tax, through 2008): quarterly momentum + 8m
regime filter.
"""
import os
import numpy as np, pandas as pd
import bhavcopy_factors as bf
import bhavcopy_strategy as bs
import bhavcopy_regime_hmm as rh

price, dates, rets, tov = bs.price, bs.dates, bs.rets, bs.tov
LO, HI = bf.EXTREME_LO, bf.EXTREME_HI
mkt_price = (1 + rh.mkt).cumprod()

def signal(i=None, top_universe=200, lookback=12):
    if i is None: i = len(dates) - 1
    asof = dates[i]
    ro = bool(mkt_price.iloc[i] > mkt_price.rolling(8).mean().iloc[i])
    px=price.iloc[i]; px1=price.iloc[i-1]; pxL=price.iloc[i-1-lookback]
    valid=px.notna()&px1.notna()&pxL.notna()
    cand=[c for c in price.columns if valid[c]]
    univ=tov.iloc[i][cand].fillna(0).sort_values(ascending=False).head(top_universe).index
    look=rets.iloc[i-lookback:i][univ]
    clean=[c for c in univ if look[c].dropna().between(LO,HI).all()]
    mom=(px1[clean]/pxL[clean]-1).sort_values(ascending=False)
    q=max(1,len(clean)//5)
    top=mom.head(q)
    return asof, ro, top

if __name__=="__main__":
    # use the most recent month that yields a full universe (last panel month can be sparse)
    i = len(dates) - 1
    while i > 20:
        asof, ro, top = signal(i)
        if len(top) >= 10: break
        i -= 1
    print("="*60)
    print(f"  LIVE SIGNAL  |  data as of month-end: {asof.date()}")
    print("="*60)
    print(f"  REGIME: {'RISK-ON (invest)' if ro else 'RISK-OFF (HOLD CASH)'}")
    if ro:
        w = 100.0/len(top)
        print(f"  TARGET PORTFOLIO: {len(top)} stocks, equal weight ({w:.1f}% each)\n")
        print(f"  {'#':<4}{'Symbol':<16}{'12-1 momentum':>14}")
        for j,(sym,m) in enumerate(top.items(),1):
            print(f"  {j:<4}{sym:<16}{m*100:>13.1f}%")
        # save to CSV for reference
        out=pd.DataFrame({"rank":range(1,len(top)+1),"symbol":top.index,
                          "momentum_12_1_pct":(top.values*100).round(1),
                          "target_weight_pct":round(100.0/len(top),2)})
        out.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "current_portfolio.csv"), index=False)
        print(f"\n  Saved -> current_portfolio.csv")
    else:
        print("  Hold cash / liquid fund until the market reclaims its 8-month average.")
    print("\n  NOTE: signal is as of the latest CLEAN month in the panel. For a truly current")
    print("  call, refresh data (run groww_fetch.py locally / add recent bhavcopy) and re-run.")
