"""
MAKE-OR-BREAK test for the survivorship-free momentum strategy:
position-level simulation with realistic Indian transaction costs AND
short-term capital-gains tax, a vol-targeting overlay, and a split-sample
out-of-sample check.

Strategy: monthly, long the top-quintile 12-1 momentum stocks within a
point-in-time top-200-by-turnover universe (survivorship-free bhavcopy data).
LAZY rebalance: persisting names ride (no needless turnover); only entries/exits
trade -> realistic turnover and tax.

Costs: 25 bps per side (STT 0.1%x2 + brokerage + slippage + stamp/exchange/GST).
Tax: short-term capital gains on realized gains of SOLD positions, paid annually.
   India STCG on equities = 15% pre-23-Jul-2024, 20% after. We headline 20%
   (current law, conservative) and also show 15% and 0% (gross) for bracketing.

CAVEATS: raw prices (split noise biases AGAINST momentum => conservative);
month-end monthly data; lazy-rebalance approximation; tax accrued at realization
and paid each December; vol overlay's own turnover cost approximated.
"""
import numpy as np, pandas as pd
import bhavcopy_factors as bf

price, tov, sym = bf.build_panels()
rets = price.pct_change()
dates = price.index; isins = list(price.columns)
LO, HI = bf.EXTREME_LO, bf.EXTREME_HI

def safe_ret(i, name):
    r = rets.iloc[i].get(name, np.nan)
    return 0.0 if (pd.isna(r) or not (LO <= r <= HI)) else r

def compute_targets(top_universe=200, lookback=12, frac=5):
    T = {}
    for i in range(lookback + 1, len(dates) - 1):
        px = price.iloc[i]; px1 = price.iloc[i-1]; pxL = price.iloc[i-1-lookback]
        valid = px.notna() & px1.notna() & pxL.notna()
        cand = [c for c in isins if valid[c]]
        if len(cand) < 50: continue
        tn = tov.iloc[i][cand].fillna(0)
        univ = list(tn.sort_values(ascending=False).head(top_universe).index)
        look = rets.iloc[i-lookback:i][univ]
        clean = [c for c in univ if look[c].dropna().between(LO, HI).all()]
        if len(clean) < 30: continue
        q = max(1, len(clean)//frac)
        mom = (px1[clean]/pxL[clean] - 1)
        T[i] = list(mom.nlargest(q).index)
    return T

def simulate(T, cost_side=0.0025, stcg=0.20):
    """Position-level NAV sim with lazy rebalance, costs, annual STCG tax."""
    keys = sorted(T.keys())
    nav = 1.0; holdings = {}; basis = {}; realized = 0.0
    navs = []; ndx = []
    # establish first portfolio at decision keys[0]
    first = keys[0]; names = T[first]
    each = nav / len(names)
    for n in names:
        buy = each * (1 - cost_side)
        holdings[n] = buy; basis[n] = buy
    nav = sum(holdings.values())
    navs.append(nav); ndx.append(dates[first])
    for k in keys[1:]:
        # 1) earn month-k return on holdings (positions chosen at previous decision)
        for n in list(holdings):
            holdings[n] *= (1 + safe_ret(k, n))
        # 2) rebalance to T[k]
        newset = set(T[k]); cur = set(holdings)
        sells = cur - newset; buys = newset - cur
        cash = 0.0
        for n in sells:
            val = holdings.pop(n)
            gain = val - basis.pop(n)
            realized += gain
            cash += val * (1 - cost_side)   # sell-side cost
        if buys:
            each = cash / len(buys) if cash > 0 else 0.0
            for n in buys:
                buy = each * (1 - cost_side)  # buy-side cost
                holdings[n] = buy; basis[n] = buy
        nav = sum(holdings.values())
        # 3) pay STCG tax each December on realized short-term gains
        if dates[k].month == 12 and stcg > 0 and realized > 0:
            tax = stcg * realized
            scale = max(0.0, (nav - tax)) / nav if nav > 0 else 1.0
            for n in holdings: holdings[n] *= scale
            nav = sum(holdings.values()); realized = 0.0
        elif dates[k].month == 12:
            realized = 0.0
        navs.append(nav); ndx.append(dates[k])
    s = pd.Series(navs, index=ndx)
    return s.pct_change().dropna()

def voltarget(r, target=0.13, window=6):
    out = r.copy(); cash_m = (1+bf.CASH_ANNUAL)**(1/12)-1
    sc = pd.Series(1.0, index=r.index)
    for i in range(len(r)):
        if i >= window:
            rv = r.iloc[i-window:i].std()*np.sqrt(12)
            s = 1.0 if (rv==0 or np.isnan(rv)) else min(1.0, target/rv)
        else:
            s = 1.0
        out.iloc[i] = s*r.iloc[i] + (1-s)*cash_m
    return out

def stats(r, label):
    r = r.dropna(); n=len(r); yrs=n/12
    cum=(1+r).prod(); cagr=cum**(1/yrs)-1 if cum>0 else np.nan
    vol=r.std()*np.sqrt(12); rf=(1+bf.CASH_ANNUAL)**(1/12)-1
    sh=(r.mean()-rf)/r.std()*np.sqrt(12) if r.std()>0 else np.nan
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()
    print(f"  {label:<40} CAGR {cagr*100:6.1f}%  Vol {vol*100:4.1f}%  Sharpe {sh:5.2f}  MaxDD {dd*100:6.1f}%")
    return dict(cagr=cagr, sharpe=sh, maxdd=dd)

if __name__ == "__main__":
    T = compute_targets(top_universe=200, lookback=12, frac=5)
    gross = simulate(T, cost_side=0.0, stcg=0.0)
    netc  = simulate(T, cost_side=0.0025, stcg=0.0)
    net15 = simulate(T, cost_side=0.0025, stcg=0.15)
    net20 = simulate(T, cost_side=0.0025, stcg=0.20)
    vt20  = voltarget(net20, target=0.13)
    nifty = pd.read_csv("data/NIFTYBEES.NS.csv", parse_dates=["date"]).set_index("date")["adjclose"]
    nifty.index = nifty.index.to_period("M").to_timestamp()
    nf = nifty.pct_change().reindex(net20.index)

    print(f"Survivorship-free momentum, position-level sim | {gross.index[0].date()} -> {gross.index[-1].date()}\n")
    print("THE TAX LADDER (this is the make-or-break):")
    stats(gross, "1. Gross (no costs, no tax)")
    stats(netc,  "2. Net of costs (25bps/side), no tax")
    stats(net15, "3. Net of costs + 15% STCG (pre-2024 rate)")
    stats(net20, "4. Net of costs + 20% STCG (current rate)")
    stats(vt20,  "5. #4 + vol-targeting overlay (investable)")
    stats(nf,    "   Nifty 50 buy & hold (benchmark)")

    print("\nOUT-OF-SAMPLE SPLIT (net of costs + 20% STCG):")
    mid = net20.index[len(net20)//2]
    stats(net20.loc[:mid], "  First half")
    stats(net20.loc[net20.index > mid], "  Second half")
