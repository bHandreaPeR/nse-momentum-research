"""
Iteration 3: 'diversified core + trend overlay', plus selection/regime-bias stress.

Tests the hypothesis from iteration 2: most of the value was diversification,
not momentum selection. So compare:
  - EW core              (equal-weight, monthly rebalance)
  - IVW core             (inverse-volatility weight, monthly rebalance)
  - EW + trend + voltgt  (core, but gate each asset by its 12m trend; vol-target)
  - IVW + trend + voltgt
against Nifty B&H and 60/40. Then repeat EXCLUDING the lucky-regime winners
(Gold, Nasdaq) and across two sub-periods, to gauge selection/regime bias.

No look-ahead: weights at t use data <= t; returns earned t->t+1.
"""
import numpy as np, pandas as pd
import backtest as bt

CASH_M = (1 + bt.CASH_ANNUAL) ** (1/12) - 1
COST = 0.0020
START_I = 12

def load(cols=None):
    p = bt.load_prices()
    if cols:
        p = p[cols].dropna(how="any")
    return p

def run(prices, scheme="ew", gate=False, vol_target=None):
    rets = prices.pct_change()
    cols = list(prices.columns)
    dates = prices.index
    # weights per month (data <= t)
    W = {}
    for i in range(START_I, len(dates) - 1):
        if scheme == "ew":
            w = pd.Series(1.0/len(cols), index=cols)
        elif scheme == "ivw":
            vol = rets.iloc[i-11:i+1].std()
            inv = (1.0/vol.replace(0, np.nan))
            w = (inv/inv.sum()).fillna(0.0)
        else:
            raise ValueError(scheme)
        if gate:
            ok = (prices.iloc[i]/prices.iloc[i-12] - 1) > 0
            w = w.where(ok, 0.0)
        W[i] = w
    # gross (for vol estimate)
    gross = pd.Series(0.0, index=dates)
    for i in range(START_I, len(dates)-1):
        w = W[i]; gross.iloc[i+1] = (w*rets.iloc[i+1]).sum() + (1.0-w.sum())*CASH_M
    # net with vol targeting + costs
    net = pd.Series(0.0, index=dates); prev = pd.Series(0.0, index=cols); cost = pd.Series(0.0, index=dates)
    for i in range(START_I, len(dates)-1):
        s = 1.0
        if vol_target:
            hist = gross.iloc[:i+1].dropna().tail(6)
            rv = hist.std()*np.sqrt(12) if len(hist) >= 6 else np.nan
            s = 1.0 if (not rv or np.isnan(rv) or rv == 0) else min(1.0, vol_target/rv)
        w = W[i]*s
        net.iloc[i+1] = (w*rets.iloc[i+1]).sum() + (1.0-w.sum())*CASH_M
        cost.iloc[i+1] = (w-prev).abs().sum()*COST; prev = w
    return (net - cost).iloc[START_I+1:]

def bench(prices, kind):
    rets = prices.pct_change()
    if kind == "nifty":
        s = rets["NIFTYBEES.NS"].copy()
    elif kind == "6040":
        s = 0.6*rets["NIFTYBEES.NS"] + 0.4*CASH_M
    s.iloc[:START_I+1] = np.nan
    return s.iloc[START_I+1:].dropna()

def line(name, r):
    st = bt.perf_stats(r.dropna())
    return f"  {name:<26} CAGR {st['cagr']*100:5.1f}%  Vol {st['vol']*100:4.1f}%  Sharpe {st['sharpe']:5.2f}  MaxDD {st['maxdd']*100:6.1f}%"

def report(prices, title):
    print(f"\n=== {title}  ({prices.index[START_I+1].date()} -> {prices.index[-1].date()}, {len(prices.columns)} assets) ===")
    print(line("EW core", run(prices, "ew")))
    print(line("IVW core", run(prices, "ivw")))
    print(line("EW + trend + voltgt13", run(prices, "ew", gate=True, vol_target=0.13)))
    print(line("IVW + trend + voltgt13", run(prices, "ivw", gate=True, vol_target=0.13)))
    print(line("Nifty 50 B&H", bench(prices, "nifty")))
    print(line("60/40 Nifty/cash", bench(prices, "6040")))

if __name__ == "__main__":
    full = load()
    report(full, "FULL universe (7 assets)")

    # selection-bias: drop the lucky-regime winners (Gold, Nasdaq)
    no_winners = [c for c in bt.RISKY if c not in ("GOLDBEES.NS", "MON100.NS")]
    report(load(no_winners), "EXCLUDING Gold & Nasdaq (selection-bias test)")

    # sub-period robustness on full universe
    mid = full.index[len(full)//2]
    report(full.loc[:mid], "Sub-period A (first half)")
    report(full.loc[full.index >= mid], "Sub-period B (second half)")
