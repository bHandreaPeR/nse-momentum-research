"""Anchored walk-forward / out-of-sample validation.

At each year-end we look ONLY at data available up to then, pick the parameter
set (TOP_N, vol target) with the best Sharpe so far, then trade the NEXT 12
months with it. We stitch those out-of-sample years into one track record.

This answers the real overfitting question: does choosing parameters with
hindsight flatter the result? If the stitched OOS record collapses vs the
hindsight-best fixed params, the strategy was overfit. If it holds up, the
edge (such as it is) is robust to not knowing the future.
"""
import itertools
import numpy as np, pandas as pd
import backtest as bt

prices = bt.load_prices()
COMBOS = [(tn, tv) for tn in (2, 3, 4) for tv in (0.10, 0.13, 0.16)]

# Precompute each parameter set's net monthly return series over all data.
series = {}
for tn, tv in COMBOS:
    bt.TOP_N = tn; bt.TARGET_VOL = tv; bt.COST_PER_SIDE = 0.0020
    out, _, _ = bt.run_backtest(prices)
    series[(tn, tv)] = out["strategy"]
    bench = out["benchmark"]; bench6040 = out["bench_6040"]; benchew = out["bench_ew"]
idx = series[(3, 0.13)].index
allret = pd.DataFrame({k: v.reindex(idx) for k, v in series.items()})
bench = bench.reindex(idx)

def sharpe(r):
    r = r.dropna()
    if len(r) < 6 or r.std() == 0:
        return -9
    rf_m = (1 + bt.CASH_ANNUAL) ** (1/12) - 1
    return (r.mean() - rf_m) / r.std() * np.sqrt(12)

# Walk forward: initial training through end-2016, OOS years 2017..2025
years = list(range(2017, 2026))
oos = pd.Series(dtype=float)
picks = []
for y in years:
    train_end = pd.Timestamp(f"{y-1}-12-31")
    fwd_start = pd.Timestamp(f"{y}-01-01"); fwd_end = pd.Timestamp(f"{y}-12-31")
    train = allret.loc[allret.index <= train_end]
    if len(train) < 36:
        continue
    best = max(COMBOS, key=lambda c: sharpe(train[c]))
    picks.append((y, best))
    fwd = series[best].loc[(series[best].index >= fwd_start) & (series[best].index <= fwd_end)]
    oos = pd.concat([oos, fwd])

oos = oos.dropna()
oos_window = (oos.index.min(), oos.index.max())

# Comparators over the SAME OOS window
fixed = series[(3, 0.13)].loc[(series[(3,0.13)].index >= oos_window[0]) & (series[(3,0.13)].index <= oos_window[1])].dropna()
oracle_combo = max(COMBOS, key=lambda c: sharpe(allret[c]))  # hindsight-best over whole sample
oracle = series[oracle_combo].loc[(series[oracle_combo].index >= oos_window[0]) & (series[oracle_combo].index <= oos_window[1])].dropna()
def win(s):
    return s.loc[(s.index >= oos_window[0]) & (s.index <= oos_window[1])].dropna()
bench_oos = win(bench); b6040_oos = win(bench6040); bew_oos = win(benchew)

def stats(r):
    s = bt.perf_stats(r)
    return f"CAGR {s['cagr']*100:5.1f}%  Vol {s['vol']*100:4.1f}%  Sharpe {s['sharpe']:.2f}  MaxDD {s['maxdd']*100:6.1f}%"

print(f"Out-of-sample window: {oos_window[0].date()} -> {oos_window[1].date()}  ({len(oos)} months)\n")
print("Parameter picks each year (chosen on data BEFORE that year):")
for y, (tn, tv) in picks:
    print(f"  {y}: top{tn}, vol{int(tv*100)}%")
print()
print("RESULTS over the identical out-of-sample window:")
print(f"  Walk-forward (no hindsight) : {stats(oos)}")
print(f"  Fixed top3/13% (our default): {stats(fixed)}")
print(f"  Hindsight-best {('top%d/%d%%'%(oracle_combo[0],int(oracle_combo[1]*100))):<10}   : {stats(oracle)}")
print(f"  Nifty 50 buy & hold        : {stats(bench_oos)}")
print(f"  60/40 Nifty/cash           : {stats(b6040_oos)}")
print(f"  Equal-weight universe      : {stats(bew_oos)}")
