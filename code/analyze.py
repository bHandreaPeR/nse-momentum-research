"""Verification + charts for the prototype: parameter-stability grid,
cost sensitivity, and an equity-curve / drawdown chart. Monkeypatches the
backtest module's config globals to test robustness without editing it."""
import os, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import backtest as bt

IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "images")
prices = bt.load_prices()

def run(top_n=3, target_vol=0.13, cost=0.0020, abs_lb=12, lookbacks=(3,6,12)):
    bt.TOP_N=top_n; bt.TARGET_VOL=target_vol; bt.COST_PER_SIDE=cost
    bt.ABS_FILTER_LB=abs_lb; bt.LOOKBACKS=list(lookbacks)
    out,_,_ = bt.run_backtest(prices)
    return bt.perf_stats(out["strategy"]), out

# ---- base case + chart ----
base, out = run()
bench = bt.perf_stats(out["benchmark"])

eq_s = (1+out["strategy"].dropna()).cumprod()
eq_b = (1+out["benchmark"].dropna()).cumprod()
eq_e = (1+out["bench_ew"].dropna()).cumprod()
dd_s = eq_s/eq_s.cummax()-1
dd_b = eq_b/eq_b.cummax()-1

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,7), gridspec_kw={"height_ratios":[2.2,1]}, sharex=True)
ax1.plot(eq_e.index, eq_e.values, color="#1E6B45", lw=1.8, label="Equal-weight universe")
ax1.plot(eq_s.index, eq_s.values, color="#1F3A5F", lw=1.8, label="Tactical strategy (net)")
ax1.plot(eq_b.index, eq_b.values, color="#9C2B2B", lw=1.4, alpha=0.8, label="Nifty 50 buy & hold")
ax1.set_yscale("log"); ax1.set_ylabel("Growth of 1 (log)"); ax1.legend(loc="upper left", frameon=False)
ax1.set_title("Indian ETF: Tactical vs Equal-Weight vs Nifty 50  (2012–2026, monthly)", color="#1F3A5F", fontsize=12, weight="bold")
ax1.grid(True, alpha=0.25)
ax2.fill_between(dd_b.index, dd_b.values*100, 0, color="#9C2B2B", alpha=0.25, label="Nifty 50")
ax2.fill_between(dd_s.index, dd_s.values*100, 0, color="#1F3A5F", alpha=0.45, label="Strategy")
ax2.set_ylabel("Drawdown %"); ax2.legend(loc="lower left", frameon=False); ax2.grid(True, alpha=0.25)
plt.tight_layout()
chart_path = os.path.join(IMG_DIR, "prototype_equity_curve.png")
plt.savefig(chart_path, dpi=130); print("Saved chart ->", chart_path)

# ---- parameter-stability grid (Sharpe / CAGR / MaxDD) ----
print("\n=== Parameter stability: Sharpe (CAGR%, MaxDD%) ===")
print(f"{'':8}", "".join(f"vol={v:>4.0%}   " for v in (0.10,0.13,0.16)))
for tn in (2,3,4):
    cells=[]
    for tv in (0.10,0.13,0.16):
        st,_ = run(top_n=tn, target_vol=tv)
        cells.append(f"{st['sharpe']:.2f}({st['cagr']*100:.0f},{st['maxdd']*100:.0f})")
    print(f"topN={tn:<3}", "  ".join(f"{c:>14}" for c in cells))

# ---- cost sensitivity ----
print("\n=== Cost sensitivity (topN=3, vol=13%) ===")
for c in (0.0, 0.0010, 0.0020, 0.0040):
    st,_ = run(cost=c)
    print(f"  {c*1e4:>4.0f} bps/side -> CAGR {st['cagr']*100:5.1f}%  Sharpe {st['sharpe']:.2f}  MaxDD {st['maxdd']*100:5.1f}%")

# ---- no-trend-filter sanity (set abs_lb filter effectively off via huge negative? )----
# Disable filter by making eligibility always pass: temporarily patch run_backtest? simplest: lookback=1 keeps filter; skip.
print("\n=== Benchmark for reference ===")
print(f"  Nifty 50 B&H -> CAGR {bench['cagr']*100:.1f}%  Sharpe {bench['sharpe']:.2f}  MaxDD {bench['maxdd']*100:.1f}%  Vol {bench['vol']*100:.1f}%")
