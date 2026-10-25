import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bhavcopy_strategy as bs
import bhavcopy_regime_hmm as rh

IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "images")
T = bs.compute_targets(top_universe=200, lookback=12, frac=5)
mkt_price = (1 + rh.mkt).cumprod()
sma8 = mkt_price > mkt_price.rolling(8).mean()
strat = rh.simulate(T, signal=sma8)            # final strategy, net of costs+tax
mkt_ret = rh.mkt.reindex(strat.index)

eq_s = (1 + strat).cumprod(); eq_m = (1 + mkt_ret.fillna(0)).cumprod()
dd_s = eq_s/eq_s.cummax() - 1; dd_m = eq_m/eq_m.cummax() - 1

fig,(a1,a2)=plt.subplots(2,1,figsize=(10,7),gridspec_kw={"height_ratios":[2.2,1]},sharex=True)
a1.plot(eq_s.index, eq_s.values, color="#1F3A5F", lw=1.9, label="Momentum + 8m regime filter (net of costs + 20% tax)")
a1.plot(eq_m.index, eq_m.values, color="#9C2B2B", lw=1.3, alpha=0.8, label="Market (EW top-200) buy & hold")
a1.set_yscale("log"); a1.set_ylabel("Growth of 1 (log)"); a1.legend(loc="upper left", frameon=False, fontsize=9)
a1.set_title("Final strategy vs market, survivorship-free, through the 2008 crash (2004-2024)", color="#1F3A5F", fontsize=12, weight="bold")
a1.grid(True, alpha=0.25)
a2.fill_between(dd_m.index, dd_m.values*100, 0, color="#9C2B2B", alpha=0.25, label="Market")
a2.fill_between(dd_s.index, dd_s.values*100, 0, color="#1F3A5F", alpha=0.45, label="Strategy")
a2.set_ylabel("Drawdown %"); a2.legend(loc="lower left", frameon=False, fontsize=9); a2.grid(True, alpha=0.25)
plt.tight_layout()
p=f"{IMG_DIR}/final_momentum_strategy.png"; plt.savefig(p,dpi=130); print("saved",p)
