"""
Cross-sectional single-name factor prototype (PRICE-ONLY): momentum + low-vol
on a large-cap NSE universe. Monthly rebalance, long top/bottom quintiles.

Honest benchmark = equal-weight of the SAME universe each month. The factor must
beat THAT (not just the Nifty) to show the signal adds value beyond the
(survivorship-biased) universe itself -- the lesson from the ETF iterations.

CAVEATS: current-constituent universe => SURVIVORSHIP BIAS (upward). Monthly
free data. Value/quality factors NOT included (need point-in-time fundamentals).
No look-ahead: momentum uses price[t-1]/price[t-13] (skips latest month);
positions earn month t+1 returns.
"""
import os, glob
import numpy as np, pandas as pd
import backtest as bt

HERE = os.path.dirname(os.path.abspath(__file__))
SDIR = os.path.join(HERE, "data", "stocks")
COST = 0.0020
CASH_M = (1 + bt.CASH_ANNUAL) ** (1/12) - 1

def load_panel():
    series = {}
    for p in glob.glob(os.path.join(SDIR, "*.csv")):
        name = os.path.basename(p).replace(".csv", "")
        s = pd.read_csv(p, parse_dates=["date"]).set_index("date")["adjclose"]
        series[name] = s
    panel = pd.DataFrame(series).sort_index().resample("ME").last()
    return panel

def load_nifty(index_like):
    s = pd.read_csv(os.path.join(HERE, "data", "NIFTYBEES.NS.csv"),
                    parse_dates=["date"]).set_index("date")["adjclose"].resample("ME").last()
    return s.pct_change().reindex(index_like)

def run():
    panel = load_panel()
    rets = panel.pct_change()
    dates = panel.index
    cols = list(panel.columns)

    ret_top = pd.Series(0.0, index=dates)      # momentum long top quintile
    ret_bot = pd.Series(0.0, index=dates)      # momentum bottom quintile
    ret_ew  = pd.Series(0.0, index=dates)      # equal-weight all eligible
    ret_lv  = pd.Series(0.0, index=dates)      # low-vol bottom-vol quintile
    cost_top = pd.Series(0.0, index=dates)
    cost_lv  = pd.Series(0.0, index=dates)
    nstocks = pd.Series(0, index=dates)
    prev_top = set(); prev_lv = set()

    for i in range(13, len(dates) - 1):
        px = panel.iloc[i]; px1 = panel.iloc[i-1]; px13 = panel.iloc[i-13]
        elig = px.notna() & px1.notna() & px13.notna()
        names = [c for c in cols if elig[c]]
        if len(names) < 10:
            continue
        nstocks.iloc[i] = len(names)
        q = max(1, len(names) // 5)

        # momentum 12-1
        mom = (px1[names] / px13[names] - 1)
        top = list(mom.nlargest(q).index); bot = list(mom.nsmallest(q).index)

        # low vol (trailing 12m)
        vols = rets.iloc[i-11:i+1][names].std()
        lv = list(vols.nsmallest(q).index)

        nxt = rets.iloc[i+1]
        ret_top.iloc[i+1] = nxt[top].mean()
        ret_bot.iloc[i+1] = nxt[bot].mean()
        ret_ew.iloc[i+1]  = nxt[names].mean()
        ret_lv.iloc[i+1]  = nxt[lv].mean()

        # turnover-based costs (one-sided fraction of names rotated)
        cost_top.iloc[i+1] = len(set(top) ^ prev_top) / max(1, len(top)) * COST
        cost_lv.iloc[i+1]  = len(set(lv) ^ prev_lv) / max(1, len(lv)) * COST
        prev_top = set(top); prev_lv = set(lv)

    start = 14
    mom_net = (ret_top - cost_top).iloc[start:]
    lv_net  = (ret_lv - cost_lv).iloc[start:]
    ls = (ret_top - ret_bot).iloc[start:]            # long-short spread (gross)
    ew = ret_ew.iloc[start:]
    nifty = load_nifty(panel.index).iloc[start:]

    def line(name, r):
        st = bt.perf_stats(r.dropna())
        return f"  {name:<34} CAGR {st['cagr']*100:6.1f}%  Vol {st['vol']*100:4.1f}%  Sharpe {st['sharpe']:5.2f}  MaxDD {st['maxdd']*100:6.1f}%"

    print(f"Universe: {len(cols)} large-cap NSE stocks | window {mom_net.dropna().index[0].date()} -> {mom_net.dropna().index[-1].date()}")
    print(f"Avg # eligible stocks: {nstocks[nstocks>0].mean():.0f}  | quintile size ~{int(nstocks[nstocks>0].mean()//5)}\n")
    print(line("Momentum: long top quintile (net)", mom_net))
    print(line("Low-vol: long low-vol quintile (net)", lv_net))
    print(line("EQUAL-WEIGHT all (honest bench)", ew))
    print(line("Nifty 50 (NIFTYBEES)", nifty))
    print()
    lsd = ls.dropna()
    ls_sharpe = lsd.mean() / lsd.std() * np.sqrt(12) if lsd.std() > 0 else float("nan")
    print(f"  {'Momentum long-SHORT spread (gross)':<34} ann.mean {lsd.mean()*1200:5.1f}%  Sharpe {ls_sharpe:5.2f}")
    print("  (long-short = academic factor evidence, not retail-implementable. A WEAK long-short")
    print("   Sharpe means the long-only top-quintile result is mostly beta + survivorship, not a")
    print("   clean momentum premium.)")
    print("\n  ** SURVIVORSHIP WARNING: universe = TODAY's large caps run backto the 1990s. The")
    print("     absolute CAGRs above are fantasy (we only hold companies that survived & grew).")
    print("     Only the RELATIVE ranking is weakly informative, and even that is biased.")

if __name__ == "__main__":
    run()
