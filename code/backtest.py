"""
Indian ETF tactical-allocation backtest engine (Phase 1 prototype).

Strategy: dual-momentum / GTAA with trend filter + volatility targeting.
  - Monthly rebalance, month-end data.
  - Each asset scored by blended 3/6/12-month total return.
  - Absolute-momentum (trend) filter: an asset is only eligible if its
    12-month return > 0. Failing assets are not held -> capital goes to cash.
  - Hold the top-N eligible assets, equal weighted. Empty slots -> cash.
  - Volatility targeting: scale gross exposure to a target annual vol,
    capped at 1.0 (we only de-risk, never lever). Unused weight -> cash.
  - Realistic Indian transaction costs charged on turnover.

NO LOOK-AHEAD: weights at month t use only data through t; the position is
held to earn month t+1's return. The vol scale at t uses gross returns
strictly before t+1.

IMPORTANT MODELLING NOTE: the 'safe asset' (cash) is modelled as a constant
annual rate (default 5.5%, ~ Indian short-term rate), NOT a real instrument
price series, because Yahoo's LIQUIDBEES series is unreliable. This is an
explicit assumption, flagged in the output. Swap in a real liquid/gilt ETF
or your paid data when available.
"""
import os, csv, datetime as dt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# ---------------- config ----------------
RISKY = ["NIFTYBEES.NS", "JUNIORBEES.NS", "BANKBEES.NS", "PSUBNKBEES.NS",
         "GOLDBEES.NS", "MON100.NS", "HNGSNGBEES.NS"]
LABELS = {"NIFTYBEES.NS": "Nifty 50", "JUNIORBEES.NS": "Nifty Next 50",
          "BANKBEES.NS": "Nifty Bank", "PSUBNKBEES.NS": "PSU Bank",
          "GOLDBEES.NS": "Gold", "MON100.NS": "Nasdaq 100", "HNGSNGBEES.NS": "Hang Seng"}
BENCHMARK = "NIFTYBEES.NS"
TOP_N = 3                 # hold up to 3 of the 5 risky assets
LOOKBACKS = [3, 6, 12]    # months, blended momentum score
ABS_FILTER_LB = 12        # absolute-momentum (trend) filter lookback, months
TARGET_VOL = 0.13         # annual vol target (Moderate risk profile)
VOL_EST_WINDOW = 6        # months of returns to estimate realized vol
MAX_LEVERAGE = 1.0        # never lever; only scale down
CASH_ANNUAL = 0.055       # modelled cash/safe-asset annual yield (ASSUMPTION)
COST_PER_SIDE = 0.0020    # 20 bps per side: brokerage + STT + exchange + slippage


def load_prices():
    series = {}
    for tkr in RISKY:
        path = os.path.join(DATA_DIR, tkr + ".csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing data file for {tkr}: {path}")
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")["adjclose"]
        series[tkr] = df
    prices = pd.DataFrame(series).sort_index()
    # normalize to calendar month-end (handles mixed monthly/weekly granularity),
    # then require all risky assets present (intersection of histories)
    prices = prices.resample("ME").last()
    prices = prices.dropna(how="any")
    if len(prices) < ABS_FILTER_LB + VOL_EST_WINDOW + 2:
        raise ValueError(f"Not enough overlapping history: only {len(prices)} months.")
    return prices


def run_backtest(prices):
    rets = prices.pct_change()                     # monthly simple returns
    cash_m = (1 + CASH_ANNUAL) ** (1 / 12) - 1     # monthly cash return
    dates = prices.index
    cols = list(prices.columns)

    # ---- 1) target risky weights at each month-end (signal uses data <= t) ----
    target_w = pd.DataFrame(0.0, index=dates, columns=cols)
    cash_w = pd.Series(0.0, index=dates)
    start_i = ABS_FILTER_LB  # need 12m of history before first signal
    for i in range(start_i, len(dates)):
        px_now = prices.iloc[i]
        score = {}
        eligible = {}
        for c in cols:
            mom = np.mean([px_now[c] / prices.iloc[i - lb][c] - 1 for lb in LOOKBACKS])
            absmom = px_now[c] / prices.iloc[i - ABS_FILTER_LB][c] - 1
            score[c] = mom
            eligible[c] = absmom > 0          # trend filter vs zero
        passers = [c for c in cols if eligible[c]]
        passers = sorted(passers, key=lambda c: score[c], reverse=True)[:TOP_N]
        if passers:
            w = 1.0 / TOP_N                   # equal weight; empty slots stay cash
            for c in passers:
                target_w.iloc[i, cols.index(c)] = w
        cash_w.iloc[i] = 1.0 - target_w.iloc[i].sum()

    # ---- 2) gross (pre-vol-scaling) strategy return, held t -> t+1 ----
    gross = pd.Series(0.0, index=dates)
    for i in range(start_i, len(dates) - 1):
        r_assets = (target_w.iloc[i] * rets.iloc[i + 1]).sum()
        r_cash = cash_w.iloc[i] * cash_m
        gross.iloc[i + 1] = r_assets + r_cash

    # ---- 3) volatility targeting (scale uses gross returns strictly < t+1) ----
    scale = pd.Series(1.0, index=dates)
    eff_w = pd.DataFrame(0.0, index=dates, columns=cols)
    eff_cash = pd.Series(1.0, index=dates)
    net = pd.Series(0.0, index=dates)
    for i in range(start_i, len(dates) - 1):
        hist = gross.iloc[:i + 1].dropna()
        rv = hist.tail(VOL_EST_WINDOW).std() * np.sqrt(12) if len(hist) >= VOL_EST_WINDOW else np.nan
        s = MAX_LEVERAGE if (not rv or np.isnan(rv) or rv == 0) else min(MAX_LEVERAGE, TARGET_VOL / rv)
        scale.iloc[i] = s
        eff_w.iloc[i] = target_w.iloc[i] * s
        eff_cash.iloc[i] = 1.0 - eff_w.iloc[i].sum()
        r_assets = (eff_w.iloc[i] * rets.iloc[i + 1]).sum()
        r_cash = eff_cash.iloc[i] * cash_m
        net.iloc[i + 1] = r_assets + r_cash

    # ---- 4) transaction costs on turnover of effective weights ----
    turnover = pd.Series(0.0, index=dates)
    cost = pd.Series(0.0, index=dates)
    prev = pd.Series(0.0, index=cols)
    for i in range(start_i, len(dates) - 1):
        cur = eff_w.iloc[i]
        to = (cur - prev).abs().sum()        # one-sided turnover of risky book
        turnover.iloc[i] = to
        cost.iloc[i + 1] = to * COST_PER_SIDE
        prev = cur
    net_after_cost = net - cost

    # ---- 5) benchmarks ----
    # (a) Nifty buy & hold; (b) 60/40 Nifty/cash; (c) equal-weight of universe.
    bench = rets[BENCHMARK].copy()
    bench_6040 = 0.6 * rets[BENCHMARK] + 0.4 * cash_m
    bench_ew = rets[cols].mean(axis=1)            # equal-weight, monthly rebalanced
    for s_ in (bench, bench_6040, bench_ew):
        s_.iloc[:start_i + 1] = np.nan

    out = pd.DataFrame({
        "strategy": net_after_cost,
        "strategy_gross_of_cost": net,
        "benchmark": bench,
        "bench_6040": bench_6040,
        "bench_ew": bench_ew,
        "scale": scale,
        "cash_w": eff_cash,
        "turnover": turnover,
    })
    # active window only
    out = out.iloc[start_i + 1:]
    return out, target_w.iloc[start_i:], eff_w


def perf_stats(r, rf_annual=CASH_ANNUAL):
    r = r.dropna()
    n = len(r)
    yrs = n / 12.0
    cum = (1 + r).prod()
    cagr = cum ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12)
    rf_m = (1 + rf_annual) ** (1 / 12) - 1
    sharpe = (r.mean() - rf_m) / r.std() * np.sqrt(12) if r.std() > 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(12)
    sortino = (r.mean() - rf_m) / (downside / np.sqrt(12)) / np.sqrt(12) * 12 if downside > 0 else np.nan
    sortino = (r.mean() - rf_m) * 12 / downside if downside > 0 else np.nan
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1)
    maxdd = dd.min()
    best = r.max(); worst = r.min()
    win = (r > 0).mean()
    return dict(months=n, years=round(yrs, 1), cagr=cagr, vol=vol, sharpe=sharpe,
                sortino=sortino, maxdd=maxdd, best_m=best, worst_m=worst, win_rate=win,
                total_return=cum - 1)


if __name__ == "__main__":
    prices = load_prices()
    out, tw, ew = run_backtest(prices)
    s = perf_stats(out["strategy"])
    sg = perf_stats(out["strategy_gross_of_cost"])
    b = perf_stats(out["benchmark"])

    print(f"Backtest window: {out.index[0].date()} -> {out.index[-1].date()}  ({s['years']} yrs, {s['months']} months)")
    print(f"Universe: {', '.join(LABELS[c] for c in RISKY)}  | hold top {TOP_N} | vol target {TARGET_VOL:.0%}")
    print(f"Cash assumption: {CASH_ANNUAL:.1%}/yr  | cost/side: {COST_PER_SIDE*1e4:.0f} bps")
    print()
    hdr = f"{'Metric':<22}{'Strategy(net)':>15}{'Strategy(gross)':>17}{'Nifty B&H':>13}"
    print(hdr); print("-" * len(hdr))
    def row(name, key, fmt):
        print(f"{name:<22}{fmt(s[key]):>15}{fmt(sg[key]):>17}{fmt(b[key]):>13}")
    pc = lambda x: f"{x*100:.1f}%"
    nm = lambda x: f"{x:.2f}"
    row("CAGR", "cagr", pc)
    row("Volatility (ann)", "vol", pc)
    row("Sharpe (rf=5.5%)", "sharpe", nm)
    row("Sortino", "sortino", nm)
    row("Max drawdown", "maxdd", pc)
    row("Best month", "best_m", pc)
    row("Worst month", "worst_m", pc)
    row("Win rate (months)", "win_rate", pc)
    row("Total return", "total_return", pc)
    print()
    # benchmark comparison
    b40 = perf_stats(out["bench_6040"]); bew = perf_stats(out["bench_ew"])
    print("Benchmarks (same window):")
    for nm2, st in [("Nifty 50 B&H", b), ("60/40 Nifty/cash", b40), ("Equal-weight universe", bew)]:
        print(f"  {nm2:<22} CAGR {st['cagr']*100:5.1f}%  Vol {st['vol']*100:4.1f}%  Sharpe {st['sharpe']:.2f}  MaxDD {st['maxdd']*100:6.1f}%")
    print()
    print(f"Avg cash weight: {out['cash_w'].mean():.0%}   Avg vol-scale: {out['scale'].replace(0,np.nan).mean():.2f}   "
          f"Annualized turnover: {out['turnover'].sum()/s['years']:.1f}x")

    out.to_csv(os.path.join(HERE, "backtest_results.csv"))
    print("\nSaved monthly results -> backtest_results.csv")
