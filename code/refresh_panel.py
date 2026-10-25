"""Append the latest bhavcopy month-end files (2025-2026) to the cached panel."""
import pickle, glob, os, datetime as dt
import pandas as pd
import bhavcopy_factors as bf

HERE=os.path.dirname(os.path.abspath(__file__))
OLD=os.path.join(HERE,"bhavcopy_panels_sym.pkl")
NEW=os.path.join(HERE,"bhavcopy_panels_refreshed.pkl")
if not os.path.exists(OLD):  # fresh clone: extend the panel built by bf.build_panels()
    OLD = NEW

price_o, tov_o, sym_o = pickle.load(open(OLD,"rb"))
# build_panels() keys months by month-start; align everything to month-end before merging
price_o.index = price_o.index.to_period("M").to_timestamp("M")
tov_o.index = tov_o.index.to_period("M").to_timestamp("M")
# read month-end files from the bhavcopy folder and code/nse_recent/ (written by fetch_nse_recent.py)
SRC = (glob.glob(os.path.join(bf.NSE_DIR, "*.csv")) + glob.glob(os.path.join(HERE, "nse_recent", "*.csv")))
prows={}; trows={}; sym=dict(sym_o)
for f in sorted(set(SRC)):
    d=dt.datetime.strptime(os.path.basename(f)[:-4],"%d%b%Y")
    rec=bf.parse_file(f); ts=pd.Timestamp(d.year,d.month,1)
    prows[ts]={k:v[0] for k,v in rec.items()}
    trows[ts]={k:v[1] for k,v in rec.items()}
    for k,v in rec.items(): sym[k]=v[2]
pn=pd.DataFrame(prows).T.sort_index().resample("ME").last()
tn=pd.DataFrame(trows).T.sort_index().resample("ME").last()
price=pd.concat([price_o, pn]).sort_index(); price=price[~price.index.duplicated(keep="last")]
tov=pd.concat([tov_o, tn]).sort_index();   tov=tov[~tov.index.duplicated(keep="last")]
pickle.dump((price,tov,sym), open(NEW,"wb"))
print("refreshed panel:", price.shape, price.index.min().date(),"->",price.index.max().date())
print("last 4 months stock counts:")
for ix in price.index[-4:]:
    print("  ", ix.date(), int(price.loc[ix].notna().sum()))
