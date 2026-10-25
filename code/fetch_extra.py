"""Probe + fetch candidate additional Indian ETFs (caches to data/)."""
import time
from data_fetch import fetch_one

CANDIDATES = [
    "M100.NS",        # Motilal Nifty Midcap 100
    "ITBEES.NS",      # Nifty IT
    "PHARMABEES.NS",  # Nifty Pharma
    "INFRABEES.NS",   # Nifty Infra
    "PSUBNKBEES.NS",  # Nifty PSU Bank
    "HNGSNGBEES.NS",  # Hang Seng (intl diversifier)
    "CPSEETF.NS",     # CPSE
    "LIQUIDBEES.NS",  # liquid (cash proxy, for reference)
    "SETFGOLD.NS",    # gold alt
    "QNIFTY.NS",      # quantum nifty
]
for s in CANDIDATES:
    try:
        print(fetch_one(s))
    except RuntimeError as e:
        print("ERROR", str(e)[:120])
    time.sleep(3.0)
