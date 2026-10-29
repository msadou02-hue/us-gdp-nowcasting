"""Download FRED series and build a stationary quarterly panel."""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"

TARGET = "GDP_Growth"

# label: (FRED code, transformation applied after averaging to quarterly)
#   "level" -> keep as is (already a rate, index or diffusion index)
#   "diff"  -> quarter-on-quarter change
#   "dlog"  -> annualised quarter-on-quarter log growth (%)
SERIES: dict[str, tuple[str, str]] = {
    TARGET: ("A191RL1Q225SBEA", "level"),  # real GDP growth, % SAAR
    "Industrial_Production": ("INDPRO", "dlog"),
    "Retail_Sales": ("RSAFS", "dlog"),
    "CPI": ("CPIAUCSL", "dlog"),
    "Oil_Price": ("DCOILWTICO", "dlog"),
    "SP500": ("SP500", "dlog"),
    "Unemployment_Rate": ("UNRATE", "diff"),
    "PPI": ("PPIACO", "dlog"),
    "Housing_Starts": ("HOUST", "dlog"),
    "Consumer_Sentiment": ("UMCSENT", "diff"),
    "M2_Money": ("M2SL", "dlog"),
    "Ten_Year_Rate": ("GS10", "diff"),
    "Nonfarm_Payrolls": ("PAYEMS", "dlog"),
    "Durable_Goods_Orders": ("DGORDER", "dlog"),
    "Building_Permits": ("PERMIT", "dlog"),
    "Capacity_Utilization": ("TCU", "diff"),
    "Personal_Income": ("DSPIC96", "dlog"),
    "Business_Inventories": ("BUSINV", "dlog"),
    "Job_Openings": ("JTSJOL", "dlog"),
    "Empire_State_Index": ("GACDISA066MSFRBNY", "level"),
    "Philly_Fed_Index": ("GACDFSA066MSFRBPHI", "level"),
    "Exports": ("BOPTEXP", "dlog"),
    "Import_Price_Index": ("IR", "dlog"),
}


def fetch_fred(code: str, retries: int = 3) -> pd.Series:
    """Fetch one FRED series as a float Series indexed by date (no API key needed)."""
    for attempt in range(retries):
        try:
            resp = requests.get(FRED_CSV.format(code=code), timeout=60)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    df = pd.read_csv(io.StringIO(resp.text))
    date_col = df.columns[0]
    s = pd.to_numeric(df[code], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    return s.dropna().rename(code)


def transform(s: pd.Series, how: str) -> pd.Series:
    if how == "level":
        return s
    if how == "diff":
        return s.diff()
    if how == "dlog":
        return 400 * np.log(s).diff()
    raise ValueError(f"unknown transformation {how!r}")


def build_quarterly_panel(cache: Path | None = Path("data/fred_raw.csv"),
                          start: str = "1990-01-01", refresh: bool = False) -> pd.DataFrame:
    """Return a quarterly DataFrame of stationary indicators plus the GDP target.

    Monthly/daily series are averaged within the quarter (what a forecaster has
    once the quarter is over but before the GDP advance release).
    """
    if cache is not None and cache.exists() and not refresh:
        raw = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        quarterly = {}
        for label, (code, _) in SERIES.items():
            s = fetch_fred(code)
            quarterly[label] = s.resample("QE-DEC").mean()
            print(f"  loaded {label:<22} ({code}) {s.index.min():%Y-%m} to {s.index.max():%Y-%m}")
        raw = pd.DataFrame(quarterly)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            raw.to_csv(cache)

    panel = pd.DataFrame({label: transform(raw[label], how)
                          for label, (_, how) in SERIES.items()})
    panel = panel.loc[start:]
    return panel[panel[TARGET].notna()]
