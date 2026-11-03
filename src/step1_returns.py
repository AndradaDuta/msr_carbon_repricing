"""Step 1: daily total returns for the historical STOXX 600 universe. Needs WRDS.

Source: notebooks/Stoxx600.ipynb. The notebook in the repo pulls from 2013; the file
every later step reads was built from the same code with a 2010 start, which is what
sample.start sets here. Runs only with `--from returns`; otherwise the pipeline starts
from the saved parquet in data/raw.

Inputs:  data/raw/compustat_stoxx600.xlsx, WRDS comp.g_secd / g_security / g_company
Output:  data/raw/stoxx600_daily_returns_2010_2025.parquet
"""
import os

import numpy as np
import pandas as pd

from src.common import load_config, raw_path, record_stats

COMPUSTAT_ISIN = "(isin) International Security Identification Number"
COMPUSTAT_THRU = "(thru) Effective Thru Date"
FACTSET_ISIN = "Unnamed: 4"

PRICES_SQL = """
    SELECT gvkey, iid, datadate, isin,
           prccd, ajexdi, trfd, cshoc, curcdd, exchg, fic
    FROM comp.g_secd
    WHERE isin IN %(isins)s
      AND datadate BETWEEN %(start)s AND %(end)s
"""
STATIC_SQL = """
    SELECT gvkey, iid, tic, isin, sedol, cusip,
           exchg, excntry, tpci, secstat, dlrsni, dldtei
    FROM comp.g_security
    WHERE isin IN %(isins)s
"""
NAMES_SQL = """
    SELECT gvkey, conm, conml, fic, gsector, ggroup, gind, gsubind
    FROM comp.g_company
    WHERE gvkey IN %(gvkeys)s
"""


def _clean_isins(series):
    return series.dropna().str.strip().str.upper()


def load_universe(constituents):
    """Union of every ISIN that was ever a constituent."""
    return _clean_isins(constituents[COMPUSTAT_ISIN]).unique().tolist()


def validate_against_factset(constituents, factset):
    """Print overlap between current Compustat constituents and the FactSet snapshot."""
    factset_isins = set(_clean_isins(factset[FACTSET_ISIN]))
    active = constituents[constituents[COMPUSTAT_THRU].isna()]
    compustat_isins = set(_clean_isins(active[COMPUSTAT_ISIN]))
    overlap = factset_isins & compustat_isins
    print(f"[1] FactSet {len(factset_isins)} | Compustat active {len(compustat_isins)} | "
          f"overlap {len(overlap)} ({len(overlap) / len(factset_isins):.1%})")


def pull_prices(db, universe, start, end):
    """Pull prices, security identifiers and company names, merged on gvkey + iid."""
    isin_tuple = tuple(universe)
    prices = db.raw_sql(PRICES_SQL, params={"isins": isin_tuple, "start": start, "end": end})
    print(f"[1] pulled {len(prices):,} rows for {prices['isin'].nunique()} securities")
    static = db.raw_sql(STATIC_SQL, params={"isins": isin_tuple})
    gvkeys = tuple(static["gvkey"].dropna().unique())
    names = db.raw_sql(NAMES_SQL, params={"gvkeys": gvkeys})

    firm_info = static.merge(names, on="gvkey", how="left")
    prices = prices.merge(firm_info, on=["gvkey", "iid"], how="left", suffixes=("", "_dup"))
    return prices[[c for c in prices.columns if not c.endswith("_dup")]]


def compute_returns(prices, artefact_cutoff, winsor):
    """Total return index, daily log return, artefact filter, winsorisation."""
    prices = prices.sort_values(["gvkey", "iid", "datadate"]).copy()
    prices["datadate"] = pd.to_datetime(prices["datadate"])

    # Total return index: adjusts for both splits and dividends.
    prices["tri"] = prices["prccd"] / prices["ajexdi"] * prices["trfd"]
    prices["ret"] = prices.groupby(["gvkey", "iid"])["tri"].transform(
        lambda x: np.log(x / x.shift(1)))
    prices = prices.dropna(subset=["ret"])  # first observation per security

    n_before = len(prices)
    panel = prices[prices["ret"].abs() < artefact_cutoff].copy()
    lower, upper = panel["ret"].quantile(winsor[0]), panel["ret"].quantile(winsor[1])
    panel["ret_w"] = panel["ret"].clip(lower, upper)
    print(f"[1] dropped {n_before - len(panel)} artefact rows | "
          f"winsor bounds [{lower:.4f}, {upper:.4f}] | {len(panel):,} rows")
    return panel


def run(cfg):
    try:
        import wrds
    except ImportError as err:
        raise SystemExit("Step 1 needs `pip install wrds` and a WRDS account.") from err
    username = os.environ.get("WRDS_USERNAME")
    if not username:
        raise SystemExit("Set the WRDS_USERNAME environment variable to run step 1.")

    constituents = pd.read_excel(raw_path(cfg, "constituents"))
    universe = load_universe(constituents)
    print(f"[1] universe: {len(universe)} unique ISINs")
    if raw_path(cfg, "factset").exists():
        validate_against_factset(constituents, pd.read_excel(raw_path(cfg, "factset")))

    db = wrds.Connection(wrds_username=username)
    prices = pull_prices(db, universe, cfg["sample"]["start"], cfg["sample"]["end"])
    panel = compute_returns(prices, cfg["returns"]["artefact_cutoff"], cfg["returns"]["winsor"])

    out = raw_path(cfg, "returns")
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    record_stats(cfg, returns_rows=len(panel))
    print(f"[1] saved {len(panel):,} rows, {panel['gvkey'].nunique()} gvkeys -> {out.name}")


if __name__ == "__main__":
    run(load_config())
