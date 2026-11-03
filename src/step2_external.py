"""Step 2: public inputs, downloaded once and then frozen as snapshots in data/raw.

Sources: download_ff_factors.py (not previously in the repo),
notebooks/Returns_ff5_Currency_solved.ipynb, Regression.ipynb cell 3.

A file already in data/raw is never re-downloaded. Keeping the thesis-run snapshots is
what makes the run replicable: Ken French and Yahoo revise history, and Yahoo's gaps
decide which dates survive into the trading-day grid that positions the event windows.
Delete a snapshot only to deliberately refresh it.

Outputs: data/raw/europe_ff5_mom_daily.csv, fx_to_usd_daily_selected_currencies.csv,
         brent_dcoilbrenteu.csv
"""
import io
import zipfile

import pandas as pd

from src.common import load_config, raw_path

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FRED_BRENT_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"
BRENT_START = "2012-01-01"  # one year before the analysis window opens


def parse_french_csv(raw_text):
    """Parse a Ken French daily CSV: prose header, YYYYMMDD dates, values in percent."""
    lines = raw_text.splitlines()
    # The date column is unnamed, so a header row always starts with a comma. Prose
    # lines can contain words like "Momentum", so token matching alone is not enough.
    tokens = ("Mkt-RF", "Mkt-Rf", "WML", "Mom", "SMB", "HML")
    header = next(i for i, line in enumerate(lines)
                  if line.lstrip().startswith(",") and any(t in line for t in tokens))
    frame = pd.read_csv(io.StringIO("\n".join(lines[header:])), skipinitialspace=True)
    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = frame["date"].astype(str).str.strip()
    frame = frame[frame["date"].str.match(r"^\d{8}$")].copy()  # drops the footer
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    factor_cols = [c for c in frame.columns if c != "date"]
    frame[factor_cols] = frame[factor_cols].apply(pd.to_numeric, errors="coerce") / 100.0
    return frame.set_index("date").sort_index()


def _fetch_french(zip_name):
    import requests

    response = requests.get(f"{FRENCH_BASE}/{zip_name}", timeout=60,
                            headers={"User-Agent": "Mozilla/5.0 (thesis-data-pipeline)"})
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open(archive.namelist()[0]) as f:
            return f.read().decode("latin-1")


def combine_factors(ff5, mom, start, end):
    """Five factors plus momentum, in decimals, as one daily table."""
    mom = mom.rename(columns={mom.columns[0]: "Mom"})
    combined = ff5.join(mom[["Mom"]], how="inner").loc[start:end]
    assert combined.isna().sum().sum() == 0, "NaN in factor table"
    assert combined[["Mkt-RF", "SMB", "HML"]].abs().max().max() < 0.20, \
        "factor magnitude too large: percent-to-decimal conversion failed"
    return combined


def fetch_factors(cfg):
    path = raw_path(cfg, "factors")
    if path.exists():
        print(f"[2] snapshot present: {path.name}")
        return
    ff5 = parse_french_csv(_fetch_french("Europe_5_Factors_Daily_CSV.zip"))
    mom = parse_french_csv(_fetch_french("Europe_Mom_Factor_Daily_CSV.zip"))
    combine_factors(ff5, mom, cfg["sample"]["start"], cfg["sample"]["end"]).to_csv(path)
    print(f"[2] downloaded -> {path.name}")


def fx_from_yahoo(data, currency, invert):
    """Turn one yfinance download into date / curcdd / fx_to_usd rows."""
    data = data.reset_index()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    frame = data[["Date", price_col]].copy()
    frame.columns = ["date", "fx_rate"]
    frame["fx_to_usd"] = 1 / frame["fx_rate"] if invert else frame["fx_rate"]
    frame["curcdd"] = currency
    frame["date"] = pd.to_datetime(frame["date"])
    return frame[["date", "curcdd", "fx_to_usd"]]


def fetch_fx(cfg):
    path = raw_path(cfg, "fx")
    if path.exists():
        print(f"[2] snapshot present: {path.name}")
        return
    import yfinance as yf

    frames = []
    for currency, (ticker, invert) in cfg["fx"]["tickers"].items():
        # `end` is exclusive in yfinance, so the last sample day has no rate.
        data = yf.download(ticker, start=cfg["sample"]["start"], end=cfg["sample"]["end"],
                           progress=False, auto_adjust=False)
        if data.empty:
            raise RuntimeError(f"no FX data downloaded for {currency}")
        frames.append(fx_from_yahoo(data, currency, invert))
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    print(f"[2] downloaded -> {path.name}")


def clean_brent(brent, end):
    """FRED CSV to date / brent rows. Positional, because FRED has renamed the date column."""
    brent = brent.iloc[:, :2].copy()
    brent.columns = ["date", "brent"]
    brent["date"] = pd.to_datetime(brent["date"])
    brent = brent[brent["date"].between(BRENT_START, end)]
    return brent.dropna(subset=["brent"])


def fetch_brent(cfg):
    path = raw_path(cfg, "brent")
    if path.exists():
        print(f"[2] snapshot present: {path.name}")
        return
    clean_brent(pd.read_csv(FRED_BRENT_URL, na_values="."), cfg["sample"]["end"]).to_csv(path, index=False)
    print(f"[2] downloaded -> {path.name}")


def run(cfg):
    raw_path(cfg, "factors").parent.mkdir(parents=True, exist_ok=True)
    fetch_factors(cfg)
    fetch_fx(cfg)
    fetch_brent(cfg)


if __name__ == "__main__":
    run(load_config())
