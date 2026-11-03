"""Generate fake raw inputs so the whole pipeline can run without licensed data.

All values are random. Files have the same names and columns as the real inputs and the
same quirks: late entrants, FX gaps, fan-out duplicates in the emissions extract, junk
rows in the method file. A known effect is planted, so the tests can check recovery:

    TRUE_BETA on sign x carbon z, in the announcement window of clean-risk events only.

Usage: python tests/make_synthetic_data.py [--firms 100] [--out data/synthetic/raw]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
CURRENCIES = ["EUR", "GBP", "CHF", "SEK", "USD"]
TRUE_BETA = 0.0030          # 30 bp per SD of carbon z
TRUE_MKT_BETA_MEAN = 1.0
SEED = 20260815


def trading_days(start="2010-01-04", end="2025-12-31"):
    return pd.bdate_range(start, end)


def clean_risk_announcement_sign(dates, events_path):
    """Date -> pv_ar_sign on clean-risk announcement windows, on a business-day grid."""
    ev = pd.read_excel(events_path, sheet_name="events", parse_dates=["event_date"])
    signs = pd.read_excel(events_path, sheet_name="channel_signs")
    ev = ev.merge(signs[["event_id", "analysis_group", "pv_ar_sign"]], on="event_id")
    out = pd.Series(0.0, index=dates)
    for _, event in ev[ev["analysis_group"] == "clean_risk"].iterrows():
        pos = dates.searchsorted(event["event_date"])
        out.iloc[pos - 1: pos + 2] = event["pv_ar_sign"]
    return out


def make_factors(dates, rng):
    scale = np.array([0.011, 0.005, 0.005, 0.003, 0.003, 0.007])
    frame = pd.DataFrame(rng.normal(0, 1, (len(dates), 6)) * scale, index=dates, columns=FACTORS)
    frame["RF"] = 0.00002
    return frame.round(6).rename_axis("date")


def make_fx(dates, rng):
    """USD per local unit. Mimics Yahoo: two shared gap days, last sample day missing."""
    gaps = pd.to_datetime(["2013-10-08", "2017-11-16", "2025-12-31"])
    frames = []
    for currency, level in zip(CURRENCIES[:-1], [1.2, 1.4, 1.0, 0.12]):
        rates = level * np.exp(np.cumsum(rng.normal(0, 0.005, len(dates))))
        frame = pd.DataFrame({"date": dates, "curcdd": currency, "fx_to_usd": rates})
        frames.append(frame[~frame["date"].isin(gaps)])
    return pd.concat(frames, ignore_index=True)


def make_prices(dates, factors, fx, carbon_z, event_sign, rng):
    """Prices whose USD excess return follows a known six-factor model plus the effect."""
    fx_wide = fx.pivot(index="date", columns="curcdd", values="fx_to_usd").reindex(dates).ffill().bfill()
    fx_ret = fx_wide.pct_change().fillna(0.0)
    rf = factors["RF"].to_numpy()
    loadings = np.column_stack([rng.normal(TRUE_MKT_BETA_MEAN, 0.25, len(carbon_z)),
                                rng.normal(0, 0.3, (len(carbon_z), 5))])
    frames = []
    for i, z in enumerate(carbon_z):
        # Most firms span the sample; some list late or exit early, so warm-up thins them.
        first = 0 if i % 7 else int(rng.integers(0, 2500))
        last = len(dates) if i % 11 else int(rng.integers(3000, len(dates)))
        idx = np.arange(first, last)
        currency = CURRENCIES[i % len(CURRENCIES)]
        fx_step = np.zeros(len(dates)) if currency == "USD" else fx_ret[currency].to_numpy()

        excess = factors[FACTORS].to_numpy() @ loadings[i] + rng.normal(0, 0.015, len(dates))
        excess += TRUE_BETA * event_sign.to_numpy() * z
        # Invert the pipeline's (1 + ret)(1 + fx) - 1 - RF, so `ret` carries the same effect.
        ret = (1 + excess + rf) / (1 + fx_step) - 1
        tri = 50 * np.exp(np.cumsum(ret[idx]))
        frames.append(pd.DataFrame({
            "gvkey": f"{i:06d}", "iid": "01W", "datadate": dates[idx],
            "isin": f"SY{i:010d}", "prccd": tri, "ajexdi": 1.0, "trfd": 1.0, "cshoc": 1e7,
            "curcdd": currency, "exchg": 104, "fic": "NLD", "tic": f"T{i}", "sedol": f"S{i}",
            "cusip": f"C{i}", "excntry": "NLD", "tpci": "0", "secstat": "A", "dlrsni": None,
            "dldtei": None, "conm": f"FIRM {i}", "conml": f"Firm {i}", "gsector": "20",
            "ggroup": "2010", "gind": "201010", "gsubind": "20101010"}))
    prices = pd.concat(frames, ignore_index=True)
    # Two split artefacts for the |ret| >= 1 filter.
    prices.loc[prices.index[[1500, 1501]], "prccd"] *= 5
    return prices, loadings


def make_emissions(firm_ids, carbon_z, rng):
    """Refinitiv-style extract: fan-out duplicates, gaps, zero and negative values."""
    rows = []
    for isin, z in zip(firm_ids, carbon_z):
        for year in range(2009, 2025):
            revenue = float(np.exp(rng.normal(21, 1)))
            intensity = np.exp(-11 + 2.5 * z + rng.normal(0, 0.1) - 0.05 * (year - 2009))
            copies = {0: 1, 1: 1, 2: 4}[int(rng.integers(0, 3))]   # n x n join signature
            rows += [{"ISIN": isin, "Date": f"{year}-12-31",
                      "CO2 Equivalent Emissions Direct, Scope 1": intensity * revenue,
                      "Revenue": revenue}] * copies
    emissions = pd.DataFrame(rows)
    emissions["Date"] = pd.to_datetime(emissions["Date"])
    emissions.loc[emissions.sample(frac=0.05, random_state=1).index, "Revenue"] = np.nan
    emissions.loc[emissions.sample(frac=0.005, random_state=2).index, "Revenue"] = 0.0
    emissions.loc[emissions.sample(frac=0.005, random_state=3).index,
                  "CO2 Equivalent Emissions Direct, Scope 1"] = 0.0
    return emissions


def make_method(emissions, rng):
    """Method flags with duplicates and the workbook's junk rows."""
    method = emissions[["ISIN", "Date"]].drop_duplicates().copy()
    method["Method"] = rng.choice(["Reported", "Median", "CO2", "Energy", None], len(method),
                                  p=[0.85, 0.07, 0.03, 0.02, 0.03])
    method["Date"] = method["Date"].astype(object)
    junk = pd.DataFrame({"ISIN": [None, "The invalid identifier ignored"],
                         "Date": ["Updated at 12:00", None], "Method": [None, None]})
    return pd.concat([junk, method], ignore_index=True)


def make_brent(dates, rng):
    prices = 80 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
    frame = pd.DataFrame({"date": dates, "brent": prices})
    gaps = frame["date"].isin(pd.to_datetime(["2015-03-04", "2022-06-20"]))  # source-calendar gaps
    return frame[(frame["date"] >= "2012-01-01") & ~gaps]


def build(out, n_firms, seed=SEED):
    import sys
    sys.path.insert(0, str(ROOT))
    from src.step1_returns import compute_returns

    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    dates = trading_days()
    carbon_z = rng.normal(0, 1, n_firms)
    event_sign = clean_risk_announcement_sign(dates, ROOT / "data/events/EU_MSR_Events_sign.xlsx")

    factors = make_factors(dates, rng)
    fx = make_fx(dates, rng)
    prices, loadings = make_prices(dates, factors, fx, carbon_z, event_sign, rng)
    returns = compute_returns(prices, 1.0, [0.001, 0.999])

    firm_ids = sorted(prices["isin"].unique())[: int(n_firms * 0.85)]  # 15% have no carbon data
    emissions = make_emissions(firm_ids, carbon_z, rng)

    returns.to_parquet(out / "stoxx600_daily_returns_2010_2025.parquet")
    factors.to_csv(out / "europe_ff5_mom_daily.csv")
    fx.to_csv(out / "fx_to_usd_daily_selected_currencies.csv", index=False)
    make_brent(dates, rng).to_csv(out / "brent_dcoilbrenteu.csv", index=False)
    emissions.to_excel(out / "emissions_revenue_with_isin_key_final.xlsx", index=False)
    with pd.ExcelWriter(out / "Emission_Method.xlsx") as writer:
        make_method(emissions, rng).to_excel(writer, sheet_name="Table2", index=False)
    pd.DataFrame(loadings, columns=FACTORS).assign(carbon_z=carbon_z).to_csv(out / "truth.csv")
    print(f"wrote synthetic inputs for {n_firms} firms -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firms", type=int, default=100)
    parser.add_argument("--out", type=Path, default=ROOT / "data/synthetic/raw")
    args = parser.parse_args()
    build(args.out, args.firms)
