# Data

Nothing in `data/raw/` or `data/processed/` is committed. Compustat and Refinitiv data are licensed and cannot be redistributed. The only data file in the repo is `data/events/EU_MSR_Events_sign.xlsx`, the hand-classified 27-event workbook.

`python run_pipeline.py` expects these files in `data/raw/`:

| File | Source | How to obtain | Licensed |
|---|---|---|---|
| `stoxx600_daily_returns_2010_2025.parquet` | Compustat Global Security Daily via WRDS | Step 1, or the saved file | Yes |
| `emissions_revenue_with_isin_key_final.xlsx` | Refinitiv/LSEG | Manual export, see below | Yes |
| `Emission_Method.xlsx` | Refinitiv/LSEG | Manual export, sheet `Table2` | Yes |
| `europe_ff5_mom_daily.csv` | Ken French Data Library | Step 2 downloads it if missing | No |
| `fx_to_usd_daily_selected_currencies.csv` | Yahoo Finance | Step 2 downloads it if missing | No |
| `brent_dcoilbrenteu.csv` | FRED `DCOILBRENTEU` | Step 2 downloads it if missing | No |

Step 1 also reads `compustat_stoxx600.xlsx` (Compustat constituents, GVKEYX 150376) and, for validation only, `factset_stoxx600.xlsx`.

**Exact replication needs the original snapshots.** Ken French and Yahoo revise history, and FX gaps set the trading-day grid that positions event windows. Use the factor and FX files saved from the thesis run. Step 2 downloads a file only when it is missing, and a fresh download is a new vintage, not a replication. The thesis run fetched Brent live and kept no copy, so the first run downloads it from FRED and the snapshot is frozen from then on.

## Required columns

**Returns parquet:** `gvkey, iid, datadate, isin, curcdd, tri, ret, ret_w`, as written by step 1 from `prccd, ajexdi, trfd`. `ret` is the daily log total return within `gvkey + iid`.

**Emissions workbook:** `ISIN`, `Date`, `CO2 Equivalent Emissions Direct, Scope 1` (`TR.CO2DirectScope1`), `Revenue` (`TR.Revenue`). Pulled by RIC in LSEG Workspace, then joined to ISIN through a RIC–ISIN crosswalk. One row per firm-year is not assumed: the extract fans out and step 5 collapses it (13,976 rows to 8,442 firm-years).

**Method workbook, sheet `Table2`:** `ISIN`, `Date`, `Method` (`TR.CO2EstimationMethod`). `Reported` is company-disclosed; `Median`, `CO2` and `Energy` are vendor estimation models. Footer rows are dropped automatically.

**Factors CSV:** `date, Mkt-RF, SMB, HML, RMW, CMA, RF, Mom`, in decimals, USD-denominated.

**FX CSV:** `date, curcdd, fx_to_usd`, as USD per one unit of local currency.

**Brent CSV:** `date, brent`.
