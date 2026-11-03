# msr_carbon_repricing

Replication code for the MSc Finance thesis *Does Carbon Regulatory Risk Reprice European Equities? Evidence from EU ETS Market Stability Reserve Legislative Announcements* (University of Amsterdam, 2026; [PDF](docs/Duta_Andrada_MSc_FIN.pdf)).

**Design.** Signed panel event study of STOXX 600 firms, 2013–2025: six-factor abnormal returns from 48-month rolling betas lagged one day, interacted with within-year z-scored log(Scope 1 / Revenue) lagged one year, with firm and date fixed effects. Identification rests on 5 supply-neutral MSR events, with channel separation through a supply placebo and reinforcing events.

**Result.** Informative null: 0.80 bp (SE 4.37) per SD of carbon intensity in the primary specification, bounded within about ±10 bp. The bound applies to residual announcement-window repricing of the identified estimand, not to the structural premium.

## Run

```bash
conda create -n msr python=3.13 -y
conda activate msr
pip install -r requirements.txt
python run_pipeline.py          # needs the inputs listed in DATA.md in data/raw/
```

Without licensed data, the same pipeline runs end to end on generated data with a planted effect:

```bash
python tests/make_synthetic_data.py
python run_pipeline.py --config tests/config_synthetic.yaml
python -m pytest tests
```

Useful flags: `--from regressions`, `--to tables`, `--only abnormal_returns`. `--from returns` also rebuilds returns from WRDS (set `WRDS_USERNAME`).

## Pipeline

| # | Script (`src/`) | Replaces | Output |
|---|---|---|---|
| 1 | `step1_returns.py` | `Stoxx600.ipynb` | Daily total returns; needs WRDS, skipped by default |
| 2 | `step2_external.py` | `download_ff_factors.py`, Yahoo and FRED cells | Frozen factor, FX and Brent snapshots |
| 3 | `step3_usd_panel.py` | `Stoxx_600_returns_ff5_mom.ipynb`, `Returns_ff5_Currency_solved.ipynb` | USD excess returns plus factors, 2010–2025 |
| 4 | `step4_abnormal_returns.py` | `Rolling_Beta_48.ipynb` cell 0 | AR for the 48- and 60-month windows |
| 5 | `step5_carbon_panel.py` | `Regression.ipynb` cell 0 | Carbon z-score, quartile and reported flags, lagged merge |
| 6 | `step6_regressions.py` | `Regression.ipynb` cells 1 and 3 | `output/tables/regression_results.csv` |
| 7 | `step7_tables.py` | Table export cells | Tables 1, 4, 5, 8 as `.tex` fragments |
| 8 | `step8_figures.py` | Figure cell | Figures 1, 2 |
| 9 | `step9_replication.py` | New | `output/replication_report.md` |

Every parameter varied in the thesis sits in `config.yaml`: rolling window and `min_nobs`, event windows, clustering per specification, event sets, and each robustness variant. 

## Replication check

Step 9 compares each run with `thesis_targets.yaml`: sample counts at every stage must match exactly, and each table cell must match to its printed precision (Tables 1, 4, 5, 8 and the Appendix C.1 bounds). The run exits non-zero and lists the mismatches if any cell differs.

 Latest run on the licensed data: 102 of 102 checks match ([report](output/replication_report.md)).

## Preserved behaviour

These are carried over unchanged so the run reproduces the thesis. They are recorded here because each is a choice a reader may want to revisit.

- **Log returns compounded as simple.** Step 3 computes `(1 + ret)(1 + fx_return) - 1`, where `ret` is a log return. The error is about r²/2 per day: 12.5 bp on a 5% move.
- **The trading-day grid inherits FX gaps.** A firm-day with no Yahoo rate is dropped (9,121 rows), event windows are counted on the surviving dates, and `end` is exclusive in yfinance, so 2025-12-31 is lost. A fresh FX download can therefore shift windows, which is why step 2 reuses snapshots.
- **Universe is the union of historical constituents.** A firm stays in the panel outside its index-membership spell, giving 841 firms per day on average in the return panel, and 619 after the carbon merge, rather than 600.
- **Holiday padding.** Compustat rows on market holidays stay in the panel with zero return. Events dated 1 January can sit on such a date.
- **Carbon carry-forward.** `merge_asof` has no staleness tolerance.
- **Window overlap across specifications.** Event 16's announcement days (22–23 June 2022) fall inside event 17's pre-window, so they carry sign −1 in the announcement specification and +1 in the pre-event one. No regression sees two signs for one date. The pipeline logs the overlap, the `pre_no_overlap` variant, which is not in the thesis, re-estimates the pre-event rows without those dates, and `python -m src.diagnostic_resolving` breaks down the resolving pre-event coefficient.

## Not ported

- `Rolling_Beta_48.ipynb` cell 1: an early exploratory regression superseded by `Regression.ipynb`.
- The ±2 window: an empty cell in `Regression.ipynb` and no reported result. Add it as a window and variant in `config.yaml`.
- Emissions extraction: done by hand in LSEG Workspace, so the workbook is a raw input (see `DATA.md`).
