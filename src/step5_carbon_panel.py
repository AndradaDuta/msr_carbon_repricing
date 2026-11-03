"""Step 5: build carbon intensity and merge it onto AR with a one-year lag.

Source: notebooks/Regression.ipynb cell 0 (Section A), plus the quartile and
reported-emissions edits made for Tables 5 and 8.

carbon_intensity_z is the within-year z-score of log(Scope 1 / Revenue), computed on the
full carbon universe before matching to returns. FY t emissions apply from 1 January
t+1 and merge_asof carries them forward with no staleness tolerance.

Inputs:  AR parquets, data/raw emissions and method workbooks
Outputs: data/processed/stoxx600_panel_step5_{w48,w60}.parquet, carbon_firm_year.parquet
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.common import CARBON, DATE, ID, load_config, processed_path, raw_path, record_stats


def load_carbon(path):
    """Scope 1 / Revenue collapsed to one row per firm-year, with z-score and quartile."""
    carbon = pd.read_excel(path)
    carbon = carbon.rename(columns={"ISIN": ID,
                                    "CO2 Equivalent Emissions Direct, Scope 1": "scope1",
                                    "Date": "carbon_date"})
    carbon[ID] = carbon[ID].astype(str).str.strip()
    carbon["carbon_date"] = pd.to_datetime(carbon["carbon_date"])
    carbon["year"] = carbon["carbon_date"].dt.year
    ladder = {"carbon_raw_rows": len(carbon), "carbon_raw_firms": carbon[ID].nunique()}

    carbon = carbon.dropna(subset=["scope1", "Revenue"])
    ladder |= {"carbon_present_rows": len(carbon), "carbon_present_firms": carbon[ID].nunique()}
    carbon = carbon[carbon["Revenue"] > 0]
    ladder["carbon_posrev_rows"] = len(carbon)
    carbon["carbon_intensity"] = carbon["scope1"] / carbon["Revenue"]

    # A hard zero is a non-report, and log(0) is undefined.
    n_zero = int((carbon["carbon_intensity"] <= 0).sum())
    carbon = carbon[carbon["carbon_intensity"] > 0].copy()
    ladder |= {"carbon_nonpositive_dropped": n_zero, "carbon_positive_rows": len(carbon),
               "carbon_positive_firms": carbon[ID].nunique()}

    # One row per (isin, year) so the merge cannot fan out.
    carbon = carbon.groupby([ID, "year"], as_index=False)["carbon_intensity"].mean()
    carbon[CARBON] = carbon.groupby("year")["carbon_intensity"].transform(
        lambda s: (np.log(s) - np.log(s).mean()) / np.log(s).std())
    carbon = carbon.dropna(subset=[CARBON])
    ladder |= {"carbon_firm_years": len(carbon), "carbon_firms": carbon[ID].nunique()}

    # Table 5: within-year Q4 = 1, Q1 = 0. Q2/Q3 stay in the z-score specs, and NaN here
    # drops them from the quartile regression only.
    quartile = carbon.groupby("year")["carbon_intensity"].transform(
        lambda s: pd.qcut(s, 4, labels=False, duplicates="drop") + 1)
    carbon["carbon_q4q1"] = np.select([quartile == 4, quartile == 1], [1.0, 0.0], default=np.nan)

    print(f"[5] carbon ladder: {ladder}")
    print(f"[5] carbon_z mean {carbon[CARBON].mean():+.2f} std {carbon[CARBON].std():.2f}")
    return carbon, ladder


def attach_method_flag(carbon, path):
    """Table 8: flag company-reported firm-years. Median, CO2 and Energy are vendor models.

    Attached to `carbon`, not to the firm-day panel, so the flag takes the same lag and
    forward-fill as the emissions themselves through merge_asof.
    """
    meth = pd.read_excel(path, sheet_name="Table2")
    meth["Date"] = pd.to_datetime(meth["Date"], errors="coerce")   # footer rows become NaT
    meth = meth[meth["Date"].notna() & meth["ISIN"].notna()].copy()
    meth[ID] = meth["ISIN"].astype(str).str.strip()
    meth["year"] = meth["Date"].dt.year
    meth = meth.rename(columns={"Method": "method"})
    meth["reported"] = meth["method"].eq("Reported")
    # Prefer a Reported label where a duplicated firm-year has one.
    meth = (meth.sort_values("reported", ascending=False)
                .drop_duplicates(subset=[ID, "year"], keep="first")
                .loc[:, [ID, "year", "method", "reported"]])

    carbon = carbon.merge(meth, on=[ID, "year"], how="left")
    carbon["reported"] = carbon["reported"].eq(True)                    # no flag = not reported
    print(f"[5] method flag matched {carbon['method'].notna().sum():,}/{len(carbon):,} firm-years | "
          f"reported {carbon['reported'].mean():.1%}")
    return carbon


def merge_onto_ar(ar, carbon):
    """Backward merge_asof: each firm-day takes the latest firm-year already available."""
    carbon = carbon.copy()
    carbon["available_from"] = pd.to_datetime((carbon["year"] + 1).astype(int).astype(str) + "-01-01")
    carbon = carbon.sort_values("available_from")
    panel = pd.merge_asof(ar.sort_values(DATE),
                          carbon[[ID, "available_from", CARBON, "carbon_q4q1", "reported", "method"]],
                          left_on=DATE, right_on="available_from", by=ID, direction="backward")
    panel = panel.dropna(subset=[CARBON]).reset_index(drop=True)
    panel["reported"] = panel["reported"].eq(True)
    return panel


def flag_estimation_sample(carbon, panel):
    """Mark sample firms and firm-years attaching to at least one day; run the tilt check."""
    carbon = carbon.copy()
    carbon["in_sample"] = carbon[ID].isin(panel[ID].unique())
    available = pd.to_datetime((carbon["year"] + 1).astype(int).astype(str) + "-01-01")
    used = set(map(tuple, panel[[ID, "available_from"]].drop_duplicates().to_numpy()))
    carbon["used"] = [(i, a) in used for i, a in zip(carbon[ID], available.to_numpy())]

    tilt = stats.ttest_ind(carbon.loc[carbon["used"], CARBON],
                           carbon.loc[~carbon["used"], CARBON], equal_var=False)
    return carbon, float(tilt.statistic), float(tilt.pvalue)


def run(cfg):
    primary = cfg["abnormal_returns"]["primary"]
    carbon, ladder = load_carbon(raw_path(cfg, "emissions"))
    carbon = attach_method_flag(carbon, raw_path(cfg, "method"))

    for window in cfg["abnormal_returns"]["windows"]:
        ar = pd.read_parquet(processed_path(cfg, "ar", window=window), columns=[ID, DATE, "AR"])
        ar[DATE] = pd.to_datetime(ar[DATE])
        ar = ar.dropna(subset=["AR"]).copy()
        ar[ID] = ar[ID].astype(str).str.strip()

        overlap = set(carbon[ID]) & set(ar[ID])
        assert overlap, "no shared ISINs: inspect both isin columns for a format mismatch"
        panel = merge_onto_ar(ar, carbon)
        print(f"[5] {window}: ISIN overlap {len(overlap)} | matched {len(panel):,}/{len(ar):,} "
              f"firm-days ({len(panel) / len(ar):.1%}) | {panel[ID].nunique()} firms")
        panel.to_parquet(processed_path(cfg, "panel", window=window), index=False)
        record_stats(cfg, **{f"panel_rows_{window}": len(panel),
                             f"panel_firms_{window}": int(panel[ID].nunique())})

        if window == primary:
            flagged, tilt_t, tilt_p = flag_estimation_sample(carbon, panel)
            flagged.to_parquet(processed_path(cfg, "carbon"), index=False)
            record_stats(cfg, **ladder, carbon_ar_overlap=len(overlap),
                         sample_firm_years=int(flagged["in_sample"].sum()),
                         used_firm_years=int(flagged["used"].sum()),
                         tilt_t=tilt_t, tilt_p=tilt_p)
            print(f"[5] firm-years: {len(flagged):,} z-scored | {flagged['in_sample'].sum():,} "
                  f"sample firms | {flagged['used'].sum():,} used | tilt t={tilt_t:.2f} p={tilt_p:.2f}")


if __name__ == "__main__":
    run(load_config())
