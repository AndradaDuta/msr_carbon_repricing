"""Step 4: rolling six-factor betas and abnormal returns, one file per beta window.

Source: notebooks/Rolling_Beta_48.ipynb cell 0, docs/AR_CONSTRUCTION_README.md.

Betas are estimated on a trailing window per firm and lagged one day, so day t's
expected return uses betas through t-1. Estimation runs on 2010-2025 and AR is trimmed
to the analysis window afterwards: 2010-2012 is warm-up only.

Input:   data/processed/stoxx_600_returns_ff5_mom_usd_2010_2025.parquet
Outputs: data/processed/stoxx600_AR_2013_2025_{w48,w60}.parquet
"""
import numpy as np
import pandas as pd
from statsmodels.regression.rolling import RollingOLS

from src.common import DATE, ID, load_config, processed_path, record_stats

KEEP = [ID, DATE, "gvkey", "iid", "pred_excess", "beta_mkt", "AR"]


def add_rolling_ar(df, y_col, factors, window, min_nobs):
    """Per firm: trailing RollingOLS, betas lagged one day, AR = realised - predicted.

    window and min_nobs are in trading-day observations. Firms with fewer than min_nobs
    rows, and every warm-up day, get AR = NaN.
    """
    cols = ["const", *factors]
    out = []
    for _, g in df.groupby(ID, sort=False):
        g = g.sort_values(DATE).copy()
        if len(g) >= min_nobs:
            X = g[factors].assign(const=1.0)[cols]
            fit = RollingOLS(g[y_col], X, window=window, min_nobs=min_nobs).fit(params_only=True)
            betas = fit.params.shift(1)                # lag: day-t pred uses betas through t-1
            g["pred_excess"] = (betas.values * X.values).sum(axis=1)
            g["beta_mkt"] = betas["Mkt-RF"].values
        else:
            g["pred_excess"] = np.nan
            g["beta_mkt"] = np.nan
        g["AR"] = g[y_col] - g["pred_excess"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def run(cfg):
    params = cfg["abnormal_returns"]
    y_col, factors = params["y_col"], params["factors"]
    analysis_start = cfg["sample"]["analysis_start"]

    panel = pd.read_parquet(processed_path(cfg, "usd_panel"),
                            columns=[ID, DATE, "gvkey", "iid", y_col, *factors])
    panel[DATE] = pd.to_datetime(panel[DATE])
    panel = panel.dropna(subset=[y_col, *factors]).sort_values([ID, DATE]).reset_index(drop=True)
    print(f"[4] loaded {len(panel):,} rows | {panel[ID].nunique()} firms")

    for name, spec in params["windows"].items():
        full = add_rolling_ar(panel, y_col, factors, spec["window"], spec["min_nobs"])
        ar = full.loc[full[DATE] >= analysis_start, KEEP].copy()  # trim AFTER computing AR

        valid = ar.dropna(subset=["AR"])
        assert (ar[DATE] < analysis_start).sum() == 0, "AR rows before the analysis window"
        print(f"[4] {name}: {len(valid):,} AR rows of {len(ar):,} | {valid[ID].nunique()} firms | "
              f"starts {valid[DATE].min().date()} | mean {valid['AR'].mean():+.2e} | "
              f"std {valid['AR'].std():.4f} | median beta_mkt {valid['beta_mkt'].median():.2f}")

        ar.to_parquet(processed_path(cfg, "ar", window=name), index=False)
        record_stats(cfg, **{f"ar_rows_{name}": len(valid),
                             f"ar_firms_{name}": int(valid[ID].nunique()),
                             f"ar_start_{name}": str(valid[DATE].min().date())})


if __name__ == "__main__":
    run(load_config())
