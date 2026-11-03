"""Step 3: merge the factors and convert returns to USD excess returns.

Source: notebooks/Stoxx_600_returns_ff5_mom.ipynb, Returns_ff5_Currency_solved.ipynb.
The Ken French European factors are in USD, so returns are converted to match.

Inputs:  data/raw returns parquet, factor and FX snapshots
Output:  data/processed/stoxx_600_returns_ff5_mom_usd_2010_2025.parquet
"""
import pandas as pd

from src.common import DATE, ID, load_config, processed_path, raw_path, record_stats


def merge_factors(stoxx, factors):
    stoxx = stoxx.rename(columns={"datadate": DATE})
    stoxx[DATE] = pd.to_datetime(stoxx[DATE])
    factors[DATE] = pd.to_datetime(factors[DATE])
    return stoxx.merge(factors, on=DATE, how="inner")


def fx_returns(fx):
    """Daily simple FX return per currency. First observation per currency is set to 0."""
    fx = fx.sort_values(["curcdd", DATE]).copy()
    fx["fx_return"] = fx.groupby("curcdd")["fx_to_usd"].transform(
        lambda s: s.ffill().pct_change(fill_method=None))
    fx["fx_return"] = fx["fx_return"].fillna(0.0)
    return fx


def convert_to_usd(merged, fx):
    """Attach FX, build USD returns and USD excess returns, drop firm-days with no rate."""
    fx = fx[fx["curcdd"] != "USD"]
    panel = merged.merge(fx_returns(fx)[[DATE, "curcdd", "fx_to_usd", "fx_return"]],
                         on=[DATE, "curcdd"], how="left")
    is_usd = panel["curcdd"] == "USD"
    panel.loc[is_usd, "fx_to_usd"] = 1.0
    panel.loc[is_usd, "fx_return"] = 0.0
    n_missing = int(panel["fx_return"].isna().sum())

    # As run in the thesis: `ret` is a LOG return compounded here as if simple. README,
    # "Preserved behaviour", gives the size of the approximation.
    panel["ret_usd"] = (1 + panel["ret"]) * (1 + panel["fx_return"]) - 1
    panel["ret_w_usd"] = (1 + panel["ret_w"]) * (1 + panel["fx_return"]) - 1
    panel["excess_return_usd"] = panel["ret_usd"] - panel["RF"]
    panel["excess_return_w_usd"] = panel["ret_w_usd"] - panel["RF"]
    return panel.dropna(subset=["excess_return_w_usd"]).copy(), n_missing


def check_panel(panel, factors, start_year):
    need = ["excess_return_usd", *factors, "RF"]
    assert panel[DATE].dt.year.min() == start_year, "panel must include the warm-up years"
    assert panel[need].isna().sum().sum() == 0, "NaN in estimation columns"
    assert not panel.duplicated(subset=[ID, DATE]).any(), "duplicate firm-day rows"


def run(cfg):
    stoxx = pd.read_parquet(raw_path(cfg, "returns"))
    stoxx.columns = stoxx.columns.astype(str).str.strip()
    factors = pd.read_csv(raw_path(cfg, "factors"))
    fx = pd.read_csv(raw_path(cfg, "fx"), parse_dates=[DATE])

    merged = merge_factors(stoxx, factors)
    print(f"[3] returns + factors: {len(merged):,} rows | "
          f"{merged[DATE].min().date()} to {merged[DATE].max().date()}")

    panel, n_missing = convert_to_usd(merged, fx)
    print(f"[3] dropped {n_missing:,} rows with no FX rate "
          f"({n_missing / len(merged):.3%}) | {len(panel):,} remain")

    check_panel(panel, cfg["abnormal_returns"]["factors"], int(cfg["sample"]["start"][:4]))
    panel.to_parquet(processed_path(cfg, "usd_panel"), index=False)
    record_stats(cfg, returns_rows=len(stoxx), merged_rows=len(merged), fx_missing_rows=n_missing,
                 usd_rows=len(panel), usd_firms=int(panel[ID].nunique()))
    print(f"[3] saved {len(panel):,} rows | {panel[ID].nunique()} ISINs")


if __name__ == "__main__":
    run(load_config())
