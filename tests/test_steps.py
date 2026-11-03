"""Unit tests for each step. Needs no licensed data or network access."""
import numpy as np
import pandas as pd
import pytest

from src.common import ROOT
from src import step1_returns as s1, step2_external as s2, step3_usd_panel as s3
from src import step4_abnormal_returns as s4, step5_carbon_panel as s5
from src import step6_regressions as s6, step7_tables as s7, step9_replication as s9

EVENTS = ROOT / "data/events/EU_MSR_Events_sign.xlsx"


# ---------------------------------------------------------------- step 1
def test_compute_returns_formula_and_artefact_filter():
    prices = pd.DataFrame({
        "gvkey": ["1"] * 4, "iid": ["01"] * 4,
        "datadate": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]),
        "prccd": [100.0, 110.0, 400.0, 404.0], "ajexdi": [1.0, 1.0, 1.0, 1.0],
        "trfd": [1.0, 1.0, 1.0, 1.0]})
    out = s1.compute_returns(prices, artefact_cutoff=1.0, winsor=[0.0, 1.0])
    # First row has no return; log(400/110) = 1.29 is dropped as an artefact.
    assert out["datadate"].tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06")]
    assert out["ret"].iloc[0] == pytest.approx(np.log(110 / 100))


def test_compute_returns_adjusts_for_splits_and_dividends():
    prices = pd.DataFrame({
        "gvkey": ["1", "1"], "iid": ["01", "01"],
        "datadate": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "prccd": [100.0, 50.0], "ajexdi": [2.0, 1.0], "trfd": [1.0, 1.01]})
    out = s1.compute_returns(prices, 1.0, [0.0, 1.0])
    assert out["ret"].iloc[0] == pytest.approx(np.log(1.01))  # 2:1 split nets out


def test_universe_is_the_union_of_cleaned_isins(capsys):
    constituents = pd.DataFrame({s1.COMPUSTAT_ISIN: [" aa ", "AA", "BB", None],
                                 s1.COMPUSTAT_THRU: [None, None, "2015-01-01", None]})
    assert s1.load_universe(constituents) == ["AA", "BB"]
    s1.validate_against_factset(constituents, pd.DataFrame({s1.FACTSET_ISIN: ["AA", "CC"]}))
    assert "overlap 1 (50.0%)" in capsys.readouterr().out


class FakeWrds:
    """Answers the three SQL pulls with minimal frames."""

    def raw_sql(self, sql, params=None):
        if "g_secd" in sql:
            return pd.DataFrame({
                "gvkey": ["1"], "iid": ["01"], "datadate": ["2010-01-04"], "isin": ["AA"],
                "prccd": [1.0], "ajexdi": [1.0], "trfd": [1.0], "cshoc": [1.0],
                "curcdd": ["EUR"], "exchg": [104], "fic": ["NLD"]})
        if "g_security" in sql:
            return pd.DataFrame({
                "gvkey": ["1"], "iid": ["01"], "tic": ["T"], "isin": ["AA"], "sedol": ["S"],
                "cusip": ["C"], "exchg": [104], "excntry": ["NLD"], "tpci": ["0"],
                "secstat": ["A"], "dlrsni": [None], "dldtei": [None]})
        return pd.DataFrame({"gvkey": ["1"], "conm": ["X"], "conml": ["X NV"], "fic": ["NLD"],
                             "gsector": ["20"], "ggroup": ["2010"], "gind": ["201010"],
                             "gsubind": ["20101010"]})


def test_pull_prices_matches_the_saved_parquet_schema():
    out = s1.pull_prices(FakeWrds(), ["AA"], "2010-01-01", "2025-12-31")
    assert out.columns.tolist() == [
        "gvkey", "iid", "datadate", "isin", "prccd", "ajexdi", "trfd", "cshoc", "curcdd", "exchg",
        "fic", "tic", "sedol", "cusip", "excntry", "tpci", "secstat", "dlrsni", "dldtei", "conm",
        "conml", "gsector", "ggroup", "gind", "gsubind"]


# ---------------------------------------------------------------- step 2
FRENCH_TEXT = """This file was created using the 202512 Bloomberg database.
Momentum factors are constructed from six portfolios, including Mom and WML sorts.

,WML
20100104,   0.50
20100105,  -1.00

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_french_parser_skips_prose_that_mentions_factor_names():
    out = s2.parse_french_csv(FRENCH_TEXT)
    assert out.columns.tolist() == ["WML"]
    assert out["WML"].tolist() == pytest.approx([0.005, -0.01])  # percent to decimal
    assert out.index[0] == pd.Timestamp("2010-01-04")


def test_combine_factors_rejects_percent_scale():
    index = pd.to_datetime(["2020-01-02"])
    ff5 = pd.DataFrame({"Mkt-RF": [1.5], "SMB": [0.1], "HML": [0.1], "RMW": [0.1], "CMA": [0.1],
                        "RF": [0.0]}, index=index)
    mom = pd.DataFrame({"WML": [0.1]}, index=index)
    with pytest.raises(AssertionError, match="percent-to-decimal"):
        s2.combine_factors(ff5, mom, "2020-01-01", "2020-12-31")


def test_fx_from_yahoo_handles_multiindex_and_inversion():
    index = pd.DatetimeIndex(["2020-01-02"], name="Date")
    columns = pd.MultiIndex.from_product([["Adj Close", "Close"], ["USDSEK=X"]])
    data = pd.DataFrame([[10.0, 10.0]], index=index, columns=columns)
    out = s2.fx_from_yahoo(data, "SEK", invert=True)
    assert out["fx_to_usd"].iloc[0] == pytest.approx(0.1)
    assert out["curcdd"].iloc[0] == "SEK"


def test_brent_cleaning_keeps_the_window_and_drops_gaps():
    raw = pd.DataFrame({"observation_date": ["2011-12-30", "2012-01-03", "2012-01-04", "2026-01-02"],
                        "DCOILBRENTEU": [100.0, np.nan, 110.0, 60.0]})
    out = s2.clean_brent(raw, "2025-12-31")
    assert out["date"].tolist() == [pd.Timestamp("2012-01-04")] and out["brent"].tolist() == [110.0]


# ---------------------------------------------------------------- step 3
def test_usd_conversion_formula_gaps_and_usd_rows():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    merged = pd.DataFrame({
        "isin": ["E"] * 3 + ["U"] * 3, "date": list(dates) * 2,
        "curcdd": ["EUR"] * 3 + ["USD"] * 3, "ret": 0.01, "ret_w": 0.01, "RF": 0.001})
    fx = pd.DataFrame({"date": dates[[0, 1]], "curcdd": "EUR", "fx_to_usd": [1.0, 1.1]})
    panel, n_missing = s3.convert_to_usd(merged, fx)

    assert n_missing == 1                              # EUR has no rate on the third day
    euro = panel[panel["isin"] == "E"].set_index("date")
    assert euro.loc[dates[0], "fx_return"] == 0.0       # first observation
    assert euro.loc[dates[1], "excess_return_usd"] == pytest.approx(1.01 * 1.10 - 1 - 0.001)
    usd = panel[panel["isin"] == "U"]
    assert len(usd) == 3 and (usd["fx_return"] == 0).all()


def test_check_panel_catches_duplicates():
    frame = pd.DataFrame({"isin": ["A", "A"], "date": pd.to_datetime(["2010-01-04"] * 2),
                          "excess_return_usd": [0.0, 0.0], "RF": [0.0, 0.0]})
    with pytest.raises(AssertionError, match="duplicate"):
        s3.check_panel(frame, [], 2010)


# ---------------------------------------------------------------- step 4
FACTORS = ["Mkt-RF", "SMB"]


def one_firm(n=400, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(0, 0.01, (n, 2)), columns=FACTORS)
    frame["isin"] = "A"
    frame["date"] = pd.bdate_range("2010-01-04", periods=n)
    frame["y"] = 0.0002 + 1.3 * frame["Mkt-RF"] - 0.4 * frame["SMB"] + rng.normal(0, 1e-6, n)
    return frame


def test_rolling_betas_recover_true_loadings():
    out = s4.add_rolling_ar(one_firm(), "y", FACTORS, window=100, min_nobs=100)
    assert out["beta_mkt"].dropna().mean() == pytest.approx(1.3, abs=1e-3)
    assert out["AR"].dropna().abs().max() < 1e-4


def test_rolling_betas_have_no_lookahead():
    base = one_firm()
    shocked = base.copy()
    shocked.loc[200, "y"] += 0.5
    a = s4.add_rolling_ar(base, "y", FACTORS, 100, 100)
    b = s4.add_rolling_ar(shocked, "y", FACTORS, 100, 100)
    # A shock on day 200 cannot change the prediction on or before day 200, only after.
    assert np.allclose(a["pred_excess"][:201], b["pred_excess"][:201], equal_nan=True)
    assert not np.isclose(a["pred_excess"][201], b["pred_excess"][201])
    assert b["AR"][200] == pytest.approx(a["AR"][200] + 0.5)


def test_warm_up_days_and_short_firms_have_no_ar():
    out = s4.add_rolling_ar(one_firm(150), "y", FACTORS, 100, 100)
    assert out["AR"].iloc[:100].isna().all() and out["AR"].iloc[100:].notna().all()
    assert s4.add_rolling_ar(one_firm(50), "y", FACTORS, 100, 100)["AR"].isna().all()


# ---------------------------------------------------------------- step 5
def write_carbon(path):
    pd.DataFrame({
        "ISIN": [" AA ", "AA", "BB", "CC", "DD", "EE", "FF"],
        "Date": ["2015-12-31"] * 7,
        "CO2 Equivalent Emissions Direct, Scope 1": [10.0, 10.0, 100.0, 1000.0, 0.0, np.nan, 5.0],
        "Revenue": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0]}).to_excel(path, index=False)


def test_load_carbon_ladder_collapse_and_zscore(tmp_path):
    path = tmp_path / "carbon.xlsx"
    write_carbon(path)
    carbon, ladder = s5.load_carbon(path)
    assert ladder["carbon_raw_rows"] == 7 and ladder["carbon_present_rows"] == 6
    assert ladder["carbon_posrev_rows"] == 5 and ladder["carbon_nonpositive_dropped"] == 1
    assert ladder["carbon_positive_rows"] == 4 and ladder["carbon_firm_years"] == 3
    assert carbon["isin"].tolist() == ["AA", "BB", "CC"]      # stripped and collapsed
    assert carbon["carbon_intensity_z"].mean() == pytest.approx(0)
    assert carbon["carbon_intensity_z"].std() == pytest.approx(1)


def test_merge_lags_one_year_and_carries_forward():
    carbon = pd.DataFrame({"isin": ["A", "A"], "year": [2014, 2016],
                           "carbon_intensity_z": [1.0, 2.0], "carbon_q4q1": [1.0, 0.0],
                           "reported": [True, False], "method": ["Reported", "Median"]})
    ar = pd.DataFrame({"isin": "A", "AR": 0.0, "date": pd.to_datetime(
        ["2014-12-31", "2015-01-01", "2016-12-31", "2017-01-02", "2025-06-30"])})
    panel = s5.merge_onto_ar(ar, carbon)
    # FY2014 is not usable during 2014; FY2016 applies from 2017 and never goes stale.
    assert panel["date"].min() == pd.Timestamp("2015-01-01")
    assert panel["carbon_intensity_z"].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert panel["reported"].tolist() == [True, True, False, False]


def test_method_flag_prefers_reported_and_drops_junk(tmp_path):
    path = tmp_path / "method.xlsx"
    sheet = pd.DataFrame({"ISIN": ["A", "A", None, "B"],
                          "Date": ["2015-12-31", "2015-06-30", "Updated at noon", "2015-12-31"],
                          "Method": ["Median", "Reported", None, "CO2"]})
    with pd.ExcelWriter(path) as writer:
        sheet.to_excel(writer, sheet_name="Table2", index=False)
    carbon = pd.DataFrame({"isin": ["A", "B", "C"], "year": [2015] * 3})
    out = s5.attach_method_flag(carbon, path)
    assert len(out) == 3 and out["reported"].tolist() == [True, False, False]


def test_sample_flags_count_firms_and_attached_years():
    carbon = pd.DataFrame({"isin": ["A", "A", "B"], "year": [2014, 2015, 2015],
                           "carbon_intensity_z": [0.5, -0.5, 1.0]})
    panel = pd.DataFrame({"isin": ["A"], "available_from": pd.to_datetime(["2016-01-01"])})
    flagged, _, _ = s5.flag_estimation_sample(carbon, panel)
    assert flagged["in_sample"].tolist() == [True, True, False]
    assert flagged["used"].tolist() == [False, True, False]


# ---------------------------------------------------------------- step 6, real workbook
def thesis_table3():
    return {"clean_risk": {2, 4, 8, 16, 17}, "clean_risk_intro": {4, 16},
            "clean_risk_resolv": {2, 8, 17}, "reinforcing": {1, 11, 14},
            "supply_placebo": {19, 23, 25},
            "pooled_signed": {1, 2, 3, 4, 6, 8, 11, 14, 16, 17, 18, 20, 21, 27}}


def test_event_sets_match_thesis_table_3():
    used = s6.load_events(EVENTS, "2013-01-01")
    assert s6.event_sets(used) == thesis_table3()
    assert len(used) == 17
    assert not set(used["event_id"]) & {5, 7, 10, 12, 13}  # month-precision TNACs excluded
    assert 22 not in set(used["event_id"])                 # neutral event gets no window


def test_windows_do_not_collide_within_a_window_type():
    used = s6.load_events(EVENTS, "2013-01-01")
    tdays = pd.Index(pd.bdate_range("2013-11-18", "2025-12-30"))
    cfg = {"announce": {"offsets": [-1, 1]}, "pre": {"offsets": [-5, -2]}}
    windows = s6.build_windows(used, tdays, cfg)
    assert len(windows["announce"]) == 51 and len(windows["pre"]) == 68


def test_event_16_announcement_sits_in_event_17_pre_window():
    used = s6.load_events(EVENTS, "2013-01-01")
    tdays = pd.Index(pd.bdate_range("2013-11-18", "2025-12-30"))
    cfg = {"announce": {"offsets": [-1, 1]}, "pre": {"offsets": [-5, -2]}}
    overlap = s6.report_cross_window_overlap(s6.build_windows(used, tdays, cfg))
    assert overlap == {pd.Timestamp("2022-06-22"), pd.Timestamp("2022-06-23")}


def test_build_window_snaps_forward_and_never_wraps_the_grid():
    tdays = pd.Index(pd.bdate_range("2022-01-03", periods=30))
    event = pd.DataFrame({"event_date": pd.to_datetime(["2022-01-08"]), "event_id": [1],
                          "analysis_group": ["x"], "event_type": ["y"], "pv_ar_sign": [1],
                          "supply_ar_sign": [0]})
    window = s6.build_window(event, tdays, -1, 1)          # Saturday snaps to Monday the 10th
    assert window["date"].dt.strftime("%m-%d").tolist() == ["01-07", "01-10", "01-11"]

    event["event_date"] = tdays[0]                         # event on the first grid date
    assert s6.build_window(event, tdays, -5, -2).empty     # previously wrapped to the whole grid


def test_pre_sample_events_are_dropped():
    used = s6.load_events(EVENTS, "2013-01-01")
    tdays = pd.Index(pd.bdate_range("2014-11-03", "2025-12-30"))
    windows = s6.build_windows(used, tdays, {"pre": {"offsets": [-5, -2]}})
    assert 1 not in set(windows["pre"]["event_id"]) and len(windows["pre"]) == 64


def test_energy_control_is_kept_on_a_source_calendar_gap():
    panel = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                          "carbon_intensity_z": [2.0, 2.0, 2.0]})
    brent = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-03"]),
                          "brent": [50.0, 55.0]})
    out = s6.add_energy_control(panel, brent)
    assert out["energy_ret"].tolist()[1:] == pytest.approx([0.0, 0.1])
    assert out["energy_x_carbon"].iloc[2] == pytest.approx(0.2)


def test_run_spec_excludes_other_events_but_never_its_own_dates():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=40)
    panel = pd.DataFrame({"isin": np.repeat([f"F{i}" for i in range(30)], 40),
                          "date": list(dates) * 30})
    panel["carbon_intensity_z"] = np.repeat(rng.normal(0, 1, 30), 40) + rng.normal(0, .1, 1200)
    panel["AR"] = rng.normal(0, 0.01, 1200)
    win = pd.DataFrame({"date": dates[[10, 11, 20]], "event_id": [1, 1, 2],
                        "pv_ar_sign": [1, 1, -1], "supply_ar_sign": [0, 0, 0]})
    _, n_events, n_obs = s6.run_spec(panel, win, {1}, "pv_ar_sign", s6.CLUSTER["firm"])
    assert n_events == 1 and n_obs == 1200 - 30            # only event 2's date removed


# ---------------------------------------------------------------- step 7
def test_stars_only_on_two_way_rows():
    assert s7.stars(0.001, "firm") == ""
    assert s7.stars(0.09, "2way") == "^{*}" and s7.stars(0.10, "2way") == ""
    assert s7.stars(0.04, "2way") == "^{**}" and s7.stars(0.009, "2way") == "^{***}"


def test_cell_format_matches_the_thesis_tables():
    row = {"beta_bp": -5.381, "se_bp": 3.162, "p": 0.089, "clust": "2way"}
    assert s7.cell(row) == r"$-5.38^{*}\;(3.16)$"
    assert s7.cell(dict(row, clust="firm"), dagger=True) == r"$-5.38^{\dagger}\;(3.16)$"


# ---------------------------------------------------------------- step 9
def test_replication_check_applies_rounding_tolerance_and_flags_gaps():
    assert s9.check("x", 0.80, 0.8049, 0.005)["pass"]
    assert not s9.check("x", 0.80, 0.8061, 0.005)["pass"]
    assert s9.check("n", 2657470, 2657470)["pass"] and not s9.check("n", 10, 11)["pass"]
    assert s9.check("d", "2013-11-18", "2013-11-18")["pass"]
    assert not s9.check("m", 1.0, None)["pass"]


def test_every_table_cell_has_a_target():
    import yaml

    with open(ROOT / "thesis_targets.yaml") as f:
        targets = yaml.safe_load(f)
    cells = {(t["variant"], t["spec"], t["window"]) for t in targets["results"]}
    assert len(cells) == len(targets["results"]) == 30
