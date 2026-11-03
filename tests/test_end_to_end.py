"""Run every step on synthetic data and check that the planted truth comes back out."""
import numpy as np
import pandas as pd
import pytest

import run_pipeline
from src.common import load_config, processed_path
from tests.make_synthetic_data import TRUE_BETA, build

ANNOUNCE, PRE = "ANNOUNCE [-1,+1]", "PRE-EVENT [-5,-2]"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    base = tmp_path_factory.mktemp("synthetic")
    build(base / "raw", n_firms=60)
    cfg = load_config()
    cfg["paths"].update(raw=str(base / "raw"), processed=str(base / "processed"),
                        tables=str(base / "tables"), figures=str(base / "figures"),
                        report=str(base / "report.md"))
    cfg["replication"]["targets"] = None
    for name in run_pipeline.select_steps("external", "replication", None):
        __import__(f"src.{run_pipeline.STEPS[name]}", fromlist=["run"]).run(cfg)
    results = pd.read_csv(base / "tables" / "regression_results.csv")
    return cfg, base, results


def pick(results, variant, spec, window):
    return results[(results["variant"] == variant) & (results["spec"] == spec)
                   & (results["window"] == window)].iloc[0]


def test_planted_effect_is_recovered(run):
    row = pick(run[2], "main", "CLEAN_RISK", ANNOUNCE)
    assert abs(row["beta_bp"] - TRUE_BETA * 1e4) < 3 * row["se_bp"]
    assert row["beta_bp"] > 0 and row["events"] == 5


def test_no_effect_where_none_was_planted(run):
    for spec, window in [("SUPPLY_PLACEBO", ANNOUNCE), ("CLEAN_RISK", PRE)]:
        row = pick(run[2], "main", spec, window)
        assert abs(row["beta_bp"]) < 3.5 * row["se_bp"], spec


def test_rolling_betas_are_close_to_truth(run):
    cfg, base, _ = run
    truth = pd.read_csv(base / "raw" / "truth.csv", index_col=0)
    ar = pd.read_parquet(processed_path(cfg, "ar", window="w48"))
    last = ar.dropna(subset=["beta_mkt"]).groupby("isin")["beta_mkt"].last()
    firm = last.index.str[2:].astype(int)
    assert np.median(np.abs(last.to_numpy() - truth.loc[firm, "Mkt-RF"].to_numpy())) < 0.1


def test_ar_trimmed_after_estimation_and_centred(run):
    cfg = run[0]
    ar = pd.read_parquet(processed_path(cfg, "ar", window="w48")).dropna(subset=["AR"])
    assert ar["date"].min() >= pd.Timestamp("2013-01-01")
    assert abs(ar["AR"].mean()) < 5e-4


def test_60_month_window_drops_event_1_only(run):
    results = run[2]
    assert pick(results, "w60", "CLEAN_RISK", ANNOUNCE)["events"] == 5
    assert pick(results, "w60", "REINFORCING", ANNOUNCE)["events"] == 2
    assert pick(results, "w60", "POOLED_SIGNED", ANNOUNCE)["events"] == 13


def test_every_variant_ran(run):
    assert set(run[2]["variant"]) == {"main", "reported", "w60", "quartile", "energy",
                                      "resolv_twoway", "pre_narrow", "pre_no_overlap"}


def test_quartile_sample_is_the_top_and_bottom_only(run):
    results = run[2]
    assert (pick(results, "quartile", "CLEAN_RISK", ANNOUNCE)["N"]
            < 0.6 * pick(results, "main", "CLEAN_RISK", ANNOUNCE)["N"])


def test_reported_sample_is_a_strict_subset(run):
    results = run[2]
    assert (pick(results, "reported", "CLEAN_RISK", ANNOUNCE)["N"]
            < pick(results, "main", "CLEAN_RISK", ANNOUNCE)["N"])


def test_overlap_variant_removes_the_shared_dates(run):
    results = run[2]
    assert (pick(results, "pre_no_overlap", "CLEAN_RISK_RESOLV", PRE)["N"]
            < pick(results, "main", "CLEAN_RISK_RESOLV", PRE)["N"])


def test_tables_and_figures_are_written(run):
    base = run[1]
    for name in ["table1_summary.csv", "table1_summary.tex", "table4_main.tex",
                 "table5_robustness.tex", "table8_reported.tex"]:
        assert (base / "tables" / name).stat().st_size > 0
    for name in ["fig_carbon_raw_vs_log.pdf", "fig_carbon_within_year_z.pdf"]:
        assert (base / "figures" / name).stat().st_size > 0


def test_every_replication_target_maps_to_a_pipeline_output(run):
    import yaml

    from src.common import ROOT, read_stats

    cfg, _, results = run
    with open(ROOT / "thesis_targets.yaml") as f:
        targets = yaml.safe_load(f)
    missing = (set(targets["counts"]) | set(targets["statistics"])) - set(read_stats(cfg))
    assert not missing, f"targets with no recorded statistic: {missing}"
    available = set(zip(results["variant"], results["spec"]))
    assert {(t["variant"], t["spec"]) for t in targets["results"]} <= available


def test_resolving_diagnostic_reproduces_the_two_way_row(run):
    from src import diagnostic_resolving

    cfg, _, results = run
    rows = diagnostic_resolving.run(cfg).set_index("row")
    twoway = pick(results, "resolv_twoway", "CLEAN_RISK_RESOLV", PRE)
    assert len(rows) == 5
    assert rows.loc["all three events", "beta_bp"] == pytest.approx(twoway["beta_bp"])
    assert rows.loc["all three events", "se_bp"] == pytest.approx(twoway["se_bp"])
    assert rows.loc["without event 17", "events"] == 2
    assert rows.loc["without 22-23 June 2022", "N"] < rows.loc["all three events", "N"]
