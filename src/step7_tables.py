"""Step 7: thesis tables, written as CSV and bare booktabs tabular fragments.

Table numbers follow the submitted thesis. Numeric content is read from the pipeline's
outputs, never transcribed.

Inputs:  regression_results.csv, the primary panel, carbon_firm_year.parquet
Outputs: output/tables/table{1,4,5,8}_*.tex, table1_summary.csv
"""
import numpy as np
import pandas as pd

from src.common import CARBON, DATE, ID, load_config, output_dir, processed_path, record_stats

ANNOUNCE, PRE = "ANNOUNCE [-1,+1]", "PRE-EVENT [-5,-2]"
BODY_START = (r"\begin{tabular}{l c c c c}" "\n" r"\toprule" "\n"
              r" & & & \multicolumn{2}{c}{Event window} \\" "\n" r"\cmidrule(lr){4-5}" "\n")
COLUMNS = (r"%s & Cluster & Events & Announcement & Pre-event \\" "\n"
           r" & & & $[-1,+1]$ & $[-5,-2]$ \\" "\n" r"\midrule" "\n")
BODY_END = r"\bottomrule" "\n" r"\end{tabular}" "\n"
PANEL_ROW = r"\multicolumn{5}{l}{\textit{%s}} \\" "\n"


def stars(p, clust):
    """Stars mark two-way clustered rows only; firm-clustered rows are not significance tests."""
    if clust != "2way":
        return ""
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.10 else ""


def cell(row, dagger=False):
    mark = r"^{\dagger}" if dagger else stars(row["p"], row["clust"])
    return f"${row['beta_bp']:.2f}{mark}\\;({row['se_bp']:.2f})$"


def lookup(results, variant, spec, window):
    match = results[(results["variant"] == variant) & (results["spec"] == spec)
                    & (results["window"] == window)]
    assert len(match) == 1, f"expected one row for {variant}/{spec}/{window}, found {len(match)}"
    return match.iloc[0]


def ladder_row(results, label, variant, spec, dagger_pre=False):
    ann, pre = lookup(results, variant, spec, ANNOUNCE), lookup(results, variant, spec, PRE)
    cluster = "2-way" if ann["clust"] == "2way" else "firm"
    return (f"{label} & {cluster} & {ann['events']} & {cell(ann)} & "
            f"{cell(pre, dagger_pre)} \\\\\n")


def table4(results):
    body = PANEL_ROW % "Panel A. Regulatory-risk channel: supply-neutral events"
    body += ladder_row(results, "Clean-risk (all)", "main", "CLEAN_RISK")
    body += ladder_row(results, r"\quad Introducing", "main", "CLEAN_RISK_INTRO")
    body += ladder_row(results, r"\quad Resolving", "main", "CLEAN_RISK_RESOLV", dagger_pre=True)
    body += r"\midrule" "\n" + PANEL_ROW % "Panel B. Channel separation: supply-active events"
    body += ladder_row(results, "Supply placebo", "main", "SUPPLY_PLACEBO")
    body += ladder_row(results, "Reinforcing", "main", "REINFORCING")
    return BODY_START + COLUMNS % "Specification" + body + BODY_END


def table5(results):
    body = ladder_row(results, r"\textit{Memo: clean-risk baseline}$^{a}$", "main", "CLEAN_RISK")
    body += r"\addlinespace" "\n" + ladder_row(results, "Pool all signed events", "main", "POOLED_SIGNED")
    body += r"\addlinespace" "\n"
    body += PANEL_ROW % "Variations on the primary (clean-risk) specification:"
    body += ladder_row(results, r"\quad 60-month beta window", "w60", "CLEAN_RISK")
    body += ladder_row(results, r"\quad Quartile carbon (Q4$-$Q1)$^{b}$", "quartile", "CLEAN_RISK")
    body += ladder_row(results, r"\quad Energy return $\times$ carbon control", "energy", "CLEAN_RISK")
    return BODY_START + COLUMNS % "Robustness check" + body + BODY_END


def table8(results):
    head = (r"\begin{tabular}{l c c c c c c}" "\n" r"\toprule" "\n"
            r" & & & \multicolumn{2}{c}{Announcement $[-1,+1]$} & "
            r"\multicolumn{2}{c}{Pre-event $[-5,-2]$} \\" "\n"
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n"
            r"Specification & Cluster & Events & Full & Reported & Full & Reported \\" "\n"
            r"\midrule" "\n")
    labels = [("Clean-risk (all)", "CLEAN_RISK"), (r"\quad Introducing", "CLEAN_RISK_INTRO"),
              (r"\quad Resolving", "CLEAN_RISK_RESOLV"), ("Supply placebo", "SUPPLY_PLACEBO"),
              ("Reinforcing", "REINFORCING"), ("Pool all signed", "POOLED_SIGNED")]
    body = ""
    for label, spec in labels:
        cells = []
        for window in (ANNOUNCE, PRE):
            dagger = spec == "CLEAN_RISK_RESOLV" and window == PRE
            cells += [cell(lookup(results, v, spec, window), dagger) for v in ("main", "reported")]
        first = lookup(results, "main", spec, ANNOUNCE)
        cluster = "2-way" if first["clust"] == "2way" else "firm"
        body += f"{label} & {cluster} & {first['events']} & " + " & ".join(cells) + " \\\\\n"
    return head + body + BODY_END


def describe(series):
    return {"N": len(series), "Mean": series.mean(), "Median": series.median(), "SD": series.std(),
            "Min": series.min(), "P25": series.quantile(0.25), "P75": series.quantile(0.75),
            "Max": series.max()}


def table1(panel, carbon):
    """Panel A: AR at firm-day level. Panel B: carbon at firm-year level, sample firms only."""
    sample = carbon[carbon["in_sample"]]
    return pd.DataFrame({
        "Abnormal return AR (bp)": describe(panel["AR"] * 1e4),
        "log carbon intensity": describe(np.log(sample["carbon_intensity"])),
        "Carbon intensity, z-score": describe(sample[CARBON]),
    }).T


def table1_tex(summary):
    head = (r"\begin{tabular}{lrrrrrrrr}" "\n" r"\toprule" "\n"
            r"Variable & N & Mean & Median & SD & Min & $P_{25}$ & $P_{75}$ & Max \\" "\n"
            r"\midrule" "\n")
    def fmt(name, label):
        row = summary.loc[name]
        stats_ = " & ".join(f"${row[c]:,.2f}$" for c in summary.columns if c != "N")
        return f"{label} & {int(row['N']):,} & {stats_} \\\\\n"
    return (head + PANEL_ROW.replace("{5}", "{9}") % "Panel A: Abnormal returns (firm-day)"
            + fmt("Abnormal return AR (bp)", r"Abnormal return $AR_{it}$ (bp)")
            + PANEL_ROW.replace("{5}", "{9}") % "Panel B: Carbon intensity (firm-year)"
            + fmt("log carbon intensity", "log carbon intensity")
            + fmt("Carbon intensity, z-score", "Carbon intensity, z-score") + BODY_END)


def run(cfg):
    tables = output_dir(cfg, "tables")
    results = pd.read_csv(tables / "regression_results.csv")
    primary = cfg["abnormal_returns"]["primary"]
    panel = pd.read_parquet(processed_path(cfg, "panel", window=primary), columns=[ID, DATE, "AR", CARBON])
    carbon = pd.read_parquet(processed_path(cfg, "carbon"))

    summary = table1(panel, carbon)
    summary.to_csv(tables / "table1_summary.csv")
    (tables / "table1_summary.tex").write_text(table1_tex(summary))
    (tables / "table4_main.tex").write_text(table4(results))
    (tables / "table5_robustness.tex").write_text(table5(results))
    (tables / "table8_reported.tex").write_text(table8(results))

    firms_per_day = panel.groupby(DATE)[ID].nunique()
    record_stats(cfg, ar_mean_bp=float(summary.loc["Abnormal return AR (bp)", "Mean"]),
                 ar_median_bp=float(summary.loc["Abnormal return AR (bp)", "Median"]),
                 ar_sd_bp=float(summary.loc["Abnormal return AR (bp)", "SD"]),
                 log_ci_mean=float(summary.loc["log carbon intensity", "Mean"]),
                 log_ci_sd=float(summary.loc["log carbon intensity", "SD"]),
                 firms_per_day=float(firms_per_day.mean()),
                 ar_carbon_corr=float(panel["AR"].corr(panel[CARBON])))
    print(summary.round(2).to_string())
    print(f"[7] average firms per trading day: {firms_per_day.mean():.0f}")
    print(f"[7] wrote Tables 1, 4, 5, 8 -> {tables}")


if __name__ == "__main__":
    run(load_config())
