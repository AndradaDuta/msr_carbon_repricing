"""Diagnostic: what drives the resolving pre-event coefficient in Table 4?

Not part of the thesis and not run by run_pipeline.py. Re-estimates CLEAN_RISK_RESOLV on
[-5,-2] under two-way clustering: on all three events, without the dates shared with
event 16's announcement window, and leaving out each resolving event in turn.

With three events the two-way standard error is itself unreliable, so read the point
estimates first.

Usage:   python -m src.diagnostic_resolving
Output:  output/tables/resolving_diagnostics.csv
"""
import sys

import pandas as pd

from src import step6_regressions as s6
from src.common import DATE, ROOT, load_config, output_dir


def estimate(label, panel, window, event_ids, extra_drop=()):
    model, n_events, n_obs = s6.run_spec(panel, window, event_ids, "pv_ar_sign",
                                         s6.CLUSTER["twoway"], extra_drop=extra_drop)
    return {"row": label, "events": n_events, "N": n_obs,
            "beta_bp": model.coef()["snx"] * s6.BP, "se_bp": model.se()["snx"] * s6.BP,
            "t": model.tstat()["snx"], "p": model.pvalue()["snx"]}


def run(cfg):
    panel = s6.load_panel(cfg, cfg["abnormal_returns"]["primary"])
    used = s6.load_events(ROOT / cfg["paths"]["events"], cfg["sample"]["analysis_start"])
    tdays = pd.Index(sorted(panel[DATE].unique()))
    windows = s6.build_windows(used, tdays, cfg["regressions"]["windows"])
    overlap = s6.report_cross_window_overlap(windows)

    pre = windows["pre"]
    resolving = s6.event_sets(used)["clean_risk_resolv"]
    rows = [estimate("all three events", panel, pre, resolving),
            estimate("without 22-23 June 2022", panel, pre[~pre[DATE].isin(overlap)], resolving, overlap)]
    rows += [estimate(f"without event {event}", panel, pre, resolving - {event})
             for event in sorted(resolving)]

    results = pd.DataFrame(rows)
    results.to_csv(output_dir(cfg, "tables") / "resolving_diagnostics.csv", index=False)
    print(results.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    return results


if __name__ == "__main__":
    run(load_config(sys.argv[1] if len(sys.argv) > 1 else None))
