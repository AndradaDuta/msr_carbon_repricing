"""Step 6: signed event windows and the specification ladder for every thesis variant.

Source: notebooks/Regression.ipynb cell 1 (Sections B-D) and cell 3 (energy control),
plus the quartile, 60-month and reported-only runs.

    AR_it = beta (sign_t x carbon_z_it) + gamma carbon_z_it + firm FE + date FE

beta is reported in basis points per one within-year SD of carbon intensity. A positive
beta places high-carbon AR in the sign-predicted direction.

Inputs:  data/processed panels, the event workbook, the Brent snapshot
Output:  output/tables/regression_results.csv
"""
import warnings

import pandas as pd
import pyfixest as pf

from src.common import CARBON, DATE, ID, load_config, output_dir, processed_path, raw_path, ROOT

# carbon_z is mostly absorbed by the firm FE, which is expected.
warnings.filterwarnings("ignore", message=r"(?s).*multicollinearity.*")

CLUSTER = {"twoway": {"CRV1": f"{ID} + {DATE}"}, "firm": {"CRV1": ID}}
CLUSTER_LABEL = {"twoway": "2way", "firm": "firm"}
BP = 1e4


def load_events(path, event_start):
    """Exact-dated, in-sample events that are treatment in at least one specification.

    Month-precision TNACs cannot anchor tight windows. Neutral events get no window, so
    their dates stay in the control pool and cannot delete a real treatment date.
    """
    ev = pd.read_excel(path, sheet_name="events", parse_dates=["event_date"])
    signs = pd.read_excel(path, sheet_name="channel_signs")
    ev = ev.merge(signs[["event_id", "analysis_group", "pv_ar_sign", "supply_ar_sign"]], on="event_id")
    ev = ev[(ev["event_date"] >= event_start) & (ev["date_precision"] == "exact")].copy()
    return ev[(ev["pv_ar_sign"] != 0) | (ev["analysis_group"] == "supply_placebo")].copy()


def event_sets(used):
    """Derive every specification's event set from analysis_group and event_type."""
    by_group = used.groupby("analysis_group")["event_id"].apply(set).to_dict()
    event_type = used.set_index("event_id")["event_type"]
    clean = by_group.get("clean_risk", set())
    return {
        "clean_risk": clean,
        "clean_risk_intro": {i for i in clean if event_type[i] == "risk_introducing"},
        "clean_risk_resolv": {i for i in clean if event_type[i] == "risk_resolving"},
        "reinforcing": by_group.get("reinforcing", set()),
        "supply_placebo": by_group.get("supply_placebo", set()),
        "pooled_signed": set().union(*[by_group.get(k, set()) for k in
                                       ["clean_risk", "confounded", "reinforcing",
                                        "risk_ambiguous_supply"]]),
    }


def build_window(events, tdays, lo_off, hi_off):
    """Dates at trading-day offsets around each event, on the panel's own grid.

    A non-trading event date snaps to the next trading day. Offsets past the grid edge
    are dropped; an empty range cannot turn into a negative slice.
    """
    columns = [DATE, "event_id", "analysis_group", "event_type", "pv_ar_sign", "supply_ar_sign"]
    records = []
    for _, event in events.iterrows():
        pos = min(tdays.searchsorted(event["event_date"]), len(tdays) - 1)
        lo, hi = max(0, pos + lo_off), min(len(tdays) - 1, pos + hi_off)
        for date in tdays[lo:hi + 1] if hi >= lo else []:
            records.append((date, event["event_id"], event["analysis_group"], event["event_type"],
                            event["pv_ar_sign"], event["supply_ar_sign"]))
    return pd.DataFrame(records, columns=columns)


def build_windows(used, tdays, window_cfg):
    """All configured windows. Within a window, treatment dates must not collide."""
    grid_start = tdays.min()
    pre_sample = used["event_date"] < grid_start
    if pre_sample.any():  # 60-month warm-up starts AR after event 1
        print(f"[6] dropping pre-sample event(s) before {grid_start.date()}: "
              f"{sorted(used.loc[pre_sample, 'event_id'])}")
    used = used[~pre_sample].copy()

    windows = {}
    for name, spec in window_cfg.items():
        windows[name] = build_window(used, tdays, *spec["offsets"])
        n_dup = int(windows[name][DATE].duplicated().sum())
        print(f"[6] {name:10s}: {len(used)} events -> {len(windows[name])} dates | overlaps: {n_dup}")
        assert n_dup == 0, f"{name}: treatment windows collide, resolve before pooling signs"
    return windows


def report_cross_window_overlap(windows):
    """Print dates in one event's announcement window AND another event's pre-window."""
    ann, pre = windows["announce"], windows["pre"]
    shared = ann.merge(pre, on=DATE, suffixes=("_announce", "_pre"))
    for _, row in shared.iterrows():
        print(f"[6] NOTE {row[DATE].date()}: announcement window of event {row['event_id_announce']} "
              f"is in the pre-window of event {row['event_id_pre']}")
    return set(shared[DATE])


def add_energy_control(panel, brent):
    """energy_ret x carbon_z only: date FE absorb the level, and EUA would be a bad control.

    Brent is forward-filled onto the panel's grid, so a treated date on a source-calendar
    gap is kept with energy_ret near zero rather than dropped.
    """
    dates = pd.Index(sorted(panel[DATE].unique()))
    series = brent.set_index(DATE)["brent"]
    series = series.reindex(dates.union(series.index)).ffill()
    energy = series.reindex(dates).pct_change(fill_method=None).rename("energy_ret")
    panel = panel.merge(energy.rename_axis(DATE).reset_index(), on=DATE, how="left")
    panel["energy_x_carbon"] = panel["energy_ret"] * panel[CARBON]
    return panel


def run_spec(panel, win, event_ids, signcol, vcov, carbon_col=CARBON, extra_rhs=(), extra_drop=()):
    """Fit one specification.

    Controls are all non-event days. Other events' dates in this window type are removed
    so they cannot sit in the residual, except any date this spec itself treats.
    """
    sel = win[win["event_id"].isin(event_ids)]
    assert not sel[DATE].duplicated().any(), "duplicate treated date within a spec"
    treat_dates = set(sel[DATE])
    drop_dates = (set(win.loc[~win["event_id"].isin(event_ids), DATE]) - treat_dates) | set(extra_drop)
    df = panel[~panel[DATE].isin(drop_dates)].copy()
    df["sn"] = df[DATE].map(sel.set_index(DATE)[signcol]).fillna(0)
    df["snx"] = df["sn"] * df[carbon_col]
    df = df.dropna(subset=["AR", carbon_col, *extra_rhs])
    rhs = " + ".join([carbon_col, "snx", *extra_rhs])
    model = pf.feols(f"AR ~ {rhs} | {ID} + {DATE}", data=df, vcov=vcov)
    return model, sel["event_id"].nunique(), len(df)


def result_row(variant, window_label, spec_name, spec, cluster, model, n_events, n_obs):
    beta, se = model.coef()["snx"], model.se()["snx"]
    lower, upper = model.confint().loc["snx"].tolist()
    return {"variant": variant, "window": window_label, "tier": spec["tier"], "spec": spec_name,
            "sign": spec["sign"].split("_")[0], "clust": CLUSTER_LABEL[cluster], "events": n_events,
            "N": n_obs, "beta_bp": beta * BP, "se_bp": se * BP, "t": model.tstat()["snx"],
            "p": model.pvalue()["snx"], "ci_lo_bp": lower * BP, "ci_hi_bp": upper * BP,
            "bound_bp": (abs(beta) + 1.96 * se) * BP}


def run_variant(name, variant, reg_cfg, sets, panel, windows, overlap):
    """Run one variant's specs across its windows and return result rows."""
    data = panel[panel["reported"]].copy() if variant.get("sample") == "reported" else panel
    carbon_col = variant.get("carbon_col", CARBON)
    extra_rhs = ("energy_x_carbon",) if variant.get("energy_control") else ()

    rows = []
    for window_name in variant.get("windows", ["announce", "pre"]):
        win = windows[window_name]
        extra_drop = ()
        if variant.get("drop_announce_overlap"):
            win, extra_drop = win[~win[DATE].isin(overlap)], overlap
        for spec_name in variant.get("specs", list(reg_cfg["specs"])):
            spec = reg_cfg["specs"][spec_name]
            ids = sets[spec["events"]]
            if not ids:
                continue
            cluster = variant.get("cluster", spec["cluster"])
            model, n_events, n_obs = run_spec(data, win, ids, spec["sign"], CLUSTER[cluster],
                                              carbon_col, extra_rhs, extra_drop)
            rows.append(result_row(name, reg_cfg["windows"][window_name]["label"], spec_name, spec,
                                   cluster, model, n_events, n_obs))
    return rows


def load_panel(cfg, window):
    panel = pd.read_parquet(processed_path(cfg, "panel", window=window))
    panel[DATE] = pd.to_datetime(panel[DATE])
    return panel


def run(cfg):
    reg_cfg, primary = cfg["regressions"], cfg["abnormal_returns"]["primary"]
    used = load_events(ROOT / cfg["paths"]["events"], cfg["sample"]["analysis_start"])
    brent = pd.read_csv(raw_path(cfg, "brent"), parse_dates=[DATE])

    panels, contexts, rows = {}, {}, []
    for name, variant in reg_cfg["variants"].items():
        window = variant.get("ar", primary)
        if window not in panels:
            panels[window] = load_panel(cfg, window)
            if window == primary:  # as run: the energy term enters on the primary panel only
                panels[window] = add_energy_control(panels[window], brent)
            tdays = pd.Index(sorted(panels[window][DATE].unique()))
            print(f"[6] {window} grid: {len(tdays):,} dates from {tdays.min().date()}")
            windows = build_windows(used, tdays, reg_cfg["windows"])
            contexts[window] = (windows, report_cross_window_overlap(windows),
                                event_sets(used[used["event_date"] >= tdays.min()]))
        windows, overlap, sets = contexts[window]
        rows += run_variant(name, variant, reg_cfg, sets, panels[window], windows, overlap)
        print(f"[6] finished variant: {name}")

    results = pd.DataFrame(rows)
    path = output_dir(cfg, "tables") / "regression_results.csv"
    results.to_csv(path, index=False)
    show = results[results["variant"] == "main"].drop(columns=["variant", "tier", "bound_bp"])
    print(show.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print(f"[6] saved {len(results)} rows -> {path.name}")


if __name__ == "__main__":
    run(load_config())
