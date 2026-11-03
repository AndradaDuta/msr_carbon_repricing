"""Step 9: compare this run with every number reported in the thesis.

Targets live in thesis_targets.yaml: sample counts must match exactly, and published
statistics must match to the precision they were printed at.

Inputs:  run_stats.json, regression_results.csv, thesis_targets.yaml
Output:  output/replication_report.md
"""
import pandas as pd
import yaml

from src.common import ROOT, load_config, output_dir, read_stats

WINDOW_LABELS = {"announce": "ANNOUNCE [-1,+1]", "pre": "PRE-EVENT [-5,-2]"}


def check(label, target, actual, tol=0.0):
    """One comparison. tol = 0 means exact equality."""
    if actual is None or (isinstance(actual, float) and pd.isna(actual)):
        return {"check": label, "thesis": target, "run": "missing", "pass": False}
    if isinstance(target, str):
        return {"check": label, "thesis": target, "run": actual, "pass": str(actual) == target}
    return {"check": label, "thesis": target, "run": round(float(actual), 4),
            "pass": abs(float(actual) - float(target)) <= tol + 1e-9}


def compare(targets, stats, results):
    checks = [check(key, value, stats.get(key)) for key, value in targets["counts"].items()]
    checks += [check(key, item["value"], stats.get(key), item["tol"])
               for key, item in targets["statistics"].items()]

    for t in targets["results"]:
        window = WINDOW_LABELS[t["window"]]
        row = results[(results["variant"] == t["variant"]) & (results["spec"] == t["spec"])
                      & (results["window"] == window)]
        label = f"{t['variant']} {t['spec']} {t['window']}"
        if row.empty:
            checks.append(check(f"{label} beta", t["beta"], None))
            continue
        row = row.iloc[0]
        checks.append(check(f"{label} beta", t["beta"], row["beta_bp"], 0.005))
        checks.append(check(f"{label} se", t["se"], row["se_bp"], 0.005))
        if "n" in t:
            checks.append(check(f"{label} N", t["n"], row["N"]))
        if "bound" in t:
            checks.append(check(f"{label} bound", t["bound"], row["bound_bp"], 0.05))
    return pd.DataFrame(checks)


def write_report(report, path):
    failed = report[~report["pass"]]
    lines = ["# Replication report", "",
             f"{report['pass'].sum()} of {len(report)} checks match the submitted thesis.", ""]
    if len(failed):
        lines += ["## Mismatches", "", failed.to_markdown(index=False), ""]
    lines += ["## All checks", "", report.to_markdown(index=False), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(cfg):
    target_file = cfg["replication"].get("targets")
    if not target_file:
        print("[9] no targets configured, skipping the comparison")
        return 0
    with open(ROOT / target_file) as f:
        targets = yaml.safe_load(f)
    results = pd.read_csv(output_dir(cfg, "tables") / "regression_results.csv")
    report = compare(targets, read_stats(cfg), results)

    path = ROOT / cfg["paths"]["report"]
    write_report(report, path)
    n_failed = int((~report["pass"]).sum())
    print(f"[9] {report['pass'].sum()} of {len(report)} checks match the thesis -> {path.name}")
    if n_failed:
        print(report[~report["pass"]].to_string(index=False))
    return n_failed


if __name__ == "__main__":
    raise SystemExit(1 if run(load_config()) else 0)
