"""Run the thesis pipeline end to end.

    python run_pipeline.py                  # steps 2-9, from the saved returns parquet
    python run_pipeline.py --from returns   # also rebuild returns from WRDS (needs an account)
    python run_pipeline.py --from regressions --to tables
    python run_pipeline.py --only abnormal_returns
    python run_pipeline.py --config tests/config_synthetic.yaml   # no licensed data needed
"""
import argparse
import importlib
import time

from src.common import load_config

STEPS = {
    "returns": "step1_returns",
    "external": "step2_external",
    "usd_panel": "step3_usd_panel",
    "abnormal_returns": "step4_abnormal_returns",
    "carbon_panel": "step5_carbon_panel",
    "regressions": "step6_regressions",
    "tables": "step7_tables",
    "figures": "step8_figures",
    "replication": "step9_replication",
}
DEFAULT_START = "external"  # step 1 needs a WRDS account; its output is a raw input


def select_steps(first, last, only):
    names = list(STEPS)
    if only:
        return [only]
    return names[names.index(first): names.index(last) + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML applied on top of config.yaml")
    parser.add_argument("--from", dest="first", choices=STEPS, default=DEFAULT_START)
    parser.add_argument("--to", dest="last", choices=STEPS, default="replication")
    parser.add_argument("--only", choices=STEPS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    failures = 0
    for name in select_steps(args.first, args.last, args.only):
        print(f"\n=== {name} ===")
        started = time.time()
        outcome = importlib.import_module(f"src.{STEPS[name]}").run(cfg)
        print(f"=== {name} done in {time.time() - started:.0f}s ===")
        if name == "replication":
            failures = outcome
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
