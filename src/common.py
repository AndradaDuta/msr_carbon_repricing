"""Config loading, path helpers and the run-statistics log shared by every step."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ID, DATE = "isin", "date"
CARBON = "carbon_intensity_z"


def _deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(override_path=None):
    """Read config.yaml, then apply a second YAML on top (used by the synthetic run)."""
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    if override_path:
        with open(override_path) as f:
            _deep_update(cfg, yaml.safe_load(f))
    return cfg


def raw_path(cfg, key):
    return ROOT / cfg["paths"]["raw"] / cfg["raw_files"][key]


def processed_path(cfg, key, **fmt):
    path = ROOT / cfg["paths"]["processed"] / cfg["processed_files"][key].format(**fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(cfg, key):
    path = ROOT / cfg["paths"][key]
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_stats(cfg, **values):
    """Append sample counts to run_stats.json; the replication step reads them back."""
    path = processed_path(cfg, "stats")
    stats = json.loads(path.read_text()) if path.exists() else {}
    stats.update(values)
    path.write_text(json.dumps(stats, indent=2, default=str))


def read_stats(cfg):
    path = processed_path(cfg, "stats")
    return json.loads(path.read_text()) if path.exists() else {}
