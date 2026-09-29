from pathlib import Path
import json

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml" # Cái này để tạm thời, sau này sẽ load riêng từng config 1 cho từng giai đoạn

def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def resolve_path(cfg: dict, key: str) -> Path:

    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise KeyError("Config is missing the 'paths' section.")
    if key not in paths:
        available = ", ".join(sorted(paths)) or "(empty)"
        raise KeyError(f"Path '{key}' not found in config. Available: {available}")

    return PROJECT_ROOT / paths[key]

def calculate_metrics(y_true, y_pred, y_score, val_loss: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    if not (len(y_true) == len(y_pred) == len(y_score)):
        raise ValueError("y_true, y_pred, and y_score must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics for an empty validation set.")

    roc_auc = (
        float(roc_auc_score(y_true, y_score))
        if np.unique(y_true).size == 2
        else float("nan")
    )

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "val_loss": float(val_loss),
    }