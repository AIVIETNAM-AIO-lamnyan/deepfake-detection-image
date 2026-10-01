"""Load train/validation metadata for one cross-validation fold."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utilities import PROJECT_ROOT, load_config


def load_path(fold: str | int, cfg: dict | None = None):
    config = cfg if cfg is not None else load_config()
    n_splits = int(config.get("cross_validation", {}).get("n_splits", 5))
    fold = f"fold{fold}" if isinstance(fold, int) else fold

    split_dir = PROJECT_ROOT / "data" / "splits"
    mappings = {
        f"fold{i}": pd.read_csv(split_dir / f"fold{i}.csv")
        for i in range(1, n_splits + 1)
    }

    val_mapping = mappings[fold]
    train_mapping = pd.concat(
        [mapping for name, mapping in mappings.items() if name != fold],
        ignore_index=True,
    )

    image_root = PROJECT_ROOT / "data" / "train"

    def resolve_path(path):
        path = Path(path)
        return str(path if path.is_absolute() else image_root / path)

    X_train = train_mapping["path"].map(resolve_path).to_numpy()
    y_train = train_mapping["class"].to_numpy(dtype="int64")
    X_val = val_mapping["path"].map(resolve_path).to_numpy()
    y_val = val_mapping["class"].to_numpy(dtype="int64")

    return X_train, y_train, X_val, y_val