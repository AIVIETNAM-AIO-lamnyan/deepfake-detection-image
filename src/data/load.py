"""Load train/validation metadata for one cross-validation fold."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utilities import PROJECT_ROOT, load_config


def load_path(fold: str | int, cfg: dict | None = None):
	"""Return train paths/labels and validation paths/labels for a fold.

	Images are kept as paths so callers can decode them lazily in a Dataset or
	DataLoader instead of loading the full 512x512 dataset into memory.
	"""
	config = cfg if cfg is not None else load_config()
	n_splits = int(config.get('cross_validation', {}).get('n_splits', 5))

	split_dir = PROJECT_ROOT / 'data' / 'splits'
	fold_names = [f'fold{fold_id}' for fold_id in range(1, n_splits + 1)]
	if fold not in fold_names:
		raise ValueError(f"Unknown fold '{fold}'. Expected one of: {', '.join(fold_names)}")
	fold_paths = {name: split_dir / f'{name}.csv' for name in fold_names}
	missing_mappings = [path for path in fold_paths.values() if not path.is_file()]
	if missing_mappings:
		missing_names = ', '.join(path.name for path in missing_mappings)
		raise FileNotFoundError(
			f'Missing fold mappings: {missing_names}. Run src/data/split.py first.'
		)

	validation_mapping = pd.read_csv(fold_paths[fold])
	train_mappings = [
		pd.read_csv(path)
		for fold_name, path in fold_paths.items()
		if fold_name != fold
	]
	train_mapping = pd.concat(train_mappings, ignore_index=True)

	required_columns = {'path', 'class', 'fold'}
	for mapping_path, mapping in [
		(fold_paths[fold], validation_mapping),
		*zip([path for fold_name, path in fold_paths.items() if fold_name != fold], train_mappings),
	]:
		missing_columns = required_columns.difference(mapping.columns)
		if missing_columns:
			raise ValueError(
				f'{mapping_path} is missing columns: {sorted(missing_columns)}'
			)
		if mapping['path'].isna().any() or mapping['class'].isna().any():
			raise ValueError(f'{mapping_path} contains empty paths or class labels.')

	if validation_mapping.empty or train_mapping.empty:
		raise ValueError(f'Fold {validation_fold} has an empty train or validation split.')
	if validation_mapping['path'].duplicated().any() or train_mapping['path'].duplicated().any():
		raise ValueError('Fold mappings contain duplicate paths.')

	image_root = PROJECT_ROOT / 'data' / 'train'
	def resolve_image_path(path: str) -> str:
		image_path = Path(path)
		return str(image_path if image_path.is_absolute() else image_root / image_path)

	X_train = np.asarray(train_mapping['path'].map(resolve_image_path), dtype=str)
	y_train = train_mapping['class'].to_numpy(dtype=np.int64)
	X_test = np.asarray(validation_mapping['path'].map(resolve_image_path), dtype=str)
	y_test = validation_mapping['class'].to_numpy(dtype=np.int64)

	missing_images = [path for path in np.concatenate((X_train, X_test)) if not Path(path).is_file()]
	if missing_images:
		raise FileNotFoundError(
			f'Image path does not exist: {missing_images[0]} '
			f'({len(missing_images)} missing total)'
		)

	return X_train, y_train, X_test, y_test
