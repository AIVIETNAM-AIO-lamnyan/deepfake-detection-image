"""Create stratified cross-validation fold mapping CSVs from a train manifest."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utilities import PROJECT_ROOT

def build_fold_mappings(
	manifest_path: Path,
	output_dir: Path,
	n_splits: int = 5,
	seed: int = 42,
) -> pd.DataFrame:
	"""Write one CSV per validation fold and return the combined mapping."""
	manifest = pd.read_csv(manifest_path)
	required_columns = {'path', 'label'}
	missing_columns = required_columns.difference(manifest.columns)
	if missing_columns:
		raise ValueError(
			f'Manifest is missing required columns: {sorted(missing_columns)}'
		)
	if manifest['path'].isna().any():
		raise ValueError('Manifest contains empty image paths.')
	if manifest['path'].duplicated().any():
		raise ValueError('Manifest contains duplicate image paths.')

	class_counts = manifest['label'].value_counts()
	if class_counts.empty or class_counts.min() < n_splits:
		raise ValueError(
			f'Each class needs at least {n_splits} samples for stratified splitting.'
		)

	fold_ids = np.zeros(len(manifest), dtype=np.int64)
	rng = np.random.default_rng(seed)
	labels = manifest['label'].to_numpy()
	for class_label in class_counts.index:
		class_indices = np.flatnonzero(labels == class_label)
		rng.shuffle(class_indices)
		class_fold_ids = np.resize(np.arange(1, n_splits + 1), len(class_indices))
		rng.shuffle(class_fold_ids)
		fold_ids[class_indices] = class_fold_ids

	if (fold_ids == 0).any():
		raise RuntimeError('Some manifest rows were not assigned to a fold.')

	mapping = pd.DataFrame({
		'path': manifest['path'].astype(str),
		'class': manifest['label'],
		'fold': fold_ids,
	})

	output_dir.mkdir(parents=True, exist_ok=True)
	for fold_id in range(1, n_splits + 1):
		fold_mapping = mapping.loc[mapping['fold'] == fold_id]
		output_path = output_dir / f'fold{fold_id}.csv'
		fold_mapping.to_csv(output_path, index=False)
		distribution = fold_mapping['class'].value_counts().sort_index().to_dict()
		print(
			f'Fold {fold_id}: {len(fold_mapping)} samples, '
			f'class distribution={distribution} -> {output_path}'
		)

	return mapping


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		'--manifest',
		type=Path,
		default=PROJECT_ROOT / 'data' / 'train' / 'manifest.csv',
		help='Path to the training manifest CSV.',
	)
	parser.add_argument(
		'--output-dir',
		type=Path,
		default=PROJECT_ROOT / 'data' / 'splits',
		help='Directory where fold mapping CSVs are written.',
	)
	parser.add_argument('--n-splits', type=int, default=5)
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()

	if args.n_splits < 2:
		parser.error('--n-splits must be at least 2')

	build_fold_mappings(
		manifest_path=args.manifest,
		output_dir=args.output_dir,
		n_splits=args.n_splits,
		seed=args.seed,
	)


if __name__ == '__main__':
	main()
