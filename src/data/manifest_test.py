from pathlib import Path

import pandas as pd

from src.utilities import PROJECT_ROOT, load_config


def create_public_test_manifest(cfg: dict) -> pd.DataFrame:
	test_cfg = cfg['public_test']
	image_dir = PROJECT_ROOT / test_cfg['images_dir']
	manifest_path = PROJECT_ROOT / test_cfg['manifest_path']
	image_paths = sorted(
		path for path in image_dir.iterdir()
		if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}
	)
	manifest = pd.DataFrame({
		'image_id': [path.stem for path in image_paths],
		'path': [str(path.relative_to(PROJECT_ROOT)) for path in image_paths],
	})
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	manifest.to_csv(manifest_path, index=False)
	print(f'Wrote public-test manifest with {len(manifest)} images to {manifest_path}')
	return manifest


if __name__ == '__main__':
	create_public_test_manifest(load_config())
