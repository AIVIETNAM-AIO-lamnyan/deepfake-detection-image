from src.pipeline.train_test import test
from src.utilities import load_config
from src.pipeline.build_pipeline import run_baseline



if __name__ == '__main__':
	run_baseline()
	# test(load_config())
