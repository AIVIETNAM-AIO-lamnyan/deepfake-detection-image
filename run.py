from src.pipeline.build_pipeline import run_baseline, save


if __name__ == '__main__':
	results = run_baseline()
	save(results)
