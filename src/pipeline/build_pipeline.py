from src.data.load import load_path
from src.pipeline.train_test import train_and_evaluate
from src.utilities import load_config

cfg = load_config()

def run_baseline():
    folds = [
    # 'fold1',
    # 'fold2',
    # 'fold3',
    # 'fold4',
    'fold5'
    ]


    for fold in folds:
        print(f'\n=== Validation case: {fold} ===')
        X_train_path, y_train_path, X_val_path, y_val_path = load_path(fold, cfg=cfg)

        results = train_and_evaluate(
            fold=fold,
            X_train=X_train_path,
            y_train=y_train_path,
            X_val=X_val_path,
            y_val=y_val_path,
            cfg=cfg,
        )
    return 1
