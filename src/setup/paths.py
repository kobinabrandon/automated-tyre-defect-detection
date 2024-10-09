import os
from pathlib import Path

PARENT_DIR = Path("__file__").parent.resolve()

DATA_DIR = PARENT_DIR/"Mushrooms"
TRAIN_DATA_DIR = DATA_DIR/"Training"
VAL_DATA_DIR = DATA_DIR/"Validation"
TEST_DATA_DIR = DATA_DIR/"Testing"

MODELS_DIR = PARENT_DIR/"models"
LOGS_DIR = PARENT_DIR/"logs"
TRIALS_DIR = MODELS_DIR/"optuna_trials"


def make_fundamental_paths():
    
    for folder in [DATA_DIR, MODELS_DIR, TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR, LOGS_DIR, TRIALS_DIR]:
        if not Path(folder).exists():
            os.mkdir(folder)
