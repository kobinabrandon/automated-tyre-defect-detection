import os
from pathlib import Path

PARENT_DIR = Path("__file__").parent.resolve()

DATA_DIR = PARENT_DIR/"data"
TRAINING_DATA_DIR = DATA_DIR/"Training"
VALIDATION_DATA_DIR = DATA_DIR/"Validation"
TEST_DATA_DIR = DATA_DIR/"Testing"
MODELS_DIR = PARENT_DIR/"models"

TRIALS_DIR = MODELS_DIR/"optuna_trials"


if __name__ == "__main__":

    # Create any of these folders if they don't already exist
    for folder in [
        DATA_DIR, MODELS_DIR, TRAINING_DATA_DIR, VALIDATION_DATA_DIR, TEST_DATA_DIR, TRIALS_DIR
    ]:
        if not Path(folder).exists():
            os.mkdir(folder)



