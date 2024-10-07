import os 
import subprocess 
from pathlib import Path

from loguru import logger 
from src.setup.paths import PARENT_DIR, DATA_DIR


def download_data(path: Path = PARENT_DIR) -> None:
    """
    Download the zip file that contains the raw data in the event that the folder that contains it is 
    either absent or incomplete.
    """
    file_size = check_folder_size(path=DATA_DIR) 
    expected_file_size = 1438485394  # The size of the full dataset in bytes

    if Path(DATA_DIR).exists() and file_size== expected_file_size:
        logger.success("The data has already been downloaded")
        return 
    elif Path(file_path).is_file() and file_size < expected_file_size:
        logger.warning("An incomplete version of the data's zip file is present -> Deleting it and starting a fresh download")
        os.remove(path=file_path)

    dataset_name = "anilkrsah/deepmushroom"
    command = ["kaggle", "datasets", "download", "-d", dataset_name, "-p", path, "--unzip"]  
    subprocess.run(command, check=True) 

def check_folder_size(path: Path):
    return sum(folder.stat().st_size for folder in Path(path).glob('**/*') if folder.is_file())

if __name__ == "__main__":
    download_data()
    