import os

from pathlib import Path
from loguru import logger
from zipfile import ZipFile

from src.setup.paths import RAW_DATA_DIR
from src.setup.config import data_config


def extract_zipfile(keep_zipfile: bool, zipfile_path: Path = RAW_DATA_DIR / data_config.zipfile_name) -> None:
    """
    Extract all of the contents of the zipfile
        
    Args:
        keep_zipfile: whether to keep te zipfile after extraction 
        zipfile_path: the path to the zipfile 
    """
    with ZipFile(file=zipfile_path, mode="r") as zipfile:
        zipfile.extractall(RAW_DATA_DIR)
        logger.success(f"Zipfile extracted to data/raw/{data_config.file_name}")
     
    if not keep_zipfile: 
        os.remove(zipfile_path) 

