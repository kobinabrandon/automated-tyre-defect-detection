import os

from pathlib import Path
from loguru import logger
from zipfile import ZipFile

from tqdm import tqdm
from requests import Response, get

from src.setup.config import data_config
from src.setup.paths import RAW_DATA_DIR, make_fundamental_paths
from src.feature_pipeline.data_extraction import extract_zipfile


def download(url: str = data_config.url, keep_zipfile: bool=False):

    make_fundamental_paths()
    file_path: Path = RAW_DATA_DIR / data_config.zipfile_name

    if Path(file_path).exists():
        logger.success("The data is already downloaded")
        try:
            extract_zipfile(keep_zipfile=keep_zipfile)
        except Exception as error:
            logger.error(error)
    else:
        response: Response = get(url, stream=True) 
        if response.status_code == 200:
        
            block_size: int = 1024
            total_size: int = int(response.headers.get("content_length", 0))

            with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading data") as bar:
                with open(file_path, "wb") as file:
                    for block in response.iter_content(block_size): 
                        bar.update(len(block))  # Push the progress bar by the length of the block 
                        file.write(block)  # Write the block of data to the disk
                
            # Complain if the file is not empty and the download is incomplete.
            if (total_size != 0) and (bar.n != total_size):
                raise RuntimeError("Could not download file")
            else:
                logger.success("Download complete")


def extract_zipfile(keep_zipfile: bool, zipfile_path: Path = RAW_DATA_DIR / data_config.zipfile_name) -> None:
    """
    Extract all of the contents of the zipfile
        
    Args:
        keep_zipfile: whether to keep te zipfile after extraction 
        zipfile_path: the path to the zipfile 
    """
    with ZipFile(file=zipfile_path, mode="r") as zipfile:
        zipfile.extractall(RAW_DATA_DIR)
        logger.success(f"Zipfile extracted to {zipfile_path}")
     
    if not keep_zipfile: 
        os.remove(zipfile_path) 


if __name__ == "__main__":
    download()
