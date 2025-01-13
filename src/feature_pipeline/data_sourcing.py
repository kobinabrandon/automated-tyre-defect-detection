from tqdm import tqdm
from pathlib import Path
from loguru import logger
from requests import Response, get

from src.setup.config import data_config
from src.setup.paths import RAW_DATA_DIR, make_fundamental_paths
from src.feature_pipeline.data_extraction import extract_zipfile


def download(url: str = data_config.url):

    make_fundamental_paths()
    file_path: Path = RAW_DATA_DIR / data_config.zipfile_name

    if Path(file_path).exists():
        logger.success("The data is already downloaded")
        try:
            extract_zipfile(keep_zipfile=True)
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

if __name__ == "__main__":
    download()
