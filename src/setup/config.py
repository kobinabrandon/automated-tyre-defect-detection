import os 
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.setup.paths import PARENT_DIR, RAW_DATA_DIR


def get_num_classes(path: Path) -> int: 
    """
    Each class of mushrooms is in a folder, and this function will look through the subdirectories of the folder where 
    the training data is kept. It will then make a list of these subdirectories, and return the length of said list.

    Returns:
        path: the path to the directory where the classes of data are stored.
    """
    classes = [name for root, sub_dirs, files in os.walk(path, topdown=True) for name in sub_dirs] 
    return len(classes)


class DataConfig(BaseSettings):

    _ = SettingsConfigDict(env_file=PARENT_DIR/".env", env_file_encoding="utf-8", extra="allow")

    if load_dotenv(find_dotenv()):
        directory_name: str = os.environ["DIRECTORY_NAME"]
        zipfile_name: str = f"{directory_name}.zip" 
        num_classes: int = get_num_classes(path=RAW_DATA_DIR / directory_name)
        url: str = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/bn7ch8tvyp-1.zip" 


class ImageConfig(BaseSettings):
    resized_image_width: int = 512
    resized_image_height: int = 512


class EnvConfig(BaseSettings):

    _ = SettingsConfigDict(env_file=PARENT_DIR/".env", env_file_encoding="utf-8", extra="allow")

    if load_dotenv(find_dotenv()):
        comet_api_key: str = os.environ["COMET_API_KEY"]
        comet_workspace: str = os.environ["COMET_WORKSPACE"]
        comet_project_name: str = os.environ["COMET_PROJECT_NAME"]


class ModelConfig(BaseSettings):
    batch_size: int = 32
    vit_base: str = "google/vit-base-patch16-224"
    vit_hybrid: str = "google/vit-hybrid-base-bit-384"
    beit_base: str = "microsoft/beit-base-patch16-224"


class ProcessingConfig(BaseSettings):
    batch_size: int = 10
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


env = EnvConfig()
data_config = DataConfig()
image_config = ImageConfig()
model_config = ModelConfig()
process_config = ProcessingConfig()

