import os 
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.setup.paths import PARENT_DIR


def get_num_classes(path: Path) -> int: 
    """
    Each class of mushrooms is in a folder, and this function will look through the subdirectories of the folder where 
    the training data is kept. It will then make a list of these subdirectories, and return the length of said list.

    Returns:
        int: the length of the list of classes 
    """
    classes = [name for root, sub_dirs, files in os.walk(path, topdown=True) for name in sub_dirs] 
    return len(classes)


class DataConfig(BaseSettings):

    _ = SettingsConfigDict(env_file=PARENT_DIR/".env", env_file_encoding="utf-8", extra="allow")
    # num_classes: int = get_num_classes()

    if load_dotenv(find_dotenv()):
        url: str = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/bn7ch8tvyp-1.zip" 
        file_name: str = os.environ["FILE_NAME"]


class ImageConfig(BaseSettings):

    _ = SettingsConfigDict(env_file=PARENT_DIR/".env", env_file_encoding="utf-8", extra="allow")
    resized_image_width: int = 512
    resized_image_height: int = 512


class EnvConfig(BaseSettings):

    _ = SettingsConfigDict(env_file=PARENT_DIR/".env", env_file_encoding="utf-8", extra="allow")

    if load_dotenv(find_dotenv()):
        comet_api_key: str
        comet_project_name: str
        comet_workspace: str


env = EnvConfig()
data_config = DataConfig()
image_config = ImageConfig()

