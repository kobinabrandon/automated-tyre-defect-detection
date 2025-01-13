import os 
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.setup.paths import PARENT_DIR, DATA_DIR


def get_num_classes(path: str = DATA_DIR) -> int: 
    """
    Each class of mushrooms is in a folder, and this function 
    will look through the subdirectories of the folder where 
    the training data is kept. It will then make a list of 
    these subdirectories, and return the length of said list.

    Returns:
        int: the length of the list of classes (the genera of mushrooms)
    """
    classes = [name for root, sub_dirs, files in os.walk(path, topdown=True) for name in sub_dirs] 
    return len(classes)


class GeneralConfig(BaseSettings):
    """ Using pydantic to validate environment variables """

    model_config = SettingsConfigDict(env_file=PARENT_DIR/".env", env_file_encoding="utf-8", extra="allow")
    num_classes: int = get_num_classes()

    # CometML
    comet_api_key: str
    comet_project_name: str
    comet_workspace: str

    # Image dimensions
    resized_image_width: int
    resized_image_height: int


config = GeneralConfig()
