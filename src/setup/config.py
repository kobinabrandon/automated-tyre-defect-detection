from pydantic_settings import BaseSettings, SettingsConfigDict

from src.setup.paths import PARENT_DIR

class Settings(BaseSettings):

    """ Using pydantic to validate environment variables """ 
 
    model_config = SettingsConfigDict(
        env_file=PARENT_DIR/".env",
        env_file_encoding="utf-8",
        extra="allow"
    )

    # CometML
    comet_api_key: str
    comet_workspace: str
    comet_project_name: str

    # Image dimensions
    resized_image_width: int
    resized_image_height: int
  

settings = Settings()