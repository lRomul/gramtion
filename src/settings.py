import torch
from pathlib import Path
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    consumer_key: str
    consumer_secret: str
    access_token: str
    access_token_secret: str
    caption_config_path: Path = "width_ablations/bicaptioning_R_50_L1_H2048.yaml"
    twitter_char_limit: int = 280
    device: str = "cpu"
    since_id: str = "old"
    log_level: str = "INFO"
    google_application_credentials: str = "/workdir/google_key.json"

    @validator("device")
    def valid_device(cls, value):
        torch.device(value)
        return value

    @validator("since_id")
    def valid_since_id(cls, value):
        if value in {"old", "new"} or value.isnumeric():
            return value
        else:
            raise ValueError(f"Since id must be 'old'/'new' or numeric")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
