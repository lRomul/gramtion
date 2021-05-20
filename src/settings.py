import torch
from pathlib import Path
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    consumer_key: str
    consumer_secret: str
    access_token: str
    access_token_secret: str
    feature_checkpoint_path: Path = "/model_data/detectron_model.pth"
    feature_config_path: Path = "/model_data/detectron_model.yaml"
    caption_checkpoint_path: Path = "/model_data/model-best.pth"
    caption_config_path: Path = "/model_data/infos_trans12-best.pkl"
    clip_model_name: str = "ViT-B/32"
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
            raise ValueError("Since id must be 'old'/'new' or numeric")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
