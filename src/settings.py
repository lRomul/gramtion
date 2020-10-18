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
    twitter_char_limit: int = 280
    device: str = "cuda"
    state_path: str = "/workdir/state.json"
    log_level: str = "INFO"

    @validator("device")
    def valid_device(cls, value):
        try:
            torch.device(value)
        except RuntimeError:
            raise ValueError(f"Device must be cpu or cuda")
        return value

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
