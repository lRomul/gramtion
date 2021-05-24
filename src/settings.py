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
    caption_beam_size: int = 64
    caption_sample_n: int = 64
    clip_min_confidence_for_caption: float = 0.0
    max_text_area_for_caption: float = 0.3
    min_text_area_for_ocr: float = 0.03
    label_score_threshold: float = 0.7
    label_max_number: int = 5
    mention_loop_sleep: float = 14.0
    clip_model_name: str = "ViT-B/32"
    twitter_char_limit: int = 280
    device: str = "cpu"
    since_id: str = "old"
    log_level: str = "INFO"
    google_application_credentials: str = "/workdir/google_key.json"
    debug: bool = False

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
