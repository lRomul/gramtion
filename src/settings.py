from pydantic import BaseSettings


class Settings(BaseSettings):
    consumer_key: str
    consumer_secret: str
    access_token: str
    access_token_secret: str
    feature_checkpoint_path: str = "/model_data/detectron_model.pth"
    feature_config_path: str = "/model_data/detectron_model.yaml"
    caption_checkpoint_path: str = "/model_data/model-best.pth"
    caption_config_path: str = "/model_data/infos_trans12-best.pkl"
    twitter_char_limit: int = 280

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
