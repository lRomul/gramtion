import sys
import logging
from typing import Optional

from pydantic import BaseModel


class Photo(BaseModel):
    media_url_https: str
    ext_alt_text: Optional[str]


class Caption(BaseModel):
    text: str
    alt_text: bool = False


def setup_logging(log_level: str = "INFO"):
    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setLevel(log_level)
    logging.basicConfig(
        format="[{asctime}][{levelname}]: {message}",
        style="{",
        level=log_level,
        handlers=[stdout],
    )
