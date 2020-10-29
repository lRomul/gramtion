from typing import Optional, List

from pydantic import BaseModel


class Caption(BaseModel):
    text: str
    alt_text: bool = False


class Photo(BaseModel):
    url: str
    caption: Optional[Caption]


class Label(BaseModel):
    name: str
    score: float


class Prediction(BaseModel):
    caption: Caption
    labels: List[Label]
