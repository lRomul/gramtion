from typing import Optional, List

from pydantic import BaseModel


class Caption(BaseModel):
    text: str
    alt_text: bool = False
    confidence: Optional[float] = None


class Photo(BaseModel):
    url: str
    caption: Optional[Caption]


class Label(BaseModel):
    name: str
    score: float


class OCRText(BaseModel):
    text: str
    locale: str
    area: float


class PhotoPrediction(BaseModel):
    caption: Caption
    labels: List[Label]
    ocr_text: OCRText
