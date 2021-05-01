import re
from typing import Optional, Dict, List

from src.pydantic_models import PhotoPrediction
from src.settings import settings


def has_font(prediction: PhotoPrediction):
    for label in prediction.labels:
        if label.name == 'Font':
            return True
    return False


class PredictionProcessor:
    def __init__(self,
                 caption_replace_dict: Optional[Dict[str, str]] = None,
                 ocr_text_min_len: int = 5,
                 clip_min_confidence: float = 0.2):
        if caption_replace_dict is None:
            caption_replace_dict = {
                "unk": "unknown",
            }
        self.caption_replace_dict = caption_replace_dict
        self.ocr_text_min_len = ocr_text_min_len
        self.clip_min_confidence = clip_min_confidence

    def process_prediction(
        self, prediction: PhotoPrediction, photo_num: int = 0
    ) -> str:
        caption = prediction.caption
        message = f"Photo {photo_num}\n"

        if not caption.alt_text and caption.confidence >= self.clip_min_confidence:
            caption_text = caption.text.lower()
            for key, value in self.caption_replace_dict.items():
                caption_text = re.sub(r"\b{}\b".format(key), value, caption_text)
            caption_text = caption_text.capitalize() + "."
            confidence = round(caption.confidence * 100)
            caption_text = f"May show ({confidence}% confidence): {caption_text}\n"
        elif caption.alt_text:
            caption_text = f"Alt text: {caption.text}\n"
        else:
            caption_text = ""

        if has_font(prediction) and len(prediction.ocr_text.text) >= self.ocr_text_min_len:
            caption_text += f"Сontains text:\n{prediction.ocr_text.text}\n"

        message += caption_text

        labels_text = ", ".join([lab.name for lab in prediction.labels])
        labels_text = f"Tags: {labels_text.capitalize()}."
        message += labels_text

        message = message[: settings.twitter_char_limit]
        return message

    def predictions_to_messages(self, predictions: List[PhotoPrediction]) -> List[str]:
        messages = []
        for num, prediction in enumerate(predictions):
            message = self.process_prediction(prediction, photo_num=num + 1)
            messages.append(message)
        return messages
