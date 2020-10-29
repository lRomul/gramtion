import re
from typing import Optional, Dict, List

from src.pydantic_models import PhotoPrediction
from src.settings import settings


class PredictionProcessor:
    def __init__(self, caption_replace_dict: Optional[Dict[str, str]] = None):
        if caption_replace_dict is None:
            caption_replace_dict = {
                "unk": "unknown",
            }
        self.caption_replace_dict = caption_replace_dict

    def process_prediction(
        self, prediction: PhotoPrediction, photo_num: int = 0
    ) -> str:
        caption = prediction.caption
        message = f"Photo {photo_num}\n"

        caption_text = caption.text
        if not caption.alt_text:
            caption_text = caption.text.lower()
            for key, value in self.caption_replace_dict.items():
                caption_text = re.sub(r"\b{}\b".format(key), value, caption_text)
            caption_text = caption_text.capitalize() + "."
            phrase = "May show"
        else:
            phrase = "Alt text"
        caption_text = f"{phrase}: {caption_text}\n"
        message += caption_text

        labels_text = ", ".join([l.name for l in prediction.labels])
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
