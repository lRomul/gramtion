import re
from typing import Optional, Dict, List

from src.pydantic_models import PhotoPrediction
from src.settings import settings


class PredictionProcessor:
    def __init__(self, replace_dict: Optional[Dict[str, str]] = None):
        if replace_dict is None:
            replace_dict = {
                "unk": "unknown",
            }
        self.replace_dict = replace_dict

    def process_prediction(
        self, prediction: PhotoPrediction, num_text: str = ""
    ) -> str:
        caption = prediction.caption
        text = caption.text
        if not caption.alt_text:
            text = caption.text.lower()
            for key, value in self.replace_dict.items():
                text = re.sub(r"\b{}\b".format(key), value, text)
            text = text.capitalize() + "."
            phrase = "may show"
        else:
            phrase = "alt text"
        text = f"{num_text} {phrase}: {text}"
        text = text[: settings.twitter_char_limit]
        return text

    def predictions_to_messages(self, predictions: List[PhotoPrediction]) -> List[str]:
        messages = []
        for num, prediction in enumerate(predictions):
            num_text = f"Photo {num + 1}" if len(predictions) > 1 else "Photo"
            message = self.process_prediction(prediction, num_text=num_text)
            messages.append(message)
        return messages
