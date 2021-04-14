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
        if not caption.alt_text and not has_font(prediction):
            caption_text = caption.text.lower()
            for key, value in self.caption_replace_dict.items():
                caption_text = re.sub(r"\b{}\b".format(key), value, caption_text)
            caption_text = caption_text.capitalize() + "."
            caption_text = f"May show: {caption_text}\n"
        elif caption.alt_text:
            caption_text = f"Alt text: {caption_text}\n"
        else:  # has_font(prediction)
            caption_text = ("Image with a lot of text. "
                            "Bot will soon support text recognition.\n")

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
