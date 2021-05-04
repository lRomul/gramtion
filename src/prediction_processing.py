import textwrap
from typing import List

from src.pydantic_models import PhotoPrediction
from src.settings import settings
from src.utils import generate_repr


def caption_has_unknown(prediction: PhotoPrediction):
    if not prediction.caption.alt_text:
        if "UNK" in prediction.caption.text:
            return True
    return False


def split_message(message: str, max_splits=9):
    if len(message) <= settings.twitter_char_limit:
        return [message]
    else:
        width = settings.twitter_char_limit - len(f" [{max_splits}/{max_splits}]")
        messages = textwrap.wrap(message, width, break_long_words=False)
        messages = messages[:max_splits]
        messages = [mes + f" [{num + 1}/{len(messages)}]"
                    for num, mes in enumerate(messages)]
    return messages


class PredictionProcessor:
    def __init__(self,
                 clip_min_confidence: float = 0.0,
                 max_text_area_for_caption: float = 0.3,
                 min_text_area_for_ocr: float = 0.03):
        self.clip_min_confidence = clip_min_confidence
        self.max_text_area_for_caption = max_text_area_for_caption
        self.min_text_area_for_ocr = min_text_area_for_ocr

    def process_prediction(
        self, prediction: PhotoPrediction, photo_num: int = 0
    ) -> str:
        caption = prediction.caption
        message = f"Image {photo_num}\n"

        caption_text = ""
        if not caption.alt_text and not caption_has_unknown(prediction) \
                and prediction.ocr_text.area < self.max_text_area_for_caption \
                and caption.confidence >= self.clip_min_confidence:
            caption_text = caption.text.lower().capitalize() + "."
            caption_text = f"May show: {caption_text}\n"
        elif caption.alt_text:
            caption_text = f"Alt text: {caption.text}\n"

        if prediction.ocr_text.area > self.min_text_area_for_ocr:
            caption_text += f"Сontains text:\n{prediction.ocr_text.text}"

        message += caption_text

        if prediction.labels:
            labels_text = ", ".join([lab.name for lab in prediction.labels])
            labels_text = f"Tags: {labels_text.capitalize()}."
            message += labels_text
        return message

    def predictions_to_messages(self, predictions: List[PhotoPrediction]) -> List[str]:
        messages = []
        for num, prediction in enumerate(predictions):
            message = self.process_prediction(prediction, photo_num=num + 1)
            messages += split_message(message)
        return messages

    def __repr__(self):
        return generate_repr(
            self,
            [
                "clip_min_confidence",
                "max_text_area_for_caption",
                "min_text_area_for_ocr",
            ],
        )
