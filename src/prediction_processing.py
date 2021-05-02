from typing import Optional, List

from src.pydantic_models import PhotoPrediction
from src.settings import settings
from src.utils import generate_repr


def has_labels(prediction: PhotoPrediction,
               labels: List[str],
               number: Optional[int] = None):
    if number is None:
        number = len(labels)
    for label in prediction.labels[:number]:
        if label.name in labels:
            return True
    return False


def caption_has_unknown(prediction: PhotoPrediction):
    if not prediction.caption.alt_text:
        if "UNK" in prediction.caption.text:
            return True
    return False


class PredictionProcessor:
    def __init__(self,
                 ocr_text_min_len: int = 5,
                 clip_min_confidence: float = 0.0):
        self.ocr_text_min_len = ocr_text_min_len
        self.clip_min_confidence = clip_min_confidence

    def process_prediction(
        self, prediction: PhotoPrediction, photo_num: int = 0
    ) -> str:
        caption = prediction.caption
        message = f"Image {photo_num}\n"

        caption_text = ""
        if not caption.alt_text and not caption_has_unknown(prediction) \
                and not has_labels(prediction, ['Font'], 1) \
                and caption.confidence >= self.clip_min_confidence:
            caption_text = caption.text.lower().capitalize() + "."
            confidence = round(caption.confidence * 100)
            caption_text = f"May show ({confidence}%): {caption_text}\n"
        elif caption.alt_text:
            caption_text = f"Alt text: {caption.text}\n"

        if has_labels(prediction, ['Font', 'Handwriting']):
            if len(prediction.ocr_text.text) >= self.ocr_text_min_len:
                caption_text += f"Сontains text:\n{prediction.ocr_text.text}"

        message += caption_text

        if prediction.labels:
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

    def __repr__(self):
        return generate_repr(
            self,
            [
                "ocr_text_min_len",
                "clip_min_confidence",
            ],
        )
