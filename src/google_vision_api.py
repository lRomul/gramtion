from typing import Optional, List, Tuple
from google.cloud import vision

from src.pydantic_models import Label, OCRText
from src.utils import generate_repr


class GoogleVisionPredictor:
    def __init__(self, score_threshold: float = 0.0, max_number: Optional[int] = None):
        self.score_threshold = score_threshold
        self.max_number = max_number
        self.client = vision.ImageAnnotatorClient()

    def predict(self, image_url) -> Tuple[List[Label], OCRText]:
        response = self.client.annotate_image(
            {
                "image": {"source": {"image_uri": image_url}},
                "features": [
                    {"type_": vision.Feature.Type.LABEL_DETECTION},
                    {"type_": vision.Feature.Type.DOCUMENT_TEXT_DETECTION}
                ],
            }
        )

        labels = []
        for label_annotation in response.label_annotations:
            label = Label(
                name=label_annotation.description, score=label_annotation.score
            )
            labels.append(label)
        if self.max_number is not None:
            labels = labels[: self.max_number]

        if response.text_annotations:
            ocr_text = response.text_annotations[0]
            ocr_text = OCRText(text=ocr_text.description, locale=ocr_text.locale)
        else:
            ocr_text = OCRText(text="", locale="")

        return labels, ocr_text

    def __repr__(self):
        return generate_repr(
            self,
            [
                "score_threshold",
                "max_number",
            ],
        )


if __name__ == "__main__":
    predictor = GoogleVisionPredictor(score_threshold=0.8, max_number=5)

    image_url = (
        "https://user-images.githubusercontent.com/"
        "11138870/96363040-6da16080-113a-11eb-83f7-3cdb65b62dbb.jpg"
    )
    print(predictor)
    print(predictor.predict(image_url))
