from typing import Optional, List, Tuple
from google.cloud import vision
from shapely.geometry import Polygon

from src.pydantic_models import Label, OCRText
from src.utils import generate_repr, load_pil_image


def text_annot2poly(text_annot):
    return Polygon([(vertex.x, vertex.y) for vertex
                    in text_annot.bounding_poly.vertices])


def calculate_area_ratio(response, image_url):
    image = load_pil_image(image_url)
    poly = Polygon([
        (0, 0),
        (0, image.size[1]),
        (image.size[0], image.size[1]),
        (image.size[0], 0),
    ])
    image_area = poly.area
    for text_annot in response.text_annotations[1:]:
        poly = poly.difference(text_annot2poly(text_annot))
    area_ratio = (image_area - poly.area) / image_area
    return area_ratio


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
            text_annot = response.text_annotations[0]
            area_ratio = calculate_area_ratio(response, image_url)
            ocr_text = OCRText(text=text_annot.description,
                               locale=text_annot.locale,
                               area=area_ratio)
        else:
            ocr_text = OCRText(text="", locale="", area=0.0)

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
