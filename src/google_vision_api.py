from typing import List
from google.cloud import vision

from src.pydantic_models import Label


class GoogleVisionPredictor:
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()

    def get_labels(self, image_url) -> List[Label]:
        response = self.client.annotate_image(
            {
                "image": {"source": {"image_uri": image_url}},
                "features": [{"type_": vision.Feature.Type.LABEL_DETECTION}],
            }
        )
        labels = []
        for label_annotation in response.label_annotations:
            label = Label(name=label_annotation.description,
                          score=label_annotation.score)
            labels.append(label)
        return labels


if __name__ == "__main__":
    predictor = GoogleVisionPredictor()

    image_url = (
        "https://user-images.githubusercontent.com/"
        "11138870/96363040-6da16080-113a-11eb-83f7-3cdb65b62dbb.jpg"
    )
    print(predictor.get_labels(image_url))
