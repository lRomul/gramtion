import torch
from PIL import Image
from typing import List

import virtex
from virtex import model_zoo
from virtex.data.transforms import DEFAULT_IMAGE_TRANSFORM

from src.pydantic_models import Caption
from src.utils import generate_repr, load_pil_image


class CaptionPredictor:
    def __init__(
        self,
        config_path: str,
        device="cpu",
    ):
        self.config_path = config_path
        self.device = torch.device(device)
        self.model = model_zoo.get(config_path, pretrained=True)
        self.model = self.model.to(device=self.device)

    def get_captions(self, image: Image) -> List[Caption]:
        pass
        return []

    def __repr__(self):
        return generate_repr(
            self,
            [
                "config_path",
                "device",
            ],
        )


if __name__ == "__main__":
    from src.settings import settings

    predictor = CaptionPredictor(
        config_path=settings.caption_config_path,
        device=settings.device,
    )

    image = load_pil_image(
        "https://user-images.githubusercontent.com/"
        "11138870/96363040-6da16080-113a-11eb-83f7-3cdb65b62dbb.jpg"
    )
    print(predictor)
    print(predictor.get_captions(image))
