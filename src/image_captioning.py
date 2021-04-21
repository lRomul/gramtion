import torch
import numpy as np
from PIL import Image
from typing import List

from virtex import model_zoo
from virtex.config import Config
from virtex.factories import TokenizerFactory
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

        config = Config("/virtex/configs/" + config_path)
        self.tokenizer = TokenizerFactory.from_config(config)

    def get_captions(self, image: Image) -> List[Caption]:
        image = np.array(image)
        image = DEFAULT_IMAGE_TRANSFORM(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))[np.newaxis]

        image = torch.tensor(image, dtype=torch.float, device=self.device)

        with torch.no_grad():
            output_dict = self.model({"image": image})

        caption = output_dict["predictions"][0]
        caption = self.tokenizer.decode(caption.tolist())
        caption = Caption(text=caption)
        return [caption]

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
