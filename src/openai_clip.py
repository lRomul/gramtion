import torch
import clip
from PIL import Image
from typing import List

from src.image_captioning import load_pil_image
from src.pydantic_models import Caption
from src.utils import generate_repr


class ClipPredictor:
    def __init__(self,
                 clip_model_name="ViT-B/32",
                 device="cpu"):
        self.clip_model_name = clip_model_name
        self.device = device
        self.model, self.preprocess = clip.load(clip_model_name,
                                                device=device)

    @torch.no_grad()
    def match_best_caption(self, image: Image, captions: List[Caption]):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize([c.text for c in captions]).to(self.device)

        logits_per_image, logits_per_text = self.model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy().ravel()
        caption = captions[probs.argmax()].copy()
        caption.confidence = float(probs.max())
        return caption

    def __repr__(self):
        return generate_repr(
            self,
            [
                "clip_model_name",
                "device",
            ],
        )


if __name__ == "__main__":
    from src.settings import settings

    predictor = ClipPredictor(
        clip_model_name=settings.clip_model_name,
        device=settings.device
    )

    image = load_pil_image(
        "https://user-images.githubusercontent.com/"
        "11138870/96363040-6da16080-113a-11eb-83f7-3cdb65b62dbb.jpg"
    )
    print(predictor)
    print(predictor.match_best_caption(
        image,
        [Caption(text=c) for c in ["a dog", "a cat"]])
    )
