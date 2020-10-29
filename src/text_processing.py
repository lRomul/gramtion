import re
from typing import Optional, Dict, List

from src.utils import Caption
from src.settings import settings


class CaptionProcessor:
    def __init__(self, replace_dict: Optional[Dict[str, str]] = None):
        if replace_dict is None:
            replace_dict = {
                "unk": "unknown",
            }
        self.replace_dict = replace_dict

    def process_caption(self, caption: Caption, num_text: str = "") -> str:
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

    def process_captions(self, captions: List[Caption]) -> List[str]:
        processed_captions = []
        for num, caption in enumerate(captions):
            num_text = f"Photo {num + 1}" if len(captions) > 1 else "Photo"
            caption = self.process_caption(caption, num_text=num_text)
            processed_captions.append(caption)
        return processed_captions
