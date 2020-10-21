import re
from typing import Optional, Dict, List

from src.settings import settings


class CaptionProcessor:
    def __init__(self, replace_dict: Optional[Dict[str, str]] = None):
        if replace_dict is None:
            replace_dict = {
                "unk": "unknown",
            }
        self.replace_dict = replace_dict

    def process_caption(self, caption: str, num_text: str = "") -> str:
        caption = caption.lower()
        for key, value in self.replace_dict.items():
            caption = re.sub(r"\b{}\b".format(key), value, caption)
        caption = caption.capitalize()
        caption = f"Photo{num_text} may show: {caption}."
        caption = caption[: settings.twitter_char_limit]
        return caption

    def process_captions(self, captions: List[str]) -> List[str]:
        processed_captions = []
        for num, caption in enumerate(captions):
            num_text = f" {num + 1}" if len(captions) > 1 else ""
            caption = self.process_caption(caption, num_text=num_text)
            processed_captions.append(caption)
        return processed_captions
