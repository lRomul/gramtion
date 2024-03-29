import sys
import logging
import requests
from PIL import Image


def load_pil_image(path):
    if path.startswith("http"):
        path = requests.get(path, stream=True).raw
    else:
        path = path
    image = Image.open(path).convert("RGB")
    return image


def setup_logging(log_level: str = "INFO"):
    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setLevel(log_level)
    logging.basicConfig(
        format="[{asctime}][{levelname}]: {message}",
        style="{",
        level=log_level,
        handlers=[stdout],
    )


def generate_repr(obj: object, attrs):
    lines = [f"{obj.__class__.__name__}("]
    for attr in attrs:
        attr_line = f"    {attr}={getattr(obj, attr, None)},"
        lines.append(attr_line)
    lines.append(")")
    return "\n".join(lines)
