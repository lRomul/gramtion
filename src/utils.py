import sys
import json
import logging


def setup_logging(log_level: str = "INFO"):
    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setLevel(log_level)
    logging.basicConfig(
        format="[{asctime}][{levelname}] - {name}: {message}",
        style="{",
        level=log_level,
        handlers=[stdout],
    )


def load_json(file_path):
    with open(file_path, "r") as file:
        dictionary = json.load(file)
    return dictionary


def save_json(dictionary, file_path):
    with open(file_path, "w") as file:
        json.dump(dictionary, file)
