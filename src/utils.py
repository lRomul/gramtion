import sys
import logging


def setup_logging(log_level: str = "INFO"):
    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setLevel(log_level)
    logging.basicConfig(
        format="[{asctime}][{levelname}]: {message}",
        style="{",
        level=log_level,
        handlers=[stdout],
    )
