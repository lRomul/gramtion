import tweepy
import logging

from pydantic import BaseModel

from src.utils import save_json, load_json


logger = logging.getLogger(__name__)


class ProcessorState(BaseModel):
    since_id: int = 1


def init_state(api, state_path):
    try:
        if state_path:
            state_dict = load_json(state_path)
            state = ProcessorState.parse_raw(state_dict)
            return state
    except FileNotFoundError as error:
        logger.info(f"State file is not found: {error}")
    except BaseException as error:
        logger.error(f"Read state error: {error}")

    # Get id of last tweet with bot mention
    cursor = tweepy.Cursor(api.mentions_timeline, count=1)
    since_id = next(cursor.items()).id
    state = ProcessorState(since_id=since_id)
    save_state(state, state_path)
    return state


def save_state(state, state_path):
    if state_path:
        save_json(state.json(), state_path)
