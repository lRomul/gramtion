import logging
from queue import Queue

from bottle import run, route

logger = logging.getLogger(__name__)
process_queue = Queue()


@route('/process_tweet/<tweet_id>', method='POST')
def process_tweet(tweet_id: int):
    process_queue.put(tweet_id)
    logger.info(f"Put tweet id '{tweet_id}' to process queue '{process_queue=}'")


def run_web_server():
    run(host="0.0.0.0", port=7518)
