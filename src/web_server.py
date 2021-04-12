import logging
from queue import Queue

from bottle import run, route, response

logger = logging.getLogger(__name__)
tweets_queue = Queue()


@route('/process_tweet/<tweet_id>', method='POST')
def process_tweet(tweet_id):

    if tweet_id.isdigit():
        tweet_id = int(tweet_id)
    else:
        response.status = 400
        return 'Validation is not passed, tweet_id must be integer'

    tweets_queue.put(tweet_id)
    logger.info(f"Put tweet id '{tweet_id}' to queue")


def run_web_server():
    logger.info("Start HTTP server")
    run(host="0.0.0.0", port=7518)
    logger.info("HTTP server started")


if __name__ == "__main__":
    run_web_server()
