import time
import tweepy
import logging

from src.settings import settings
from src.prediction import CaptionPredictor, load_pil_image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
predictor = CaptionPredictor(
    "/model_data/detectron_model.pth",
    "/model_data/detectron_model.yaml",
    "/model_data/model-best.pth",
    "/model_data/infos_trans12-best.pkl",
    beam_size=5,
    sample_n=1,
    device="cuda",
)


def check_mentions(api, since_id):
    logger.info("Retrieving mentions")
    new_since_id = since_id
    for tweet in tweepy.Cursor(api.mentions_timeline, since_id=since_id).items():
        new_since_id = max(tweet.id, new_since_id)
        if tweet.in_reply_to_status_id is not None:
            continue

        logger.info(f"Answering to {tweet.user.name}")

        if 'media' in tweet.entities:
            media = tweet.entities['media'][0]
            if media['type'] == 'photo':
                image = load_pil_image(media['media_url_https'])
                caption = predictor.get_captions(image)[0]

                try:
                    api.update_status(
                        status=caption,
                        in_reply_to_status_id=tweet.id,
                    )
                except tweepy.TweepError as error:
                    logging.error(f"Raised error: {error}")
                    if error.api_code != 187:
                        raise error
    return new_since_id


if __name__ == "__main__":
    auth = tweepy.OAuthHandler(settings.consumer_key, settings.consumer_secret)
    auth.set_access_token(settings.access_token, settings.access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    api.verify_credentials()

    since_id = 1
    while True:
        since_id = check_mentions(api, since_id)
        logger.info("Waiting...")
        time.sleep(15)
