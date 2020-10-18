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


def tweet_has_photo(tweet):
    if 'media' in tweet.entities:
        media = tweet.entities['media'][0]
        if media['type'] == 'photo':
            return True
    return False


def get_photo_urls(tweet):
    photo_urls = []
    for media in tweet.extended_entities['media']:
        if media['type'] == 'photo':
            photo_urls.append(media['media_url_https'])
    return photo_urls


def tweet_is_reply(tweet):
    return tweet.in_reply_to_status_id is not None


def predict_and_post_captions(photo_urls, tweet_to_reply, mention_name):
    text = []
    if mention_name:
        text.append(f"@{mention_name},")

    for num, photo_url in enumerate(photo_urls):
        image = load_pil_image(photo_url)
        caption = predictor.get_captions(image)[0]
        num = f" {num + 1}" if len(photo_urls) > 1 else ""
        text.append(f"The photo{num} may show: {caption.capitalize()}.")

    text = "\n".join(text)
    logger.info(f"Tweet to {tweet_to_reply.id}: {text}")
    try:
        api.update_status(
            status=text,
            in_reply_to_status_id=tweet_to_reply.id,
            auto_populate_reply_metadata=True
        )
    except tweepy.TweepError as error:
        logger.error(f"Raised error: {error}")
        if error.api_code != 187:
            raise error


def check_mentions(api, since_id):
    logger.info("Retrieving mentions")
    new_since_id = since_id
    for tweet in tweepy.Cursor(api.mentions_timeline, since_id=since_id).items():
        new_since_id = max(tweet.id, new_since_id)
        if tweet.user.id == api.me().id:
            continue

        mention_name = ""
        photo_urls = []
        if tweet_has_photo(tweet):
            photo_urls = get_photo_urls(tweet)
        elif tweet_is_reply(tweet):
            mention_name = tweet.user.screen_name
            tweet = api.get_status(tweet.in_reply_to_status_id)
            if tweet_has_photo(tweet):
                photo_urls = get_photo_urls(tweet)

        if photo_urls:
            predict_and_post_captions(photo_urls, tweet, mention_name)
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
