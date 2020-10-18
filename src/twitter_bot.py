import time
import tweepy
import logging

from src.settings import settings
from src.prediction import CaptionPredictor, load_pil_image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def tweet_has_photo(tweet):
    if "media" in tweet.entities:
        media = tweet.entities["media"][0]
        if media["type"] == "photo":
            return True
    return False


def get_photo_urls(tweet):
    photo_urls = []
    for media in tweet.extended_entities["media"]:
        if media["type"] == "photo":
            photo_urls.append(media["media_url_https"])
    return photo_urls


def tweet_is_reply(tweet):
    return tweet.in_reply_to_status_id is not None


def tweet_text_to(api, tweet, text):
    logger.info(f"Tweet to {tweet.id}: {text}")
    try:
        tweet = api.update_status(
            status=text,
            in_reply_to_status_id=tweet.id,
            auto_populate_reply_metadata=True,
        )
    except tweepy.TweepError as error:
        logger.error(f"Raised Tweep error: {error}")
        raise error
    return tweet


def predict_and_post_captions(api, predictor, photo_urls, tweet_to_reply, mention_name):
    text = []
    if mention_name:
        text.append(f"@{mention_name},")
        logger.info(f"Add mention of user '{mention_name}'")

    for num, photo_url in enumerate(photo_urls):
        image = load_pil_image(photo_url)
        caption = predictor.get_captions(image)[0]
        num = f" {num + 1}" if len(photo_urls) > 1 else ""
        photo_caption_text = f"The photo{num} may show: {caption.capitalize()}."
        text.append(photo_caption_text)
        logger.info(f"Tweet '{tweet_to_reply.id}' - {photo_caption_text}")

    text = "\n".join(text)
    tweet_text_to(api, tweet_to_reply, text)


def process_tweet(api, predictor, tweet):
    logger.info(f"Start processing tweet '{tweet.id}'")
    if tweet.user.id == api.me().id:
        logger.info(f"Skip tweet by me '{tweet.user.id}'")
        return

    mention_name = ""
    photo_urls = []
    if tweet_has_photo(tweet):
        photo_urls = get_photo_urls(tweet)
        logger.info(f"Tweet '{tweet.id}' has photos: {photo_urls}")
    elif tweet_is_reply(tweet):
        mention_name = tweet.user.screen_name
        tweet = api.get_status(tweet.in_reply_to_status_id)
        if tweet_has_photo(tweet):
            photo_urls = get_photo_urls(tweet)
            logger.info(f"Replied tweet '{tweet.id}' has photos: {photo_urls}")

    if photo_urls:
        predict_and_post_captions(api, predictor, photo_urls, tweet, mention_name)
    logger.info(f"Finish processing tweet '{tweet.id}'")


def check_mentions(api, predictor, since_id):
    logger.info(f"Retrieving mentions since_id '{since_id}'")
    new_since_id = since_id
    for tweet in tweepy.Cursor(api.mentions_timeline, since_id=since_id).items():
        new_since_id = max(tweet.id, new_since_id)
        try:
            process_tweet(api, predictor, tweet)
        except BaseException as error:
            logger.info(f"Error while processing tweet '{tweet.id}': {error}")
    return new_since_id


if __name__ == "__main__":
    auth = tweepy.OAuthHandler(settings.consumer_key, settings.consumer_secret)
    auth.set_access_token(settings.access_token, settings.access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    api.verify_credentials()
    logger.info("Credentials verified")

    predictor = CaptionPredictor(
        settings.feature_checkpoint_path,
        settings.feature_config_path,
        settings.caption_checkpoint_path,
        settings.caption_config_path,
        beam_size=5,
        sample_n=1,
        device="cuda",
    )
    logger.info("Predictor loaded")

    since_id = 1
    logger.info(f"Starting with since_id: '{since_id}'")
    while True:
        since_id = check_mentions(api, predictor, since_id)
        logger.info("Waiting...")
        time.sleep(15)
