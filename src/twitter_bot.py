import time
import tweepy
import logging

from src.prediction import CaptionPredictor, load_pil_image
from src.utils import setup_logging
from src.settings import settings


logger = logging.getLogger(__name__)


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
        logger.info(f"New tweet id '{tweet.id}'")
    except tweepy.TweepError as error:
        logger.error(f"Raised Tweep error: {error}")
        raise error
    return tweet


def predict_and_post_captions(api, predictor, photo_urls, tweet_to_reply):
    text_lst = []

    # Generate caption for each photo
    for num, photo_url in enumerate(photo_urls):
        image = load_pil_image(photo_url)
        caption = predictor.get_captions(image)[0]
        num = f" {num + 1}" if len(photo_urls) > 1 else ""
        photo_caption_text = f"Photo{num} may show: {caption.capitalize()}."
        text_lst.append(photo_caption_text[: settings.twitter_char_limit])
        logger.info(f"Tweet '{tweet_to_reply.id}' - {photo_caption_text}")

    text = ""
    # Chunk large text into several tweets
    for num, line in enumerate(text_lst):
        if len(text) + len(line) >= settings.twitter_char_limit:
            tweet_to_reply = tweet_text_to(api, tweet_to_reply, text)
            text = ""
        if num:
            text += "\n"
        text += line
    if text:
        tweet_text_to(api, tweet_to_reply, text)


class TwitterMentionProcessor:
    def __init__(self, api, predictor, since_id="old", sleep=14.0):
        self.api = api
        self.predictor = predictor
        self.sleep = sleep
        self.me = api.me()
        self.since_id = self.init_since_id(since_id)

    def init_since_id(self, since_id: str) -> int:
        if since_id in {"old", "new"}:
            if since_id == "old":
                logger.info(f"Get id of last tweet by bot")
                tweets = self.api.user_timeline(id=self.me.id, count=1)
            else:
                logger.info(f"Get id of last tweet with bot mention")
                tweets = self.api.mentions_timeline(count=1)
            since_id = 1
            if tweets:
                since_id = tweets[0].id
        return int(since_id)

    def process_tweet(self, tweet):
        logger.info(f"Start processing tweet '{tweet.id}'")

        photo_urls = []
        if tweet_has_photo(tweet):
            photo_urls = get_photo_urls(tweet)
            logger.info(f"Tweet '{tweet.id}' has photos: {photo_urls}")
        elif tweet_is_reply(tweet):
            replied_tweet = self.api.get_status(
                tweet.in_reply_to_status_id, tweet_mode="extended"
            )
            if tweet_has_photo(replied_tweet):
                photo_urls = get_photo_urls(replied_tweet)
                logger.info(
                    f"Replied tweet '{replied_tweet.id}' has photos: {photo_urls}"
                )

        if photo_urls:
            predict_and_post_captions(self.api, self.predictor, photo_urls, tweet)
        logger.info(f"Finish processing tweet '{tweet.id}'")

    def process_mentions(self):
        logger.info(f"Retrieving mentions since_id '{self.since_id}'")
        for tweet in tweepy.Cursor(
            self.api.mentions_timeline, since_id=self.since_id, tweet_mode="extended"
        ).items():
            try:
                self.process_tweet(tweet)
                self.since_id = max(tweet.id, self.since_id)
            except BaseException as error:
                logger.info(f"Error while processing tweet '{tweet.id}': {error}")

    def process(self):
        logger.info(f"Starting with since_id: '{self.since_id}'")
        while True:
            start = time.time()
            self.process_mentions()
            sleep = max(0.0, self.sleep - time.time() + start)
            logger.info(f"Waiting {sleep} seconds")
            time.sleep(sleep)


if __name__ == "__main__":
    setup_logging(settings.log_level)

    auth = tweepy.OAuthHandler(settings.consumer_key, settings.consumer_secret)
    auth.set_access_token(settings.access_token, settings.access_token_secret)
    twitter_api = tweepy.API(
        auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True
    )
    twitter_api.verify_credentials()
    logger.info("Credentials verified")

    predictor_params = {
        "feature_checkpoint_path": settings.feature_checkpoint_path,
        "feature_config_path": settings.feature_config_path,
        "caption_checkpoint_path": settings.caption_checkpoint_path,
        "caption_config_path": settings.caption_config_path,
        "beam_size": 5,
        "sample_n": 1,
        "device": settings.device,
    }
    caption_predictor = CaptionPredictor(**predictor_params)
    logger.info(f"Predictor loaded with params: {predictor_params}")

    processor = TwitterMentionProcessor(
        twitter_api, caption_predictor, since_id=settings.since_id, sleep=14.0
    )
    processor.process()
