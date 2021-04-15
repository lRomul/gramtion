import time
import tweepy
import logging
from typing import List, Optional
from queue import Queue
from threading import Thread

from src.image_captioning import CaptionPredictor, load_pil_image
from src.google_vision_api import GoogleVisionPredictor
from src.prediction_processing import PredictionProcessor
from src.pydantic_models import Photo, Caption, PhotoPrediction
from src.web_server import run_web_server, tweets_queue
from src.utils import setup_logging
from src.settings import settings


logger = logging.getLogger(__name__)


def tweet_has_photo(tweet) -> bool:
    if "media" in tweet.entities:
        media = tweet.entities["media"][0]
        if media["type"] == "photo":
            return True
    return False


def get_photos(tweet) -> List[Photo]:
    photos_lst = []
    for media in tweet.extended_entities["media"]:
        if media["type"] == "photo":
            caption = None
            if media["ext_alt_text"] is not None:
                caption = Caption(text=media["ext_alt_text"], alt_text=True)
            photo = Photo(url=media["media_url_https"], caption=caption)
            photos_lst.append(photo)
    return photos_lst


def tweet_is_reply(tweet) -> bool:
    return tweet.in_reply_to_status_id is not None


def tweet_text_to(api, tweet, text: str):
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


def split_text_to_tweets(texts):
    tweet_texts = []
    text = ""
    # Chunk large text into several tweets
    for num, caption in enumerate(texts):
        if len(text) + len(caption) >= settings.twitter_char_limit:
            tweet_texts.append(text)
            text = ""
        if num:
            text += "\n\n"
        text += caption
    if text:
        tweet_texts.append(text)
    return tweet_texts


class TwitterMentionProcessor:
    def __init__(
        self,
        api,
        caption_predictor: CaptionPredictor,
        google_predictor: GoogleVisionPredictor,
        since_id: str = "old",
        sleep: float = 14.0,
        tweets_queue: Optional[Queue] = None
    ):
        self.api = api
        self.caption_predictor = caption_predictor
        self.google_predictor = google_predictor
        self.sleep = sleep
        self.tweets_queue = tweets_queue
        self.me = api.me()
        self.caption_processor = PredictionProcessor()
        self.since_id = self.init_since_id(since_id)
        self._thread = None
        self._stopped = True

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

    def process_photos(self, photos: List[Photo]) -> List[str]:
        predictions = []

        # Generate caption for each photo
        for photo in photos:
            caption = photo.caption
            if caption is None:
                image = load_pil_image(photo.url)
                caption = self.caption_predictor.get_captions(image)[0]

            labels, ocr_text = self.google_predictor.predict(photo.url)
            predictions.append(PhotoPrediction(caption=caption,
                                               labels=labels,
                                               ocr_text=ocr_text))

        messages = self.caption_processor.predictions_to_messages(predictions)
        tweet_texts = split_text_to_tweets(messages)
        return tweet_texts

    def process_tweet(self, tweet, post=True):
        logger.info(f"Start processing tweet '{tweet.id}'")

        photos = []
        if tweet_has_photo(tweet):
            photos = get_photos(tweet)
            logger.info(f"Tweet '{tweet.id}' has photos: {photos}")
        elif tweet_is_reply(tweet):
            replied_tweet = self.api.get_status(
                tweet.in_reply_to_status_id,
                tweet_mode="extended",
                include_ext_alt_text=True,
            )
            if tweet_has_photo(replied_tweet):
                photos = get_photos(replied_tweet)
                logger.info(f"Replied tweet '{replied_tweet.id}' has photos: {photos}")

        tweet_texts = []
        if photos:
            tweet_texts = self.process_photos(photos)

        if tweet_texts and post:
            for text in tweet_texts:
                tweet = tweet_text_to(self.api, tweet, text)

        logger.info(f"Finish processing, tweets: {tweet_texts}")
        return tweet_texts

    def process_mentions(self):
        logger.info(f"Retrieving mentions since_id '{self.since_id}'")
        for tweet in tweepy.Cursor(
            self.api.mentions_timeline,
            since_id=self.since_id,
            tweet_mode="extended",
            include_ext_alt_text=True,
        ).items():
            try:
                self.since_id = max(tweet.id, self.since_id)
                self.process_tweet(tweet, post=True)
            except BaseException as error:
                logger.error(f"Error while processing tweet '{tweet.id}': {error}")

    def process_tweet_id(self, tweet_id: int):
        try:
            tweet = self.api.get_status(
                tweet_id,
                tweet_mode="extended",
                include_ext_alt_text=True
            )
            self.process_tweet(tweet)
        except BaseException as error:
            logger.error(f"Error while processing tweet id '{tweet_id}': {error}")

    def _run_processing(self):
        logger.info(f"Starting with since_id: '{self.since_id}'")
        self._stopped = False
        prev_time = 0
        while not self._stopped:
            now_time = time.time()
            if now_time - prev_time > self.sleep:
                prev_time = now_time
                self.process_mentions()

            if self.tweets_queue is not None:
                while not self.tweets_queue.empty():
                    self.process_tweet_id(self.tweets_queue.get())

            sleep = max(0.0, self.sleep - time.time() + prev_time)
            sleep = min(1.0, sleep)
            time.sleep(sleep)

    def start(self):
        if self._thread is None:
            self._thread = Thread(target=self._run_processing)
            self._thread.daemon = True
            self._thread.start()

    def stop(self):
        self._stopped = True

    def wait(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None


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
    logger.info(f"Caption predictor loaded: {caption_predictor}")
    google_predictor = GoogleVisionPredictor(score_threshold=0.8, max_number=5)
    logger.info(f"Google predictor loaded: {google_predictor}")

    processor = TwitterMentionProcessor(
        twitter_api,
        caption_predictor,
        google_predictor,
        since_id=settings.since_id,
        sleep=14.0,
        tweets_queue=tweets_queue
    )
    processor.start()
    run_web_server()

    processor.stop()
    processor.wait()
