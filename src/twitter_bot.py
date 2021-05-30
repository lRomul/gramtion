import time
import tweepy
import logging
from typing import List, Optional
from queue import Queue
from threading import Thread

from src.image_captioning import CaptionPredictor
from src.openai_clip import ClipPredictor
from src.google_vision_api import GoogleVisionPredictor
from src.prediction_processing import PredictionProcessor
from src.pydantic_models import Photo, Caption, PhotoPrediction
from src.web_server import run_web_server, tweets_queue
from src.utils import setup_logging, load_pil_image
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


def get_tweet(api, tweet_id):
    tweet = api.get_status(
        tweet_id,
        tweet_mode="extended",
        include_ext_alt_text=True
    )
    return tweet


def tweet_is_reply(tweet) -> bool:
    return tweet.in_reply_to_status_id is not None


def tweet_text_to(api, tweet, text: str, resend_long: bool = True):
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
        if resend_long and error.api_code == 186:
            logger.info("Split too long tweet and resend")
            split_index = len(text) // 2
            tweet = tweet_text_to(api, tweet, text[:split_index], resend_long=False)
            tweet = tweet_text_to(api, tweet, text[split_index:], resend_long=False)
        else:
            raise error
    return tweet


def merge_small_messages(messages):
    tweet_texts = []
    text = ""
    for num, message in enumerate(messages):
        carriage_len = 2 if num else 0
        if len(text) + len(message) + carriage_len >= settings.twitter_char_limit:
            tweet_texts.append(text)
            text = ""
        if num:
            text += "\n\n"
        text += message
    if text:
        tweet_texts.append(text)
    return [t for t in tweet_texts if t]


class TwitterMentionProcessor:
    def __init__(
        self,
        api,
        caption_predictor: CaptionPredictor,
        clip_predictor: ClipPredictor,
        google_predictor: GoogleVisionPredictor,
        caption_processor: PredictionProcessor,
        since_id: str = "old",
        sleep: float = 14.0,
        tweets_queue: Optional[Queue] = None,
        debug: bool = False
    ):
        self.api = api
        self.caption_predictor = caption_predictor
        self.clip_predictor = clip_predictor
        self.google_predictor = google_predictor
        self.sleep = sleep
        self.tweets_queue = tweets_queue
        self.debug = debug
        self.me = api.me()
        self.caption_processor = caption_processor
        self.since_id = self.init_since_id(since_id)
        self._thread = None
        self._stopped = True

    def init_since_id(self, since_id: str) -> int:
        if since_id in {"old", "new"}:
            if since_id == "old":
                logger.info("Get id of last tweet by bot")
                tweets = self.api.user_timeline(id=self.me.id, count=1)
            else:
                logger.info("Get id of last tweet with bot mention")
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
                captions = self.caption_predictor.get_captions(image)
                caption = self.clip_predictor.match_best_caption(image, captions)

            labels, ocr_text = self.google_predictor.predict(photo.url)
            predictions.append(PhotoPrediction(caption=caption,
                                               labels=labels,
                                               ocr_text=ocr_text))
        logger.info(f"Photo predictions: {predictions}")

        messages = self.caption_processor.predictions_to_messages(predictions)
        tweet_texts = merge_small_messages(messages)
        return tweet_texts

    def fetch_photos(self, tweet):
        photos = []
        if tweet_has_photo(tweet):
            photos = get_photos(tweet)
            logger.info(f"Tweet '{tweet.id}' has photos: {photos}")
        elif tweet_is_reply(tweet):
            replied_tweet = get_tweet(self.api, tweet.in_reply_to_status_id)
            if tweet_has_photo(replied_tweet):
                photos = get_photos(replied_tweet)
                logger.info(f"Replied tweet '{replied_tweet.id}' has photos: {photos}")
            elif replied_tweet.is_quote_status:
                quoted_tweet = get_tweet(self.api, replied_tweet.quoted_status_id)
                if tweet_has_photo(quoted_tweet):
                    photos = get_photos(quoted_tweet)
                    logger.info(f"Replied quoted tweet '{quoted_tweet.id}' "
                                f"has photos: {photos}")
        elif tweet.is_quote_status:
            quoted_tweet = get_tweet(self.api, tweet.quoted_status_id)
            if tweet_has_photo(quoted_tweet):
                photos = get_photos(quoted_tweet)
                logger.info(f"Quoted tweet '{quoted_tweet.id}' has photos: {photos}")
        return photos

    def post_tweets(self, tweet_texts, tweet, post_attempts: int = 3):
        for text in tweet_texts:
            for attempt in range(post_attempts):
                try:
                    tweet = tweet_text_to(self.api, tweet, text)
                    break  # exit attempt loop if error not caused
                except tweepy.TweepError as error:
                    logger.error(f"On {attempt + 1} attempt to post a tweet"
                                 f" caused error: {error}")
                    time.sleep(1.0)

    def process_tweet(self, tweet):
        logger.info(f"Start processing tweet '{tweet.id}'")

        photos = self.fetch_photos(tweet)
        tweet_texts = []
        if photos:
            tweet_texts = self.process_photos(photos)
            if not self.debug:
                self.post_tweets(tweet_texts, tweet)

        logger.info(f"Finish processing, tweets: {tweet_texts}")
        return tweet_texts

    def process_mentions(self):
        logger.info(f"Retrieving mentions since_id '{self.since_id}'")
        try:
            for tweet in tweepy.Cursor(
                self.api.mentions_timeline,
                since_id=self.since_id,
                tweet_mode="extended",
                include_ext_alt_text=True,
            ).items():
                try:
                    self.since_id = max(tweet.id, self.since_id)
                    self.process_tweet(tweet)
                except BaseException as error:
                    logger.error(f"Error while processing tweet '{tweet.id}': {error}")
        except BaseException as error:
            logger.error(f"Error while mentions processing: {error}")

    def process_tweet_id(self, tweet_id: int):
        try:
            tweet = get_tweet(self.api, tweet_id)
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
        "beam_size": settings.caption_beam_size,
        "sample_n": settings.caption_sample_n,
        "device": settings.device,
    }
    caption_predictor = CaptionPredictor(**predictor_params)
    logger.info(f"Caption predictor loaded: {caption_predictor}")

    google_predictor = GoogleVisionPredictor(
        score_threshold=settings.label_score_threshold,
        max_number=settings.label_max_number
    )
    logger.info(f"Google predictor loaded: {google_predictor}")

    clip_predictor = ClipPredictor(
        clip_model_name=settings.clip_model_name,
        device=settings.device
    )
    logger.info(f"Clip predictor loaded: {clip_predictor}")

    caption_processor = PredictionProcessor(
        clip_min_confidence=settings.clip_min_confidence_for_caption,
        max_text_area_for_caption=settings.max_text_area_for_caption,
        min_text_area_for_ocr=settings.min_text_area_for_ocr
    )
    logger.info(f"Prediction processor created: {caption_processor}")

    processor = TwitterMentionProcessor(
        twitter_api,
        caption_predictor,
        clip_predictor,
        google_predictor,
        caption_processor,
        since_id=settings.since_id,
        sleep=settings.mention_loop_sleep,
        tweets_queue=tweets_queue,
        debug=settings.debug
    )
    processor.start()
    run_web_server()

    processor.stop()
    processor.wait()
