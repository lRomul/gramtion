import cv2
import torch
import requests
import numpy as np
from typing import List

from PIL import Image

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

import captioning.utils.misc
import captioning.models

from src.pydantic_models import Caption


def load_pil_image(path):
    if path.startswith("http"):
        path = requests.get(path, stream=True).raw
    else:
        path = path
    image = Image.open(path).convert("RGB")
    return image


def image_transform(image):
    image = np.array(image).astype(np.float32)
    image = image[:, :, ::-1]
    image -= np.array([102.9801, 115.9465, 122.7717])
    im_shape = image.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    image_scale = float(800) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(image_scale * im_size_max) > 1333:
        image_scale = float(1333) / float(im_size_max)
    image = cv2.resize(
        image,
        None,
        None,
        fx=image_scale,
        fy=image_scale,
        interpolation=cv2.INTER_LINEAR,
    )
    img = torch.from_numpy(image).permute(2, 0, 1)
    return img, image_scale


class FeatureExtractor:
    def __init__(self, checkpoint_path, config_path, device="cpu"):
        self.device = torch.device(device)
        self.detection_model = self._build_detection_model(checkpoint_path, config_path)

    @torch.no_grad()
    def __call__(self, image):
        return self.get_detectron_features(image)

    def _build_detection_model(self, checkpoint_path, config_path):
        cfg.merge_from_file(config_path)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _process_feature_extraction(output, im_scales, feat_name="fc6"):
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feat_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]

            max_conf = torch.zeros((scores.shape[0])).to(cur_device)

            for cls_ind in range(1, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep]
                )

            keep_boxes = torch.argsort(max_conf, descending=True)[:100]
            feat_list.append(feats[i][keep_boxes])
        return feat_list

    @torch.no_grad()
    def get_detectron_features(self, image):
        img, img_scale = image_transform(image)
        img_tensor, im_scales = [img], [img_scale]
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to(self.device)
        output = self.detection_model(current_img_list)
        feat_list = self._process_feature_extraction(output, im_scales, "fc6")
        return feat_list[0]


class CaptionPredictor:
    def __init__(
        self,
        feature_checkpoint_path,
        feature_config_path,
        caption_checkpoint_path,
        caption_config_path,
        beam_size=5,
        sample_n=5,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.beam_size = beam_size
        self.sample_n = sample_n
        self.feature_extractor = FeatureExtractor(
            checkpoint_path=feature_checkpoint_path,
            config_path=feature_config_path,
            device=device,
        )
        self.caption_model = self._build_caption_model(
            caption_checkpoint_path, caption_config_path
        )

    def _build_caption_model(self, checkpoint_path, config_path):
        infos = captioning.utils.misc.pickle_load(open(config_path, "rb"))
        infos["opt"].vocab = infos["vocab"]
        caption_model = captioning.models.setup(infos["opt"])
        caption_model.to(self.device)
        caption_model.load_state_dict(torch.load(checkpoint_path))
        return caption_model

    def get_captions(self, image: Image) -> List[Caption]:
        image_feature = self.feature_extractor(image)
        sequence = self.caption_model(
            image_feature.mean(0)[None],
            image_feature[None],
            mode="sample",
            opt={
                "beam_size": self.beam_size,
                "sample_method": "beam_search",
                "sample_n": self.sample_n,
            },
        )[0]
        captions = self.caption_model.decode_sequence(sequence)
        captions = [Caption(text=capt) for capt in captions]
        return captions


if __name__ == "__main__":
    from src.settings import settings

    predictor = CaptionPredictor(
        settings.feature_checkpoint_path,
        settings.feature_config_path,
        settings.caption_checkpoint_path,
        settings.caption_config_path,
        beam_size=5,
        sample_n=5,
        device=settings.device,
    )

    image = load_pil_image(
        "https://user-images.githubusercontent.com/"
        "11138870/96363040-6da16080-113a-11eb-83f7-3cdb65b62dbb.jpg"
    )
    print(predictor.get_captions(image))
