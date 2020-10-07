import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd

import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from IPython.display import display, HTML, clear_output
from ipywidgets import widgets, Layout
from io import BytesIO

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

import captioning
import captioning.utils.misc
import captioning.models


class FeatureExtractor:
    def __init__(self, checkpoint_path, config_path, device="cpu"):
        self.device = torch.device(device)
        self.detection_model = self._build_detection_model(checkpoint_path, config_path)

    def __call__(self, url):
        with torch.no_grad():
            detectron_features = self.get_detectron_features(url)

        return detectron_features

    def _build_detection_model(self, checkpoint_path, config_path):
        cfg.merge_from_file(config_path)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to(self.device)
        model.eval()
        return model

    def _image_transform(self, image_path):
        img = Image.open(image_path).convert("RGB")
        im = np.array(img).astype(np.float32)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(800) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > 1333:
            im_scale = float(1333) / float(im_size_max)
        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)
        return img, im_scale

    def _process_feature_extraction(self, output, im_scales, feat_name="fc6"):
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

    def get_detectron_features(self, image_path):
        im, im_scale = self._image_transform(image_path)
        img_tensor, im_scales = [im], [im_scale]
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to(self.device)
        with torch.no_grad():
            output = self.detection_model(current_img_list)
        feat_list = self._process_feature_extraction(output, im_scales, "fc6")
        return feat_list[0]


if __name__ == "__main__":
    device = "cpu"
    feature_extractor = FeatureExtractor(
        checkpoint_path="model_data/detectron_model.pth",
        config_path="model_data/detectron_model.yaml",
        device=device,
    )

    infos = captioning.utils.misc.pickle_load(
        open("model_data/infos_trans12-best.pkl", "rb")
    )
    infos["opt"].vocab = infos["vocab"]

    model = captioning.models.setup(infos["opt"])
    model.to(device)
    model.load_state_dict(torch.load("model_data/model-best.pth"))

    def get_captions(img_feature):
        # Return the 5 captions from beam serach with beam size 5
        return model.decode_sequence(
            model(
                img_feature.mean(0)[None],
                img_feature[None],
                mode="sample",
                opt={"beam_size": 5, "sample_method": "beam_search", "sample_n": 5},
            )[0]
        )

    captions = get_captions(feature_extractor('test.png'))
    print(captions)
