import cv2
import torch
import numpy as np

from PIL import Image

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

import captioning.utils.misc
import captioning.models


class FeatureExtractor:
    def __init__(self, checkpoint_path, config_path, device="cpu"):
        self.device = torch.device(device)
        self.detection_model = self._build_detection_model(checkpoint_path, config_path)

    @torch.no_grad()
    def __call__(self, url):
        return self.get_detectron_features(url)

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
    def _image_transform(image_path):
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
    def get_detectron_features(self, image_path):
        im, im_scale = self._image_transform(image_path)
        img_tensor, im_scales = [im], [im_scale]
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

    def get_captions(self, image_path):
        image_feature = self.feature_extractor(image_path)
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
        return captions


if __name__ == "__main__":
    predictor = CaptionPredictor(
        "model_data/detectron_model.pth",
        "model_data/detectron_model.yaml",
        "model_data/model-best.pth",
        "model_data/infos_trans12-best.pkl",
        beam_size=5,
        sample_n=5,
        device="cpu",
    )

    print(predictor.get_captions("test.jpg"))
