"""
A two-view sparse feature matching pipeline.

This model contains sub-models for each step:
    feature extraction, feature matching, outlier filtering, pose estimation.
Each step is optional, and the features or matches can be provided as input.
Default: SuperPoint with nearest neighbor matching.

Convention for the matches: m0[i] is the index of the keypoint in image 1
that corresponds to the keypoint i in image 0. m0[i] = -1 if i is unmatched.
"""

from omegaconf import OmegaConf

from . import get_model
from .base_model import BaseModel
import numpy as np
import numba
import math
import torch
import torch.nn.functional as F
from .extractors.dino_extractor import dino_feature_extractor_fix

to_ctr = OmegaConf.to_container  # convert DictConfig to dict


class TwoViewPipeline(BaseModel):
    default_conf = {
        "extractor": {
            "name": None,
            "trainable": False,
        },
        "matcher": {"name": None},
        "filter": {"name": None},
        "solver": {"name": None},
        "pixel_re": {"name": "liot"},
        "ground_truth": {"name": None},
        "allow_no_extract": False,
        "run_gt_in_forward": False,
    }
    required_data_keys = ["view0", "view1"]
    strict_conf = False  # need to pass new confs to children models
    components = [
        "extractor",
        "matcher",
        "filter",
        "solver",
        "ground_truth",
    ]

    def _init(self, conf):
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(to_ctr(conf.extractor))

        if conf.matcher.name:
            self.matcher = get_model(conf.matcher.name)(to_ctr(conf.matcher))

        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(to_ctr(conf.filter))

        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(to_ctr(conf.solver))

        if conf.ground_truth.name:
            self.ground_truth = get_model(conf.ground_truth.name)(
                to_ctr(conf.ground_truth)
            )
        # add pixel_relation extractor
        if conf.pixel_re.name:
            self.extract_re = True
            #self.sd_feature = sd_feature_extractor(conf.sd_feature)
            self.dino_feature = dino_feature_extractor_fix(conf.sd_feature)


    def extract_view(self, data, i):
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract
        if self.conf.extractor.name and not skip_extract:
            pred_i = {**pred_i, **self.extractor(data_i)}
        elif self.conf.extractor.name and not self.conf.allow_no_extract:
            pred_i = {**pred_i, **self.extractor({**data_i, **pred_i})}
        return pred_i
    
    
    def ex_sdfeature_view(self, data, pred_ex, i):
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        keypoint_i = pred_ex["keypoints"]
        #pred_i = {**pred_i, **self.sd_feature.extract_feature(data_i, keypoint_i)}
        pred_i = {**pred_i, **self.dino_feature.extract_feature(data_i, keypoint_i)}
        return pred_i

    def _forward(self, data):
        pred0 = self.extract_view(data, "0")
        pred1 = self.extract_view(data, "1")
        
        if (self.extract_re):
            extract0 = self.ex_sdfeature_view(data, pred0, "0")
            extract1 = self.ex_sdfeature_view(data, pred1, "1")
            pred = {
                **{k + "0": v for k, v in pred0.items()},
                **{k + "1": v for k, v in pred1.items()},
                **{k + "0": v for k, v in extract0.items()},
                **{k + "1": v for k, v in extract1.items()},
            }
            # pred["descriptors0"] = torch.cat([pred["descriptors0"], pred["relation0"]], dim=-1)
            # pred["descriptors1"] = torch.cat([pred["descriptors1"], pred["relation1"]], dim=-1)
            
        else:
            pred = {
                **{k + "0": v for k, v in pred0.items()},
                **{k + "1": v for k, v in pred1.items()},
            }
        # keypoint [18,512,2]; des [18,512,256]; score [18,512]
        if self.conf.matcher.name:
            pred = {**pred, **self.matcher({**data, **pred})}
        if self.conf.filter.name:
            pred = {**pred, **self.filter({**data, **pred})}
        if self.conf.solver.name:
            pred = {**pred, **self.solver({**data, **pred})}

        if self.conf.ground_truth.name and self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})
        return pred

    def loss(self, pred, data):
        losses = {}
        metrics = {}
        total = 0

        # get labels
        if self.conf.ground_truth.name and not self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

        for k in self.components:
            apply = True
            if "apply_loss" in self.conf[k].keys():
                apply = self.conf[k].apply_loss
            if self.conf[k].name and apply:
                try:
                    losses_, metrics_ = getattr(self, k).loss(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                losses = {**losses, **losses_}
                metrics = {**metrics, **metrics_}
                total = losses_["total"] + total
        return {**losses, "total": total}, metrics

import sys
sys.path.append("/home/yepeng_liu/code_python/third_repos/dift/")
from src.models.dift_sd import SDFeaturizer4Eval
from PIL import Image



def sample_keypoint_desc(keypoints, descriptors):
        """ Interpolate descriptors at keypoint locations """
        # 输入的尺寸keypoint[18*512*2], des:[18*256*96*96]; 输出是【18*4*512】
        
        b, c, h, w = descriptors.shape
        descriptors = descriptors / 255.0
        keypoints = keypoints.clone().float()

        keypoints /= h
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)

        args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
        descriptors = torch.nn.functional.grid_sample(
            descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)

        descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1)
        
        descriptors = descriptors.permute(0, 2, 1)
        return descriptors