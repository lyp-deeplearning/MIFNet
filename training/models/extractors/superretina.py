"""
superRetina的简单版本，只包含推理部分进行lightglue训练的时候前端推理
"""

import random
from ..base_model import BaseModel
from torch.nn import functional as F
import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple
import cv2




def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores

def pad_to_length(
    x,
    length: int,
    pad_dim: int = -2,
    mode: str = "zeros",  # zeros, ones, random, random_c
    bounds: Tuple[int] = (None, None),
):
    shape = list(x.shape)
    d = x.shape[pad_dim]
    assert d <= length
    if d == length:
        return x
    shape[pad_dim] = length - d
   

    low, high = bounds

    if mode == "zeros":
        xn = torch.zeros(*shape, device=x.device, dtype=x.dtype)
    elif mode == "ones":
        xn = torch.ones(*shape, device=x.device, dtype=x.dtype)
    elif mode == "random":
        low = low if low is not None else x.min()
        high = high if high is not None else x.max()
        xn = torch.empty(*shape, device=x.device).uniform_(low, high)
    elif mode == "random_c":
        low, high = bounds  # we use the bounds as fallback for empty seq.
        xn = torch.cat(
            [
                torch.empty(*shape[:-1], 1, device=x.device).uniform_(
                    x[..., i].min() if d > 0 else low,
                    x[..., i].max() if d > 0 else high,
                )
                for i in range(shape[-1])
            ],
            dim=-1,
        )
    else:
        raise ValueError(mode)
    return torch.cat([x, xn], dim=pad_dim)

def pad_and_stack(
    sequences: List[torch.Tensor],
    length: Optional[int] = None,
    pad_dim: int = -2,
    **kwargs,
):
    if length is None:
        length = max([x.shape[pad_dim] for x in sequences])

    y = torch.stack([pad_to_length(x, length, pad_dim, **kwargs) for x in sequences], 0)
    return y

# class SuperRetina(BaseModel):
#     def _init(self, config):
#         self.PKE_learn = True
#         self.relu = torch.nn.ReLU(inplace=True)
#         self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
#         c1, c2, c3, c4, c5, d1, d2 = 64, 64, 128, 128, 256, 256, 256
#         # Shared Encoder.
#         self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
#         self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)

#         self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
#         self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)

#         self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
#         self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)

#         self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
#         self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

#         # Descriptor Head.
#         self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
#         self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=4, stride=2, padding=0)
#         self.convDc = torch.nn.Conv2d(d1, d2, kernel_size=1, stride=1, padding=0)

#         self.trans_conv = nn.ConvTranspose2d(d1, d2, 2, stride=2)

#         # Detector Head
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.dconv_up3 = double_conv(c3 + c4, c3)
#         self.dconv_up2 = double_conv(c2 + c3, c2)
#         self.dconv_up1 = double_conv(c1 + c2, c1)
#         n_class=1
#         self.conv_last = nn.Conv2d(c1, n_class, kernel_size=1)

#         if config is not None:
#             self.config = config
#             self.nms_size = 10
#             self.nms_thresh = 0.05
#             self.scale = 8
#             self.max_kps = config.max_num_keypoints
#         self.force_num_keypoints = True
#         # 加载初始化模型的参数
#         checkpoint = torch.load("/home/yepeng_liu/code_base/awesome-detector/glue-factory-main/pretrain_weight/SuperRetina.pth", map_location="cuda")
#         self.load_state_dict(checkpoint['net'])
       

#     def network(self, data):
#         # ([2, 1, 768, 768]), need single channel image input
#         x = data["image"]
#         if x.shape[1] == 3:  # RGB
#             scale = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
#             x = (x * scale).sum(1, keepdim=True)
#         x = self.relu(self.conv1a(x))
#         conv1 = self.relu(self.conv1b(x))
#         x = self.pool(conv1)
#         x = self.relu(self.conv2a(x))
#         conv2 = self.relu(self.conv2b(x))
#         x = self.pool(conv2)
#         x = self.relu(self.conv3a(x))
#         conv3 = self.relu(self.conv3b(x))
#         x = self.pool(conv3)
#         x = self.relu(self.conv4a(x))
#         x = self.relu(self.conv4b(x))

#         # Descriptor Head.
#         cDa = self.relu(self.convDa(x))
#         cDb = self.relu(self.convDb(cDa))
#         desc = self.convDc(cDb)

#         dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
#         desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

#         desc = self.trans_conv(desc)

#         cPa = self.upsample(x)
#         cPa = torch.cat([cPa, conv3], dim=1)

#         cPa = self.dconv_up3(cPa)
#         cPa = self.upsample(cPa)
#         cPa = torch.cat([cPa, conv2], dim=1)

#         cPa = self.dconv_up2(cPa)
#         cPa = self.upsample(cPa)
#         cPa = torch.cat([cPa, conv1], dim=1)

#         cPa = self.dconv_up1(cPa)

#         semi = self.conv_last(cPa)
#         semi = torch.sigmoid(semi)

#         return semi, desc
    
#     def simple_nms(self, scores, nms_radius: int):
#         """ Fast Non-maximum suppression to remove nearby points """
#         assert (nms_radius >= 0)

#         size = nms_radius * 2 + 1
#         avg_size = 2
#         def max_pool(x):
#             return torch.nn.functional.max_pool2d(
#                 x, kernel_size=size, stride=1, padding=nms_radius)

#         zeros = torch.zeros_like(scores)
#         max_mask = scores == max_pool(scores)
#         max_mask_ = torch.rand(max_mask.shape).to(max_mask.device) / 10
#         max_mask_[~max_mask] = 0
#         mask = ((max_mask_ == max_pool(max_mask_)) & (max_mask_ > 0))

#         return torch.where(mask, scores, zeros)
    
#     def remove_borders(self, keypoints, scores, border: int, height: int, width: int):
#         """ Removes keypoints too close to the border """
#         mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
#         mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
#         mask = mask_h & mask_w
#         return keypoints[mask], scores[mask]
    
#     def sample_keypoint_desc(self, keypoints, descriptors, s: int = 8):
#         """ Interpolate descriptors at keypoint locations """
#         b, c, h, w = descriptors.shape
#         keypoints = keypoints.clone().float()
        
#         #keypoints /= torch.tensor([(w * s - 1), (h * s - 1)]).to(keypoints)[None]
#         keypoints /= 768.0
#         keypoints = keypoints * 2 - 1  # normalize to (-1, 1)

#         args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
#         descriptors = torch.nn.functional.grid_sample(
#             descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)

#         descriptors = torch.nn.functional.normalize(
#             descriptors.reshape(b, c, -1), p=2, dim=1)
#         return descriptors

#     def _forward(self, x):
#         """
#         In interface phase, only need to input x
#         :param x: retinal images
#         :param label_point_positions: positions of keypoints on labels
#         :param value_map: value maps, used to record history learned geo_points
#         :param learn_index: index of input data with detector labels
#         :param phase: distinguish dataset
#         :return: if training, return loss, else return predictions
#         """

#         detector_pred, descriptor_pred = self.network(x)
       
#         # decode the score
#         scores = self.simple_nms(detector_pred, self.nms_size)

#         b, _, h, w = detector_pred.shape
#         scores = scores.reshape(-1, h, w)

#         keypoints = [
#             torch.nonzero(s > self.nms_thresh)
#             for s in scores]
        
#         scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

#         # Discard keypoints near the image borders
#         keypoints, scores = list(zip(*[
#             self.remove_borders(k, s, 4, h, w)
#             for k, s in zip(keypoints, scores)]))

#         keypoints = [torch.flip(k, [1]).float().data for k in keypoints]
#         keypoints = [k for k in keypoints]
#         # Keep the k keypoints with highest score
#         if self.max_kps > 0:
#             keypoints, scores = list(
#                 zip(*[
#                         top_k_keypoints(k, s, self.max_kps) for k, s in zip(keypoints, scores)
#                             ]
#                         )
#                     )
#             keypoints, scores = list(keypoints), list(scores)
       
#         if self.force_num_keypoints:
#                 keypoints = pad_and_stack(
#                     keypoints,
#                     self.max_kps,
#                     -2,
#                     mode="zeros",
#                     bounds=(
#                         0,
#                         768.0,
#                     ),
#                 )
#                 scores = pad_and_stack(scores, self.max_kps, -1, mode="zeros")
        
#         # sample descriptors
#         descriptors = [self.sample_keypoint_desc(k[None], d[None], 8)[0]
#                        for k, d in zip(keypoints, descriptor_pred)]
#         descriptors = [des.permute(1,0) for des in descriptors]
#         descriptors = torch.stack(descriptors, 0)      
#         pred = {
#             "keypoints": keypoints,
#             "keypoint_scores": scores,
#             "descriptors": descriptors,
#         }

#         # visulize the image with keypoints for check training pipeline
#         # image_with_keypoints = visulize_img(x, pred["keypoints"])
#         # cv2.imwrite("/home/yepeng_liu/code_base/awesome-detector/glue-factory-main/demo1.jpg", image_with_keypoints)
#         return pred
    
#     def loss(self, pred, data):
#         raise NotImplementedError
    

################## 采用mrnet ###################
class SuperRetina(BaseModel):
    def _init(self, config):
        self.PKE_learn = True
        self.relu = torch.nn.PReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1, d2 = 64, 64, 128, 128, 256, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)

        # LKUNET Block with K = 5
        self.conv2 = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv2c = torch.nn.Conv2d(c2, c2, kernel_size=5, stride=1, padding=2)
        
        self.conv3 = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c3, c3, kernel_size=1, stride=1, padding=0)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv3c = torch.nn.Conv2d(c3, c3, kernel_size=5, stride=1, padding=2)
        
        self.conv4 = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c4, c4, kernel_size=1, stride=1, padding=0)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.conv4c = torch.nn.Conv2d(c4, c4, kernel_size=5, stride=1, padding=2)
         
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=4, stride=2, padding=0)
        self.convDc = torch.nn.Conv2d(d1, d2, kernel_size=1, stride=1, padding=0)

        self.trans_conv = nn.ConvTranspose2d(d1, d2, 2, stride=2)

        # Detector Head
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = double_conv(c3 + c4, c3)
        self.dconv_up2 = double_conv(c2 + c3, c2)
        self.dconv_up1 = double_conv(c1 + c2, c1)
        self.conv_last = nn.Conv2d(c1, 1, kernel_size=1)

        if config is not None:
            self.config = config
            self.nms_size = 10
            self.nms_thresh = 0.05
            self.scale = 8
            self.max_kps = config.max_num_keypoints
        self.force_num_keypoints = True
        # 加载初始化模型的参数
        checkpoint = torch.load("/home/yepeng_liu/code_python/157_beifen/awesome-diffusion/dift/retina_fire_test/models/weights/mrnet.pth", map_location="cuda")
        self.load_state_dict(checkpoint['net'])
       

    def network(self, data):
        # ([2, 1, 768, 768]), need single channel image input
        x = data["image"]
        if x.shape[1] == 3:  # RGB
            scale = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            x = (x * scale).sum(1, keepdim=True)
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))
        x = self.pool(conv1)

        # LKUNET STARTS
        x = self.relu(self.conv2(x))
        x = x + self.conv2a(x) + self.conv2b(x) + self.conv2c(x)  
        conv2 = self.relu(x)
        x = self.pool(conv2)

        x = self.relu(self.conv3(x))
        x = x + self.conv3a(x) + self.conv3b(x) + self.conv3c(x)
        conv3 = self.relu(x)
        x = self.pool(conv3)

        x = self.relu(self.conv4(x))
        x = x + self.conv4a(x) + self.conv4b(x) + self.conv4c(x)
        x = self.relu(x)

        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        cDb = self.relu(self.convDb(cDa))
        desc = self.convDc(cDb)

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        desc = self.trans_conv(desc)

        cPa = self.upsample(x)
        cPa = torch.cat([cPa, conv3], dim=1)

        cPa = self.dconv_up3(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)

        cPa = self.dconv_up2(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)

        cPa = self.dconv_up1(cPa)

        semi = self.conv_last(cPa)
        semi = torch.sigmoid(semi)


        return semi, desc
    
    def simple_nms(self, scores, nms_radius: int):
        """ Fast Non-maximum suppression to remove nearby points """
        assert (nms_radius >= 0)

        size = nms_radius * 2 + 1
        avg_size = 2
        def max_pool(x):
            return torch.nn.functional.max_pool2d(
                x, kernel_size=size, stride=1, padding=nms_radius)

        zeros = torch.zeros_like(scores)
        max_mask = scores == max_pool(scores)
        max_mask_ = torch.rand(max_mask.shape).to(max_mask.device) / 10
        max_mask_[~max_mask] = 0
        mask = ((max_mask_ == max_pool(max_mask_)) & (max_mask_ > 0))

        return torch.where(mask, scores, zeros)
    
    def remove_borders(self, keypoints, scores, border: int, height: int, width: int):
        """ Removes keypoints too close to the border """
        mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
        mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
        mask = mask_h & mask_w
        return keypoints[mask], scores[mask]
    
    def sample_keypoint_desc(self, keypoints, descriptors, s: int = 8):
        """ Interpolate descriptors at keypoint locations """
        b, c, h, w = descriptors.shape
        keypoints = keypoints.clone().float()
        
        #keypoints /= torch.tensor([(w * s - 1), (h * s - 1)]).to(keypoints)[None]
        keypoints /= 768.0
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)

        args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
        descriptors = torch.nn.functional.grid_sample(
            descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)

        descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1)
        return descriptors

    def _forward(self, x):
        """
        In interface phase, only need to input x
        :param x: retinal images
        :param label_point_positions: positions of keypoints on labels
        :param value_map: value maps, used to record history learned geo_points
        :param learn_index: index of input data with detector labels
        :param phase: distinguish dataset
        :return: if training, return loss, else return predictions
        """

        detector_pred, descriptor_pred = self.network(x)
       
        # decode the score
        scores = self.simple_nms(detector_pred, self.nms_size)

        b, _, h, w = detector_pred.shape
        scores = scores.reshape(-1, h, w)

        keypoints = [
            torch.nonzero(s > self.nms_thresh)
            for s in scores]
        
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            self.remove_borders(k, s, 4, h, w)
            for k, s in zip(keypoints, scores)]))

        keypoints = [torch.flip(k, [1]).float().data for k in keypoints]
        keypoints = [k for k in keypoints]
        # Keep the k keypoints with highest score
        if self.max_kps > 0:
            keypoints, scores = list(
                zip(*[
                        top_k_keypoints(k, s, self.max_kps) for k, s in zip(keypoints, scores)
                            ]
                        )
                    )
            keypoints, scores = list(keypoints), list(scores)
       
        if self.force_num_keypoints:
                keypoints = pad_and_stack(
                    keypoints,
                    self.max_kps,
                    -2,
                    mode="zeros",
                    bounds=(
                        0,
                        768.0,
                    ),
                )
                scores = pad_and_stack(scores, self.max_kps, -1, mode="zeros")
        
        # sample descriptors
        descriptors = [self.sample_keypoint_desc(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptor_pred)]
        descriptors = [des.permute(1,0) for des in descriptors]
        descriptors = torch.stack(descriptors, 0)      
        pred = {
            "keypoints": keypoints,
            "keypoint_scores": scores,
            "descriptors": descriptors,
        }

        # visulize the image with keypoints for check training pipeline
        # image_with_keypoints = visulize_img(x, pred["keypoints"])
        # cv2.imwrite("/home/yepeng_liu/code_base/awesome-detector/glue-factory-main/demo1.jpg", image_with_keypoints)
        return pred
    
    def loss(self, pred, data):
        raise NotImplementedError











import numpy as np
def visulize_img(x, keypoints):
    # add keypoints to tensor
   
    # 选择第一个图像的关键点和图像
   
    keypoints_first_image = keypoints[2]  # (512, 2)
    image_first =  x["image"][2]  # (3, 768, 768)
    # 转换关键点为整数类型
    keypoints_first_image = keypoints_first_image.int()
    # 创建一个单通道的图像用于绘制关键点
    image_with_keypoints = np.ascontiguousarray(image_first.cpu().permute(1,2,0).numpy() * 255)
    # 缩放因子，适应图像大小
    
    # print(image_with_keypoints.shape)
    # # 在图像上绘制关键点
    for point in keypoints_first_image:
        x, y = point
        x, y = int(x ), int(y )
        cv2.circle(image_with_keypoints, (x, y),3,(255,255,255))
    return image_with_keypoints