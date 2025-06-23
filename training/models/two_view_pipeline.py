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

## add gmm semantic consistent
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


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
            self.sd_feature = sd_feature_extractor(conf.sd_feature, conf.matcher.input_dim)


    def extract_view(self, data, i):
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract
        if self.conf.extractor.name and not skip_extract:
            pred_i = {**pred_i, **self.extractor(data_i)}
        elif self.conf.extractor.name and not self.conf.allow_no_extract:
            pred_i = {**pred_i, **self.extractor({**data_i, **pred_i})}
        return pred_i
    
    def ex_relation_view(self, data, pred_ex, i):
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        keypoint_i = pred_ex["keypoints"]
        pred_i = {**pred_i, **liot_fast(data_i, keypoint_i)}
        return pred_i
    
    def ex_sdfeature_view(self, data, pred_ex, i):
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        keypoint_i = pred_ex["keypoints"]
        pred_i = {**pred_i, **self.sd_feature.extract_feature(data_i, keypoint_i)}
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

class sd_feature_extractor:
    def __init__(self, config, feature_dim = 256) -> None:
        self.dift = SDFeaturizer4Eval()
        self.config = config
        self.gmm_cluster_ = True
        self.feature_dim = feature_dim

        
    def extract_feature(self, image_list, keypoint_list):
        """
        deal the sd_feature
        input: image_list:['image':[18, 3, 768, 768], 'H_', 'coords', 'image_size'] , (keypoint_list: [18, 300, 2])
        """
        # stable diffusion接口好像只能处理一张图片单次
        des_list = []
        class_labels = []
        center_list = []
        for index in range(image_list['image'].shape[0]):
            img = image_list['image'][index,:,:,:] # shape,(3,768,768)
            w, h = img.shape[1], img.shape[2]
            # 将张量数据归一化到[0, 255]并转换为uint8
           
            img_normalized = (img - img.min()) / (img.max() - img.min()) * 255
            img_normalized = img_normalized.byte()
            # 改变形状为(768, 768, 3)
            img_normalized = img_normalized.transpose(0, 1).transpose(1, 2).cpu().numpy()
            # 转换为PIL图像
            img_normalized = Image.fromarray(img_normalized)   
            coord_query = keypoint_list[index,:,:] 
            c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(coord_query.device).float()
            coord_norm_q= (coord_query - c) / c # -1:1, [1,780,2] 
            coord_norm_q = coord_norm_q.unsqueeze(0) 
            
            # fea:([1, 640, 96, 96]) 
            feat = self.dift.forward(img_normalized,
                                    img_size=(self.config['img_size'], self.config['img_size']),
                                    t=self.config['t'],
                                    up_ft_index=self.config['up_ft_index'],
                                    ensemble_size=self.config['ensemble_size'])
            feat = feat
            feat_q = F.grid_sample(feat, coord_norm_q.unsqueeze(2)).squeeze(-1) #([1, 640, 96, 96])
            feat_q = feat_q.transpose(1, 2) #[1, 780, 640]),每个点是640维
            #desc_q = feat_q.squeeze(0).detach().cpu().numpy() #[780,640]
            des_list.append(feat_q)
            if (self.gmm_cluster_ == False):
                # feat_q: [1, 512,1280]
                reduce_feature_q = reduce_dimensionality(feat_q.squeeze().cpu().numpy(), self.feature_dim)
                # labels:[512, ], centers_a : [5,256]
                #find_K(reduce_feature_q) 选择5看上去是合适的
                labels_a, centers_a = gmm_clustering(reduce_feature_q, 5)
                class_labels.append(labels_a.unsqueeze(0))
                center_list.append(centers_a.unsqueeze(0))
                
           
        concatenated_tensor = torch.cat(des_list, dim=0)
        # concat_sem_labels: [4, 512]; concat_sem_centers: [4, 5, 256]
       
       
        # shape :[4,300,640]
        predict = {"relation": concatenated_tensor,
                  }
        return predict

############################ add gmm clustering ############################
"""
传入聚类的特征和（N * 50）聚类的数量
"""
def reduce_dimensionality(features, reduced_dim):
    pca = PCA(n_components=reduced_dim)
    reduced_features = pca.fit_transform(features)  # (num_keypoints, reduced_dim)
    return reduced_features

def gmm_clustering(features, num_clusters):
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    labels = gmm.fit_predict(features)  # Labels for each keypoint
    cluster_centers = gmm.means_  # Cluster centers
    return torch.tensor(labels).cuda(), torch.tensor(cluster_centers).cuda()

import matplotlib.pyplot as plt
def find_K(features):
    log_likelihoods = []
    K_values = range(1, 10)

    for K in K_values:
        gmm = GaussianMixture(n_components=K, random_state=42)
        gmm.fit(features)
        log_likelihoods.append(gmm.lower_bound_)

    plt.plot(K_values, log_likelihoods, marker='o')
    plt.xlabel('Number of Components (K)')
    plt.ylabel('Log-Likelihood')
    plt.show()
    plt.savefig("./best_k.png")  






@numba.jit()
def distance_weight_binary_pattern_faster(img):
    print(img["image"].shape)
    img = np.asarray(img)#input image H*W*C
    gray_img= img[:,:,1]#convert to gray; if not retinal dataset, you can use standard grayscale api
    pad_img = np.pad(gray_img, ((8,8)), 'constant')
    Weight = pad_img.shape[0]
    Height = pad_img.shape[1]
    sum_map = np.zeros((gray_img.shape[0], gray_img.shape[1], 4)).astype(np.uint8)
    directon_map = np.zeros((gray_img.shape[0], gray_img.shape[1], 8)).astype(np.uint8)
    for direction in range(0,4):
        for postion in range(0,8):
            if direction == 0:#Right
                new_pad = pad_img[postion + 9: Weight - 7 + postion, 8:-8]  # from low to high
				#new_pad = pad_img[16-postion: Weight - postion, 8:-8]  	# from high to low
            elif direction==1:#Left
				#new_pad = pad_img[7 - postion:-1 * (9 + postion), 8:-8]  	#from low to high
                new_pad = pad_img[postion:-1 * (16 - postion), 8:-8]  	  	#from high to low
            elif direction==2:#Up
                new_pad = pad_img[8:-8, postion + 9:Height - 7 + postion]  	# from low to high
				#new_pad = pad_img[8:-8, 16 - postion: Height - postion]   	#from high to low
            elif direction==3:#Down
				#new_pad = pad_img[8:-8, 7 - postion:-1 * (9 + postion)]  	# from low to high
                new_pad = pad_img[8:-8, postion:-1 * (16 - postion)]  		#from high to low
            tmp_map = gray_img.astype(np.int64) - new_pad.astype(np.int64)
            tmp_map[tmp_map > 0] = 1
            tmp_map[tmp_map <= 0] = 0
            directon_map[:,:,postion] = tmp_map * math.pow( 2, postion)
        sum_direction = np.sum(directon_map,2)
        sum_map[:,:,direction] = sum_direction
    return sum_map

def liot_fast(imgs_list, keypoint_i):
    """
    fast method
    """
    # 输入的张量是nchw, 最后取绿色通道变成nhw
    imgs = imgs_list["image"][:,1,:,:]
    gray_imgs = imgs.cpu().numpy()
    pad_imgs = np.pad(gray_imgs, ((0,), (8,), (8,)))
    direct_imgs = []
    # down
    direct_imgs.append(
        np.reshape(np.lib.stride_tricks.sliding_window_view(pad_imgs[:, 9:, 8:-8], window_shape=(8,1), axis=(-2, -1)), (*gray_imgs.shape, -1))[..., ::-1]
    )
    # up
    direct_imgs.append(
        np.reshape(np.lib.stride_tricks.sliding_window_view(pad_imgs[:, :-9, 8:-8], window_shape=(8,1), axis=(-2, -1)), (*gray_imgs.shape, -1))[..., ::-1]
    )
    # right
    direct_imgs.append(
        np.reshape(np.lib.stride_tricks.sliding_window_view(pad_imgs[:, 8:-8, 9:], window_shape=(1, 8), axis=(-2, -1)), (*gray_imgs.shape, -1))[..., ::-1]
    )
    # left
    direct_imgs.append(
        np.reshape(np.lib.stride_tricks.sliding_window_view(pad_imgs[:, 8:-8, :-9], window_shape=(1,8), axis=(-2, -1)), (*gray_imgs.shape, -1))[..., ::-1]
    )

    direct_imgs = np.stack(direct_imgs, axis=1)
    contrast_binary = gray_imgs[:, None, ... , None] > direct_imgs

    contrast = np.packbits(contrast_binary, axis=-1).squeeze()
    # 输出的维度是（n,4,h,w)
    contrast = torch.from_numpy(contrast).to("cuda")
    contrast_ = sample_keypoint_desc(keypoint_i, contrast)
    predict = {"relation": contrast_}
    return predict

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