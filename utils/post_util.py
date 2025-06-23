import os
import numpy as np
from PIL import Image
import cv2
import torch
import sys
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from models.matcher.mifnet import MIFNet


def read_shape(img_path):
    img = cv2.imread(img_path)
    w,h,c = img.shape
    return w, h 

def pre_processing(data):
    """ Enhance retinal images """
    train_imgs = datasets_normalized(data)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs / 255.
    return train_imgs.astype(np.float32)


def rgb2gray(rgb):
    """ Convert RGB image to gray image """
    r, g, b = rgb.split()
    return g


def clahe_equalized(images):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    images_equalized = np.empty(images.shape)
    images_equalized[:, :] = clahe.apply(np.array(images[:, :],
                                                  dtype=np.uint8))
    return images_equalized


def datasets_normalized(images):
    images_std = np.std(images)
    images_mean = np.mean(images)
    images_normalized = (images - images_mean) / (images_std + 1e-6)
    minv = np.min(images_normalized)
    images_normalized = ((images_normalized - minv) /
                         (np.max(images_normalized) - minv)) * 255
    return images_normalized


def adjust_gamma(images, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    new_images = np.empty(images.shape)
    new_images[:, :] = cv2.LUT(np.array(images[:, :],
                                        dtype=np.uint8), table)
    return new_images

def mnn_matcher(descriptors_a, descriptors_b, metric='cosine'):
    device = descriptors_a.device
    if metric == 'cosine':
        descriptors_a = F.normalize(descriptors_a)
        descriptors_b = F.normalize(descriptors_b)
        sim = descriptors_a @ descriptors_b.t()
    elif metric == 'l2':
        dist = torch.sum(descriptors_a**2, dim=1, keepdim=True) + torch.sum(descriptors_b**2, dim=1, keepdim=True).t() - \
           2 * descriptors_a.mm(descriptors_b.t())
        sim = -dist
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


class cv_match:
    def __init__(self, metric='knn',knn_thresh=0.9) -> None:
        self.knn_matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.metric = metric
        self.knn_thresh = knn_thresh

    def match_point_pairs(self, descriptors_a, descriptors_b):
        goodMatch = []
        status = []
        matches = []
       
        if (self.metric == 'knn'):
            try:
                matches = self.knn_matcher.knnMatch(descriptors_a, descriptors_b, k=2)
                for m, n in matches:
                    if m.distance < self.knn_thresh * n.distance:
                        goodMatch.append(m)
                        status.append(True)
                    else:
                        status.append(False)
                        
            except Exception:
                pass
        return goodMatch
    
    def calculate_homography(self, query_pts, refer_pts, goodMatch,  type="lmeds"):
        
        kp_query = [query_pts[m.queryIdx].pt for m in goodMatch]
        kp_query = np.float32(kp_query).reshape(-1, 1, 2)
        kp_refer = [refer_pts[m.trainIdx].pt for m in goodMatch]
        kp_refer = np.float32(kp_refer).reshape(-1, 1, 2)
        
        if (kp_query.shape[0] < 4 ):
            H_m = np.ones((1))
            inliers_num_rate = 0
            return inliers_num_rate, H_m
        if (type == "lmeds"):
            print("matching number is: ", kp_query.shape[0])
            H_m, mask = cv2.findHomography(kp_query, kp_refer, cv2.LMEDS)
           
        elif (type == "rho"):
            print("matching number is: ", kp_query.shape[0])
            H_m, mask = cv2.findHomography(kp_query, kp_refer, cv2.RHO)
        elif (type == "ransac"):
            print("matching number is: ", kp_query.shape[0])
            H_m, mask = cv2.findHomography(kp_query, kp_refer, cv2.RANSAC)

        if (H_m is None ):
            H_m = np.ones((1))
            inliers_num_rate = 0
            return inliers_num_rate, H_m
        goodMatch_ = np.array(goodMatch)[mask.ravel() == 1]
        inliers_num_rate = mask.sum() / len(mask.ravel())

        match_kp_query = [query_pts[m.queryIdx].pt for m in goodMatch_]
        match_kp_query = np.float32(match_kp_query).reshape(-1,  2)
        match_kp_refer = [refer_pts[m.trainIdx].pt for m in goodMatch_]
        match_kp_refer = np.float32(match_kp_refer).reshape(-1,  2)
        return inliers_num_rate, H_m
    
    

def load_mifnet_weight(weight_path, features, input_dim, class_threshold= 0.15):
    sl_predictor = MIFNet(features=features, filter_threshold = class_threshold, depth_confidence=-1, width_confidence = -1,
                             input_dim =input_dim).eval().to("cuda")
    checkpoint = torch.load(weight_path)
    checkpoint = checkpoint["model"]
    if checkpoint:
            # rename old state dict entries
            for i in range(10):
                pattern = f"matcher.transformers.{i}.self_attn", f"transformers.{i}.self_attn"
                checkpoint = {k.replace(*pattern): v for k, v in checkpoint.items()}
                pattern = f"matcher.transformers.{i}.cross_attn", f"transformers.{i}.cross_attn"
                checkpoint = {k.replace(*pattern): v for k, v in checkpoint.items()}
                pattern = f"matcher.log_assignment.{i}", f"log_assignment.{i}"
                checkpoint = {k.replace(*pattern): v for k, v in checkpoint.items()}
                pattern = f"matcher.token_confidence.{i}", f"token_confidence.{i}"
                checkpoint = {k.replace(*pattern): v for k, v in checkpoint.items()}
                pattern = f"matcher.posenc", f"posenc"
                checkpoint = {k.replace(*pattern): v for k, v in checkpoint.items()}
                pattern = f"matcher.input_proj", f"input_proj"
                checkpoint = {k.replace(*pattern): v for k, v in checkpoint.items()}
                # pattern = f"matcher.proj_com", f"proj_com"
                # checkpoint = {k.replace(*pattern): v for k, v in checkpoint.items()}
    # 查找匹配的键和未匹配的键
    keys_a = set(sl_predictor.state_dict().keys())
    keys_b = set(checkpoint.keys())
    matching_keys = keys_a.intersection(keys_b)
    unmatched_keys_a = keys_a.difference(keys_b)
    unmatched_keys_b = keys_b.difference(keys_a)
    # 打印结果
    print(f'匹配的键数量：{len(matching_keys)}')
    print(f'未匹配的键数量（模型 a 中的未匹配键）：{len(unmatched_keys_a)}', unmatched_keys_a)
    print(f'未匹配的键数量（模型 b 中的未匹配键）：{len(unmatched_keys_b)}', unmatched_keys_b)
    sl_predictor.load_state_dict(checkpoint, strict=False)
    return sl_predictor

"""
input:
fea1["keypoints"], ([1, 77, 2])
fea1["descriptors"], ([1, 77, 274])
fea1["keypoint_scores"], ([1, 77])
fea1["image_size"], ([1, 2])
output:
"""
def descriptor_combine_sp_sd(sp_result, sd_result):
    # 融合两个输入的描述子作为lightglue输入
    fea = {}
    fea["keypoints"] = sp_result["keypoints"].cuda()
    # concat_des = torch.cat((sp_result["descriptors"], sd_result), dim=2)  # dim=2指的是第三个维度
    fea["descriptors"] = sp_result["descriptors"].cuda()
    fea["relation"] = sd_result.cuda()
    fea["image_size"] = torch.from_numpy(np.array([768, 768])).unsqueeze(0).to(torch.float32).cuda()
    return fea  

######## dino 特征经过lightglue以后的输出融合
def descriptor_cross(sp_result, sd_result):
    # 融合两个输入的描述子作为lightglue输入
    fea = {}
    fea["keypoints"] = sp_result["keypoints"].cuda()
    fea["descriptors"] = sp_result["descriptors"].cuda()
    fea["relation"] = sd_result.cuda()
    fea["image_size"] = torch.from_numpy(np.array([768, 768])).unsqueeze(0).to(torch.float32).cuda()
    return fea 


import matplotlib.cm as cm
def post_lg_glue(query_keypoints, refer_keypoints,  sg_predict):
    query = np.float32([pt.pt for pt in query_keypoints])
    refer = np.float32([pt.pt for pt in refer_keypoints])
    query = np.float32(query).reshape(-1,  2)
    refer = np.float32(refer).reshape(-1,  2)

    match_index = sg_predict["matches"][0].cpu().numpy()
    match_confidence = sg_predict["matching_scores0"][0].detach().cpu().numpy()
   
    
    match_kp0 = query[match_index[..., 0]]
    match_kp1 = refer[match_index[..., 1]]
    color = cm.jet(match_confidence[match_index[..., 0]])
    print("=====match index", match_index.shape)
    # ************************ plot motivation fig ************************ #
    # match_des0 = sg_predict["des0"].detach()[0].cpu().numpy()
    # match_des1 = sg_predict["des1"].detach()[0].cpu().numpy()
    # return match_des0,match_des1,query,refer,match_index
    
    kp_query = np.float32(match_kp0).reshape(-1, 1, 2) 
    kp_refer = np.float32(match_kp1).reshape(-1, 1, 2)
   
    if (kp_query.shape[0] >4 ):
        H_m, mask = cv2.findHomography(kp_query, kp_refer, cv2.LMEDS)#cv2.RANSAC,cv2.LMEDS
        if (H_m is None ):
            H_m = np.ones((1))
            inliers_num_rate = 0
            return match_kp0, match_kp1, color, H_m, inliers_num_rate
        inliers_num_rate = mask.sum() / len(mask.ravel())
    else:
        H_m = np.ones((1))
        inliers_num_rate = 0 
    return match_kp0, match_kp1, color, H_m, inliers_num_rate



