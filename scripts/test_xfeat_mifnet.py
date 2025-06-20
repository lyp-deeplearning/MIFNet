import os
import sys
import cv2
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from models.feature_extractor.xfeat_engine import Predictor_xfeat
from utils.post_util import cv_match, post_lg_glue, load_mifnet_weight, descriptor_combine_sp_sd


# from src.models.dift_sd import SDFeaturizer4Eval
from third_party.dift.src.models.dift_sd import SDFeaturizer4Eval
# Set environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)

def visualize_matches(img1, img2, kp1, kp2, matches, output_path):
    img1 = cv2.resize(cv2.imread(img1), (768, 768))
    img2 = cv2.resize(cv2.imread(img2), (768, 768))
    H, W = max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1] + 15
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:img2.shape[0], :img2.shape[1], :] = img2
    out[:img1.shape[0], img2.shape[1] + 15:, :] = img1

    kp1 = np.round(kp1).astype(int)
    kp2 = np.round(kp2).astype(int)
    color_ = (255, 255, 0)

    for (x1, y1), (x2, y2) in zip(kp1, kp2):
        cv2.line(out, (x2, y2), (x1 + 15 + img2.shape[1], y1), color=color_, thickness=3, lineType=cv2.LINE_AA)
        cv2.circle(out, (x2, y2), 1, color_, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + 15 + img2.shape[1], y1), 1, color_, -1, lineType=cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, out)
    print(f"Matching result saved to {output_path}")

def test_cffa_xfeat(input_image1, input_image2):
    config = load_config("../configs/xfeat.yaml")
    detector = Predictor_xfeat(config['PREDICT'])
    query_result = detector.model_run_one_image(image_path=input_image1)
    refer_result = detector.model_run_one_image(image_path=input_image2)
    matcher = cv_match(knn_thresh=0.85)
    matches = matcher.match_point_pairs(query_result['descriptors'].squeeze().cpu().numpy(),
                                        refer_result['descriptors'].squeeze().cpu().numpy())
    kp1 = [cv2.KeyPoint(int(i[0]), int(i[1]), 30) for i in query_result['keypoints'].cpu().squeeze()]
    kp2 = [cv2.KeyPoint(int(i[0]), int(i[1]), 30) for i in refer_result['keypoints'].cpu().squeeze()]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    visualize_matches(input_image1, input_image2, pts1, pts2, matches, "./output_images/matching_result_xfeat.jpg")

def test_cffa_xfeat_mifnet(step, lg_path, up_id, input_image1, input_image2, class_threshold = 0.15):
    config = load_config("../configs/xfeat.yaml")
    detector = Predictor_xfeat(config['PREDICT'])
    dift = SDFeaturizer4Eval()
    config['STABLE']['t'] = step
    config['STABLE']['up_ft_index'] = up_id
    glue_model = load_mifnet_weight(weight_path=lg_path, features="xfeat", input_dim=64, class_threshold=class_threshold)
    
    query_result = detector.model_run_one_image(image_path=input_image1)
    refer_result = detector.model_run_one_image(image_path=input_image2)

    def extract_feat(img_path, keypoints):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        kp = keypoints.clone()
        kp[:, :, 0] *= w / 768
        kp[:, :, 1] *= h / 768
        center = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(kp.device)
        norm_kp = (kp - center) / center
        feat = dift.forward(img, img_size=(config['STABLE']['img_size'], config['STABLE']['img_size']),
                            t=config['STABLE']['t'], up_ft_index=config['STABLE']['up_ft_index'],
                            ensemble_size=config['STABLE']['ensemble_size'])
        return F.grid_sample(feat, norm_kp.unsqueeze(2).cuda()).squeeze(-1).transpose(1, 2)

    feat_q = extract_feat(input_image1, query_result['keypoints'])
    feat_r = extract_feat(input_image2, refer_result['keypoints'])
    fea1 = descriptor_combine_sp_sd(query_result, feat_q)
    fea2 = descriptor_combine_sp_sd(refer_result, feat_r)
    matches = glue_model({"image0": fea1, "image1": fea2})

    kp1 = [cv2.KeyPoint(int(i[0]), int(i[1]), 30) for i in query_result['keypoints'].cpu().squeeze()]
    kp2 = [cv2.KeyPoint(int(i[0]), int(i[1]), 30) for i in refer_result['keypoints'].cpu().squeeze()]
    pts1, pts2, _, _, _ = post_lg_glue(kp1, kp2, matches)
    visualize_matches(input_image1, input_image2, np.float32(pts1), np.float32(pts2), matches, "./output_images/matching_result_mifnet.jpg")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal image matching with xfeat")
    parser.add_argument("--mode", type=str, choices=["cf-fa", "cf-oct", "ema-octa", "opt-sar", "opt-nir"], required=True,
                        help="Choose matching mode")
    parser.add_argument("--mifnet_path", type=str, default=None, help="Path to mifnet weights")
    parser.add_argument("--step", type=int, default=0, help="Step for diffusion model")
    parser.add_argument("--up_id", type=int, default=1, help="Upsample layer index")  
    args = parser.parse_args()
    if args.mode == "opt-nir" or "opt-sar":
        mifnet_path = "../checkpoints/mifnet_remote_xfeat.tar" if args.mifnet_path is None else args.mifnet_path
    else:
        mifnet_path = "../checkpoints/mifnet_retina_xfeat.tar" if args.mifnet_path is None else args.mifnet_path
    if args.mode == "cf-fa":
        img1 = "../example_imgs/cf_fa_1.jpg"
        img2 = "../example_imgs/cf_fa_2.jpg"
        test_cffa_xfeat(img1, img2)
        test_cffa_xfeat_mifnet(args.step, mifnet_path, args.up_id, img1, img2)
    elif args.mode == "cf-oct":
        img1 = "../example_imgs/cf_oct_1.png"
        img2 = "../example_imgs/cf_oct_2.png"
        test_cffa_xfeat(img1, img2)
        test_cffa_xfeat_mifnet(args.step, mifnet_path, args.up_id, img1, img2)
    elif args.mode == "ema-octa":
        img1 = "../example_imgs/ema_octa_1.png"
        img2 = "../example_imgs/ema_octa_2.png"
        test_cffa_xfeat(img1, img2)
        test_cffa_xfeat_mifnet(args.step, mifnet_path, args.up_id, img1, img2)
    elif args.mode == "opt-sar":
        img1 = "../example_imgs/opt_sar_1.png"
        img2 = "../example_imgs/opt_sar_2.png"
        test_cffa_xfeat(img1, img2)
        test_cffa_xfeat_mifnet(args.step, mifnet_path, args.up_id, img1, img2, class_threshold = 0.2)
    elif args.mode == "opt-nir":
        img1 = "../example_imgs/opt_nir_1.png"
        img2 = "../example_imgs/opt_nir_2.png"
        test_cffa_xfeat(img1, img2)
        test_cffa_xfeat_mifnet(args.step, mifnet_path, args.up_id, img1, img2)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

