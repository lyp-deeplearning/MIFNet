import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

# from gluefactory.models.base_model import BaseModel
from ..base_model import BaseModel

class BasicLayer(nn.Module):
	"""
	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									  nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									  nn.BatchNorm2d(out_channels, affine=False),
									  nn.ReLU(inplace = True),
									)

	def forward(self, x):
	  return self.layer(x)

class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions. """ 
    def __init__(self, mode = 'bicubic', align_corners = False): 
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode = self.mode , align_corners = False)
        return x.permute(0,2,3,1).squeeze(-2)

class XFeat(BaseModel):
    # default_conf = {
    #     "model_name": "xfeat",
    #     "detection_threshold": 0.02,
    #     "force_num_keypoints": False,
    #     "pretrained": True,
    #     "nms_radius": 5,
    # }

    # n_limit_max = 20000

    # required_data_keys = ["image"]

    def _init(self, conf):
        #self.conf = dict(self.default_conf,**conf)
        self.conf = conf
        # if conf.force_num_keypoints:
        #     assert conf.detection_threshold <= 0 and conf.max_num_keypoints > 0
        # get configurations
        self.norm = nn.InstanceNorm2d(1)


		########### ⬇️ CNN Backbone & Heads ⬇️ ###########
        self.skip1 = nn.Sequential(	 nn.AvgPool2d(4, stride = 4),
                                        nn.Conv2d (1, 24, 1, stride = 1, padding=0) )

        self.block1 = nn.Sequential(
                                        BasicLayer( 1,  4, stride=1),
                                        BasicLayer( 4,  8, stride=2),
                                        BasicLayer( 8,  8, stride=1),
                                        BasicLayer( 8, 24, stride=2),
                                    )

        self.block2 = nn.Sequential(
                                        BasicLayer(24, 24, stride=1),
                                        BasicLayer(24, 24, stride=1),
                                        )

        self.block3 = nn.Sequential(
                                        BasicLayer(24, 64, stride=2),
                                        BasicLayer(64, 64, stride=1),
                                        BasicLayer(64, 64, 1, padding=0),
                                        )
        self.block4 = nn.Sequential(
                                        BasicLayer(64, 64, stride=2),
                                        BasicLayer(64, 64, stride=1),
                                        BasicLayer(64, 64, stride=1),
                                        )

        self.block5 = nn.Sequential(
                                        BasicLayer( 64, 128, stride=2),
                                        BasicLayer(128, 128, stride=1),
                                        BasicLayer(128, 128, stride=1),
                                        BasicLayer(128,  64, 1, padding=0),
                                        )

        self.block_fusion =  nn.Sequential(
                                        BasicLayer(64, 64, stride=1),
                                        BasicLayer(64, 64, stride=1),
                                        nn.Conv2d (64, 64, 1, padding=0)
                                        )

        self.heatmap_head = nn.Sequential(
                                        BasicLayer(64, 64, 1, padding=0),
                                        BasicLayer(64, 64, 1, padding=0),
                                        nn.Conv2d (64, 1, 1),
                                        nn.Sigmoid()
                                    )


        self.keypoint_head = nn.Sequential(
                                        BasicLayer(64, 64, 1, padding=0),
                                        BasicLayer(64, 64, 1, padding=0),
                                        BasicLayer(64, 64, 1, padding=0),
                                        nn.Conv2d (64, 65, 1),
                                    )


        ########### ⬇️ Fine Matcher MLP ⬇️ ###########

        self.fine_matcher =  nn.Sequential(
                                            nn.Linear(128, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 64),
                                        )

        # load pretrained
        # if conf.pretrained:
        #     state_dict = torch.hub.load_state_dict_from_url(
        #         self.checkpoint_url.format(conf.model_name), map_location="cpu"
        #     )
        #     self.load_state_dict(state_dict, strict=True)
        
        state_dict = torch.load("/home/yepeng_liu/code_python/157_beifen/awesome-detector/glue-factory-main/pretrain_weight/xfeat.pt", map_location="cpu")
        self.load_state_dict(state_dict, strict=True)
        

    def preprocess_tensor(self, x):
        """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
        if isinstance(x, np.ndarray) and len(x.shape) == 3:
            x = torch.tensor(x).permute(2,0,1)[None]
        x = x.float()

        H, W = x.shape[-2:]
        _H, _W = (H//32) * 32, (W//32) * 32
        rh, rw = H/_H, W/_W

        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        return x, rh, rw
    
    def _unfold2d(self, x, ws = 2):
        """
            Unfolds tensor in 2D with desired ws (window size) and concat the channels
        """
        B, C, H, W = x.shape
        x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
            .reshape(B, C, H//ws, W//ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)
    
    def extract_raw_map(self, x):
        with torch.no_grad():
            x = x.mean(dim=1, keepdim = True)
            x = self.norm(x)
        #main backbone
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        #pyramid fusion
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion( x3 + x4 + x5 )

        #heads
        heatmap = self.heatmap_head(feats) # Reliability map
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits

        return feats, keypoints, heatmap
    
    def get_kpts_heatmap(self, kpts, softmax_temp = 1.0):
        scores = F.softmax(kpts*softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
        return heatmap

    def NMS(self, x, threshold = 0.05, kernel_size = 5):
        B, _, H, W = x.shape
        pad=kernel_size//2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max([len(x) for x in pos_batched])
        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

        #Pad kpts and build (B, N, 2) tensor
        for b in range(len(pos_batched)):
            pos[b, :len(pos_batched[b]), :] = pos_batched[b]

        return pos
    
    def _forward(self, data):
        
        image = data["image"] * 255
        x, rh1, rw1 = self.preprocess_tensor(image)
        B, _, _H1, _W1 = x.shape
        M1, K1, H1 = self.extract_raw_map(x)
        M1 = F.normalize(M1, dim=1)

        #Convert logits to heatmap and extract kpts
        K1h = self.get_kpts_heatmap(K1)
        mkpts = self.NMS(K1h, threshold=0.05, kernel_size=self.conf["nms_radius"])

        #Compute reliability scores
        _nearest = InterpolateSparse2d('nearest')
        _bilinear = InterpolateSparse2d('bilinear')
        scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1

        if self.conf["max_num_keypoints"] > 0:
            #Select top-k features
            idxs = torch.argsort(-scores)
            mkpts_x  = torch.gather(mkpts[...,0], -1, idxs)[:, :self.conf["max_num_keypoints"]]
            mkpts_y  = torch.gather(mkpts[...,1], -1, idxs)[:, :self.conf["max_num_keypoints"]]
            mkpts = torch.cat([mkpts_x[...,None], mkpts_y[...,None]], dim=-1)
            scores = torch.gather(scores, -1, idxs)[:, :self.conf["max_num_keypoints"]]
        else:
            #Select top-k features
            idxs = torch.argsort(-scores)
            mkpts_x  = torch.gather(mkpts[...,0], -1, idxs)
            mkpts_y  = torch.gather(mkpts[...,1], -1, idxs)
            mkpts = torch.cat([mkpts_x[...,None], mkpts_y[...,None]], dim=-1)
            scores = torch.gather(scores, -1, idxs)

        #Interpolate descriptors at kpts positions
        _interpolator=InterpolateSparse2d('bicubic')
        feats = _interpolator(M1, mkpts, H = _H1, W = _W1)

        #L2-Normalize
        feats = F.normalize(feats, dim=-1)

        #Correct kpt scale
        mkpts = mkpts * torch.tensor([rw1,rh1], device=mkpts.device).view(1, 1, -1)

        valid = scores > self.conf["detection_threshold"]
       
        return {
            "keypoints":mkpts,
            "keypoint_scores":scores,
            "descriptors":feats
        }

    def loss(self, pred, data):
        raise NotImplementedError
    
# if __name__ == "__main__":
#     image_path="/home/yepeng_liu/code_python/157_beifen/awesome-detector/glue-factory-main/assets/boat1.png"
#     import cv2
#     data={}
#     data["image"]=cv2.imread(image_path)
#     conf={
#         "max_num_keypoints":4096,
#         "detection_threshold":0.02
#     }
#     xfeat=XFeat(conf)
#     import pdb;pdb.set_trace()
#     pred=xfeat._forward(data)
#     print(pred["keypoints"].shape)
    