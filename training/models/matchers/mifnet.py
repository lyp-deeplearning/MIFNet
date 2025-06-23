import warnings
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...settings import DATA_PATH
from ..utils.losses import NLLLoss
from ..utils.metrics import matcher_metrics
## add gmm semantic consistent
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

torch.backends.cudnn.deterministic = True


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )

    def loss(self, desc0, desc1, la_now, la_final):
        logit0 = self.token[0](desc0.detach()).squeeze(-1) #未经过sigmod处理
        logit1 = self.token[0](desc1.detach()).squeeze(-1)
        la_now, la_final = la_now.detach(), la_final.detach() # 当前标签和最后标签
        correct0 = (
            la_final[:, :-1, :].max(-1).indices == la_now[:, :-1, :].max(-1).indices
        ) #计算当前标签和最终标签匹配情况
        correct1 = (
            la_final[:, :, :-1].max(-2).indices == la_now[:, :, :-1].max(-2).indices
        )
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1)
            + self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE

        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if FLASH_AVAILABLE:
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
                return v if mask is None else v.nan_to_num()
        elif FLASH_AVAILABLE:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(
                qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            desc0 = self.self_attn(desc0, encoding0)
            desc1 = self.self_attn(desc1, encoding1)
            return self.cross_attn(desc0, desc1)

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)

class semantic_aggragate_v2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        semantic0,
        semantic1,
        encoding0,
        encoding1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(semantic0, semantic1, encoding0, encoding1, mask0, mask1)
        else:
            # 原始的注意力就是分成三步，分别计算自注意力，然后计算交叉注意力
            semantic0 = self.self_attn(semantic0, encoding0)
            semantic1 = self.self_attn(semantic1, encoding1)
            cross0, cross1 = self.cross_attn(semantic0, semantic1)
            return cross0, cross1

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)
    


def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True) # 计算每个描述子的匹配能力
        self.final_proj = nn.Linear(dim, dim, bias=True) #对描述子进行最终投影，计算相似性

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1) #计算描述子的相似性矩阵
        z0 = self.matchability(desc0) # 计算描述子的匹配能力
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1) #得到最终的匹配分数
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1) # 返回描述子的匹配能力

# 从匹配得分矩阵（对数分配矩阵）中过滤出有效的匹配对
def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    #分别沿最后一个维度和倒数第二个维度计算分数矩阵的最大值，获取每个特征点的最佳匹配
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    #分别提取最佳匹配的索引。
    m0, m1 = max0.indices, max1.indices
    #建一个与特征点数量相等的索引序列，用于后续的互斥匹配检查
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    #将最大对数分数转换回原始分数
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    #仅保留互斥匹配的分数，非互斥匹配的分数设置为0
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    #确定有效的匹配对，即既是互斥匹配又满足分数阈值的匹配。
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1

class input_proj_geo(nn.Module):
    def __init__(self, conf):
        super(input_proj_geo, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(conf.input_dim, 256, bias=True),\
                                                nn.ReLU(),
                                               ])

    def forward(self, x):
        # 顺序通过所有层
        for layer in self.layers:
            x = layer(x)
        return x
######### sd reproject
class input_proj_seman(nn.Module):
    def __init__(self, conf):
        super(input_proj_seman, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(1280, 640, bias=True),\
                                     nn.ReLU(),\
                                     nn.Linear(640, 480, bias=True), \
                                     nn.ReLU(),\
                                     nn.Linear(480, conf.descriptor_dim, bias=True),\
                                     nn.ReLU()])

    def forward(self, x):
        # 顺序通过所有层
        for layer in self.layers:
            x = layer(x)
        return x

  
class MIFNet(nn.Module):
    default_conf = {
        "name": "mifnet",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 10,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,
        },
    }

    required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]

   

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)
        self.gmm_cluster_ = True
        double_encode = True
        self.input_proj = input_proj_geo(conf=conf)
        self.input_proj_sem = input_proj_seman(conf=conf) #input_proj_seman(conf=conf); input_proj_seman_dino(conf=conf)
       

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori, head_dim, head_dim
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

    
        trans_list = []
        for index in range(n):
            if (index == 0):
                trans_list.append(semantic_aggragate_v2(d, h, conf.flash))
            else:
                trans_list.append(TransformerLayer(d, h, conf.flash))
        self.transformers = nn.ModuleList(trans_list)
        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n - 1)]
        )

        self.loss_fn = NLLLoss(conf.loss)

        state_dict = None
        if conf.weights is not None:
            # weights can be either a path or an existing file from official LG
            if Path(conf.weights).exists():
                state_dict = torch.load(conf.weights, map_location="cpu")
            elif (Path(DATA_PATH) / conf.weights).exists():
                state_dict = torch.load(
                    str(DATA_PATH / conf.weights), map_location="cpu"
                )
            else:
                fname = (
                    f"{conf.weights}_{conf.weights_from_version}".replace(".", "-")
                    + ".pth"
                )
                state_dict = torch.hub.load_state_dict_from_url(
                    self.url.format(conf.weights_from_version, conf.weights),
                    file_name=fname,
                )

        if state_dict:
            # rename old state dict entries
            for i in range(self.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)

    def compile(self, mode="reduce-overhead"):
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        for i in range(self.conf.n_layers):
            self.transformers[i] = torch.compile(
                self.transformers[i], mode=mode, fullgraph=True
            )
    
    def forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        if "view0" in data.keys() and "view1" in data.keys():
            size0 = data["view0"].get("image_size")
            size1 = data["view1"].get("image_size")
       
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()
        

        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0 = torch.cat(
                [
                    kpts0,
                    sc0 if sc0.dim() == 3 else sc0[..., None],
                    o0 if o0.dim() == 3 else o0[..., None],
                ],
                -1,
            )
            kpts1 = torch.cat(
                [
                    kpts1,
                    sc1 if sc1.dim() == 3 else sc1[..., None],
                    o1 if o1.dim() == 3 else o1[..., None],
                ],
                -1,
            )
        
        desc0 = data["descriptors0"].contiguous()
        desc1 = data["descriptors1"].contiguous()
        sem0 = data["relation0"].contiguous()
        sem1 = data["relation1"].contiguous()
        
        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()
            sem0 = sem0.half()
            sem1 = sem1.half()
        geo0 = self.input_proj(desc0)
        geo1 = self.input_proj(desc1)
        sem0 = self.input_proj_sem(sem0)
        sem1 = self.input_proj_sem(sem1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        all_desc0, all_desc1 = [], []
        token0, token1 = None, None
        for i in range(self.conf.n_layers):
            ## diffmatch
            if self.conf.checkpointed and self.training:
                if (i == 0):
                    sem0, sem1 = checkpoint(self.transformers[i],sem0, sem1, encoding0, encoding1)
                    sem_label0, sem_center0 = self.get_sklearn_gmm(sem0, 5)
                    sem_label1, sem_center1 = self.get_sklearn_gmm(sem1, 5)
                    intra_loss_0, inter_loss_0 = self.compute_intra_inter_loss(sem_label0, \
                                                                               sem0)
                    intra_loss_1, inter_loss_1 = self.compute_intra_inter_loss(sem_label1, \
                                                                               sem1)  
                    intra_loss_0 = intra_loss_0 + inter_loss_0 * 0.02
                    intra_loss_1 = intra_loss_1 + inter_loss_1 * 0.02
                    
                    desc0 = sem0 + geo0
                    desc1 = sem1 + geo1
                else:
                    desc0, desc1 = checkpoint(self.transformers[i],desc0, desc1, encoding0, encoding1)
            else:
                if (i == 0):
                    sem0, sem1 = self.transformers[i](sem0, sem1, encoding0, encoding1)
                    desc0 = sem0 + geo0
                    desc1 = sem1 + geo1

                    sem_label0, sem_center0 = self.get_sklearn_gmm(sem0, 5)
                    sem_label1, sem_center1 = self.get_sklearn_gmm(sem1, 5)
                    intra_loss_0, inter_loss_0 = self.compute_intra_inter_loss(sem_label0, \
                                                                               sem0)
                    intra_loss_1, inter_loss_1 = self.compute_intra_inter_loss(sem_label1, \
                                                                               sem1)  
                    intra_loss_0 = intra_loss_0 + inter_loss_0 * 0.02
                    intra_loss_1 = intra_loss_1 + inter_loss_1 * 0.02
                    
                else:
                    desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)
            
           
            if self.training or i == self.conf.n_layers - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                continue  # no early stopping or adaptive width at last layer

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)
        # 根据point pruning的信息去更新匹配矩阵的信息，重新得到match对
        
        prune0 = torch.ones_like(mscores0) * self.conf.n_layers
        prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "ref_descriptors0": torch.stack(all_desc0, 1),
            "ref_descriptors1": torch.stack(all_desc1, 1),
            "log_assignment": scores,
            "prune0": prune0,
            "prune1": prune1,
            "intra_loss0": intra_loss_0,
            "intra_loss1": intra_loss_1,
        }

        return pred

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.conf.depth_confidence

    def pruning_min_kpts(self, device: torch.device):
        if self.conf.flash and FLASH_AVAILABLE and device.type == "cuda":
            return self.pruning_keypoint_thresholds["flash"]
        else:
            return self.pruning_keypoint_thresholds[device.type]

    def gmm_clustering(self, features, num_clusters):
        class_labels = []
        center_list = []
        features_ = features
        for batch_i in range(features_.shape[0]):
            gmm = GaussianMixture(5, 256, covariance_type="diag").cuda()
            gmm.fit(features_[batch_i])
            labels, cluster_centers = gmm.predict(features_[batch_i])
            class_labels.append(labels.unsqueeze(0))
            if cluster_centers.dim() == 2:
                cluster_centers = cluster_centers.unsqueeze(0)
            center_list.append(cluster_centers) 
        # concat_sem_labels: [4, 512]; concat_sem_centers: [4, 5, 256]
        concat_sem_labels = torch.cat(class_labels, dim=0)  
        concat_sem_centers = torch.cat(center_list, dim=0)
        return concat_sem_labels, concat_sem_centers
    
    def get_sklearn_gmm(self, features, num_clusters):
        class_labels = []
        center_list = []
        for batch_i in range(features.shape[0]):
            features_ = features[batch_i].detach().cpu().numpy()
            gmm = GaussianMixture(n_components=num_clusters, random_state=42)
            labels = gmm.fit_predict(features_)  # Labels for each keypoint
            cluster_centers = gmm.means_  # Cluster centers
            class_labels.append(torch.tensor(labels).unsqueeze(0).cuda())
            center_list.append(torch.tensor(cluster_centers).unsqueeze(0).cuda()) 
            
        # concat_sem_labels: [4, 512]; concat_sem_centers: [4, 5, 256]
        concat_sem_labels = torch.cat(class_labels, dim=0)  
        concat_sem_centers = torch.cat(center_list, dim=0)
        return concat_sem_labels, concat_sem_centers


    def _get_sample_num(self, x, min_sample):
        """
        初始化中心点，确保第一维不同。
        """
        check_acc = 0
        for batch_i in range(x.shape[0]):
            temp_x = x[batch_i]
            unique_values = set()
            for time_s in range(min_sample):
                if temp_x[time_s, 0].item() not in unique_values:  # 检查第一维是否唯一
                    unique_values.add(temp_x[time_s, 0].item())
            if (len(unique_values) > 5):
                check_acc += 1
        if (check_acc == x.shape[0]):
            return True
        else:
            return False

       
    
    # (4) Compute intra-class and inter-class losses using PyTorch
    def compute_intra_inter_loss(self, labels, features):
        batch_size, num_points, feature_dim = features.shape
        num_clusters = 4

        intra_loss_list = []
        inter_loss_list = []
        centers = torch.zeros(batch_size, num_clusters, feature_dim, device=features.device)

        # Process each batch separately
        for batch_idx in range(batch_size):
            # Extract data for the current batch
            batch_labels = labels[batch_idx]  # Shape: [num_points]
            batch_features = features[batch_idx]  # Shape: [num_points, feature_dim]
            batch_centers = centers[batch_idx]  # Shape: [num_clusters, feature_dim]

            intra_loss = torch.tensor(0.0, device=features.device)
            inter_loss = torch.tensor(0.0, device=features.device)

            # Compute intra-class loss (cosine similarity to center)
            cluster_counts = torch.zeros(num_clusters, device=features.device)
            for cluster_id in range(num_clusters):
                # Mask for points in the current cluster
                cluster_mask = batch_labels == cluster_id  # Shape: [num_points]
                if cluster_mask.sum() > 0:
                    cluster_features = batch_features[cluster_mask]  # Shape: [num_cluster_points, feature_dim]
                    #cluster_center = batch_centers[cluster_id]  # Shape: [feature_dim]
                    cluster_center = cluster_features.mean(dim=0) 
                    centers[batch_idx, cluster_id] = cluster_center
                    
                    # Normalize features and centers
                    normalized_features = torch.nn.functional.normalize(cluster_features, dim=1)
                    normalized_center = torch.nn.functional.normalize(cluster_center, dim=0)
                    
                    # Compute cosine similarity loss
                    cosine_similarity = torch.sum(normalized_features * normalized_center, dim=1)  # Shape: [num_cluster_points]
                    intra_loss += torch.sum(1 - cosine_similarity)  # Cosine distance: 1 - similarity
                    
                    cluster_counts[cluster_id] += cluster_mask.sum()

            # Normalize intra-class loss by total number of points
            intra_loss /= cluster_counts.sum()
            
            # Compute inter-class loss
            for i in range(num_clusters):
                for j in range(i + 1, num_clusters):
                    # Distance between cluster centers
                    distance = torch.norm(centers[batch_idx, i] - centers[batch_idx, j])
                    if distance > 0:  # Avoid division by zero
                        inter_loss += 1.0 / distance
            
            # Append the losses for the current batch to the lists
            intra_loss_list.append(intra_loss)
            inter_loss_list.append(inter_loss)
        intra_loss_list = torch.stack(intra_loss_list, dim=0)
        inter_loss_list = torch.stack(inter_loss_list, dim=0)

        return intra_loss_list, inter_loss_list

        
    
    
    def loss(self, pred, data):
        def loss_params(pred, i):
            la, _ = self.log_assignment[i](
                pred["ref_descriptors0"][:, i], pred["ref_descriptors1"][:, i]
            )
            return {
                "log_assignment": la,
            }
        
        sum_weights = 1.0
        nll, gt_weights, loss_metrics = self.loss_fn(loss_params(pred, -1), data)
        N = pred["ref_descriptors0"].shape[1]
       
        losses = {"total": nll, "last": nll.clone().detach(), **loss_metrics}

        if self.training:
            losses["confidence"] = 0.0
        
        # B = pred['log_assignment'].shape[0]
        losses["row_norm"] = pred["log_assignment"].exp()[:, :-1].sum(2).mean(1)
        # 计算损失的时候需要每层都遍历去计算
        for i in range(N - 1):
            params_i = loss_params(pred, i)
            nll, _, _ = self.loss_fn(params_i, data, weights=gt_weights)

            if self.conf.loss.gamma > 0.0:
                weight = self.conf.loss.gamma ** (N - i - 1)
            else:
                weight = i + 1
            sum_weights += weight
            losses["total"] = losses["total"] + nll * weight

            losses["confidence"] += self.token_confidence[i].loss(
                pred["ref_descriptors0"][:, i],
                pred["ref_descriptors1"][:, i],
                params_i["log_assignment"],
                pred["log_assignment"],
            ) / (N - 1)

            del params_i
        losses["total"] /= sum_weights
       

        # confidences
        if self.training:
            ## 处理新加入的语义类别损失
            sem_loss_0 = pred["intra_loss0"]
            sem_loss_1 = pred["intra_loss1"]
            #  base是设置的5，现在测试一下=10会怎么样,20
            losses["total"] = losses["total"] + 5 * sem_loss_0 + 5 * sem_loss_1
            losses["total"] = losses["total"] + losses["confidence"]
        # print("total loss:", losses["total"])
        # print("intra_loss0:", 5 * sem_loss_0 + 5 * sem_loss_1)
        if not self.training:
            # add metrics
            metrics = matcher_metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics


__main_model__ = MIFNet
