import torch 
import torch.nn as nn
from functools import partial 

from lfm_norm import RMSNorm
from lfm_rope import apply_rope
from sdpa import SDPA
from lfm_config import LFM2Config

class GQAttention(nn.Module):
    def __init__(self, config: LFM2Config):
        super().__init__()
        d_out = config.heads * config.head_dim
        kv_d_out = config.nkv_groups * config.head_dim

        self.q_norm = RMSNorm(config.head_dim)
        self.k_norm = RMSNorm(config.head_dim)

        lin = partial(nn.Linear, bias=False, dtype=config.dtype)
        self.Wq = lin(config.d_model, d_out)
        self.Wk = lin(config.d_model, kv_d_out)
        self.Wv = lin(config.d_model, kv_d_out) 
        self.Wo = lin(d_out, config.d_model)

        self.heads = config.heads
        self.nkv_groups = config.nkv_groups
        self.group_size = config.heads // config.nkv_groups

    def forward(self, x, cos, sin, mask=None):
        dtype = x.dtype
        q, k, v = (
            self.Wq(x),
            self.Wk(x),
            self.Wv(x),
        )
        q = q.view(*q.shape[:2], self.heads, -1)
        k = k.view(*k.shape[:2], self.nkv_groups, -1)
        v = v.view(*v.shape[:2], self.nkv_groups, -1)
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2)
        )
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin, dtype=dtype)
        k, v = (
            k.repeat_interleave(self.group_size, dim=1),
            v.repeat_interleave(self.group_size, dim=1)
        )
        out = SDPA(q, k, v, mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(*out.shape[:2], -1)
        out = self.Wo(out)
        return out