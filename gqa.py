import torch 
import torch.nn as nn

from lfm_norm import RMSNorm
from lfm_rope import apply_rope
from sdpa import SDPA

class GQAttention(nn.Module):
    def __init__(self, d_model, heads, head_dim, 
        nkv_groups, dtype=torch.float32):
        super().__init__()
        d_out = heads * head_dim
        kv_d_out = nkv_groups * head_dim

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        self.Wq = nn.Linear(d_model, d_out, bias=False)
        self.Wk = nn.Linear(d_model, kv_d_out, bias=False)
        self.Wv = nn.Linear(d_model, kv_d_out, bias=False) 
        self.Wo = nn.Linear(d_out, d_model)

        self.heads = heads
        self.head_dim = head_dim
        self.nkv_groups = nkv_groups
        self.group_size = heads // nkv_groups

    def forward(self, x, sin, cos, mask=None):
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