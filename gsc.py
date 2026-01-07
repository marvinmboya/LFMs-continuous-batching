import torch.nn as nn 
from functools import partial 
from lfm_config import LFM2Config

class GatedShortBlock(nn.Module):
    def __init__(self, config: LFM2Config):
        super().__init__()
        d_model = config.d_model
        k_size = config.k_size
        dtype = config.dtype
        self.conv = nn.Conv1d(
            d_model, d_model, k_size,
            bias=False, groups=d_model, padding=k_size-1,
            dtype=dtype
        )
        lin = partial(nn.Linear, bias=False, dtype=dtype)
        self.w1 = lin(d_model, 3 * d_model)
        self.w2 = lin(d_model, d_model)

    def forward(self, x):
        seq_len = x.shape[1] 
        BCx = self.w1(x).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)
        conv_out = self.conv(B * x)[..., :seq_len]
        x = C * conv_out
        x = x.transpose(-1, -2).contiguous()
        return self.w2(x)