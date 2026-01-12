import torch.nn as nn
from lfm_config import LFM2Config
from gqa import GQAttention
from gsc import GatedShortBlock
from lfm_norm import RMSNorm
from feedforward import FeedForward

class BackBone(nn.Module):
    def __init__(self, 
        is_attn_on_conv_off, config: LFM2Config):
        super().__init__()
        d_model, d_hidden = config.d_model, config.d_hidden
        dtype = config.dtype

        self.norm1 = RMSNorm(d_model, dtype=dtype)
        self.core = GQAttention(config) if is_attn_on_conv_off else GatedShortBlock(config)
        self.norm2 = RMSNorm(d_model, dtype=dtype)
        self.ff = FeedForward(d_model, d_hidden, dtype)

        self.is_attn_on_conv_off = is_attn_on_conv_off

    def forward(
            self, x, cos, sin, mask, l_idx,
            hybrid_cache, cache_pos_ids):
        shortcut = x
        x = self.norm1(x)
        if self.is_attn_on_conv_off:
            x = self.core(x, cos, sin, l_idx, hybrid_cache, mask)
        else:
            x = self.core(x, l_idx, hybrid_cache, cache_pos_ids)
        x += shortcut 
        shortcut = x 
        x = self.norm2(x)
        x = self.ff(x)
        x += shortcut 
        return x
    

