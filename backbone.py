import torch.nn as nn
from lfm_config import LFM2Config
from gqa import GQAttention
from gsc import GatedShortBlock

class BackBone(nn.Module):
    def __init__(self, 
                is_attn_on_conv_off, 
                config: LFM2Config):
        super().__init__()
        if is_attn_on_conv_off:
            self.core = GQAttention(config)
        else:
            self.core = GatedShortBlock(config)
        self.is_attn_on_conv_off = is_attn_on_conv_off

    def forward(self,x, cos, sin):
        if self.is_attn_on_conv_off:
            x = self.core(x, cos, sin)
        else:
            x = self.core(x)
        return x

