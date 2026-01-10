import torch 
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
        self.k_size = k_size

    def forward(self, x, l_idx, hybrid_cache, cache_pos_ids):
        k_size = self.k_size

        seq_len = x.shape[1] 
        BCx = self.w1(x).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)
        Bx = B * x
        if cache_pos_ids[0] > 0:
            conv_state = hybrid_cache.conv_cache[l_idx]
            cache_pos_ids = cache_pos_ids.clamp(0, k_size - 1)
            conv_state = conv_state.roll(shifts=-1, dims=-1)
            conv_state[:, :, cache_pos_ids] = Bx.to(
                device=conv_state.device, dtype=conv_state.dtype
            )
            hybrid_cache.conv_cache[l_idx].copy_(conv_state)
            conv_out = torch.sum(
                conv_state.to(Bx.device) * self.conv.weight[:, 0, :], dim=-1
            )
            conv_out = conv_out.unsqueeze(-1)
        else:
            conv_state = nn.functional.pad(Bx, (k_size - Bx.shape[-1], 0))
            hybrid_cache.conv_cache[l_idx].copy_(conv_state)
            conv_out = self.conv(Bx)[..., :seq_len]
        x = C * conv_out
        x = x.transpose(-1, -2).contiguous()
        return self.w2(x)