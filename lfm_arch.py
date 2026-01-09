import torch
import torch.nn as nn 

from lfm_norm import RMSNorm
from backbone import BackBone
from lfm_rope import compute_rope
from lfm_config import LFM2Config

class LFM2350M(nn.Module):
    def __init__(self, config: LFM2Config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.n_vocab, config.d_model, 
            padding_idx=0, dtype=config.dtype)
        attn_indeces = (2, 5, 8, 10, 12, 14)
        self.backbones = nn.ModuleList([ 
            BackBone(i in attn_indeces, config)
            for i in range(16)
        ])
        self.norm_out = RMSNorm(config.d_model, dtype=config.dtype)
        self.lin_out = nn.Linear(
            config.d_model, config.n_vocab, dtype=config.dtype
        )

        cos, sin = compute_rope(
            config.context_len, config.head_dim,
            config.theta_base, config.dtype
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    def forward(self, x, hybrid_cache):
        seq_len = x.size(1)
        device = x.device
        _start = hybrid_cache.get_seq_length()
        _end = _start + seq_len
        cache_pos_ids = torch.arange(
            _start, _start + seq_len, device=device
        )
        x = self.embedding(x)
        cos = self.cos[_start:_end, :].to(x.device)
        sin = self.sin[_start:_end, :].to(x.device)
        for l_idx, backbone in enumerate(self.backbones):
            x = backbone(
                x, cos, sin, l_idx,
                hybrid_cache, cache_pos_ids,
        )
        x = self.norm_out(x)
        x = self.lin_out(x)
        return x