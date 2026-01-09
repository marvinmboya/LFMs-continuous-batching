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
    def forward(self, x):
        _, seq_len = x.shape
        x = self.embedding(x)
        cos = self.cos[:seq_len, :].to(x.device)
        sin = self.sin[:seq_len, :].to(x.device)
        for backbone in self.backbones:
            x = backbone(x, cos, sin)
        x = self.norm_out(x)
        x = self.lin_out(x)
        return x