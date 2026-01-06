from torch import bfloat16
from dataclasses import dataclass

@dataclass
class LFM2Config:
    n_vocab = 65_536
    d_model = 1_024
    heads = 16
    head_dim = 64
    nkv_groups = 8
    theta_base = 1_000_000.0
    dtype = bfloat16