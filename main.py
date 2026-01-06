import torch
import torch, torch.nn as nn
import torch.nn.functional as F

from lfm_tokenizer import Lfm2Tokenizer
from lfm_norm import RMSNorm
from lfm_rope import compute_rope
from gqa import GQAttention

from lfm_config import LFM2Config

prompt = "The ruler of a kingdom is a"
tokenizer = Lfm2Tokenizer("tokenizer.json")
encoded_prompt = tokenizer.encode(prompt)
encoded_prompt = torch.tensor(
    encoded_prompt,
    device = "cpu", # for now
).unsqueeze(0) # create batch dim
device="cpu"

class LFM2350M(nn.Module):
    def __init__(self, config: LFM2Config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.n_vocab, config.d_model, 
            padding_idx=0, dtype=config.dtype)
        self.norm = RMSNorm(config.d_model, False)
        self.gqa = GQAttention(config)
        cos, sin = compute_rope(
            config.context_len, config.head_dim,
            config.theta_base, config.dtype
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    def forward(self, x):
        _, seq_len = x.shape
        x = self.embedding(x)
        x = self.norm(x)
        cos = self.cos[:seq_len, :].to(x.device)
        sin = self.sin[:seq_len, :].to(x.device)
        x = self.gqa(x, cos, sin)
        return x

model = LFM2350M(LFM2Config)
model.to(device)
out = model(encoded_prompt)
print(out.shape)