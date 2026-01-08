import torch 
import torch.nn as nn
import torch.nn.functional as F
from functools import partial 

class FeedForward(nn.Module):
    def __init__(self, 
                d_model, d_hidden, dtype=torch.float32):
        super().__init__()
        lin = partial(nn.Linear, bias=False, dtype=dtype)
        self.w1 = lin(d_model, d_hidden)
        self.v = lin(d_model, d_hidden)
        self.w2 = lin(d_hidden, d_model)

    def forward(self,x):
        x = F.silu(self.w1(x)) * self.v(x)
        x = self.w2(x)
        return x