import torch 
import torch.nn as nn 

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, dtype=torch.float32):
        super().__init__()
        self.weights = torch.ones(d_model)
        self.weight = nn.Parameter(torch.ones(d_model,dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_model,dtype=dtype))
        self.eps = eps

    def forward(self, x):
        x_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim = -1, keepdim = True)
        norm_x = x * torch.rsqrt(var + self.eps)
        rms_norm = norm_x * self.weight + self.bias 
        return rms_norm.to(x_dtype)