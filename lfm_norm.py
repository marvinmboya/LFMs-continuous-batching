import torch 
import torch.nn as nn 

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5):
        super().__init__()
        self.weights = torch.ones(d_model)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(dim = -1, keepdim = True)
        norm_x = x * torch.rsqrt(var + self.eps)
        return norm_x * self.weight + self.bias 