import torch 
import torch.nn.functional as F

def SDPA(q, k, v, mask, dtype):
        scale = k.shape[-1]
        qk = q @ k.transpose(-1, -2)
        mask_qk = (qk.masked_fill_(mask, -torch.inf) if 
        mask is not None else qk)
        mask_qk = mask_qk.to(dtype)
        scores = F.softmax(mask_qk / scale**.5, dim=-1)
        out = scores @ v
        return out