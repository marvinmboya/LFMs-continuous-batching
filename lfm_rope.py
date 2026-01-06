import torch 

def compute_rope(
        context_len, head_dim, 
        theta_base = 10_000, dtype = torch.float32
    ):
    thetas = theta_base ** (
        -torch.arange(0, head_dim, 2, dtype = dtype)[:(head_dim//2)].float() / head_dim
    )
    m = torch.arange(context_len, dtype=dtype)
    mthetas = torch.outer(m, thetas)
    freqs = torch.cat([mthetas, mthetas],dim=1)
    return torch.cos(freqs), torch.sin(freqs)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin, dtype=torch.float32):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    print(f"{q.shape = } {cos.shape = } {sin.shape = }")
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(dtype), k_embed.to(dtype)