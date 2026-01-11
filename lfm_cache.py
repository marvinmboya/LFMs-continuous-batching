import torch 
from lfm_config import LFM2Config

class HybridCache:
    def __init__(self, config: LFM2Config, 
                batch, dtype=torch.float32, device=None
        ):
        self.k_cache, self.v_cache = [], []
        self.is_inter = False 
        self.inter_batch = 0
        self.conv_cache: list[torch.Tensor] = []
        for _ in range(16):
            conv_state = torch.zeros(
                batch, config.d_model, config.k_size,
                dtype=dtype, device=device
            )
            self.conv_cache.append(conv_state)
            self.k_cache.append(torch.tensor([]))
            self.v_cache.append(torch.tensor([]))

    def update(self, k, v, l_idx):
        if self.is_inter:
            k, v = self.update_at_batch_i(k, v, l_idx)
            return k.unsqueeze(0), v.unsqueeze(0)
        
        if self.k_cache[l_idx].numel() == 0:
            self.k_cache[l_idx] = k
            self.v_cache[l_idx] = v
        else:
            self.k_cache[l_idx] = torch.cat([self.k_cache[l_idx], k], dim=-2)
            self.v_cache[l_idx] = torch.cat([self.v_cache[l_idx], v], dim=-2)
        return self.k_cache[l_idx], self.v_cache[l_idx]

    def get_seq_length(self, l_idx = 0):
        l_idx = 2
        if len(self.k_cache) <= l_idx or self.k_cache[l_idx].numel() == 0:
            return 0
        return self.k_cache[l_idx].shape[-2]

    def update_at_batch_i(self, k, v, l_idx):
        inter_batch = self.inter_batch
        self.k_cache[l_idx][inter_batch, ...] = k
        self.v_cache[l_idx][inter_batch, ...] = v
        return (self.k_cache[l_idx][inter_batch,...], 
                self.v_cache[l_idx][inter_batch,...])
    
    def reset(self):
        for l_idx in range(len(self.conv_cache)):
            self.conv_cache[l_idx].zero_()