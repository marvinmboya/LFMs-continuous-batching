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

    def repack_kv_caches(self, breaks):
        batches = len(breaks) - 1
        max_seq_len = max([i-j for i, j in zip(breaks[1:], breaks)])

        for i in range(16):
            others = self.conv_cache[i].shape[1:]
            dtype, device = self.conv_cache[i].dtype, self.conv_cache[i].device
            self.conv_cache[i] =  torch.zeros(batches, *others, dtype=dtype, device=device)
        for l_idx in (2, 5, 8, 10, 12, 14):
            _, heads, _, head_dim = self.k_cache[l_idx].shape
            k_cache = torch.zeros(batches, heads, max_seq_len, head_dim, dtype=dtype, device=device)
            v_cache = torch.zeros(batches, heads, max_seq_len, head_dim, dtype=dtype, device=device)

            for i in range(batches):
                s, e = breaks[i], breaks[i + 1]
                k_cache[i, :, s - e:, :] = self.k_cache[l_idx][:, :, s:e, :].contiguous()
                v_cache[i, :, s - e:, :] = self.v_cache[l_idx][:, :, s:e, :].contiguous()
            self.k_cache[l_idx] = k_cache
            self.v_cache[l_idx] = v_cache
            
    def reset(self):
        for l_idx in range(len(self.conv_cache)):
            self.conv_cache[l_idx].zero_()