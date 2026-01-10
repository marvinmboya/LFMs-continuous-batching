import torch
from lfm_cache import HybridCache

from lfm_tokenizer import Lfm2Tokenizer
from lfm_decode import decode_next_token
from lfm_config import LFM2Config
from lfm_arch import LFM2350M

prompts = ["What is 2 + 2^5?",
           "What is the capital of Kenya?"]

tokenizer = Lfm2Tokenizer("tokenizer.json")
batch_size, device = len(prompts), "cpu"
encode = lambda x: tokenizer.encode(x)
inputs = [encode(prompt) for prompt in prompts]
seq_len = max(len(_in) for _in in inputs)

encoded_prompts_d = torch.empty(batch_size, seq_len, 
            dtype=torch.int64, device=device).\
            fill_(tokenizer.eos_token_id)

for i, in_ in enumerate(inputs):
    sz = len(in_); st = seq_len - sz
    encoded_prompts_d[i, st:st + sz] = torch.tensor(in_)

model = LFM2350M(LFM2Config)

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from lfm_weight import transferLFMWeights

hf_hub_download(
    repo_id = "LiquidAI/LFM2-350M",
    local_dir = "./",
    filename = "model.safetensors",
    revision="3dbef32"
)

pretrained_state_dict = load_file("model.safetensors")
transferLFMWeights(model, pretrained_state_dict)
del pretrained_state_dict

model.to(device).eval()
from lfm_cache import HybridCache
hybrid_cache = HybridCache(LFM2Config, batch_size, LFM2Config.dtype, device)
with torch.no_grad():
    decode_next_token(
        model, tokenizer, encoded_prompts_d, hybrid_cache,
        tokenizer.eos_token_id, temperature=0.3, max_tokens=100
    )