import torch
from lfm_cache import HybridCache

from lfm_tokenizer import Lfm2Tokenizer
from lfm_decode import decode_next_token
from lfm_config import LFM2Config
from lfm_arch import LFM2350M

in_print = lambda x: print(f"\x1B[0;32m{x}\x1B[0m")
out_print = lambda x: print(f"\x1B[38;5;216;1m{x}\x1B[0m")

all_prompts = ["What is 2 + 2^5?",
           "What is the capital of Kenya?",
           "What's the best landmark in France?",
           "What's hello in Spanish?"]

batch_size = 2
batch_size = min(batch_size, len(all_prompts))
prompts = all_prompts[:batch_size]
other_prompts = all_prompts[batch_size:]

tokenizer = Lfm2Tokenizer("tokenizer.json")
device = "cpu"
encode = lambda x: tokenizer.encode(x)

encoded_prompts = []
prompt_breaks = [0]
for prompt in prompts:
    tokens = tokenizer.encode(prompt)
    encoded_prompts.extend(tokens)
    prompt_breaks.append(prompt_breaks[-1] + len(tokens))
encoded_prompts_d = torch.tensor(encoded_prompts, device=device).unsqueeze(0)

t_sz = encoded_prompts_d.numel()
ragged_mask = torch.ones((t_sz, t_sz), device=device)
for i in range(len(prompt_breaks)-1):
     s, e = prompt_breaks[i], prompt_breaks[i+1]
     block = torch.triu(torch.ones(e-s,e-s, dtype=torch.bool), diagonal=1)
     ragged_mask[s:e,s:e] = block
ragged_mask = ragged_mask.bool()

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

responses = [""] * len(prompts)
done = [False] * len(prompts)
processing = set(range(batch_size))

model.to(device).eval()
from lfm_cache import HybridCache
hybrid_cache = HybridCache(LFM2Config, batch_size, LFM2Config.dtype, device)
with torch.no_grad():
    for next_token_ids, prompt_meta, avg_spt in decode_next_token(
        model, encoded_prompts_d, other_prompts, tokenizer, hybrid_cache,
        tokenizer.eos_token_id, temperature=0.3, max_tokens=1000,
        done=done):
        for i, each_token in enumerate(next_token_ids):
            if prompt_meta[1] and i == prompt_meta[0]:
                in_print(prompts[i])
                out_print(responses[i])
                print("*"*100)
                prompts[i] = prompt_meta[2]
                responses[i] = tokenizer.decode([each_token])
            elif not done[i]:
                responses[i] += tokenizer.decode([each_token])
            if done[i] and i in processing:
                processing.remove(i)
                in_print(prompts[i])
                out_print(responses[i])
                print("*"*100)

rem = processing.pop()
in_print(prompts[rem])
out_print(responses[rem])
colorprint = lambda x: print(f"\n\x1B[0;33m[{x} tokens/second]\x1B[0m")
colorprint(f"DECODE: {1/avg_spt:.1f}")