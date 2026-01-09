import torch

from lfm_tokenizer import Lfm2Tokenizer
from lfm_decode import decode_next_token
from lfm_config import LFM2Config
from lfm_arch import LFM2350M

prompt = "The ruler of a kingdom is a"
tokenizer = Lfm2Tokenizer("tokenizer.json")
encoded_prompt = tokenizer.encode(prompt)
encoded_prompt_d = torch.tensor(
    encoded_prompt,
    device = "cpu", # for now
).unsqueeze(0) # create batch dim
device="cpu"

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
with torch.no_grad():
    decode_next_token(
        model, tokenizer, encoded_prompt_d, 
        tokenizer.eos_token_id, temperature=0.3
    )