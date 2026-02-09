import torch, torch.nn as nn
from pathlib import Path 
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from lfm_config import LFM2Config
from lfm_arch import LFM2350M
from lfm_weight import transferLFMWeights

from lfm_to_bin import saveLFMWeightsToBin
from tok_to_bin import saveLFMTokenizerToBin

weights_name = "model.safetensors"
tokenizer_name = "tokenizer.json"

BASE = Path("files")
bin_path = BASE / "fp32_bins"
weights = BASE / weights_name
tokenizer = BASE / tokenizer_name

bin_path.mkdir(parents=True, exist_ok=True)
def download(file: str, dir: Path):
    hf_hub_download(
        repo_id = "LiquidAI/LFM2-350M", local_dir = dir,
        filename = file, revision="3dbef32"
    )

if not weights.exists():
    download(weights_name, BASE)
if not tokenizer.exists():
    download(tokenizer_name, BASE)

model = LFM2350M(LFM2Config)
pretrained_state_dict = load_file(str(weights))
transferLFMWeights(model, pretrained_state_dict)
del pretrained_state_dict

with torch.no_grad():
    saveLFMWeightsToBin(model)
saveLFMTokenizerToBin(
    str(tokenizer), 
    str(BASE / "tokenizer.bin")
)