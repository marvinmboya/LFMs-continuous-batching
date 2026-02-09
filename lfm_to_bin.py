import torch

def store_to_file_bf16(in_: torch.Tensor, _name: str):
    assert in_.dtype == torch.bfloat16
    in_ = in_.view(torch.int16)
    in_ = in_.cpu().flatten()
    _name = f"./all_bins/{_name}"
    in_.numpy().tofile(_name)

def store_to_file(in_: torch.Tensor, _name: str):
    if in_.dtype == torch.bfloat16:
        in_ = in_.float()
    assert in_.dtype == torch.float32, f"Expected float32, got {in_.dtype}"
    in_ = in_.cpu().flatten()
    _name = f"files/fp32_bins/{_name}"
    in_.numpy().tofile(_name)

def saveLFMWeightsToBin(model):
    store_to_file(model.embedding.weight, "embed.bin")
    attn_layer_indeces = (2,5,8,10,12,14)
    for l in range(16):
        if l not in attn_layer_indeces:
             continue
        block = model.backbones[l]
        attn = block.core
        wqkv = torch.cat([
            attn.Wq.weight.transpose(0, 1).reshape(-1),
            attn.Wk.weight.transpose(0, 1).reshape(-1),
            attn.Wv.weight.transpose(0, 1).reshape(-1)
        ])
        store_to_file(wqkv, f"wqkv_{l}.bin")
        wo = attn.Wo.weight.transpose(0, 1).reshape(-1)
        store_to_file(wo, f"wo_{l}.bin")
        # QK norms
        store_to_file(attn.q_norm.weight, f"qnorm_{l}.bin")
        store_to_file(attn.k_norm.weight, f"knorm_{l}.bin")
        # norm before
        store_to_file(block.norm1.weight, f"rms_before_{l}.bin")
        # Feedforward weights
        ffw1 = block.ff.w1.weight.transpose(0, 1).reshape(-1)
        store_to_file(ffw1, f"ffw1_{l}.bin")
        ffv = block.ff.v.weight.transpose(0, 1).reshape(-1)
        store_to_file(ffv, f"ffv_{l}.bin")
        ffw2 = block.ff.w2.weight.transpose(0, 1).reshape(-1)
        store_to_file(ffw2, f"ffw2_{l}.bin")
        # norm after
        store_to_file(block.norm2.weight, f"rms_after_{l}.bin")
    
    for l in range(16):
        if l in attn_layer_indeces:
            continue 
        block = model.backbones[l]
        gate = block.core
        store_to_file(gate.conv.weight, f"gate_conv_{l}.bin")
        w1 = gate.w1.weight.transpose(0, 1).reshape(-1)
        store_to_file(w1, f"gate_w1_{l}.bin")
        w2 = gate.w2.weight.transpose(0, 1).reshape(-1)
        store_to_file(w2, f"gate_w2_{l}.bin")
        # norm before
        store_to_file(block.norm1.weight, f"rms_before_{l}.bin")
        # Feedforward weights
        ffw1 = block.ff.w1.weight.transpose(0, 1).reshape(-1)
        store_to_file(ffw1, f"ffw1_{l}.bin")
        ffv = block.ff.v.weight.transpose(0, 1).reshape(-1)
        store_to_file(ffv, f"ffv_{l}.bin")
        ffw2 = block.ff.w2.weight.transpose(0, 1).reshape(-1)
        store_to_file(ffw2, f"ffw2_{l}.bin")
        # norm after
        store_to_file(block.norm2.weight, f"rms_after_{l}.bin")
    # Final normalization and output head
    store_to_file(model.norm_out.weight, f"rms_out.bin")
    linout = model.lin_out.weight.transpose(0, 1).reshape(-1)
    store_to_file(linout, f"lin_out.bin")