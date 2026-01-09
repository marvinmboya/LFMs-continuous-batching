import torch

def assign(left, right, tensor_name="unknown", cast_to=None):
        cast_to = cast_to or right.clone().detach().dtype
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(
            right.clone().detach().to(cast_to) 
            if isinstance(right, torch.Tensor) 
            else torch.tensor(right)
        )

def transferLFMWeights(model, params):
    model.embedding.weight = assign(model.embedding.weight, 
        params["model.embed_tokens.weight"], "model.embed_tokens.weight")
    
    attn_layer_indeces = (2,5,8,10,12,14)

    for l in range(16):
        if l not in attn_layer_indeces:
             continue
        block = model.backbones[l]
        attn = block.core
        attn.Wq.weight = assign(attn.Wq.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight")
        attn.Wk.weight = assign(attn.Wk.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight")
        attn.Wv.weight = assign(attn.Wv.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight")
        # Output projection
        attn.Wo.weight = assign(attn.Wo.weight,
            params[f"model.layers.{l}.self_attn.out_proj.weight"],
            f"model.layers.{l}.self_attn.out_proj.weight")
        # QK norms
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            attn.q_norm.weight = assign(attn.q_norm.weight,
                params[f"model.layers.{l}.self_attn.q_layernorm.weight"],
                f"model.layers.{l}.self_attn.q_layernorm.weight")
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            attn.k_norm.weight = assign(attn.k_norm.weight,
                params[f"model.layers.{l}.self_attn.k_layernorm.weight"],
                f"model.layers.{l}.self_attn.k_layernorm.weight")
        # Attention layernorm
        block.norm1.weight = assign(block.norm1.weight,
            params[f"model.layers.{l}.operator_norm.weight"],
            f"model.layers.{l}.operator_norm.weight")
        # Feedforward weights
        block.ff.w1.weight = assign(
            block.ff.w1.weight,
            params[f"model.layers.{l}.feed_forward.w1.weight"],
            f"model.layers.{l}.feed_forward.w1.weight")
        block.ff.v.weight = assign(
            block.ff.v.weight,
            params[f"model.layers.{l}.feed_forward.w3.weight"],
            f"model.layers.{l}.feed_forward.w3.weight")
        block.ff.w2.weight = assign(
            block.ff.w2.weight,
            params[f"model.layers.{l}.feed_forward.w2.weight"],
            f"model.layers.{l}.feed_forward.w2.weight")
        block.norm2.weight = assign(
            block.norm2.weight,
            params[f"model.layers.{l}.ffn_norm.weight"],
            f"model.layers.{l}.ffn_norm.weight")
    
    for l in range(16):
        if l in attn_layer_indeces:
            continue 
        block = model.backbones[l]
        gate = block.core
        gate.conv.weight = assign(
            gate.conv.weight,
            params[f"model.layers.{l}.conv.conv.weight"],
            f"model.layers.{l}.conv.conv.weight")
        gate.w1.weight = assign(
            gate.w1.weight,
            params[f"model.layers.{l}.conv.in_proj.weight"],
            f"model.layers.{l}.conv.in_proj.weight")
        gate.w2.weight = assign(
            gate.w2.weight,
            params[f"model.layers.{l}.conv.out_proj.weight"],
            f"model.layers.{l}.conv.out_proj.weight")
        # Gated layernorm
        block.norm1.weight = assign(block.norm1.weight,
            params[f"model.layers.{l}.operator_norm.weight"],
            f"model.layers.{l}.operator_norm.weight")
        # Feedforward weights
        block.ff.w1.weight = assign(
            block.ff.w1.weight,
            params[f"model.layers.{l}.feed_forward.w1.weight"],
            f"model.layers.{l}.feed_forward.w1.weight")
        block.ff.v.weight = assign(
            block.ff.v.weight,
            params[f"model.layers.{l}.feed_forward.w3.weight"],
            f"model.layers.{l}.feed_forward.w3.weight")
        block.ff.w2.weight = assign(
            block.ff.w2.weight,
            params[f"model.layers.{l}.feed_forward.w2.weight"],
            f"model.layers.{l}.feed_forward.w2.weight")
        block.norm2.weight = assign(
            block.norm2.weight,
            params[f"model.layers.{l}.ffn_norm.weight"],
            f"model.layers.{l}.ffn_norm.weight")

    # Final normalization and output head
    model.norm_out.weight = assign(model.norm_out.weight, params["model.embedding_norm.weight"], "model.embedding_norm.weight")
    if "lm_head.weight" in params:
        model.lin_out.weight = assign(model.lin_out.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        print("Model uses weight tying.")
        model.lin_out.weight = assign(model.lin_out.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
