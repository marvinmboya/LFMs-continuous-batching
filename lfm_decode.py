import torch 

def sample_next_token(logits, top_k=None, temperature=1.0):
    if top_k is not None:
        top_logits, _ = torch.topk(logits, top_k)
        min_val = top_logits[:, -1]
        logits = torch.where(logits < min_val, -torch.inf, logits)
    if temperature > 0.0:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
    else:
        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
    return next_token_id

def decode_next_token(
        model, tokenizer, init_tokens, eos_token_id, 
        top_k=None, temperature=1.0, max_tokens=10,
    ):
    token_ids = init_tokens
    logits = model(init_tokens)[:, -1]
    for _ in range(max_tokens):
        next_token_id = sample_next_token(
            logits, top_k, temperature
        )
        if (eos_token_id is not None and 
            torch.all(next_token_id == eos_token_id)):
            break
        _next_token_id = next_token_id.flatten().tolist()
        next_token = tokenizer.decode(_next_token_id)
        print(next_token, flush=True, end="")
        token_ids = torch.cat((token_ids, next_token_id), dim=1)
        logits = model(token_ids)[:, -1]
