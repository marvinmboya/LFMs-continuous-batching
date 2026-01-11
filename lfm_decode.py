import torch 
import time 

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
    model, init_tokens, other_prompts, tokenizer, hybrid_cache, 
    eos_token_id, top_k=None, temperature=1.0, max_tokens=10, done=None):
    token_ids = init_tokens
    logits = model(init_tokens, hybrid_cache)[:, -1]

    avg_seconds_per_token = 0.0
    seq_len = init_tokens.size(1)
    for index in range(max_tokens):
        prompt_meta = (0, False, "")
        next_token_id = sample_next_token(
            logits, top_k, temperature
        )
        for i, each_token in enumerate(next_token_id):
            if each_token == eos_token_id:
                done[i] = True
                if other_prompts:
                    new_prompt = other_prompts.pop(0)
                    prompt_meta = (i, True, new_prompt)
                    new_token_id = tokenizer.encode(new_prompt)
                    new_batch_id = torch.empty(
                        1, seq_len + index, dtype=torch.int64,
                        device = init_tokens.device
                    ).fill_(eos_token_id)
                    sz = len(new_token_id)
                    st = (seq_len + index) - sz
                    new_token_id = torch.tensor(new_token_id)
                    new_batch_id[0, st:st + sz] = new_token_id
                    hybrid_cache.is_inter = True
                    hybrid_cache.inter_batch = i
                    logits = model(new_batch_id, hybrid_cache)[:, -1]
                    hybrid_cache.is_inter = False 
                    hybrid_cache.inter_batch = 0
                    new_prefill_id = sample_next_token(logits, top_k, temperature)
                    next_token_id[i, 0] = new_prefill_id.item()
                    done[i] = False
                    ##########
            if done[i]:
                next_token_id[i] = eos_token_id
        start = time.perf_counter()
        if (eos_token_id is not None and 
            torch.all(next_token_id == eos_token_id)):
            break
        _next_token_id = next_token_id.flatten().tolist()
        token_ids = torch.cat((token_ids, next_token_id), dim=1)
        logits = model(next_token_id, hybrid_cache)[:, -1]
        end = time.perf_counter()
        avg_seconds_per_token = (1/(index + 1)) * (
            (index * avg_seconds_per_token) + (end - start)
        )
        yield _next_token_id, prompt_meta, avg_seconds_per_token
    
    
