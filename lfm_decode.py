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

colorprint = lambda x: print(f"\n\x1B[0;33m[{x} tokens/second]\x1B[0m")
def decode_next_token(
        model, tokenizer, init_tokens, hybrid_cache, 
        eos_token_id, top_k=None, temperature=1.0, 
        max_tokens=10):
    token_ids = init_tokens
    logits = model(init_tokens, hybrid_cache)[:, -1]

    avg_seconds_per_token = 0.0
    cum_avgs = []
    for index in range(max_tokens):
        next_token_id = sample_next_token(
            logits, top_k, temperature
        )
        start = time.perf_counter()
        if (eos_token_id is not None and 
            torch.all(next_token_id == eos_token_id)):
            break
        _next_token_id = next_token_id.flatten().tolist()
        next_token = tokenizer.decode(_next_token_id)
        print(next_token, flush=True, end="")
        token_ids = torch.cat((token_ids, next_token_id), dim=1)
        logits = model(next_token_id, hybrid_cache)[:, -1]
        end = time.perf_counter()
        avg_seconds_per_token = (1/(index + 1)) * (
            (index * avg_seconds_per_token) + (end - start)
        )
        cum_avgs.append(avg_seconds_per_token)
    colorprint(f"DECODE: {1/avg_seconds_per_token:.1f}")
    torch.save(torch.tensor(cum_avgs), "avgs_seconds_per_token.pt")
