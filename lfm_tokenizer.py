import re
from tokenizers import Tokenizer

class Lfm2Tokenizer:
    SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")
    SPECIALS = ["<|startoftext|>", "<|im_start|>", "<|im_end|>"]

    def __init__(self, file: str):
        self.tokenizer = Tokenizer.from_file(file)
        self.set_special_ids()
        self.set_eos_token()
        
    def encode(self, prompt):
        ids = []
        _wrap = self.wrap_chat(prompt)  
        if _wrap in self.special_to_id and "\n" not in _wrap:
            return [self.special_to_id[_wrap]]
        
        _split = self.SPLIT_RE.split(_wrap)
        _split = filter(None,_split)
        for part in _split:
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            else:
                ids.extend(self.tokenizer.encode(part).ids[1:])
        return ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def wrap_chat(self, prompt):
        s = (f"<|startoftext|>"
             f"<|im_start|>user\n{prompt}<|im_end|>\n"
             f"<|im_start|>assistant\n")
        return s 
    
    def set_special_ids(self):
        self.special_to_id = {t: self.tokenizer.token_to_id(t)
                              for t in self.SPECIALS}
    def set_eos_token(self):
        eos_token = "<|im_end|>"
        self.eos_token_id = self.special_to_id[eos_token]