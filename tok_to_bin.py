import json
import struct

def saveLFMTokenizerToBin(json_path, output_path):
    """Save tokenizer to compact binary format"""
    with open(json_path, "r", encoding="utf-8") as f:
        tok = json.load(f)
    
    vocab = tok["model"]["vocab"]
    merges = tok["model"]["merges"]
    
    with open(output_path, "wb") as f:
        # Header: magic number + vocab size + merges count
        f.write(b"BTOK")  # Magic number
        f.write(struct.pack("<I", len(vocab)))
        f.write(struct.pack("<I", len(merges)))
        
        # Write vocab: for each token write (id, length, utf8_bytes)
        for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
            token_bytes = token.encode("utf-8")
            f.write(struct.pack("<I", token_id))
            f.write(struct.pack("<H", len(token_bytes)))
            f.write(token_bytes)
        
        # Write merges: for each merge write (len1, bytes1, len2, bytes2)
        for pair in merges:
            if len(pair) != 2:
                raise ValueError(f"Invalid merge: {pair}")
            b1 = pair[0].encode("utf-8")
            b2 = pair[1].encode("utf-8")
            f.write(struct.pack("<H", len(b1)))
            f.write(b1)
            f.write(struct.pack("<H", len(b2)))
            f.write(b2)