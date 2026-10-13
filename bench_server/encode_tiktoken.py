"""Encode a whole file with OpenAI tiktoken (Rust core), using a vocab trained by HF.

  python3 encode_tiktoken.py <tokenizer.json> <corpus.txt> <pretokenizer> <threads> <out_dir>

Byte-level pre-tokenizers only (bytelevel = GPT-2, llama3 = Llama-3, gpt4o = GPT-4o regex): tiktoken works on bytes.
Vocab: HF token string -> raw bytes, rank = HF id, so ids are directly comparable with HF.
tiktoken needs every single byte in its vocab; bytes HF never saw get extra ranks after the vocab (never used).
Same pieces as encode_hf.py (common.pieces), 32 per encode_ordinary_batch call with <threads> threads.
Writes the final token sequence (uint32 LE) to common.ids_path(out_dir) (RAM by default, see run.sh) and out_dir/merges_used.txt.
Prints: encode_s=<seconds for read + encode + write> tokens=<count>
"""
import json
import os
import sys
import time

import tiktoken

import common
import pretokenizers

tok_json, corpus, pretok, threads, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
BATCH = 32


def byte_decoder():
    """HF ByteLevel writes each byte as a printable char; this maps the char back to the byte."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


# Not timed: HF vocab {token string: id} -> tiktoken ranks {token bytes: id}
dec = byte_decoder()
vocab = json.load(open(tok_json, encoding="utf-8"))["model"]["vocab"]
ranks = {bytes(dec[ch] for ch in s): i for s, i in vocab.items()}
nxt = max(ranks.values()) + 1
for b in range(256):
    if bytes([b]) not in ranks:
        ranks[bytes([b])] = nxt
        nxt += 1
enc = tiktoken.Encoding(name="bench", pat_str=pretokenizers.REGEX[pretok],
                        mergeable_ranks=ranks, special_tokens={})

tokens = 0
t0 = time.perf_counter()
with open(common.ids_path(out), "wb") as f:
    batch = []
    for p in common.pieces(corpus, pretok):
        batch.append(p)
        if len(batch) == BATCH:
            for ids in enc.encode_ordinary_batch(batch, num_threads=threads):
                common.write_ids(f, ids)
                tokens += len(ids)
            batch = []
    for ids in enc.encode_ordinary_batch(batch, num_threads=threads):
        common.write_ids(f, ids)
        tokens += len(ids)
print(f"encode_s={time.perf_counter() - t0:.3f} tokens={tokens}")

common.save_merges_and_vocab(tok_json, os.path.join(out, "merges_used.txt"), os.path.join(out, "vocab_used.txt"))
