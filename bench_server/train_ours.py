"""Train BPE with our Rust trainer (bucket_bpe), through the Python binding rust_bpe_py.

  python3 train_ours.py <corpus.txt> <pretokenizer> <vocab_size> <out_dir> [normalizer]

Threads: BB_THREADS env var. SIMD or not: which build of rust_bpe_py is on PYTHONPATH (set by run.sh).
Writes out_dir/merges.txt (one line per merge step), out_dir/vocab.txt (final vocab),
and out_dir/tokenizer.json (HF format, built from those two after the timer stops; used for encoding).
Prints: the Rust stats line, then train_s=<seconds inside the Rust call>
"""
import os
import sys
import time

import rust_bpe_py
from tokenizers import Tokenizer, decoders, models

import normalizers
import pretokenizers


def byte_level_chars():
    """HF ByteLevel's byte -> printable char table (GPT-2's bytes_to_unicode)."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

corpus, pretok, vocab_size, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
norm = sys.argv[5] if len(sys.argv) > 5 else "none"
merges_txt, vocab_txt = os.path.join(out, "merges.txt"), os.path.join(out, "vocab.txt")

os.environ["BB_PRETOK"] = pretokenizers.OURS[pretok]
os.environ["BB_NORMALIZER"] = normalizers.spec(norm)
os.environ["BB_VOCAB_OUT"] = vocab_txt
os.environ["BB_MODEL_OUT"] = os.path.join(out, "model.txt")
os.environ["BB_SPECIAL"] = "" if pretok in pretokenizers.BYTE_LEVEL else "[UNK]"   # same as train_hf.py
print("rust_bpe_py build:", rust_bpe_py.build_variant(), flush=True)

t0 = time.perf_counter()
rust_bpe_py.train(["bucket_bpe", corpus, "0", vocab_size, merges_txt])
print(f"train_s={time.perf_counter() - t0:.3f}", flush=True)

# Not timed: wrap the result as an HF tokenizer.json so encode_*.py can load it. Read from model.txt
# (raw token bytes in hex: tokens may contain spaces or newlines, which merges.txt / vocab.txt can't carry).
toks = {}
merges = []
with open(os.path.join(out, "model.txt"), encoding="ascii") as f:
    for line in f:
        if line.strip() == "merges":
            break
        i, h = line.split()
        toks[int(i)] = bytes.fromhex(h)
    for line in f:
        l, r = line.split()
        merges.append((int(l), int(r)))
if pretok in pretokenizers.BYTE_LEVEL:
    bl = byte_level_chars()
    name = {i: "".join(bl[x] for x in b) for i, b in toks.items()}
else:
    name = {i: b.decode("utf-8") for i, b in toks.items()}
vocab = {name[i]: i for i in toks}
merges = [(name[l], name[r]) for l, r in merges]
if pretok in pretokenizers.BYTE_LEVEL:
    tk = Tokenizer(models.BPE(vocab, merges))
    tk.decoder = decoders.ByteLevel()
else:
    tk = Tokenizer(models.BPE(vocab, merges, unk_token="[UNK]"))
    tk.add_special_tokens(["[UNK]"])          # as HF's trainer does (special_tokens=["[UNK]"])
tk.pre_tokenizer = pretokenizers.make(pretok)
if norm != "none":
    tk.normalizer = normalizers.make(norm)
tk.save(os.path.join(out, "tokenizer.json"))
