"""Train BPE with HF `tokenizers`.

  python3 train_hf.py <corpus.txt> <pretokenizer> <vocab_size> <out_dir> [normalizer]

Threads: RAYON_NUM_THREADS env var (set by run.sh).
Writes out_dir/tokenizer.json, out_dir/merges.txt (one line per merge step), out_dir/vocab.txt (final vocab).
Prints: train_s=<seconds inside tk.train()>
"""
import os
import sys
import time

from tokenizers import Tokenizer, decoders, models, trainers

import common
import normalizers
import pretokenizers

corpus, pretok, vocab_size, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
norm = sys.argv[5] if len(sys.argv) > 5 else "none"

if pretok in pretokenizers.BYTE_LEVEL:
    tk = Tokenizer(models.BPE())
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False)
else:
    tk = Tokenizer(models.BPE(unk_token="[UNK]"))
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False, special_tokens=["[UNK]"])
tk.pre_tokenizer = pretokenizers.make(pretok)
if norm != "none":
    tk.normalizer = normalizers.make(norm)

t0 = time.perf_counter()
tk.train([corpus], trainer)
print(f"train_s={time.perf_counter() - t0:.3f}")

tk.save(os.path.join(out, "tokenizer.json"))
common.save_merges_and_vocab(os.path.join(out, "tokenizer.json"),
                             os.path.join(out, "merges.txt"), os.path.join(out, "vocab.txt"))
