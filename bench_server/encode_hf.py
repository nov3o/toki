"""Encode a whole file with HF `tokenizers`.

  python3 encode_hf.py <tokenizer.json> <corpus.txt> <out_dir> [pretokenizer]

Threads: RAYON_NUM_THREADS env var (set by run.sh).
The file is encoded as ONE text, exactly like our encoder: it is cut into ~256 KB pieces only at points where
pre-tokenization restarts identically ([A-Za-z0-9] '\n' | [A-Za-z0-9]), and 32 pieces go to encode_batch_fast at a time.
Writes the final token sequence (uint32 LE) to common.ids_path(out_dir) (RAM by default, see run.sh) and out_dir/merges_used.txt (merges, in order).
Prints: encode_s=<seconds for read + encode + write> tokens=<count>
"""
import os
import sys
import time

from tokenizers import Tokenizer

import common

tok_json, corpus, out = sys.argv[1], sys.argv[2], sys.argv[3]
pretok = sys.argv[4] if len(sys.argv) > 4 else "bytelevel"   # decides where the file may be cut (common.safe_cut_at)
BATCH = 32           # pieces per encode_batch_fast call (~8 MB of text: HF keeps a heavy object per token)


tk = Tokenizer.from_file(tok_json)
tokens = 0
t0 = time.perf_counter()
with open(common.ids_path(out), "wb") as f:
    batch = []
    for p in common.pieces(corpus, pretok):
        batch.append(p)
        if len(batch) == BATCH:
            for enc in tk.encode_batch_fast(batch, add_special_tokens=False):
                common.write_ids(f, enc.ids)
                tokens += len(enc.ids)
            batch = []
    for enc in tk.encode_batch_fast(batch, add_special_tokens=False):
        common.write_ids(f, enc.ids)
        tokens += len(enc.ids)
print(f"encode_s={time.perf_counter() - t0:.3f} tokens={tokens}")

common.save_merges_and_vocab(tok_json, os.path.join(out, "merges_used.txt"), os.path.join(out, "vocab_used.txt"))
