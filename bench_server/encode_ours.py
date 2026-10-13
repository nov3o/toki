"""Encode a whole file with our Rust encoder (hf_bpe_enc, stream mode), through the Python binding rust_bpe_py.

  python3 encode_ours.py <tokenizer.json> <corpus.txt> <out_dir>

Threads: HB_THREADS env var. SIMD or not: which build of rust_bpe_py is on PYTHONPATH (set by run.sh).
Byte-level tokenizers only. Before the timer: tokenizer.json -> out_dir/model.hfbp (our flat format).
Writes the final token sequence (uint32 LE) to common.ids_path(out_dir) (RAM by default, see run.sh) and out_dir/merges_used.txt (merges, in order).
Prints: the Rust stats line, then encode_s=<seconds inside the Rust call (read + encode + write)> and tokens=<count>
"""
import os
import subprocess
import sys
import time

import rust_bpe_py

import common

tok_json, corpus, out = sys.argv[1], sys.argv[2], sys.argv[3]
here = os.path.dirname(os.path.abspath(__file__))
hfbp = os.path.join(out, "model.hfbp")
subprocess.run([sys.executable, os.path.join(here, "export_hfbp.py"), tok_json, hfbp], check=True)
print("rust_bpe_py build:", rust_bpe_py.build_variant(), flush=True)

t0 = time.perf_counter()
rust_bpe_py.encode(["hf_bpe_enc", hfbp, "stream", corpus, common.ids_path(out)])
print(f"encode_s={time.perf_counter() - t0:.3f}", flush=True)
print(f"tokens={os.path.getsize(common.ids_path(out)) // 4}")

common.save_merges_and_vocab(tok_json, os.path.join(out, "merges_used.txt"), os.path.join(out, "vocab_used.txt"))
