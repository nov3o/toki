# toki — exact, fast BPE

[![Toki](https://img.youtube.com/vi/kbyIS9vdarM/hqdefault.jpg)](https://www.youtube.com/watch?v=kbyIS9vdarM)

BPE training and encoding that reproduce [HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers)
**bit for bit** (every merge, the vocab, every token id), measured against the fastest code that exists today:
HF `main` (the 1.0 rewrite) and Yikai-Liao's parallel trainer [huggingface/tokenizers#2501](https://github.com/huggingface/tokenizers/pull/2501).

**Where we stand.** Our own encoder and trainer are 10-30× faster than the `tokenizers` 0.22 release, but HF `main`
has since caught up: its encoder is 3× faster than ours per core, and PR #2501's merge loop is parallel where ours is
serial. Each side does something the other does not, so we combined them:

| | what it takes from HF | what it takes from us | vs the best of the two |
|---|---|---|---|
| **union encoder** ([`union/encode`](union/encode)) | `main`'s per-word engine (SIMD pre-tokenizer, word cache, whole-word fold, merge engines) | the driver: parallel `pread` at safe cut points, work stealing, streaming, map-reduce output | **14× HF main, 3.6× ours** (10 GB, 64 cores) |
| **union trainer** ([`union/train`](union/train)) | PR #2501's parallel merge loop | parallel counting (chunked `pread`, arena word maps, sharded parallel reduce) | **1.9× PR #2501** (10 GB, 16 cores) |

Ids and merges of both unions are identical to HF's in every run below.

| dir | what |
|---|---|
| [`union/`](union) | the two unions (Rust, built against HF `main` and PR #2501: [`union/build.sh`](union/build.sh)) |
| [`rust_bpe/`](rust_bpe) | our own trainer `bucket_bpe`, encoder `hf_bpe_enc`, pre-tokenizers, normalizers |
| [`rust_bpe_py/`](rust_bpe_py) | Python binding (pyo3) |
| [`bench_server/`](bench_server) | benchmark harness and token-by-token equivalence check (`verify.py`); Rust drivers for HF main (`hfenc`, `hfbench`) |
| [`PLAN.md`](PLAN.md) | what is done and what is next |

## Which HF version

| version | released | fast 1.0 encoder | parallel trainer |
|---|---|---|---|
| 0.22.0 | 2025-08-29 | – | – |
| 0.23.3 (latest `pip install tokenizers`) | 2026-10-09 | – | – |
| 1.0.0rc1 / rc2 (`pip install --pre`, no Python training API) | 2026-09-21 | ✅ | – |
| `main` @ ff8ae1b (what we benchmark) | 2026-10-07 | ✅ | – |
| PR #2501 (open) | 2026-10-09 | ✅ | ✅ |

The 0.2x releases come from a maintenance branch; the 1.0 rewrite (`tk-encode` / `tk-train`) only exists on `main` and
in the release candidates. Its encoder speedups landed July-September 2026 (fast encode PoC, word cache, merge engine,
whole-word fold, SIMD pre-tokenizer "bitcannon", per-thread scratch pools).

**Threads:** HF 1.0 ignores `RAYON_NUM_THREADS` and uses every core it may run on. Every run here is pinned to exactly N
cores (`taskset`); pinning and `set_num_threads(N)` give the same times within 2 %.

## Union encoder

`union-encode`: each worker takes the next ~1 MB chunk, finds its edges itself (first safe cut point after the chunk
start, where pre-tokenization restarts identically), reads it with `pread`, encodes it with HF main's `encode_into`
(serial on the worker, no `Encoding` objects) and writes its ids. Nothing is serial but a counter. Output: one file
(each chunk `pwrite`s at its prefix-sum offset) or, map-reduce style, one shard per worker plus an index
(`HB_SHARDS=1`; concatenating the shards in index order gives the same file).

English, GPT-2 byte-level, vocab 128k (trained on held-out text), wall time / peak memory, ids identical:

| 10 GB | HF main | ours | **union** |
|---|---|---|---|
| 16 cores | 22.9 s / 10.1 GB | 24.1 s / 0.19 GB | **7.3 s / 0.6 GB** |
| 64 cores | 35.3 s / 10.4 GB | 8.9 s / 0.68 GB | **2.4 s / 2.0 GB** |

| 0.5 GB | HF main | ours | **union** |
|---|---|---|---|
| 1 core | 4.1 s / 0.61 GB | 12.8 s / 0.10 GB | **3.9 s / 0.11 GB** |
| 8 cores | 1.87 s / 0.72 GB | 2.52 s / 0.11 GB | **1.29 s / 0.26 GB** |
| 16 cores | 1.70 s / 0.75 GB | 1.73 s / 0.15 GB | **0.98 s / 0.44 GB** |

What limits each:
- **HF main** reads, cuts and writes on one thread (and holds the whole file): it stops scaling at ~1.7 s per 0.5 GB,
  and gets slower past 16 cores.
- **ours** scales (7.4× from 1 to 16 cores) but its per-word engine costs ~3× HF's per core (one heap for every
  word, a one-slot cache).
- **union**: 10 GB at 64 cores is 1.8 s of encoding (5.5 GB/s) plus 0.6 s of model loading, which is still serial
  (JSON parse + table build in HF). Writing one shared output file instead of shards costs 2.5× at 64 cores (kernel
  contention on the file), hence the sharded output.

## Union trainer

`union-train`: workers `pread` chunks cut at safe points, pre-tokenize line by line, and count words in per-thread
arena hash maps split into 64 shards by hash; shard k of every thread is reduced by one thread in parallel, then the
word counts go to PR #2501's `BpeTrainer::do_train`. Merges identical to HF 0.22, HF main, PR #2501 and ours.

English, GPT-2 byte-level, vocab 128k:

| | PR #2501 | ours | **union** |
|---|---|---|---|
| 0.5 GB, 1 core | 12.8 s | 6.6 s | 7.5 s |
| 0.5 GB, 16 cores | 2.17 s | 2.13 s | **1.37 s** |
| 10 GB, 16 cores | 27.2 s | – | **14.4 s** |
| 10 GB, 64 cores | – | – | **7.3 s** |

0.5 GB at 16 cores: counting 0.50 s (PR #2501: 1.30 s, ours: 0.55 s) + merging 0.81 s (PR #2501's loop; ours is a
serial 1.26 s). The merge loop is now the limit: it stops scaling at ~8 cores.

## Held-out encoding benchmark: HF main vs tiktoken vs ours

Each 1 GB dataset sample is split at a line boundary into a train half and a test half (0.5 GB each). For every
setting a vocab is trained on the train half of that dataset (byte-level vocabs get all 256 bytes), and every encoder
encodes the test half. One axis varies at a time from the default (8 threads, byte-level, vocab 128k, no normalizer).
**380 encodings compared token by token with HF main's ids: 380 identical.**

Average of the 4 datasets (all axes: [AVERAGE.md](bench_server/results_enc_heldout/AVERAGE.md), per dataset:
[BY_DATASET.md](bench_server/results_enc_heldout/BY_DATASET.md)), wall time / peak memory:

| threads | HF main | tiktoken | ours |
|---|---|---|---|
| 2 | **4.54 s** / 0.73 GB | 31.5 s / 0.21 GB | 10.3 s / **0.10 GB** |
| 4 | **3.21 s** / 0.74 GB | 18.9 s / 0.22 GB | 5.70 s / **0.11 GB** |
| 8 | **2.47 s** / 0.78 GB | 13.0 s / 0.22 GB | 3.33 s / **0.12 GB** |
| 16 | **2.11 s** / 0.84 GB | 11.4 s / 0.23 GB | 2.14 s / **0.15 GB** |

HF main is the fastest of the three at every setting (ours ties it only at 16 threads); ours uses 5-7× less memory
and is 3-5× faster than tiktoken. That is why the union keeps HF's engine.

### Data

| dataset | source |
|---|---|
| english | [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample/100BT) |
| multilingual | [fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), 7 languages |
| code | [github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean) |
| mix | english + multilingual + code, 1:1:1 |

Each sample is drawn at random from a 100 GB pool. Machine: 2× AMD EPYC 7742 (128 cores, AVX2).

## Against the 0.22 release

Before HF main existed as a baseline, all comparisons were against 0.22 (still what most users have). Default setting,
1 GB, 8 threads, average of the 4 datasets: ours trains ~10× faster with 3-4× less memory and encodes 10-33× faster.
PR #2501's own setting (256 MiB, vocab 100k), training, 8 cores (HF main and PR #2501 unpinned in this run, so they
may have used more cores; rerun pending):

| 256 MiB | HF 0.22 | HF main | PR #2501 | ours |
|---|---|---|---|---|
| english | 15.3 s / 0.66 GB | 7.8 s / 0.65 GB | 2.07 s / 0.16 GB | 1.95 s / 0.25 GB |
| code | 30.2 s / 1.53 GB | 22.6 s / 1.53 GB | 3.43 s / 0.33 GB | 4.09 s / 0.61 GB |
| multilingual | 109 s / 8.09 GB | 100 s / 8.09 GB | 4.38 s / 1.80 GB | 14.5 s / 2.29 GB |
| mix | 54.8 s / 3.82 GB | 43.8 s / 3.78 GB | 3.47 s / 0.93 GB | 7.50 s / 1.33 GB |

The full 0.22-era tables (11 pre-tokenizers, 12 normalizers, 10 GB, scaling laws) are in
[bench_server/tables/SUMMARY.md](bench_server/tables/SUMMARY.md) and [SCALING_LAWS.md](SCALING_LAWS.md).

## Other tokenizers

SentencePiece (Unigram, BPE) and WordPiece are different algorithms with different tokens, so they are compared on
speed and memory only (1 GB, 8 threads, vocab 131 072; HF rows = 0.22):

| | english train | english encode | mix train | mix encode |
|---|---|---|---|---|
| ours (BPE) | 4.5 s / 0.44 GB | 4.0 s / 0.10 GB | 22.4 s / 3.20 GB | 6.0 s / 0.12 GB |
| HF WordPiece | 51.2 s / 1.13 GB | 44.3 s / 0.47 GB | 127 s / 7.31 GB | 50.3 s / 0.58 GB |
| SentencePiece BPE | 106 s / 3.51 GB | 51.0 s / 0.11 GB | 415 s / 6.60 GB | 53.6 s / 0.08 GB |
| HF Unigram | 310 s / 1.95 GB | 159 s / 0.50 GB | crashed | – |
| SentencePiece Unigram | 1660 s / 21.83 GB | 62.2 s / 0.14 GB | > 60 min | – |

## Next (details: [PLAN.md](PLAN.md))

- union encoder: remove the 0.6 s serial model load (binary model file, or a parallel table build in HF)
- union trainer: profile PR #2501's merge loop past 8 cores; counting on few cores (1 core: 6.1 s vs ours 4.3 s)
- both unions on all 4 datasets, all pre-tokenizers / normalizers, 1 GB and 10 GB
- upstream: the driver ideas (parallel safe-cut reading, streaming, sharded output) as PRs to HF main; credit PR #2501

## Reproduce

```bash
cd bench_server
PYTHON=python3.11 ./setup.sh
venv-x86_64/bin/python prepare_data.py data 100 1 10
BASE=0 ./enc_heldout.sh results_enc_ho     # held-out encoding benchmark (needs hfenc built in hfenc-main)
python3 tables_heldout.py results_enc_ho/results.tsv results_enc_ho
../union/build.sh <tokenizers main checkout> <tokenizers PR #2501 checkout>
HB_THREADS=16 HB_SHARDS=1 taskset -c 0-15 ../union/encode/target/release/union-encode tokenizer.json data/english_10GB.txt bytelevel out
HB_THREADS=16 taskset -c 0-15 ../union/train/target/release/union-train data/english_10GB.txt 131072 vocab_out
```
