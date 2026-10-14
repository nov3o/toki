# toki — exact, fast BPE

[![Toki](https://img.youtube.com/vi/kbyIS9vdarM/hqdefault.jpg)](https://www.youtube.com/watch?v=kbyIS9vdarM)

BPE training and encoding that reproduce [HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers)
**bit for bit** (every merge, the vocab, every token id), faster and with less memory than the fastest code that
exists: HF `main` (the 1.0 rewrite) for encoding and Yikai Liao's parallel trainer
[huggingface/tokenizers#2501](https://github.com/huggingface/tokenizers/pull/2501) for training.

**Ours** combines both sides:
- **encoding**: HF `main`'s per-word engine (SIMD pre-tokenizer, word cache, merge engines) inside our driver: every
  thread finds its own chunk at a safe cut point, reads it with `pread`, encodes it and writes its ids; streaming, so
  memory stays flat;
- **training**: our counting (parallel chunked reads, arena word maps, bounded per-thread maps flushed into a sharded
  global map) feeding PR #2501's parallel merge loop.

| dir | what |
|---|---|
| [`ours/`](ours) | ours: `toki-encode` (built against HF `main`) and `toki-train` (against PR #2501): [`ours/build.sh`](ours/build.sh) |
| [`patches/`](patches) | the upstream patches: [HF `main`: encode one long input on every thread](patches/hf-main-encode-long-input.diff), [PR #2501: train from entries](patches/pr2501-do-train-entries.diff) |
| [`rust_bpe/`](rust_bpe) | pre-tokenizers, normalizers and safe cut points shared by ours; our first trainer and encoder |
| [`bench_server/`](bench_server) | benchmark ([`toki_axes.sh`](bench_server/toki_axes.sh), [`toki_tables.py`](bench_server/toki_tables.py)), HF drivers (`hfenc`, `hfbench`) |

## Experiments

**Default setting:** 1 GB (10⁹ bytes), 8 threads, GPT-2 byte-level pre-tokenizer, vocab 131 072, no normalizer,
2× AMD EPYC 7742. Each table below swaps **one** axis of the default and keeps the rest; every value runs on all 4
datasets. **Training**: ours vs PR #2501 (its Rust `BpeTrainer`, with HF's own pre-tokenizer and normalizers), on the
whole sample. **Encoding**: ours vs HF `main` (from Rust, its batch `encode`), on the held-out half of each sample, with
the vocab trained on the other half. Every run is pinned to exactly N cores (HF 1.0 ignores `RAYON_NUM_THREADS`);
all runs from Rust, no Python (see [Rust-only vs Python binding](#rust-only-vs-python-binding)); time = median of 2 runs, bytes/s = bytes processed (training: the whole sample; encoding: the held-out half) per second, MEM = peak memory in GB (ours: change vs the baseline in parentheses). Per-dataset tables:
[BY_DATASET.md](bench_server/results_toki/BY_DATASET.md).

**Exactly what is compared:**

| | version | how it is run |
|---|---|---|
| HF `main` (encoding baseline) | [`ff8ae1ba`](https://github.com/huggingface/tokenizers/commit/ff8ae1baabca0707b7fa9f22854652e231a933de) (2026-10, the 1.0 rewrite; not on PyPI) | from Rust, no Python: [`hfenc`](bench_server/hfenc/src/main.rs) cuts the whole file into ~256 KB pieces at safe cut points and passes them to HF's batch `encode`, which uses its own thread pool |
| PR #2501 (training baseline) | [`e4f787dc`](https://github.com/Yikai-Liao/tokenizers/commit/e4f787dc189d9be7192107490d652096cde7480e) (its head: `main` `ff8ae1ba` + the PR) | from Rust: [`hfbench`](bench_server/hfbench/src/main.rs) feeds each line (with its `\n`), normalized and pre-tokenized by HF's own parts from a `tokenizer.json`, to its `BpeTrainer` |
| ours | this repo, built against the two commits above | [`toki-encode`](ours/encode/src/main.rs), [`toki-train`](ours/train/src/main.rs) |

- **Same input for everyone:** the whole file. No baseline gets a smaller or pre-split input.
- **Time and memory:** wall time and peak RSS of the whole process (`/usr/bin/time`), including loading the
  tokenizer, reading the input and writing the ids or the vocab.
- **Correctness:** checked on every run. Our merges and vocab must equal PR #2501's, and our ids must equal HF
  `main`'s, token by token.
- **Build:** Rust 1.99.0, `--release`, no `target-cpu=native`, Linux 6.4.

### Data

| dataset | source |
|---|---|
| english | [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample/100BT) |
| multilingual | [fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), 7 languages |
| code | [github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean) |
| mix | english + multilingual + code, 1:1:1 |

Each sample is drawn at random from a 100 GB pool.

### Results (average of the 4 datasets)

<!-- tables -->
Equivalence: 216 training runs of ours, 216 with merges identical to PR #2501's; 216 encoding runs, 216 with ids identical to HF main's (token by token).

#### Default setting

| dataset | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| english | 5.21 | 192 MB/s | 0.37 | **2.48** (2.1×) | **402 MB/s** | **0.31** (−15%) | 1.80 | 278 MB/s | 0.69 | **0.99** (1.8×) | **503 MB/s** | **0.31** (−54%) |
| multilingual | 16.2 | 62 MB/s | 4.04 | **13.5** (1.2×) | **74 MB/s** | **3.98** (−1%) | 2.99 | 167 MB/s | 0.77 | **1.83** (1.6×) | **272 MB/s** | **0.37** (−52%) |
| code | 9.34 | 107 MB/s | 1.19 | **4.71** (2.0×) | **212 MB/s** | **0.87** (−27%) | 2.33 | 215 MB/s | 0.87 | **1.23** (1.9×) | **407 MB/s** | **0.37** (−57%) |
| mix | 11.1 | 90 MB/s | 2.19 | **7.88** (1.4×) | **127 MB/s** | **2.14** (−3%) | 2.75 | 182 MB/s | 0.80 | **1.46** (1.9×) | **342 MB/s** | **0.36** (−55%) |
| **average** | 10.5 | 95 MB/s | 1.95 | **7.15** (1.5×) | **140 MB/s** | **1.83** (−6%) | 2.47 | 203 MB/s | 0.78 | **1.38** (1.8×) | **362 MB/s** | **0.35** (−55%) |

#### Threads

| threads | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 27.9 | 36 MB/s | 1.84 | **17.3** (1.6×) | **58 MB/s** | **1.85** (+0%) | 4.47 | 112 MB/s | 0.73 | **3.71** (1.2×) | **135 MB/s** | **0.15** (−79%) |
| 4 | 15.7 | 64 MB/s | 1.87 | **10.5** (1.5×) | **96 MB/s** | **1.87** (+0%) | 3.12 | 160 MB/s | 0.74 | **2.20** (1.4×) | **227 MB/s** | **0.21** (−71%) |
| 8 | 10.5 | 95 MB/s | 1.95 | **7.15** (1.5×) | **140 MB/s** | **1.83** (−6%) | 2.47 | 203 MB/s | 0.78 | **1.38** (1.8×) | **362 MB/s** | **0.35** (−55%) |
| 16 | 8.47 | 118 MB/s | 2.17 | **5.43** (1.6×) | **184 MB/s** | **1.91** (−12%) | 2.04 | 245 MB/s | 0.84 | **0.99** (2.1×) | **507 MB/s** | **0.60** (−28%) |

#### Vocab

| vocab | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32k | 9.22 | 108 MB/s | 1.58 | **5.99** (1.5×) | **167 MB/s** | **1.60** (+1%) | 2.03 | 247 MB/s | 0.74 | **1.10** (1.8×) | **456 MB/s** | **0.34** (−54%) |
| 64k | 9.78 | 102 MB/s | 1.76 | **6.57** (1.5×) | **152 MB/s** | **1.71** (−3%) | 2.09 | 239 MB/s | 0.78 | **1.17** (1.8×) | **427 MB/s** | **0.34** (−56%) |
| 128k | 10.5 | 95 MB/s | 1.95 | **7.15** (1.5×) | **140 MB/s** | **1.83** (−6%) | 2.47 | 203 MB/s | 0.78 | **1.38** (1.8×) | **362 MB/s** | **0.35** (−55%) |
| 256k | 11.2 | 89 MB/s | 2.15 | **7.93** (1.4×) | **126 MB/s** | **1.98** (−8%) | 3.21 | 156 MB/s | 0.80 | **1.86** (1.7×) | **268 MB/s** | **0.37** (−54%) |

#### Pre-tokenizer

| pre-tokenizer | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bytelevel | 10.5 | 95 MB/s | 1.95 | **7.15** (1.5×) | **140 MB/s** | **1.83** (−6%) | 2.47 | 203 MB/s | 0.78 | **1.38** (1.8×) | **362 MB/s** | **0.35** (−55%) |
| llama3 | 10.8 | 93 MB/s | 2.18 | **7.45** (1.4×) | **134 MB/s** | **2.06** (−6%) | 2.39 | 209 MB/s | 0.78 | **1.41** (1.7×) | **355 MB/s** | **0.34** (−56%) |
| gpt4o | 10.6 | 95 MB/s | 2.02 | **7.54** (1.4×) | **133 MB/s** | **1.88** (−7%) | 2.44 | 205 MB/s | 0.77 | **1.43** (1.7×) | **349 MB/s** | **0.34** (−56%) |
| whitespace | 8.56 | 117 MB/s | 1.44 | **5.34** (1.6×) | **187 MB/s** | **1.28** (−11%) | 2.16 | 231 MB/s | 0.72 | **1.25** (1.7×) | **401 MB/s** | **0.33** (−54%) |
| whitespace_split | 12.2 | 82 MB/s | 2.71 | **8.98** (1.4×) | **111 MB/s** | **2.55** (−6%) | 2.68 | 186 MB/s | 0.72 | **1.79** (1.5×) | **279 MB/s** | **0.36** (−51%) |
| metaspace | 13.8 | 73 MB/s | 3.19 | **10.5** (1.3×) | **95 MB/s** | **3.09** (−3%) | 3.59 | 139 MB/s | 0.84 | **2.69** (1.3×) | **186 MB/s** | **0.57** (−32%) |
| bert | 8.16 | 123 MB/s | 1.53 | **7.66** (1.1×) | **131 MB/s** | **1.33** (−13%) | 2.17 | 230 MB/s | 0.73 | **1.20** (1.8×) | **417 MB/s** | **0.34** (−54%) |
| punctuation | 30.6 | 33 MB/s | 8.05 | **30.6** (1.0×) | **33 MB/s** | **7.68** (−5%) | 5.09 | 98 MB/s | 0.76 | **4.13** (1.2×) | **121 MB/s** | **0.36** (−52%) |
| digits | 37.8 | 26 MB/s | 9.72 | **34.3** (1.1×) | **29 MB/s** | **9.38** (−3%) | 6.51 | 77 MB/s | 0.77 | **5.69** (1.1×) | **88 MB/s** | **0.42** (−46%) |
| unicode_scripts | 29.3 | 34 MB/s | 7.90 | **29.9** (1.0×) | **33 MB/s** | **7.64** (−3%) | 4.83 | 104 MB/s | 0.73 | **3.95** (1.2×) | **127 MB/s** | **0.26** (−64%) |
| char_delimiter | 12.6 | 80 MB/s | 3.12 | **9.42** (1.3×) | **106 MB/s** | **2.98** (−4%) | 2.92 | 171 MB/s | 0.74 | **2.03** (1.4×) | **246 MB/s** | **0.38** (−49%) |

#### Normalizer

| normalizer | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 10.5 | 95 MB/s | 1.95 | **7.15** (1.5×) | **140 MB/s** | **1.83** (−6%) | 2.47 | 203 MB/s | 0.78 | **1.38** (1.8×) | **362 MB/s** | **0.35** (−55%) |
| nfc | 10.6 | 94 MB/s | 1.93 | **10.5** (1.0×) | **95 MB/s** | **1.83** (−5%) | 2.53 | 197 MB/s | 0.78 | **1.95** (1.3×) | **256 MB/s** | **0.36** (−54%) |
| nfd | 10.8 | 92 MB/s | 1.74 | **8.90** (1.2×) | **112 MB/s** | **1.65** (−5%) | 2.99 | 167 MB/s | 0.79 | **2.02** (1.5×) | **247 MB/s** | **0.41** (−48%) |
| nfkc | 10.9 | 91 MB/s | 1.95 | **10.3** (1.1×) | **97 MB/s** | **1.80** (−8%) | 3.41 | 147 MB/s | 0.78 | **2.39** (1.4×) | **209 MB/s** | **0.37** (−53%) |
| nfkd | 11.0 | 91 MB/s | 1.75 | **8.91** (1.2×) | **112 MB/s** | **1.65** (−6%) | 3.02 | 165 MB/s | 0.79 | **2.00** (1.5×) | **250 MB/s** | **0.41** (−49%) |
| lowercase | 10.7 | 93 MB/s | 1.82 | **7.74** (1.4×) | **129 MB/s** | **1.72** (−5%) | 2.65 | 189 MB/s | 0.78 | **1.62** (1.6×) | **308 MB/s** | **0.39** (−50%) |
| strip_accents | 11.3 | 88 MB/s | 1.95 | **9.46** (1.2×) | **106 MB/s** | **1.80** (−8%) | 3.20 | 156 MB/s | 0.79 | **2.20** (1.5×) | **227 MB/s** | **0.41** (−48%) |
| nfkc_lowercase | 11.2 | 89 MB/s | 1.82 | **10.5** (1.1×) | **95 MB/s** | **1.68** (−7%) | 3.63 | 138 MB/s | 0.78 | **2.63** (1.4×) | **190 MB/s** | **0.37** (−52%) |
| replace | 10.6 | 95 MB/s | 1.95 | **7.66** (1.4×) | **131 MB/s** | **1.83** (−6%) | 2.50 | 200 MB/s | 0.79 | **1.41** (1.8×) | **355 MB/s** | **0.37** (−54%) |
| bert | 13.7 | 73 MB/s | 0.82 | **10.5** (1.3×) | **96 MB/s** | **0.66** (−19%) | 5.11 | 98 MB/s | 0.78 | **4.08** (1.3×) | **123 MB/s** | **0.32** (−58%) |

#### Size

| size | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 GB | 10.5 | 95 MB/s | 1.95 | **7.15** (1.5×) | **140 MB/s** | **1.83** (−6%) | 2.47 | 203 MB/s | 0.78 | **1.38** (1.8×) | **362 MB/s** | **0.35** (−55%) |
| 10 GB | 105 | 95 MB/s | 11.34 | **59.3** (1.8×) | **169 MB/s** | **11.12** (−2%) | 49.7 | 201 MB/s | 9.72 | **28.5** (1.7×) | **351 MB/s** | **0.35** (−96%) |


<!-- /tables -->

### Training vs PR #2501 (its three figures, rerun)

256 MiB per corpus, vocab 100 000, pinned cores; merges identical in all 176 runs. Feed + train is 1.0-1.7× faster
(the counting); train alone is the same merge loop; peak memory 84-100% of PR #2501's at 1 GiB.

![feed + train](bench_server/results_plots/throughput_feed_train.png)
![train only](bench_server/results_plots/throughput_train.png)
![memory](bench_server/results_plots/memory.png)

### Rust-only vs Python binding

**Every other number in this README is Rust-only:** HF `main` and PR #2501 run from Rust drivers
([`hfenc`](bench_server/hfenc/src/main.rs), [`hfbench`](bench_server/hfbench/src/main.rs)), ours as the Rust tools; no
Python in the timed path. For the Python cost, the default setting (1 GB, 8 threads) from Python
([`py_bench.py`](bench_server/py_bench.py)): HF `main`'s and our fork's ([nov3o/tokenizers](https://github.com/nov3o/tokenizers))
`tokenizers` wheels, the same code as the Rust runs. Encoding: HF `main` gets the file in ~256 KB pieces through
`encode_batch`, ours the whole file through one `encode`; training: our `train_bpe`. PR #2501 has no Python training
API (the 1.0 bindings have no trainers). Ids and merges are the same as from Rust in every run. Seconds, Rust /
Python (Python ÷ Rust), median of 2 runs, whole process:

| | english | multilingual | code | mix |
|---|---|---|---|---|
| HF `main` encode | 2.21 / 5.18 (2.3×) | 3.33 / 6.72 (2.0×) | 2.97 / 15.44 (5.2×) | 2.86 / 8.81 (3.1×) |
| **ours** encode | 1.22 / 5.22 (4.3×) | 2.08 / 4.10 (2.0×) | 1.38 / 3.75 (2.7×) | 1.68 / 3.12 (1.9×) |
| **ours** train | 3.04 / 2.71 (0.9×) | 14.02 / 13.45 (1.0×) | 4.96 / 4.91 (1.0×) | 8.38 / 8.22 (1.0×) |
| **ours** `encode_file` | 1.22 / 1.29 (1.06×) | 2.08 / 2.23 (1.07×) | 1.38 / 1.50 (1.09×) | 1.68 / 1.94 (1.15×) |

Peak memory, GB, Rust / Python (Python ÷ Rust):

| | english | multilingual | code | mix |
|---|---|---|---|---|
| HF `main` encode | 0.69 / 3.72 (5.4×) | 0.77 / 3.33 (4.3×) | 0.86 / 3.93 (4.6×) | 0.80 / 3.75 (4.7×) |
| **ours** encode | 0.30 / 3.90 (13×) | 0.34 / 3.37 (9.9×) | 0.35 / 4.64 (13×) | 0.36 / 4.00 (11×) |
| **ours** `encode_file` | 0.30 / 0.24 (0.8×) | 0.34 / 0.26 (0.8×) | 0.35 / 0.29 (0.8×) | 0.36 / 0.29 (0.8×) |

Through `encode` / `encode_batch`, encoding from Python takes 2-5× the Rust time and ~3.5-4.5 GB instead of
0.3-0.9 GB, for both: the text is read into a Python string and every token comes back in an `Encoding` (ids, type
ids, mask, offsets). Our fork adds `Tokenizer.encode_file(path)`, which streams instead: Rust threads read and encode
the file piece by piece and the iterator yields each piece's ids, in file order, as a `uint32` numpy array (no
copy), running at most `4 × threads` pieces ahead of the consumer:

```python
for ids in tok.encode_file("corpus.txt", threads=8):
    ids.tofile(out)
```

From Python that is 6-15% over Rust-only, with less memory than HF `main` from Rust; ids are identical. Training from
Python costs nothing extra in time (it hands `train_bpe` a path and gets back the vocab and merges) and 0.1-0.9 GB more
memory (no mimalloc inside Python).

## Upstream

- **HF `main`, encoding** ([patch](patches/hf-main-encode-long-input.diff), 77 lines added, 7 removed): `main` splits a
  long input across threads only at special tokens, so one long text is encoded on one thread. The patch also cuts long
  text before a `\n` between ASCII letters/digits when the pre-tokenizer always ends a word there (GPT regexes,
  whitespace, BERT) and the normalizers work char by char. One 0.5 GB input on 16 cores: 3.95 s → 0.49 s; ids identical
  on all 68 tested configurations; `tk-encode` tests pass.
- **PR #2501, training** ([patch](patches/pr2501-do-train-entries.diff), 6 lines): `do_train_entries`, training from
  `(word, count)` entries without building a map of them.

## Reproduce

```bash
cd bench_server
PYTHON=python3.11 ./setup.sh
venv-x86_64/bin/python prepare_data.py data 100 1 10
BASE=0 ./enc_heldout.sh results_enc_ho3             # held-out halves and their vocabs
../ours/build.sh <tokenizers main checkout> <tokenizers PR #2501 checkout>
BASE=0 ./toki_axes.sh results_toki
python3 toki_tables.py results_toki/results.tsv results_toki
```

## History

I started by implementing models and ML algorithms from scratch. At the same time I got very excited about
pre-training, partly through my project [chsm8](https://huggingface.co/LegumMagister/chsm8): seeing which data leads
to which result made me read a lot about pre-training and its best practices. After a couple of models I moved on to
tokenizers, to figure out how they work.

I wrote my own BPE in C++ without looking at how it is done in practice; my competitive programming background helped
me work out how I would implement it. My implementation (not BPE itself) turned out a bit different, and with an
agent I made it more parallel: a 10-20× speedup for training and 10-33× for encoding.

When I was about to open a PR, I found [Yikai](https://github.com/Yikai-Liao)'s [PR](https://github.com/huggingface/tokenizers/pull/2501) and realized
I had compared against a year-old release: HF [`main`](https://github.com/huggingface/tokenizers) had moved on, Yikai's trainer is much faster than mine, and
`main`'s encoder is a bit faster than mine. Since we got our speedups in different ways, combining them should beat
both. So with an agent I built (HF `main`, mine) for encoding and (PR #2501, mine) for training.

I'm really glad [HF tokenizers](https://github.com/huggingface/tokenizers) made such a leap in just one year: compared with the [0.22 release](https://github.com/huggingface/tokenizers/releases/tag/v0.22.0), `main` encodes 12-20×
faster with about the same memory and trains 1.1-2.4× faster, and Yikai's PR #2501 trains 8-20× faster with 4-5× less
memory. Combined with mine, 10 GB is now encoded in seconds, not minutes.

In the beginning I could follow; by the end I no longer understood the changes going on:
[this](https://www.youtube.com/watch?v=W_dxteeedgs).
