# toki — exact, fast BPE

[![Toki](https://img.youtube.com/vi/kbyIS9vdarM/hqdefault.jpg)](https://www.youtube.com/watch?v=kbyIS9vdarM)

A BPE trainer and encoder that reproduce [HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers) **bit for bit**
(every merge step, the final vocab, every token id) and run ~10× faster to train and ~15× faster to encode,
with 3–4× less memory. Encoding is also faster than [OpenAI `tiktoken`](https://github.com/openai/tiktoken).

| dir | what |
|---|---|
| [`rust_bpe/`](rust_bpe) | the algorithm: trainer `bucket_bpe`, encoder `hf_bpe_enc`, pre-tokenizers, normalizers, SIMD |
| [`rust_bpe_py/`](rust_bpe_py) | Python binding (pyo3), built with and without SIMD |
| [`bench_server/`](bench_server) | benchmark harness (`run.sh`) + equivalence check (`verify.py`); see its [README](bench_server/README.md) |
| [`bench_server/tables/SUMMARY.md`](bench_server/tables/SUMMARY.md) | every finished run: time + peak RAM per implementation |

## Experiments

**Default setting:** 1 GB (10⁹ bytes), 8 threads, GPT-2 byte-level pre-tokenizer, vocab 131 072, no normalizer,
AMD EPYC 7742 (AVX2). Each table below swaps **one** axis of the default for other values and keeps the rest;
every value is run on all 4 datasets. HF = `tokenizers` 0.22; `tiktoken` encodes with HF's trained vocab.

**Equivalence:** we tested 116 settings (dataset × configuration). In every one of them our merges, vocab,
merges used and token ids were identical to HF's (1 048 output files compared, 0 different; 10 GB ids compared token by token); tiktoken's ids too, where it applies.
Ids were checked by sha256; 9 of the settings were re-encoded and compared token by token (22 comparisons, up to 330 M ids each, all identical).

### Data

| dataset | source | 1 GB | 10 GB |
|---|---|---|---|
| english | [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample/100BT) | ✅ | 🕐 |
| multilingual | [fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), 7 languages | ✅ | 🕐 |
| code | [github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean) | ✅ | 🕐 |
| mix | english + multilingual + code, 1:1:1 | ✅ | 🕐 |

Each sample is drawn at random from a 100 GB pool (10× the largest size). ✅ done · 🕐 yet to be done

### Headline: the standard pipeline at its best setting

llama3 pre-tokenizer, no normalizer (what current models use), English 1 GB, vocab 32k; threads chosen per task.

| | HF | tiktoken | **ours** |
|---|---|---|---|
| train, 2 threads | 140 s / 1.32 GB | – | **13.2 s (10.6×) / 0.38 GB** |
| train, 16 threads | 35.0 s / 1.50 GB | – | **3.3 s (10.5×) / 0.75 GB** |
| encode, 2 threads | 186 s / 0.30 GB | 86.2 s / 0.12 GB | **19.4 s (9.6×, 4.4×) / 0.03 GB** |
| encode, 16 threads | 60.9 s / 0.79 GB | 33.1 s / 0.13 GB | **1.8 s (33.7×, 18.3×) / 0.13 GB** |

**Ours encodes 1 GB of English in 1.8 s: 34× faster than HF and 18× faster than tiktoken.**
The best training speedup measured on any single axis is **18.6×** (English, 2 threads, GPT-2 byte-level, vocab 128k).
These runs shared the machine (on separate cores) with the 10 GB runs; a rerun alone is queued.

### Results (average over the 4 datasets; s = wall time, MEM = peak memory in GB)

**ours** in parentheses: speedup vs HF, then vs tiktoken.
`ours` is the SIMD build; a build without SIMD was within 2-3 %, so SIMD had little effect.
🕐 = not run yet; n/a = not supported (tiktoken has only regex pre-tokenizers and no normalizers; prepend/strip are train-only).
Per-dataset numbers with throughput (MB/s): [BY_DATASET.md](bench_server/tables/BY_DATASET.md).

#### Best setting per axis

| axis | best for training | train speedup | best for encoding | enc speedup vs HF | vs tiktoken |
|---|---|---|---|---|---|
| threads | 2, english | **18.6×** | 16, code | **33.5×** | 14.8× |
| vocab | 32k, english | **13.6×** | 32k, english | **29.2×** | 6.8× |
| pre-tokenizer | gpt4o, code | **12.6×** | bytelevel, english | **24.4×** | 5.9× |
| normalizer | strip, english | **11.7×** | none, english | **24.2×** | 5.7× |

#### Threads

| threads | HF train s | MEM | **ours** train s | MEM | HF enc s | MEM | tiktoken enc s | MEM | **ours** enc s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 311 | 9.73 | **24.8** (12.6×) | **2.91** | 195 | 0.47 | 65.0 | 0.19 | **19.4** (10.1×, 3.3×) | **0.10** |
| 4 | 280 | 10.15 | **22.2** (12.6×) | **2.93** | 121 | 0.58 | 39.2 | 0.18 | **10.3** (11.8×, 3.8×) | **0.10** |
| 8 | 220 | 10.62 | **20.7** (10.6×) | **2.97** | 84.9 | 0.77 | 27.4 | 0.17 | **5.6** (15.0×, 4.9×) | **0.11** |
| 16 | 213 | 11.17 | **21.5** (9.9×) | **3.04** | 71.6 | 1.21 | 28.3 | 0.17 | **3.3** (21.6×, 8.5×) | **0.15** |

#### Vocab

| vocab | HF train s | MEM | **ours** train s | MEM | HF enc s | MEM | tiktoken enc s | MEM | **ours** enc s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|
| 32k | 199 | 10.06 | **18.0** (11.0×) | **2.42** | 84.8 | 0.72 | 28.7 | 0.14 | **4.7** (18.0×, 6.1×) | **0.08** |
| 64k | 220 | 10.36 | **19.3** (11.4×) | **2.63** | 86.3 | 0.72 | 27.9 | 0.15 | **5.1** (16.8×, 5.4×) | **0.09** |
| 128k | 220 | 10.62 | **20.7** (10.6×) | **2.97** | 84.9 | 0.77 | 27.4 | 0.17 | **5.6** (15.0×, 4.9×) | **0.11** |
| 256k | 232 | 10.97 | **23.0** (10.1×) | **3.14** | 85.3 | 0.87 | 28.0 | 0.23 | **6.6** (12.8×, 4.2×) | **0.18** |

#### Pre-tokenizer

| pre-tokenizer | HF train s | MEM | **ours** train s | MEM | HF enc s | MEM | tiktoken enc s | MEM | **ours** enc s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|
| bytelevel | 219 | 10.78 | **20.6** (10.6×) | **2.97** | 88.1 | 0.77 | 27.9 | 0.17 | **5.6** (15.7×, 5.0×) | **0.11** |
| llama3 | 238 | 11.55 | **22.2** (10.7×) | **3.09** | 81.2 | 0.78 | 24.1 | 0.18 | **5.8** (14.0×, 4.2×) | **0.10** |
| gpt4o | 215 | 10.40 | **27.7** (7.8×) | **2.90** | 85.9 | 0.76 | 23.1 | 0.21 | **7.9** (10.9×, 2.9×) | **0.11** |
| whitespace | 134 | 7.02 | **17.7** (7.6×) | **1.96** | 57.7 | 0.59 | n/a | n/a | **4.5** (12.8×) | **0.13** |
| whitespace_split | 298 | 14.62 | **39.2** (7.6×) | **3.79** | 48.7 | 0.51 | n/a | n/a | **6.6** (7.3×) | **0.13** |
| metaspace | 354 | 17.30 | **44.4** (8.0×) | **3.98** | 74.2 | 0.74 | n/a | n/a | **9.2** (8.1×) | **0.15** |
| bert | 122 | 6.13 | **18.9** (6.5×) | **1.83** | 73.2 | 0.59 | n/a | n/a | **7.4** (9.9×) | **0.12** |
| punctuation | 1052 | 48.09 | **141** (7.5×) | **10.71** | 60.2 | 0.56 | n/a | n/a | **18.1** (3.3×) | **0.13** |
| digits | 1319 | 57.54 | **196** (6.7×) | **14.28** | 43.7 | 0.56 | n/a | n/a | **25.2** (1.7×) | **0.15** |
| unicode_scripts | 967 | 46.91 | **133** (7.3×) | **10.52** | 52.3 | 0.55 | n/a | n/a | **17.0** (3.1×) | **0.13** |
| char_delimiter | 306 | 15.79 | **41.7** (7.3×) | **3.89** | 47.9 | 0.51 | n/a | n/a | **7.4** (6.5×) | **0.13** |

#### Normalizer

| normalizer | HF train s | MEM | **ours** train s | MEM | HF enc s | MEM | tiktoken enc s | MEM | **ours** enc s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 220 | 10.77 | **21.1** (10.4×) | **2.96** | 84.8 | 0.77 | 27.7 | 0.17 | **5.6** (15.1×, 4.9×) | **0.11** |
| nfc | 225 | 10.57 | **27.3** (8.2×) | **2.97** | 84.8 | 0.78 | n/a | n/a | **8.9** (9.5×) | **0.11** |
| nfd | 212 | 9.58 | **25.9** (8.2×) | **2.78** | 88.1 | 0.82 | n/a | n/a | **7.5** (11.7×) | **0.14** |
| nfkc | 231 | 10.47 | **27.3** (8.5×) | **2.97** | 89.5 | 0.82 | n/a | n/a | **8.8** (10.1×) | **0.11** |
| nfkd | 211 | 9.55 | **26.3** (8.0×) | **2.75** | 91.7 | 0.79 | n/a | n/a | **7.6** (12.1×) | **0.13** |
| lowercase | 216 | 10.47 | **24.6** (8.8×) | **2.92** | 85.7 | 0.75 | n/a | n/a | **7.3** (11.7×) | **0.11** |
| strip_accents | 215 | 10.51 | **38.5** (5.6×) | **2.98** | 84.5 | 0.79 | n/a | n/a | **7.9** (10.7×) | **0.13** |
| nfkc_lowercase | 229 | 10.11 | **40.0** (5.7×) | **2.89** | 89.6 | 0.79 | n/a | n/a | **9.9** (9.0×) | **0.12** |
| replace | 220 | 10.46 | **23.3** (9.4×) | **2.97** | 81.5 | 0.76 | n/a | n/a | **6.0** (13.6×) | **0.14** |
| bert | 127 | 4.14 | **18.5** (6.9×) | **1.19** | 108 | 0.86 | n/a | n/a | **11.6** (9.3×) | **0.14** |
| prepend | 226 | 10.64 | **22.5** (10.0×) | **2.99** | n/a | n/a | n/a | n/a | n/a | n/a |
| strip | 230 | 10.85 | **21.3** (10.8×) | **2.97** | n/a | n/a | n/a | n/a | n/a | n/a |

#### Size (per dataset; 10 GB tiktoken ran next to other experiments, a rerun alone is queued)

| size | dataset | HF train s | MEM | **ours** train s | MEM | HF enc s | MEM | tiktoken enc s | MEM | **ours** enc s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 GB | english | 52.4 | 1.38 | **4.5** (11.6×) | **0.44** | 92.7 | 0.60 | 22.1 | 0.15 | **4.0** (23.5×, 5.6×) | **0.10** |
|  | multilingual | 484 | 25.49 | **45.1** (10.7×) | **7.00** | 70.7 | 0.63 | 23.7 | 0.16 | **8.5** (8.3×, 2.8×) | **0.11** |
|  | code | 109 | 4.19 | **10.9** (10.0×) | **1.24** | 91.4 | 1.01 | 35.8 | 0.19 | **4.1** (22.2×, 8.7×) | **0.12** |
|  | mix | 236 | 11.42 | **22.4** (10.5×) | **3.20** | 84.9 | 0.84 | 28.2 | 0.18 | **6.0** (14.1×, 4.7×) | **0.12** |
| 10 GB | english | 426 | 5.60 | **32.7** (13.0×) | **1.47** | 740 | 0.68 | 468 | 0.18 | **41.5** (17.8×, 11.3×) | **0.09** |
|  | multilingual | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
|  | code | 1169 | 24.98 | **97.3** (12.0×) | **5.50** | 1374 | 1.13 | 565 | 0.29 | **42.8** (32.2×, 13.2×) | **0.12** |
|  | mix | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |

### Still to do

| | status |
|---|---|
| 10 GB baseline: english, code ✅; multilingual, mix | 🕐 running |
| tiktoken 10 GB and the headline runs, rerun with nothing else on the machine | 🕐 queued |
| token-by-token id check for all 1 GB settings (done for 11 of 114 and all 10 GB) | 🕐 |
| minimal patches to HF `tokenizers` and `tiktoken`, before vs after: [PATCH_EXPERIMENTS.md](PATCH_EXPERIMENTS.md) | 🕐 designed |
| faster: gpt4o training, normalizers, character-level encoders, multilingual training memory | 🕐 |

## Scaling laws (1 GB, average of the 4 datasets; power-law fits in [SCALING_LAWS.md](SCALING_LAWS.md))

Threads: fit of time = serial + parallel / threads (Amdahl) over 2-16 threads. Vocab: time growth per doubling, 32k → 256k.

| | serial share | max speedup from threads | 2 → 16 threads | per vocab doubling |
|---|---|---|---|---|
| HF train | 46 % | 2.2× | 1.4× faster | +5 % |
| **ours** train | 70 % | 1.4× | 1.2× faster | +8 % |
| HF encode | 15 % | 6.6× | 2.7× faster | +0 % |
| tiktoken encode | 18 % | 5.7× | 2.3× faster | −1 % |
| **ours** encode | **3 %** | **36×** | **5.9× faster** | +12 % |

Our encoder scales almost linearly with threads, so its lead over HF and tiktoken grows with every core added.
Our trainer is already 10× faster at 2 threads; what remains is the merge loop, which is sequential by nature
(each merge depends on the previous one), so extra threads only shorten the counting pass.
Encoding cost grows with the vocab for us (longer merge chains per word) and not for HF, whose time is dominated by
per-token overhead, so our lead is largest at small vocabs (18× at 32k, 13× at 256k).
How the axes interact (threads × vocab on 4 datasets) and how other tokenizers scale: 🕐 running.

## Other tokenizers

SentencePiece (Unigram, BPE) and WordPiece are different algorithms with different tokens, so they are compared on
speed and memory only. Same data and setting as above (1 GB, 8 threads, vocab 131 072).

| | english train | english encode | mix train | mix encode |
|---|---|---|---|---|
| **ours** (BPE) | **4.5 s / 0.44 GB** | **4.0 s / 0.10 GB** | **22.4 s / 3.20 GB** | **6.0 s / 0.12 GB** |
| HF BPE | 52.4 s / 1.38 GB | 92.7 s / 0.60 GB | 236 s / 11.42 GB | 84.9 s / 0.84 GB |
| HF WordPiece | 51.2 s / 1.13 GB | 44.3 s / 0.47 GB | 127 s / 7.31 GB | 50.3 s / 0.58 GB |
| SentencePiece BPE | 106 s / 3.51 GB | 51.0 s / 0.11 GB | 415 s / 6.60 GB | 53.6 s / 0.08 GB |
| HF Unigram | 310 s / 1.95 GB | 159 s / 0.50 GB | crashed | – |
| SentencePiece Unigram | 1660 s / 21.83 GB | 62.2 s / 0.14 GB | > 60 min | – |

Some of them are faster than HF's BPE (WordPiece encodes 2× faster, SentencePiece 1.7×), but ours is 11-23× faster
than the fastest of them. HF Unigram crashed on mix ("likelihood is NAN. Input sentence may be too long").

## Terascale (10 TB, extrapolated from the measured 1 GB and 10 GB runs, English, 8 threads, vocab 128k)

**Encoding** streams the input, so memory stays flat; time scaled linearly from 1 to 10 GB (ours 10.4× for 10× the data).

| encode 10 TB | time | MEM |
|---|---|---|
| HF `tokenizers` | ~8.6 days | ~0.7 GB |
| `tiktoken` | ~5.4 days (rerun alone pending) | ~0.2 GB |
| **ours** | **~12 hours** | **~0.1 GB** |

At the best setting (16 threads, 1.8 s per GB) ours needs **~5 hours**.

**Training** grows slower than the data (distinct words grow slower than text). Power law through the 1 GB and
10 GB points, time ∝ size^b, memory ∝ size^b:

| train 10 TB | time | MEM | b (time, memory) |
|---|---|---|---|
| HF `tokenizers` | ~2.7 days | ~358 GB | 0.91, 0.60 |
| **ours** | **~4 hours** | **~48 GB** | 0.86, 0.50 |

Two points only: indicative, not a measurement.

## Reproduce

```bash
cd bench_server
PYTHON=python3.11 ./setup.sh
venv-x86_64/bin/python prepare_data.py data 100 1 10
SIZES=1 ./run.sh data results
venv-x86_64/bin/python verify.py results
```
