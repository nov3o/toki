# Results

bristen login node (AMD EPYC 7742, x86_64, AVX2), 1 GB per dataset, 8 threads, GPT-2 byte-level pre-tokenizer,
vocab 131072. Time = seconds inside the train / encode call; memory = peak RSS of the process (MB).
Every merge step, the final vocab and every token id are identical to HF tokenizers (verify.py).

## 2026-10-12

| dataset | HF train | ours train | HF encode | tiktoken encode | ours encode | HF train MB | ours train MB | HF enc MB | tiktoken enc MB | ours enc MB |
|---|---|---|---|---|---|---|---|---|---|---|
| english | 48.8 | 4.1 | 78.5 | 21.5 | 3.1 | 1399 | 467 | 520 | 161 | 104 |
| multilingual | 471.2 | 45.2 | 67.9 | 22.9 | 7.5 | 26428 | 7169 | 657 | 165 | 107 |
| code | 117.2 | 10.2 | 92.5 | 35.7 | 3.2 | 4296 | 1244 | 1096 | 218 | 113 |
| mix | 230.6 | 22.7 | 88.4 | 27.9 | 5.0 | 11414 | 3240 | 802 | 186 | 124 |

ours = simd build. The nosimd build is within ~3% (the gains are threads, layout and prefetching, not SIMD).
Raw rows: results_base (HF, tiktoken, ours before the trainer changes) and results_v2 (ours after) on bristen.

## Overnight runs (1 GB, 4 datasets, one axis at a time around the baseline)

All finished settings are in [tables/SUMMARY.md](tables/SUMMARY.md) (time + peak RAM for HF, ours simd/nosimd, tiktoken);
raw rows in `tables/results_*.tsv`, per-file equivalence in `tables/results_*_verify.tsv` (rebuild: `python3 tables/summarize.py`).

| axis | settings x datasets | equivalence vs HF (merges, vocab, merges used, ids) |
|---|---|---|
| pre-tokenizer (all 11) | 44 | 376 comparisons, 0 different |
| threads 2/4/8/16 + vocab 32k-256k | 28 | 280 comparisons, 0 different |
| normalizers (12) | 48 | 360 comparisons, 0 different |
| 10 GB baseline | queued | — |

Trainer A/B (multilingual 1 GB, 3 runs each): previous 41.5 s vs current 41.4 s, merges identical (no regression, no gain).
