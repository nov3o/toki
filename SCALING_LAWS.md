# Scaling laws

Power-law fits `y = a · x^b` (least squares in log-log space) of wall time and peak memory, averaged over the
4 datasets (english, multilingual, code, mix). Default setting: 1 GB, 8 threads, GPT-2 byte-level, vocab 131 072,
no normalizer; each law varies one axis and keeps the rest. R² is the quality of the fit.

## Threads: time ∝ threads^b (2, 4, 8, 16 threads)

b = −1 is perfect scaling (twice the threads, half the time); b = 0 is no scaling.

| | b | R² | reading |
|---|---|---|---|
| HF train | −0.20 | 0.93 | 2 → 16 threads: 1.5× faster |
| **ours** train | −0.07 | 0.69 | already 10× faster at 2 threads; the merge loop is sequential |
| HF encode | −0.49 | 0.96 | time ∝ 1/√threads; 2 → 16: 2.7× faster |
| tiktoken encode | −0.41 | 0.85 | |
| **ours** encode | **−0.85** | **1.00** | close to perfect: 2 → 16 threads: 5.9× faster |

Amdahl view (time = serial + parallel / threads): serial share HF train 46 %, ours train 70 %, HF encode 15 %,
tiktoken encode 18 %, ours encode 3 % (so up to ~36× from threads alone).

## Vocab: time ∝ vocab^b, memory ∝ vocab^b (32k, 64k, 128k, 256k)

| | time b | R² | memory b | R² |
|---|---|---|---|---|
| HF train | 0.07 | 0.85 | 0.04 | 1.00 |
| **ours** train | 0.12 | 0.99 | 0.13 | 0.98 |
| HF encode | 0.00 | 0.00 | 0.09 | 0.87 |
| tiktoken encode | −0.01 | 0.36 | 0.24 | 0.91 |
| **ours** encode | 0.16 | 0.97 | 0.37 | 0.85 |

Doubling the vocab costs ours ~8 % (train) and ~12 % (encode) more time; HF and tiktoken encode are flat because
their time is per-token overhead, not merging. So our lead is largest at small vocabs.

## Data size: time ∝ size^b, memory ∝ size^b

🕐 running: 1/8, 1/4, 1/2, 1 GB, then 10 GB. Training memory grows with the number of distinct words (Heaps' law),
so its exponent is expected below 1; this fit is what the 10 TB training estimate will use.

## Axes combined

🕐 running: threads {2, 16} × vocab {32k, 256k}; llama3 / gpt4o pre-tokenizer at 16 and 64 threads.
