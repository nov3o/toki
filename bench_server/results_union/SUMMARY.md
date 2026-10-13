Identical outputs (ids token by token, merges byte by byte): 256 of 256.

## Encode 10 GB (byte-level, vocab 128k): HF main vs union

| dataset | cores | HF main | **union** | speedup | ids |
|---|---|---|---|---|---|
| english | 16 | 25.3 s / 9.6 GB | **7.34 s / 0.6 GB** | 3.4× | same |
| english | 64 | 20.6 s / 9.9 GB | **2.52 s / 1.9 GB** | 8.2× | same |
| multilingual | 16 | 37.1 s / 9.8 GB | **14.18 s / 0.8 GB** | 2.6× | same |
| multilingual | 64 | 29.9 s / 10.4 GB | **4.29 s / 2.9 GB** | 7.0× | same |
| code | 16 | 30.1 s / 10.0 GB | **9.26 s / 0.7 GB** | 3.2× | same |
| code | 64 | 25.3 s / 10.8 GB | **3.10 s / 2.4 GB** | 8.1× | same |
| mix | 16 | 30.7 s / 9.8 GB | **11.00 s / 0.7 GB** | 2.8× | same |
| mix | 64 | 24.8 s / 10.7 GB | **3.39 s / 2.9 GB** | 7.3× | same |

## Encode 0.5 GB held-out, by pre-tokenizer (average of 4 datasets): HF main vs union

| pre-tokenizer | HF main 8 cores | **union 8 cores** | HF main 16 cores | **union 16 cores** |
|---|---|---|---|---|
| bytelevel | 3.00 s | **1.44 s** (2.1×) | 2.12 s | **1.09 s** (1.9×) |
| llama3 | 2.49 s | **1.50 s** (1.7×) | 2.10 s | **1.10 s** (1.9×) |
| gpt4o | 2.51 s | **1.52 s** (1.6×) | 2.11 s | **1.09 s** (1.9×) |
| whitespace | 2.27 s | **1.34 s** (1.7×) | 1.90 s | **0.95 s** (2.0×) |
| whitespace_split | 2.75 s | **1.89 s** (1.5×) | 2.11 s | **1.21 s** (1.7×) |
| metaspace | 3.64 s | **2.81 s** (1.3×) | 2.66 s | **1.80 s** (1.5×) |
| bert | 2.25 s | **1.30 s** (1.7×) | 1.91 s | **0.94 s** (2.0×) |
| punctuation | 5.18 s | **4.24 s** (1.2×) | 3.41 s | **2.42 s** (1.4×) |
| digits | 6.63 s | **5.83 s** (1.1×) | 4.07 s | **3.23 s** (1.3×) |
| unicode_scripts | 4.94 s | **4.06 s** (1.2×) | 3.23 s | **2.29 s** (1.4×) |
| char_delimiter | 3.02 s | **2.12 s** (1.4×) | 2.21 s | **1.34 s** (1.7×) |

## Encode 0.5 GB held-out, by normalizer (average of 4 datasets): HF main vs union

| normalizer | HF main 8 cores | **union 8 cores** | HF main 16 cores | **union 16 cores** |
|---|---|---|---|---|
| nfc | 2.61 s | **2.02 s** (1.3×) | 2.17 s | **1.43 s** (1.5×) |
| nfd | 3.09 s | **2.13 s** (1.5×) | 2.44 s | **1.40 s** (1.7×) |
| nfkc | 3.49 s | **2.51 s** (1.4×) | 2.63 s | **1.59 s** (1.6×) |
| nfkd | 3.12 s | **2.12 s** (1.5×) | 2.46 s | **1.40 s** (1.8×) |
| lowercase | 2.75 s | **1.73 s** (1.6×) | 2.26 s | **1.22 s** (1.9×) |
| strip_accents | 3.28 s | **2.31 s** (1.4×) | 2.51 s | **1.50 s** (1.7×) |
| nfkc_lowercase | 3.74 s | **2.73 s** (1.4×) | 2.79 s | **1.70 s** (1.6×) |
| replace | 2.50 s | **1.51 s** (1.7×) | 2.12 s | **1.11 s** (1.9×) |
| bert | 5.24 s | **4.19 s** (1.3×) | 3.54 s | **2.45 s** (1.4×) |

## Train 1 GB, vocab 128k, 16 cores, by pre-tokenizer (average of 4 datasets): ours vs union

| pre-tokenizer | ours | **union** | speedup |
|---|---|---|---|
| bytelevel | 21.6 s | **6.1 s** / 3.6 GB | 3.5× |
| llama3 | 22.7 s | **6.4 s** / 3.8 GB | 3.5× |
| gpt4o | 20.7 s | **6.3 s** / 3.6 GB | 3.3× |
| whitespace | 17.4 s | **4.8 s** / 2.8 GB | 3.7× |
| whitespace_split | 41.4 s | **8.2 s** / 4.9 GB | 5.0× |
| metaspace | 45.4 s | **9.5 s** / 5.9 GB | 4.8× |
| bert | 16.9 s | **5.8 s** / 2.9 GB | 2.9× |
| punctuation | 156.8 s | **26.1 s** / 11.7 GB | 6.0× |
| digits | 252.3 s | **28.9 s** / 12.6 GB | 8.7× |
| unicode_scripts | 142.8 s | **25.4 s** / 11.3 GB | 5.6× |
| char_delimiter | 44.3 s | **8.9 s** / 5.8 GB | 5.0× |

## Train 1 GB, vocab 128k, 16 cores, by normalizer (average of 4 datasets): ours vs union

| normalizer | ours | **union** | speedup |
|---|---|---|---|
| nfc | 27.4 s | **9.9 s** / 3.5 GB | 2.8× |
| nfd | 25.8 s | **10.6 s** / 3.2 GB | 2.4× |
| nfkc | 26.6 s | **11.4 s** / 3.4 GB | 2.3× |
| nfkd | 25.9 s | **10.0 s** / 3.1 GB | 2.6× |
| lowercase | 23.1 s | **8.9 s** / 3.3 GB | 2.6× |
| strip_accents | 30.3 s | **10.7 s** / 3.5 GB | 2.8× |
| nfkc_lowercase | 32.7 s | **13.5 s** / 3.4 GB | 2.4× |
| replace | 21.6 s | **6.4 s** / 3.6 GB | 3.4× |
| bert | 12.8 s | **7.0 s** / 1.6 GB | 1.8× |
| prepend | 21.6 s | **6.6 s** / 3.4 GB | 3.2× |
| strip | 21.4 s | **6.2 s** / 3.4 GB | 3.4× |
