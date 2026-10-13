## Results (average of the 4 datasets; s = wall time, MEM = peak memory in GB)

Encoding the 0.5 GB held-out half of each dataset. **ours** in parentheses: speedup vs HF main, then vs tiktoken.
🕐 = not run yet; n/a = not supported (tiktoken: regex pre-tokenizers only, no normalizers).

#### Threads

| threads | HF main s | MEM | tiktoken s | MEM | **ours** s | MEM |
|---|---|---|---|---|---|---|
| 2 | 4.54 | 0.73 | 31.5 | 0.21 | **10.3** (0.4×, 3.0×) | **0.10** |
| 4 | 3.21 | 0.74 | 18.9 | 0.22 | **5.70** (0.6×, 3.3×) | **0.11** |
| 8 | 2.47 | 0.78 | 13.0 | 0.22 | **3.33** (0.7×, 3.9×) | **0.12** |
| 16 | 2.11 | 0.84 | 11.4 | 0.23 | **2.14** (1.0×, 5.3×) | **0.15** |

#### Vocab

| vocab | HF main s | MEM | tiktoken s | MEM | **ours** s | MEM |
|---|---|---|---|---|---|---|
| 32k | 2.05 | 0.73 | 13.1 | 0.16 | **2.59** (0.8×, 5.1×) | **0.08** |
| 64k | 2.16 | 0.77 | 12.9 | 0.18 | **2.87** (0.8×, 4.5×) | **0.09** |
| 128k | 2.47 | 0.78 | 13.0 | 0.22 | **3.33** (0.7×, 3.9×) | **0.12** |
| 256k | 3.27 | 0.79 | 13.5 | 0.35 | **4.24** (0.8×, 3.2×) | **0.19** |

#### Pre-tokenizer

| pre-tokenizer | HF main s | MEM | tiktoken s | MEM | **ours** s | MEM |
|---|---|---|---|---|---|---|
| bytelevel | 2.47 | 0.78 | 13.0 | 0.22 | **3.33** (0.7×, 3.9×) | **0.12** |
| llama3 | 2.46 | 0.77 | 11.2 | 0.22 | **3.38** (0.7×, 3.3×) | **0.11** |
| gpt4o | 2.48 | 0.77 | 10.7 | 0.22 | **3.57** (0.7×, 3.0×) | **0.11** |
| whitespace | 2.23 | 0.72 | n/a | n/a | **2.74** (0.8×) | **0.13** |
| whitespace_split | 2.73 | 0.72 | n/a | n/a | **3.79** (0.7×) | **0.14** |
| metaspace | 3.62 | 0.83 | n/a | n/a | **4.95** (0.7×) | **0.15** |
| bert | 2.20 | 0.72 | n/a | n/a | **4.12** (0.5×) | **0.13** |
| punctuation | 5.23 | 0.76 | n/a | n/a | **9.39** (0.6×) | **0.14** |
| digits | 6.71 | 0.76 | n/a | n/a | **13.0** (0.5×) | **0.15** |
| unicode_scripts | 4.97 | 0.73 | n/a | n/a | **11.3** (0.4×) | **0.13** |
| char_delimiter | 2.98 | 0.74 | n/a | n/a | **4.16** (0.7×) | **0.13** |

#### Normalizer

| normalizer | HF main s | MEM | tiktoken s | MEM | **ours** s | MEM |
|---|---|---|---|---|---|---|
| none | 2.47 | 0.78 | 13.0 | 0.22 | **3.33** (0.7×, 3.9×) | **0.12** |
| nfc | 2.59 | 0.78 | n/a | n/a | **4.66** (0.6×) | **0.12** |
| nfd | 3.08 | 0.79 | n/a | n/a | **4.18** (0.7×) | **0.14** |
| nfkc | 3.56 | 0.78 | n/a | n/a | **4.65** (0.8×) | **0.12** |
| nfkd | 3.08 | 0.79 | n/a | n/a | **4.19** (0.7×) | **0.14** |
| lowercase | 2.67 | 0.77 | n/a | n/a | **3.98** (0.7×) | **0.12** |
| strip_accents | 3.26 | 0.79 | n/a | n/a | **4.38** (0.7×) | **0.13** |
| nfkc_lowercase | 3.80 | 0.78 | n/a | n/a | **5.21** (0.7×) | **0.13** |
| replace | 2.45 | 0.79 | n/a | n/a | **3.47** (0.7×) | **0.15** |
| bert | 5.17 | 0.78 | n/a | n/a | **6.18** (0.8×) | **0.15** |

Equivalence: 380 encodings compared token by token with HF main's ids, 380 identical.

## Headline (llama3 pre-tokenizer, no normalizer, vocab 32k, english)

| | HF main | tiktoken | **ours** |
|---|---|---|---|
| 2 threads | 2.89 s / 0.64 GB | 19.9 s / 0.13 GB | **6.03 s** (0.5×, 3.3×) / **0.04 GB** |
| 16 threads | 1.22 s / 0.71 GB | 8.19 s / 0.14 GB | **1.06 s** (1.1×, 7.7×) / **0.14 GB** |

## Thread scaling (time ∝ threads^b, 2 → 16 threads)

| | b | 2 → 16 threads |
|---|---|---|
| hfmain | -0.37 | 2.2× faster |
| tiktoken | -0.49 | 2.8× faster |
| ours | -0.76 | 4.8× faster |
