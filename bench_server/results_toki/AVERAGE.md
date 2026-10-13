Equivalence: 143 training runs of ours, 143 with merges identical to PR #2501's; 142 encoding runs, 142 with ids identical to HF main's (token by token).

#### Default setting

| dataset | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| english | 5.21 | 192 MB/s | 0.37 | **2.48** (2.1×) | **402 MB/s** | **0.31** | 1.80 | 278 MB/s | 0.69 | **0.99** (1.8×) | **503 MB/s** | **0.31** |
| multilingual | 16.2 | 62 MB/s | 4.04 | **13.5** (1.2×) | **74 MB/s** | **3.98** | 2.99 | 167 MB/s | 0.77 | **1.83** (1.6×) | **272 MB/s** | **0.37** |
| code | 9.34 | 107 MB/s | 1.19 | **4.71** (2.0×) | **212 MB/s** | **0.87** | 2.33 | 215 MB/s | 0.87 | **1.23** (1.9×) | **407 MB/s** | **0.37** |
| mix | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| **average** | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |

#### Threads

| threads | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| 4 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| 8 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| 16 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |

#### Vocab

| vocab | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32k | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| 64k | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| 128k | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| 256k | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |

#### Pre-tokenizer

| pre-tokenizer | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bytelevel | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| llama3 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| gpt4o | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| whitespace | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| whitespace_split | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| metaspace | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| bert | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| punctuation | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| digits | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| unicode_scripts | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| char_delimiter | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |

#### Normalizer

| normalizer | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| nfc | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| nfd | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| nfkc | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| nfkd | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| lowercase | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| strip_accents | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| nfkc_lowercase | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| replace | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| bert | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |

#### Size

| size | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 GB | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| 10 GB | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
