Equivalence: 88 training runs of ours, 88 with merges identical to PR #2501's; 86 encoding runs, 86 with ids identical to HF main's (token by token).

#### Default setting

| dataset | PR #2501 train s | MEM | **ours** train s | MEM | HF main enc s | MEM | **ours** enc s | MEM |
|---|---|---|---|---|---|---|---|---|
| english | 5.21 | 0.37 | **2.48** (2.1×) | **0.31** | 1.80 | 0.69 | **0.99** (1.8×) | **0.31** |
| multilingual | 16.2 | 4.04 | **13.5** (1.2×) | **3.98** | 2.99 | 0.77 | **1.83** (1.6×) | **0.37** |
| code | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| mix | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| **average** | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |

#### Threads

| threads | PR #2501 train s | MEM | **ours** train s | MEM | HF main enc s | MEM | **ours** enc s | MEM |
|---|---|---|---|---|---|---|---|---|
| 2 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| 4 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| 8 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| 16 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |

#### Vocab

| vocab | PR #2501 train s | MEM | **ours** train s | MEM | HF main enc s | MEM | **ours** enc s | MEM |
|---|---|---|---|---|---|---|---|---|
| 32k | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| 64k | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| 128k | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| 256k | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |

#### Pre-tokenizer

| pre-tokenizer | PR #2501 train s | MEM | **ours** train s | MEM | HF main enc s | MEM | **ours** enc s | MEM |
|---|---|---|---|---|---|---|---|---|
| bytelevel | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| llama3 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| gpt4o | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| whitespace | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| whitespace_split | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| metaspace | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| bert | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| punctuation | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| digits | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| unicode_scripts | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| char_delimiter | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |

#### Normalizer

| normalizer | PR #2501 train s | MEM | **ours** train s | MEM | HF main enc s | MEM | **ours** enc s | MEM |
|---|---|---|---|---|---|---|---|---|
| none | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| nfc | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| nfd | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| nfkc | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| nfkd | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| lowercase | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| strip_accents | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| nfkc_lowercase | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| replace | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| bert | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |

#### Size

| size | PR #2501 train s | MEM | **ours** train s | MEM | HF main enc s | MEM | **ours** enc s | MEM |
|---|---|---|---|---|---|---|---|---|
| 1 GB | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
| 10 GB | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |
