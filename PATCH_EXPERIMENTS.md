# Upstream patches: before vs after

Goal: for each upstream repo, report **their code at a pinned commit (before)** against **the same commit plus our
minimal patch (after)**, built the same way, run on the same data and machine, with identical outputs.
Ours (`rust_bpe`) is shown next to them only as a reference line.

| repo | pinned base | patch | size target |
|---|---|---|---|
| huggingface/tokenizers | `main` @ fb49a29 (1.0-dev) | **H1** trainer merge loop | one file, `tk-train/src/trainers/bpe/mod.rs` |
| | | **H2** encoder (only if main is still slower than ours, see step 1) | decided after profiling |
| openai/tiktoken | `main` @ 4e71bbe | **T1** per-thread word cache in `CoreBPE` | `src/lib.rs`, ~50 lines |
| | | **T2** hand-written pre-tokenizer for the cl100k / o200k patterns, falls back to the regex otherwise | separate PR, later |

Each patch is its own branch and its own PR; one change per PR, no API changes, no new dependencies.

## 0. What each patch changes

- **H1** keeps `BpeTrainer`, its config, inputs (word counts) and outputs (merges, vocab). Inside `do_train` it
  replaces the `OctonaryHeap` of pairs + `AHashSet` of positions with our count-bucket queue + per-pair position
  lists. Tie-break stays theirs (count, then pair order), so merges are identical by construction.
- **H2** is chosen from the profile in step 1 (counting pass, word cache, or merge engine), not decided up front.
- **T1** caches `piece -> ids` per thread (tiktoken already keeps per-thread regex clones the same way), bounded size,
  so repeated words skip the merge. No change to `byte_pair_merge`.
- **T2** recognizes the exact cl100k / o200k pattern strings and uses a hand-written scanner with the same output;
  any other pattern keeps `fancy_regex`.

## Parity target: patched upstream ≈ ours

A patch is done only when **their code + our patch runs at about our speed** on the same workload, not merely
faster than before. For every workload we report:

- **gap closed** = (before − after) / (before − ours): 0 % = no change, 100 % = as fast as ours.
- **target**: after ≤ 1.2 × ours (gap closed ≥ ~90 %) on the default setting, all 4 datasets, 8 and 16 threads.

Fair comparison, so the remaining gap is really their code and not the setup:
- ours rebuilt **with the same flags** as theirs for this comparison (release, no `target-cpu=native`, no `simd` feature);
- measured at two levels: **Rust** (a small bench binary calling their trainer / encoder on the same words, no Python)
  and **Python** (their public API, as users call it); Rust-level parity is the requirement, the Python-level gap is
  reported and attributed (e.g. HF building `Encoding` objects, tiktoken returning Python lists);
- same input path: both read the same file the same way (their trainer gets its words from their own
  pre-tokenizer, so the patch run also shows how much of the gap is counting / pre-tokenization, not merging).

**Patch ladder** (cumulative, each step timed, so the gap is explained step by step):

| step | HF | tiktoken |
|---|---|---|
| 0 | before | before |
| 1 | + merge loop (H1) | + word cache (T1) |
| 2 | + counting pass (parallel chunked read, per-thread maps) | + id-pair rank table instead of byte-string hashing |
| 3 | + encoder piece chosen from the profile (H2) | + hand-written pre-tokenizer (T2) |
| ours | reference | reference |

If parity needs more than the minimal patch, the PRs are split along this ladder: step 1 first (small, easy to
review), each later step as its own PR with its own before/after numbers.

## 1. Profile before writing a patch

On bristen, one run at a time, `perf record -g` of the **before** builds:
- HF main: `train` and `encode`, 1 GB english and multilingual, 8 threads, GPT-2 byte-level, vocab 131 072.
- tiktoken: `encode_ordinary_batch`, 1 GB english, cl100k_base and o200k_base, 8 threads.

Output: time share per phase (read + split, pre-tokenize, count, merge loop, encode merge, cache, Python glue).
A patch targets only the phase that dominates; if HF main's encoder is already within ~1.5× of ours, H2 is dropped.

## 2. Builds

Four builds from source, same Rust toolchain, `--release`, **no** `target-cpu=native` (as their PyPI wheels):

| build | source |
|---|---|
| `hf_before` / `hf_after` | tokenizers @ fb49a29, without / with H1 (+H2) — `maturin build --release` in `bindings/python` |
| `tik_before` / `tik_after` | tiktoken @ 4e71bbe, without / with T1 — `pip install .` |

Each goes in its own venv (`venv-hf_before`, ...), so `run.sh` can pick one by `IMPLS`.
Released versions (tokenizers 0.22, tiktoken from PyPI) stay in as a third line: "what users have today".

## 3. Correctness (all must pass before any timing is reported)

1. **Their tests**, before and after:
   - tokenizers: `cargo test` in `tokenizers/` (incl. the `tk-train` BPE tests) and `pytest bindings/python/tests`.
   - tiktoken: `cargo test` and `pytest tests`.
2. **After = before, token by token**, with `verify.py` on every timed run: merges, vocab, merges used, ids
   (token count, first differing token). Any difference stops the experiment.
3. **After = ours**, same check, as a cross-check of the HF trainer patch.

## 4. Benchmarks

### 4a. Their own benchmarks (what reviewers will rerun)

| repo | benchmark | settings |
|---|---|---|
| tokenizers | `cargo bench` in `tk-serialize` (`encode`, `scratch_scaling`) | as shipped |
| tokenizers | there is no train bench: add a small Criterion bench for `BpeTrainer` on a fixed file in the H1 PR | 16 MB, vocab 32k |
| tiktoken | `scripts/benchmark.py` (`encode_ordinary_batch`) | `RAYON_NUM_THREADS` = 1, 8, 16 |

### 4b. Our harness (`run.sh`, one axis at a time, 4 datasets × 1 GB, then 10 GB)

| axis | values | applies to |
|---|---|---|
| threads | 1, 2, 4, 8, 16, 64 | both |
| vocab | 32k, 64k, 128k, 256k | HF train |
| pre-tokenizer | bytelevel, llama3, gpt4o, whitespace, metaspace | HF |
| encoding | cl100k_base, o200k_base (their real vocabs) | tiktoken |
| input shape | one 1 GB stream; 1 KB documents in batches of 1000 (typical API use: small texts, cold cache) | both |

The **small-documents** workload is the regression check for T1/H2: a cache helps long streams but costs on short,
unique texts. A patch that slows any workload by more than 2 % is not proposed as is.

### 4c. Repeats and noise

- bristen login node, nothing else of ours running (`more.sh`-style waiting, never in parallel).
- Order **ABBA** per configuration (before, after, after, before), 3 rounds → 6 runs per build.
- Report the **median**, with min–max; a speedup counts only if the ranges do not overlap.
- Record `uptime` load next to every run.

## 5. Report

One table per repo, plus their benchmark output verbatim:

| workload | before (s / MEM) | after (s / MEM) | speedup | ours (s / MEM) | gap closed | identical |
|---|---|---|---|---|---|---|
| train, english 1 GB, 8 thr, 128k | | | | | | ✅ / ❌ |
| ... | | | | | | |

Plus: test-suite results (passed / total, before and after), the patch diff size (`git diff --stat`), and the base
commit hashes, so the reviewer can reproduce it with one script (`patch_bench.sh <repo> <base> <patch>`).

## 6. Order of work

1. Profile HF main and tiktoken (step 1), decide H2 / confirm T1.
2. Write H1, T1 on branches of local clones; run their tests.
3. Correctness check (step 3) on 100 MB, then 1 GB.
4. Benchmarks (step 4), ABBA, after the running queue on bristen has finished.
5. Report table → PR descriptions.
