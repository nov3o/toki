# Plan

🏃 running · 🕐 to do · ✅ done.

## Decisions (2026-10-13)

- Baselines are the fastest code that exists: HF `main` (1.0 rewrite) for encoding, PR #2501 (Yikai-Liao) for training.
  The 0.22 / 0.23 releases are no longer the reference.
- Our own encoder and trainer are slower than these per core; the way forward is the **union**: their per-word engine /
  merge loop inside our parallel, streaming, map-reduce driver. Credit PR #2501 for the merge loop.
- Every run pinned to exactly N cores (`taskset`): HF 1.0 ignores `RAYON_NUM_THREADS`.
- Held-out text: vocab trained on one half of each 1 GB sample, encoding on the other half.
- No SIMD build of ours (no consistent effect); SIMD comes with HF's pre-tokenizer (bitcannon) in the union.
- toki-encoder is folded back into this repo.

## 1. Encoder

| | what | status |
|---|---|---|
| 1.1 | held-out encoding, HF main vs tiktoken vs ours, 4 datasets, all axes, ids token by token (380/380 identical) | ✅ |
| 1.2 | union encoder: HF main engine in our driver, parallel pwrite / sharded output | ✅ 10 GB: 2.4 s at 64 cores (HF main 35 s) |
| 1.3 | serial model load (0.6 s: JSON + table build): binary model file or parallel build | 🕐 |
| 1.4 | union on all datasets / pre-tokenizers / normalizers, 1 GB and 10 GB | 🕐 |
| 1.5 | HF PR: parallel safe-cut reading + streaming `encode_file` (memory: 10 GB → 0.6 GB) | 🕐 |

## 2. Trainer

| | what | status |
|---|---|---|
| 2.1 | union trainer: our counting + PR #2501 merge loop, merges identical | ✅ 10 GB: 14.4 s at 16 cores (PR #2501 27.2 s) |
| 2.2 | counting on few cores: 64 shard maps per thread cost 1.5× at 1 core; count into one map, shard at reduce | 🕐 |
| 2.3 | PR #2501 merge loop stops scaling at ~8 cores: profile | 🕐 |
| 2.4 | other pre-tokenizers / normalizers in the union trainer (HF's pre-tokenizer or ours) | 🕐 |
| 2.5 | rerun 256 MiB comparison pinned (HF main / PR #2501 were unpinned) | 🕐 |

## Note for PR #2501 (Yikai-Liao)

HF 1.0 ignores `RAYON_NUM_THREADS`: training and encoding use `available_parallelism()` unless `set_num_threads` is
called. His runs were pinned to CPUs 0..N-1, so they are fine; any comparison against it must pin too.

## 3. tiktoken (paused)

Patch done (hand-written pre-tokenizer, ids identical, 15-20 % faster from Python). Paused.

## 4. Repo / outreach

| | what | status |
|---|---|---|
| 4.1 | private toki repo with this overhaul; old toki → toki-archive | 🕐 user creates / renames |
| 4.2 | emails to Yikai and Arthur (Gmail drafts) | 🕐 user sends |
