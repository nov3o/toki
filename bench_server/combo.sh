#!/bin/bash
# combo.sh <data_dir> <results_dir>: best settings at 1 GB, 4 datasets, one step at a time.
# Standard pipeline fixed (llama3 pre-tokenizer, no normalizer); the free axes set to their best measured value:
#   training best: 2 threads, vocab 32k   (HF vs ours)
#   encoding best: 16 threads, vocab 32k  (HF vs tiktoken vs ours)
DATA=$1; RES=$2
for t in 2 16; do
    BASE_THREADS=$t BASE_VOCAB=32768 BASE_PRETOK=llama3 BASE_NORM=none ALL_THREADS= ALL_PRETOKS= ALL_VOCABS= ALL_NORMS= \
        IMPLS="hf ours_simd" SIZES=1 ./run.sh "$DATA" "$RES"
done
