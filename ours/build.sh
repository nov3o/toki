#!/bin/bash
# build.sh <tokenizers main checkout> <tokenizers PR #2501 checkout>: build ours/encode against HF main and ours/train
# against PR #2501 with patches/pr2501-do-train-entries.diff applied (6 lines: train from (word, count) entries without
# a map), with rust_bpe from this repo. Binaries: ours/encode/target/release/toki-encode, ours/train/target/release/toki-train.
set -e
cd "$(dirname "$0")"; RB=$(cd ../rust_bpe && pwd); HF=$(cd "$1" && pwd); PR=$(cd "$2" && pwd)
git -C "$PR" apply --check ../patches/pr2501-do-train-entries.diff 2>/dev/null && git -C "$PR" apply "$(pwd)/../patches/pr2501-do-train-entries.diff"
sed -e "s|@HF@|$HF|g" -e "s|@RB@|$RB|g" encode/Cargo.toml.in > encode/Cargo.toml
sed -e "s|@PR@|$PR|g" -e "s|@RB@|$RB|g" train/Cargo.toml.in > train/Cargo.toml
(cd encode && cargo build --release) && (cd train && cargo build --release)
