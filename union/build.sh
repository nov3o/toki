#!/bin/bash
# build.sh <hf-main checkout> <hf-pr2501 checkout>: build union/encode (against HF main) and union/train (against
# huggingface/tokenizers PR #2501), with rust_bpe from this repo. Binaries: union/{encode,train}/target/release/.
set -e
cd "$(dirname "$0")"; RB=$(cd ../rust_bpe && pwd); HF=$(cd "$1" && pwd); PR=$(cd "$2" && pwd)
sed -e "s|@HF@|$HF|g" -e "s|@RB@|$RB|g" encode/Cargo.toml.in > encode/Cargo.toml
sed -e "s|@PR@|$PR|g" -e "s|@RB@|$RB|g" train/Cargo.toml.in > train/Cargo.toml
(cd encode && cargo build --release) && (cd train && cargo build --release)
