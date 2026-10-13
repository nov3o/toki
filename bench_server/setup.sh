#!/bin/bash
# One-time setup: Python venv (HF tokenizers + data download libs) and our Rust binding (simd + nosimd builds).
# Needs: python >= 3.9 (on clar/bristen: PYTHON=python3.11 ./setup.sh), cargo (https://rustup.rs;
# on bristen: x86 toolchain in $SCRATCH/rust-x86_64, see ../rust_bpe_py/build.sh).
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
ARCH=$(uname -m)
V=venv-$ARCH                       # one venv per arch (clar = aarch64, bristen = x86_64 share this folder)
[ -d $V ] || $PYTHON -m venv $V
$V/bin/pip install --upgrade pip
$V/bin/pip install "tokenizers==0.22.*" tiktoken pyarrow huggingface_hub
PYO3_PYTHON=$PWD/$V/bin/python ../rust_bpe_py/build.sh
for v in simd nosimd; do
    PYTHONPATH=../rust_bpe_py/build/$ARCH/$v $V/bin/python -c "import rust_bpe_py; print('rust_bpe_py', rust_bpe_py.build_variant())"
done
$V/bin/python -c "import tokenizers; print('tokenizers', tokenizers.__version__)"
