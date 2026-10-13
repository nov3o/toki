#!/bin/bash
# Build the Python binding twice:
#   build/<arch>/simd/rust_bpe_py.so    -C target-cpu=native + hand-written SIMD (feature simd: AVX2 on x86, NEON on ARM)
#   build/<arch>/nosimd/rust_bpe_py.so  generic CPU, loop + SLP auto-vectorization turned off
# Use one with: PYTHONPATH=rust_bpe_py/build/$(uname -m)/simd python3 -c "import rust_bpe_py"
set -e
cd "$(dirname "$0")"

ARCH=$(uname -m)            # x86_64 or aarch64: clar and bristen share this folder, so builds are kept per arch
# Rust toolchain per arch: $SCRATCH/rust-<arch> if present (bristen), else ~/.cargo (clar)
if [ -x "$SCRATCH/rust-$ARCH/cargo/bin/cargo" ]; then
    export CARGO_HOME=$SCRATCH/rust-$ARCH/cargo RUSTUP_HOME=$SCRATCH/rust-$ARCH/rustup
    export PATH=$CARGO_HOME/bin:$PATH
elif [ -x "$HOME/.cargo/bin/cargo" ]; then
    export PATH=$HOME/.cargo/bin:$PATH
fi
build() {   # build <variant> <RUSTFLAGS> [cargo args]
    echo "== $ARCH $1: RUSTFLAGS=$2 $3"
    RUST_BPE_VARIANT=$1 RUSTFLAGS="$2" cargo build --release --target-dir target/$ARCH-$1 $3
    mkdir -p build/$ARCH/$1
    cp target/$ARCH-$1/release/librust_bpe_py.so build/$ARCH/$1/rust_bpe_py.so
}

build simd   "-C target-cpu=native" "--features simd"
build nosimd "-C no-vectorize-loops -C no-vectorize-slp"
ls -la build/$ARCH/*/rust_bpe_py.so
