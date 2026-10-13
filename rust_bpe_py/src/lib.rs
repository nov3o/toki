//! Python binding: the exact same code as the two command-line programs, called in-process.
//!
//!   import rust_bpe_py
//!   rust_bpe_py.train(["bucket_bpe", corpus, "0", vocab_size, merges_out])     # env: BB_THREADS, BB_PRETOK, BB_VOCAB_OUT
//!   rust_bpe_py.encode(["hf_bpe_enc", model.hfbp, "stream", corpus, ids_out])  # env: HB_THREADS
//!
//! Both release the GIL while they run. Their stats line is printed to stdout.
use pyo3::prelude::*;

#[path = "../../rust_bpe/src/bin/bucket_bpe.rs"]
#[allow(dead_code)]
mod bucket_bpe;
#[path = "../../rust_bpe/src/bin/hf_bpe_enc.rs"]
#[allow(dead_code)]
mod hf_bpe_enc;

#[pyfunction]
fn train(py: Python<'_>, args: Vec<String>) {
    py.detach(|| bucket_bpe::run(args));
}

#[pyfunction]
fn encode(py: Python<'_>, args: Vec<String>) {
    py.detach(|| hf_bpe_enc::run(args));
}

/// Which build this is ("simd" or "nosimd"), set at compile time by build.sh.
#[pyfunction]
fn build_variant() -> &'static str {
    option_env!("RUST_BPE_VARIANT").unwrap_or("unknown")
}

#[pymodule]
fn rust_bpe_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(build_variant, m)?)?;
    Ok(())
}
