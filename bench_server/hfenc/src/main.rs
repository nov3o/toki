//! hfenc <tokenizer.json> <corpus> <pretok> <ids_out>: encode a whole file with HF `tokenizers` main (1.0) from Rust,
//! no Python in the timed path. The file is cut into ~256 KB pieces at the same safe points our encoder and the
//! Python harness use (pre-tokenization restarts identically there), pieces go to HF's batch `encode` (its own
//! thread pool: as many threads as cores this process may run on, or HF_THREADS), ids are written as u32 LE.
//! Prints encode_s= tokens=.
use std::io::Write;
use tk_encode::pipeline::{EncodeOptions, Input, Inputs};

const PIECE: usize = 256 << 10;
const BATCH: usize = 256; // pieces per encode call (~64 MB)

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (tok, corpus, pretok, out) = (&a[1], &a[2], &a[3], &a[4]);
    let json = tk_convert::canonicalize_file(tok).unwrap();
    let tk = tk_serialize::from_json(&json).unwrap();
    let kind = rust_bpe::pretok::Kind::from_name(pretok).unwrap();
    let opts = EncodeOptions::no_specials();
    // HF_THREADS=n sets HF's thread count in code (it ignores RAYON_NUM_THREADS); unset = the cores it may run on
    if let Ok(n) = std::env::var("HF_THREADS") {
        tk_encode::parallelism::set_num_threads(n.parse().unwrap());
    }
    let t0 = std::time::Instant::now();
    let text = std::fs::read(corpus).unwrap();
    // cut points: first safe cut at or after every PIECE bytes (same rule as bench_server/common.py pieces)
    let mut cuts = vec![0];
    let mut p = PIECE;
    while p < text.len() {
        while p < text.len() && !rust_bpe::pretok::safe_cut(kind, &text, p) {
            p += 1;
        }
        if p < text.len() {
            cuts.push(p);
        }
        p += PIECE;
    }
    cuts.push(text.len());
    let mut f = std::io::BufWriter::with_capacity(1 << 24, std::fs::File::create(out).unwrap());
    let mut tokens = 0usize;
    for chunk in cuts.windows(2).collect::<Vec<_>>().chunks(BATCH) {
        let batch: Vec<Input> = chunk
            .iter()
            .map(|w| Input::Single(String::from_utf8_lossy(&text[w[0]..w[1]]).into_owned()))
            .collect();
        for enc in tk.encode(Inputs::Batch(batch), &opts).wait().unwrap() {
            for t in enc.ids() {
                f.write_all(&t.id().to_le_bytes()).unwrap();
            }
            tokens += enc.len();
        }
    }
    f.flush().unwrap();
    println!("encode_s={:.3} tokens={tokens}", t0.elapsed().as_secs_f64());
}
