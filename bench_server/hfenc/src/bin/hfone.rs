//! hfone <tokenizer.json> <corpus> <ids_out>: HF main used the simplest way, the whole file as ONE input to `encode`
//! (HF splits a long sequence into chunks and encodes them on its own thread pool), ids written in one call.
//! The fairest HF-only baseline: no cutting or batching by us. Prints read_s= encode_s= write_s= tokens=.
use std::io::Write;
use tk_encode::pipeline::{EncodeOptions, Input, Inputs};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let tk = tk_serialize::from_json(&tk_convert::canonicalize_file(&a[1]).unwrap()).unwrap();
    if let Ok(n) = std::env::var("HF_THREADS") {
        tk_encode::parallelism::set_num_threads(n.parse().unwrap());
    }
    let t0 = std::time::Instant::now();
    let text = String::from_utf8_lossy(&std::fs::read(&a[2]).unwrap()).into_owned();
    let t1 = std::time::Instant::now();
    let enc = tk.encode(Inputs::Single(Input::Single(text)), &EncodeOptions::no_specials()).wait().unwrap();
    let t2 = std::time::Instant::now();
    let ids: Vec<u32> = enc[0].ids().iter().map(|t| t.id()).collect();
    let bytes = unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const u8, ids.len() * 4) };
    std::fs::File::create(&a[3]).unwrap().write_all(bytes).unwrap();
    println!("read_s={:.3} encode_s={:.3} write_s={:.3} tokens={}", (t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64(), t2.elapsed().as_secs_f64(), ids.len());
}
