//! hfbench <tokenizer.json> <corpus> <vocab_size> <out_dir> <byte_level 0|1> [special tokens, comma-separated]
//!
//! Trains with the `BpeTrainer` of the HF `tokenizers` checkout this crate is built against (main, or a PR).
//! HF 1.0 has no tokenizer-level `train`, so this does what `Tokenizer::train` did in 0.22, with HF's own parts:
//! the normalizers and pre-tokenizer come from <tokenizer.json> (loaded by `tk_serialize`), each line of the
//! corpus (with its '\n') is normalized and pre-tokenized by them, byte-level words are mapped to the GPT-2 byte
//! alphabet, and the words are fed to the trainer. Threads: RAYON_NUM_THREADS.
//! Writes out_dir/merges.txt and out_dir/vocab.txt like train_hf.py. Prints feed_s= train_s=.
use std::cell::RefCell;
use std::io::{BufRead, BufReader, Write};
use tk_encode::pipeline::PreTokenizerScratch;
use tk_encode::pipeline::{normalize_all, PreTokenizer};
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;
use tk_train::Trainer;
use tk_train::trainers::BpeTrainer;

fn byte_chars() -> [char; 256] {
    // GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to 256 + n
    let mut bs: Vec<u32> = (33..127).chain(161..173).chain(174..256).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut t = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs) {
        t[*b as usize] = char::from_u32(c).unwrap();
    }
    t
}

thread_local! {
    static SCRATCH: RefCell<(PreTokenizerScratch, Vec<bitcannon::Span>)> = RefCell::new(Default::default());
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (tok, corpus, vocab, out, byte_level) = (&a[1], &a[2], a[3].parse::<usize>().unwrap(), &a[4], a[5] == "1");
    let special: Vec<AddedToken> = a.get(6).map_or(vec![], |s| s.split(',').filter(|x| !x.is_empty()).map(|x| AddedToken::from(x.to_string(), true)).collect());
    std::fs::create_dir_all(out).unwrap();
    // Only the normalizers and the pre-tokenizer of <tokenizer.json> are used. HF main refuses a byte-level vocab that
    // lacks one of the 256 bytes, so the model is replaced by the bare alphabet; then HF's own converter turns the
    // 0.22 file into the canonical (2.0) one that main reads.
    let mut j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(tok).unwrap()).unwrap();
    if byte_level {
        let alpha: serde_json::Map<String, serde_json::Value> = byte_chars().iter().enumerate().map(|(i, c)| (c.to_string(), i.into())).collect();
        j["model"]["vocab"] = alpha.into();
        j["model"]["merges"] = serde_json::Value::Array(vec![]);
    }
    let tk = tk_serialize::from_json(&tk_convert::canonicalize_str(&j.to_string()).unwrap()).unwrap();
    if std::env::var("HB_DUMP").is_ok() {
        // print the pre-tokens of each line of <corpus>, one per line, "<line no>\t<pre-token as JSON>"
        let mut scratch = PreTokenizerScratch::default();
        let mut spans = Vec::new();
        for (k, l) in std::fs::read_to_string(corpus).unwrap().split_inclusive('\n').enumerate() {
            let text = normalize_all(tk.get_normalizers(), l, 0).unwrap();
            spans.clear();
            tk.get_pre_tokenizer().pre_tokenize(&text, &mut scratch, &mut spans).unwrap();
            for sp in &spans {
                println!("{k}\t{}", serde_json::Value::from(&text[sp.start as usize..sp.end as usize]));
            }
        }
        return;
    }
    let norms = tk.get_normalizers();
    let pre = tk.get_pre_tokenizer();
    let map = byte_chars();
    let mut trainer = BpeTrainer::builder().vocab_size(vocab).show_progress(false).special_tokens(special).build();
    let t0 = std::time::Instant::now();
    let reader = BufReader::with_capacity(1 << 20, std::fs::File::open(corpus).unwrap());
    let lines = reader.split(b'\n').map(|l| {
        let mut l = l.unwrap();
        l.push(b'\n');
        String::from_utf8_lossy(&l).into_owned()
    });
    trainer
        .feed(lines, |s: &str| {
            let text = normalize_all(norms, s, 0)?;
            SCRATCH.with(|c| {
                let (scratch, spans) = &mut *c.borrow_mut();
                spans.clear();
                pre.pre_tokenize(&text, scratch, spans)?;
                Ok(spans
                    .iter()
                    .map(|sp| {
                        let w = &text[sp.start as usize..sp.end as usize];
                        if byte_level { w.bytes().map(|x| map[x as usize]).collect() } else { w.to_string() }
                    })
                    .collect())
            })
        })
        .unwrap();
    let feed_s = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let (vocab, merges, _) = trainer.train_vocab().unwrap();
    let train_s = t1.elapsed().as_secs_f64();
    let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{out}/merges.txt")).unwrap());
    for (l, r) in &merges {
        writeln!(f, "{l} {r}").unwrap();
    }
    let mut v: Vec<_> = vocab.iter().collect();
    v.sort_by_key(|x| *x.1);
    let mut g = std::io::BufWriter::new(std::fs::File::create(format!("{out}/vocab.txt")).unwrap());
    for (t, i) in v {
        writeln!(g, "{i}\t{t}").unwrap();
    }
    println!("feed_s={feed_s:.3} train_s={train_s:.3}");
}
