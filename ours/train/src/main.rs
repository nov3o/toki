//! toki-train <corpus> <vocab_size> <out_dir>: our feed + PR #2501's merge. Workers pread chunks cut at line starts
//! (as our encoder), pre-tokenize them with GPT-2 byte-level (rust_bpe::next_gpt2) and count words as byte slices in
//! per-thread maps; the maps are merged, mapped to byte-level chars once per unique word and handed to tk-train's
//! `BpeTrainer::do_train`. env: HB_THREADS (1), HB_BLOCK_KB (4096); HF's pool sized by set_num_threads(HB_THREADS).
//! Writes out_dir/merges.txt and vocab.txt like hfbench. Prints feed_s= train_s=.
use compact_str::CompactString;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use rust_bpe::{env_or, pretok};
use tk_train::trainers::BpeTrainer;

// mimalloc (run with MIMALLOC_PURGE_DELAY=0): returns freed memory at once, so what counting freed is not
// still held while the trainer peaks
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const SHARDS: usize = 64;

#[derive(Clone, Copy, Default)]
struct Slot {
    h: u64,
    cl: u64,       // count in the low 56 bits, word length in the top 8 (words >= 255 bytes: 255, length in the arena)
    key: [u8; 16], // the word itself when <= 16 bytes, else its arena offset (u64 LE) and length (u64 LE)
}

const CNT: u64 = (1 << 56) - 1;

/// unique word -> count, open addressing on the word's hash; words of <= 16 bytes live in the slot (one memory access
/// per lookup), longer ones in the map's own arena
struct WordMap {
    s: Vec<Slot>,
    arena: Vec<u8>,
    n: usize,
    mask: usize,
}

impl WordMap {
    fn new() -> Self {
        WordMap { s: vec![Slot::default(); 1 << 10], arena: Vec::new(), n: 0, mask: (1 << 10) - 1 }
    }
    #[inline]
    fn word<'a>(&'a self, x: &'a Slot) -> &'a [u8] {
        let l = (x.cl >> 56) as usize;
        if l <= 16 {
            &x.key[..l]
        } else {
            let off = u64::from_le_bytes(x.key[..8].try_into().unwrap()) as usize;
            let len = u64::from_le_bytes(x.key[8..].try_into().unwrap()) as usize;
            &self.arena[off..off + len]
        }
    }
    #[inline]
    fn add(&mut self, w: &[u8], h: u64, c: u64) {
        let l8 = (w.len().min(255) as u64) << 56;
        let mut i = h as usize & self.mask;
        while self.s[i].h != 0 {
            let x = &self.s[i];
            if x.h == h && x.cl & !CNT == l8 && self.word(x) == w {
                self.s[i].cl += c;
                return;
            }
            i = (i + 1) & self.mask;
        }
        let mut key = [0u8; 16];
        if w.len() <= 16 {
            key[..w.len()].copy_from_slice(w);
        } else {
            key[..8].copy_from_slice(&(self.arena.len() as u64).to_le_bytes());
            key[8..].copy_from_slice(&(w.len() as u64).to_le_bytes());
            self.arena.extend_from_slice(w);
        }
        self.s[i] = Slot { h, cl: l8 | c, key };
        self.n += 1;
        if self.n * 2 > self.s.len() {
            let old = std::mem::take(&mut self.s);
            self.s = vec![Slot::default(); old.len() * 2];
            self.mask = self.s.len() - 1;
            for x in old {
                if x.h != 0 {
                    let mut j = x.h as usize & self.mask;
                    while self.s[j].h != 0 {
                        j = (j + 1) & self.mask;
                    }
                    self.s[j] = x;
                }
            }
        }
    }
    /// Empty the map, keeping its allocations (a worker's local map is flushed and reused).
    fn clear(&mut self) {
        self.s.fill(Slot::default());
        self.arena.clear();
        self.n = 0;
    }
    fn words(&self) -> impl Iterator<Item = (&[u8], u64, u64)> {
        self.s.iter().filter(|x| x.h != 0).map(|x| (self.word(x), x.h, x.cl & CNT))
    }
}

/// Small direct-mapped front for WordMap: words of <= 16 bytes with the key inline, counts accumulated here and
/// flushed to the big map on eviction and at the end. Frequent words never touch the big map (tens of MB, misses the
/// CPU caches). env HB_FRONT_BITS (12; 0 = off).
#[derive(Clone, Copy, Default)]
struct FSlot {
    key: u128,
    h: u64,
    cnt: u32,
    len: u32,
}

struct Front {
    s: Vec<FSlot>,
    mask: usize,
}

impl Front {
    fn new(bits: u32) -> Self {
        Front { s: vec![FSlot::default(); 1 << bits], mask: (1 << bits) - 1 }
    }
    #[inline]
    fn add(&mut self, w: &[u8], h: u64, back: &mut WordMap) {
        if w.len() > 16 || self.mask == 0 {
            back.add(w, h, 1);
            return;
        }
        let mut b = [0u8; 16];
        b[..w.len()].copy_from_slice(w);
        let key = u128::from_le_bytes(b);
        let s = &mut self.s[(h >> 24) as usize & self.mask];
        if s.h == h && s.key == key && s.len as usize == w.len() {
            s.cnt += 1;
            if s.cnt == u32::MAX {
                Self::flush(s, back);
            }
            return;
        }
        if s.h != 0 {
            Self::flush(s, back);
        }
        *s = FSlot { key, h, cnt: 1, len: w.len() as u32 };
    }
    #[inline]
    fn prefetch(&self, h: u64) {
        rust_bpe::prefetch(&self.s[(h >> 24) as usize & self.mask]);
    }
    #[inline]
    fn flush(s: &mut FSlot, back: &mut WordMap) {
        if s.h != 0 && s.cnt > 0 {
            back.add(&s.key.to_le_bytes()[..s.len as usize], s.h, s.cnt as u64);
        }
        s.cnt = 0;
        if s.h != 0 {
            s.h = 0;
        }
    }
    fn drain(&mut self, back: &mut WordMap) {
        for s in self.s.iter_mut() {
            Self::flush(s, back);
        }
    }
}

fn byte_chars() -> [char; 256] {
    // GPT-2 bytes_to_unicode
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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (corpus, vocab, out) = (&a[1], a[2].parse::<usize>().unwrap(), &a[3]);
    std::fs::create_dir_all(out).unwrap();
    let t_n = env_or("HB_THREADS", 1usize);
    tk_encode::parallelism::set_num_threads(t_n);
    let block = env_or("HB_BLOCK_KB", 4096usize) << 10;
    // HB_PRETOK: pre-tokenizer name (rust_bpe::pretok, default bytelevel); HB_NORMALIZER: normalizer spec
    // (rust_bpe::normalize, as BB_NORMALIZER of our trainer), applied to every line first
    let kind = pretok::Kind::from_name(&std::env::var("HB_PRETOK").unwrap_or("bytelevel".into())).expect("HB_PRETOK");
    let norm = rust_bpe::normalize::parse(&std::env::var("HB_NORMALIZER").unwrap_or_default()).expect("HB_NORMALIZER");
    let byte_level = kind.byte_level();
    let t0 = std::time::Instant::now();
    let f = File::open(corpus).unwrap();
    let size = f.metadata().unwrap().len() as usize;
    let cut = |x: usize, scratch: &mut Vec<u8>| -> usize {
        if x == 0 || x >= size {
            return x.min(size);
        }
        let mut from = x - 2;
        loop {
            let n = (64usize << 10).min(size - from);
            scratch.resize(n, 0);
            f.read_exact_at(scratch, from as u64).unwrap();
            for c in (x.max(from + 2))..from + n {
                if scratch[c - from - 1] == b'\n' {
                    return c;
                }
            }
            if from + n >= size {
                return size;
            }
            from += n - 2;
        }
    };
    let n_chunks = size.div_ceil(block).max(1);
    let simd = env_or("HB_SIMD", 1usize) == 1;
    let strip_nl = env_or("HB_STRIP_NL", 0usize) == 1;
    let stage = env_or("HB_STAGE", 2usize); // profiling: 0 pre-tokenize only, 1 + hash, 2 full count
    let next = AtomicUsize::new(0);
    // Counts: each worker counts into a local map; when it holds more than HB_LOCAL_WORDS distinct words
    // it is flushed into one global map split in SHARDS parts by hash, each behind its own lock, and reused. So a
    // word is stored once globally, plus a bounded local map per worker (memory does not grow with the workers).
    // the locals together hold at most ~512k words (65536 each at 8 workers), so memory does not grow with workers
    let local_limit = if t_n == 1 { usize::MAX } else { env_or("HB_LOCAL_WORDS", (1usize << 19) / t_n) };
    let global: Vec<std::sync::Mutex<WordMap>> = (0..SHARDS).map(|_| std::sync::Mutex::new(WordMap::new())).collect();
    let flush = |m: &mut WordMap, by: &mut Vec<Vec<u32>>| {
        by.iter_mut().for_each(Vec::clear);
        for (i, x) in m.s.iter().enumerate() {
            if x.h != 0 {
                by[(x.h >> 58) as usize % SHARDS].push(i as u32);
            }
        }
        for (k, idx) in by.iter().enumerate() {
            if idx.is_empty() {
                continue;
            }
            let mut g = global[k].lock().unwrap();
            for &i in idx {
                let x = &m.s[i as usize];
                g.add(m.word(x), x.h, x.cl & CNT);
            }
        }
        m.clear();
    };
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..t_n)
            .map(|_| {
                let (next, cut, f, norm, flush) = (&next, &cut, &f, &norm, &flush);
                sc.spawn(move || {
                    let (mut data, mut scratch) = (Vec::new(), Vec::new());
                    let mut m = WordMap::new();
                    let mut by: Vec<Vec<u32>> = (0..SHARDS).map(|_| Vec::new()).collect();
                    let mut front = Front::new(env_or("HB_FRONT_BITS", 0u32));
                    let (mut tags, mut starts, mut flag, mut spans) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    let mut sink = 0u64;
                    let (mut nbuf, mut wbuf) = (String::new(), Vec::new());
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n_chunks {
                            std::hint::black_box(sink);
                            front.drain(&mut m);
                            flush(&mut m, &mut by);
                            break;
                        }
                        if m.n > local_limit {
                            flush(&mut m, &mut by);
                        }
                        let (lo, hi) = (cut(i * block, &mut scratch), cut((i + 1) * block, &mut scratch));
                        data.resize(hi - lo, 0);
                        f.read_exact_at(&mut data, lo as u64).unwrap();
                        // per line (with its '\n'), as HF's feed and ours: a pre-token never spans a line end.
                        // HB_SIMD=1: HF's bitcannon (SIMD classify + bitstream GPT-2 grammar) finds the token starts
                        for line in data.split_inclusive(|&c| c == b'\n') {
                            // invalid UTF-8 (e.g. a file cut mid-character) becomes U+FFFD, as HF's feed reads it
                            let lossy;
                            let line: &[u8] = match std::str::from_utf8(line) {
                                Ok(_) => line,
                                Err(_) => {
                                    lossy = String::from_utf8_lossy(line).into_owned();
                                    lossy.as_bytes()
                                }
                            };
                            // HB_STRIP_NL=1: drop the line's "\n" / "\r\n", as a feed of `BufRead::lines()` does
                            let line = if strip_nl { line.strip_suffix(b"\n").map_or(line, |l| l.strip_suffix(b"\r").unwrap_or(l)) } else { line };
                            let line: &[u8] = if norm.is_empty() {
                                line
                            } else {
                                rust_bpe::normalize::apply(&norm, std::str::from_utf8(line).unwrap(), &mut nbuf);
                                nbuf.as_bytes()
                            };
                            if simd && kind == pretok::Kind::Gpt2 {
                                let n = line.len();
                                tags.resize(n.max(tags.len()), 0);
                                starts.resize(n.div_ceil(64).max(starts.len()), 0);
                                flag.resize(n.div_ceil(64).max(flag.len()), 0);
                                spans.resize(n.max(spans.len()), bitcannon::Span { start: 0, end: 0 });
                                bitcannon::classify::classify(line, &mut tags[..n]);
                                let k = bitcannon::bitcannon_byte_level(line, &tags[..n], &mut starts, &mut flag, &mut spans);
                                for sp in &spans[..k] {
                                    let w = &line[sp.start as usize..sp.end as usize];
                                    front.add(w, rust_bpe::hbytes(w) | 1, &mut m);
                                }
                                continue;
                            }
                            if kind == pretok::Kind::Gpt2 {
                                let mut p = 0;
                                while p < line.len() {
                                    let e = rust_bpe::next_gpt2(line, p);
                                    let w = &line[p..e];
                                    match stage {
                                        0 => sink = sink.wrapping_add(e as u64),
                                        1 => sink ^= rust_bpe::hbytes(w),
                                        _ => front.add(w, rust_bpe::hbytes(w) | 1, &mut m),
                                    }
                                    p = e;
                                }
                                continue;
                            }
                            pretok::for_each_word(kind, line, &mut wbuf, |w| {
                                if !w.is_empty() {
                                    front.add(w, rust_bpe::hbytes(w) | 1, &mut m);
                                }
                            });
                        }
                    }
                })
            })
            .collect();
        hs.into_iter().for_each(|h| h.join().unwrap());
    });
    let t_count = t0.elapsed().as_secs_f64();
    let rss_count = rust_bpe::peak_rss_mb();
    // each global shard is turned into (word, count) entries on its own thread, mapping byte-level words to chars
    let map = byte_chars();
    let parts: Vec<Vec<(CompactString, u64)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = global
            .into_iter()
            .map(|g| {
                let map = &map;
                sc.spawn(move || {
                    let all = g.into_inner().unwrap();
                    all.words()
                        .map(|(w, _, c)| {
                            let k: CompactString = if byte_level { w.iter().map(|&b| map[b as usize]).collect() } else { std::str::from_utf8(w).unwrap().into() };
                            (k, c)
                        })
                        .collect()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let t_reduce = t0.elapsed().as_secs_f64();
    let rss_reduce = rust_bpe::peak_rss_mb();
    // unique (word, count) entries straight to the trainer, no map of them (PR #2501's own feed does the same)
    let mut wc: Vec<(CompactString, u64)> = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for p in parts {
        wc.extend(p);
    }
    let n_words = wc.len();
    // hand the memory counting freed back to the system before training (HB_TRIM=0 keeps it)
    if env_or("HB_TRIM", 1usize) == 1 {
        unsafe { libmimalloc_sys::mi_collect(true) };
    }
    let feed_s = t0.elapsed().as_secs_f64();
    let rss_feed = rust_bpe::peak_rss_mb();
    let t1 = std::time::Instant::now();
    // the merge loop's own worker count (HB_MERGE_THREADS, default = HB_THREADS): it stops scaling before counting does
    tk_encode::parallelism::set_num_threads(env_or("HB_MERGE_THREADS", t_n));
    // non-byte-level pre-tokenizers train with an [UNK] special token, as train_hf.py / train_ours.py
    // (HB_UNK=0: no special token); HB_BYTE_ALPHABET=1: all 256 byte-level chars in the vocab, as initial_alphabet
    let specials = if byte_level || env_or("HB_UNK", 1usize) == 0 { vec![] } else { vec![tk_encode::vocab::bucket_added_vocabulary::AddedToken::from("[UNK]", true)] };
    let mut builder = BpeTrainer::builder().vocab_size(vocab).min_frequency(env_or("HB_MIN_FREQ", 0u64)).special_tokens(specials).show_progress(false);
    if byte_level && env_or("HB_BYTE_ALPHABET", 0usize) == 1 {
        builder = builder.initial_alphabet(byte_chars().into_iter().collect());
    }
    let trainer = builder.build();
    let (vocab, merges, _) = trainer.do_train_entries(wc).unwrap();
    let train_s = t1.elapsed().as_secs_f64();
    let mut f = std::io::BufWriter::new(File::create(format!("{out}/merges.txt")).unwrap());
    for (l, r) in &merges {
        writeln!(f, "{l} {r}").unwrap();
    }
    let mut v: Vec<_> = vocab.iter().collect();
    v.sort_by_key(|x| *x.1);
    let mut g = std::io::BufWriter::new(File::create(format!("{out}/vocab.txt")).unwrap());
    for (t, i) in v {
        writeln!(g, "{i}\t{t}").unwrap();
    }
    println!("count_s={t_count:.3} reduce_s={:.3} collect_s={:.3} feed_s={feed_s:.3} ", t_reduce - t_count, feed_s - t_reduce);
    println!("train_s={train_s:.3} words={n_words}");
    println!("peak_rss_mb: count {rss_count:.0} reduce {rss_reduce:.0} feed {rss_feed:.0} train {:.0}", rust_bpe::peak_rss_mb());
}
