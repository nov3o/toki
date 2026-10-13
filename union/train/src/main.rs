//! union-train <corpus> <vocab_size> <out_dir>: our feed + PR #2501's merge. Workers pread chunks cut at safe points
//! (as our encoder), pre-tokenize them with GPT-2 byte-level (rust_bpe::next_gpt2) and count words as byte slices in
//! per-thread maps; the maps are merged, mapped to byte-level chars once per unique word and handed to tk-train's
//! `BpeTrainer::do_train`. env: HB_THREADS (1), HB_BLOCK_KB (4096); HF's pool sized by set_num_threads(HB_THREADS).
//! Writes out_dir/merges.txt and vocab.txt like hfbench. Prints feed_s= train_s=.
use ahash::AHashMap;
use compact_str::CompactString;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use rust_bpe::{env_or, pretok};
use tk_train::trainers::BpeTrainer;

const SHARDS: usize = 64;

#[derive(Clone, Copy, Default)]
struct Slot {
    h: u64,
    off: u64,
    len: u32,
    cnt: u64,
}

/// unique word -> count, open addressing on the word's hash, keys in the map's own arena (as rust_bpe's trainer)
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
    fn add(&mut self, w: &[u8], h: u64, c: u64) {
        let len = w.len() as u32;
        let mut i = h as usize & self.mask;
        while self.s[i].h != 0 {
            let x = &mut self.s[i];
            if x.h == h && x.len == len && &self.arena[x.off as usize..x.off as usize + w.len()] == w {
                x.cnt += c;
                return;
            }
            i = (i + 1) & self.mask;
        }
        self.s[i] = Slot { h, off: self.arena.len() as u64, len, cnt: c };
        self.arena.extend_from_slice(w);
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
    fn words(&self) -> impl Iterator<Item = (&[u8], u64, u64)> {
        self.s.iter().filter(|x| x.h != 0).map(|x| (&self.arena[x.off as usize..x.off as usize + x.len as usize], x.h, x.cnt))
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
    let kind = pretok::Kind::Gpt2;
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
                if pretok::safe_cut(kind, scratch, c - from) {
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
    let next = AtomicUsize::new(0);
    // per-thread counts keyed by the word's bytes (owned: chunks are reused)
    let maps: Vec<Vec<WordMap>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..t_n)
            .map(|_| {
                let (next, cut, f) = (&next, &cut, &f);
                sc.spawn(move || {
                    let (mut data, mut scratch) = (Vec::new(), Vec::new());
                    let mut m: Vec<WordMap> = (0..SHARDS).map(|_| WordMap::new()).collect();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n_chunks {
                            break m;
                        }
                        let (lo, hi) = (cut(i * block, &mut scratch), cut((i + 1) * block, &mut scratch));
                        data.resize(hi - lo, 0);
                        f.read_exact_at(&mut data, lo as u64).unwrap();
                        // per line (with its '\n'), as HF's feed and ours: a pre-token never spans a line end
                        for line in data.split_inclusive(|&c| c == b'\n') {
                            let mut p = 0;
                            while p < line.len() {
                                let e = rust_bpe::next_gpt2(line, p);
                                let w = &line[p..e];
                                let h = rust_bpe::hbytes(w) | 1;
                                m[(h >> 58) as usize % SHARDS].add(w, h, 1);
                                p = e;
                            }
                        }
                    }
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // shard k of every thread goes to one merging thread, which also maps its words to byte-level chars
    let mut by_shard: Vec<Vec<WordMap>> = (0..SHARDS).map(|_| Vec::new()).collect();
    for ms in maps {
        for (k, m) in ms.into_iter().enumerate() {
            by_shard[k].push(m);
        }
    }
    let map = byte_chars();
    let parts: Vec<Vec<(CompactString, u64)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = by_shard
            .into_iter()
            .map(|ms| {
                let map = &map;
                sc.spawn(move || {
                    let mut it = ms.into_iter();
                    let mut all = it.next().unwrap();
                    for m in it {
                        for (w, h, c) in m.words() {
                            all.add(w, h, c);
                        }
                    }
                    all.words().map(|(w, _, c)| (w.iter().map(|&b| map[b as usize]).collect(), c)).collect()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut wc: AHashMap<CompactString, u64> = AHashMap::with_capacity(parts.iter().map(|p| p.len()).sum());
    for p in parts {
        wc.extend(p);
    }
    let feed_s = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let trainer = BpeTrainer::builder().vocab_size(vocab).show_progress(false).build();
    let (vocab, merges, _) = trainer.do_train(&wc).unwrap();
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
    println!("feed_s={feed_s:.3} train_s={train_s:.3} words={}", wc.len());
}
