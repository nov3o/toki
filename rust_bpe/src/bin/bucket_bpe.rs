//! Rust port of bucket_bpe_r2.cpp: BPE training whose merge list is identical
//! to HuggingFace's BpeTrainer.
//!
//! usage: bucket_bpe <input> <n_bytes|0=all> <vocab_size> [merges_out]
//! env:   BB_THREADS=T, BB_PRETOK=<name in pretok.rs: bytelevel|gpt2, llama3, gpt4o, whitespace, whitespace_split,
//!        metaspace, bert, punctuation, digits, unicode_scripts, char_delimiter>, BB_BLOCK_MB (chunk size, 4)
//!        BB_NORMALIZER=<spec, see normalize.rs>  normalize every line first (HF normalizers)
//!        BB_SPECIAL=tok1,tok2  special tokens: ids 0.. in the written vocab, count toward vocab_size (like HF's special_tokens)
//!        BB_VOCAB_OUT=path     write the final vocab "id<TAB>token"
use rust_bpe::*;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::os::unix::fs::FileExt;

#[derive(Clone, Copy, Default)]
struct Slot {
    h: u64,
    off: u64,
    len: u32,
    cnt: u64,
}

/// unique word -> count; keys live in this map's own arena.
/// One per thread, aligned to 128 bytes so two threads' maps never share a cache line (false sharing).
#[repr(align(128))]
struct WordMap {
    s: Vec<Slot>,
    arena: Vec<u8>,
    n: usize,
    mask: usize,
}

impl WordMap {
    fn new() -> Self {
        WordMap { s: vec![Slot::default(); 1 << 12], arena: Vec::new(), n: 0, mask: (1 << 12) - 1 }
    }
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
}

struct PairMap {
    k: Vec<u64>,
    v: Vec<u32>,
    n: usize,
    mask: usize,
}

impl PairMap {
    fn new() -> Self {
        PairMap { k: vec![!0u64; 1 << 16], v: vec![0; 1 << 16], n: 0, mask: (1 << 16) - 1 }
    }
    #[inline]
    fn get_or_add(&mut self, key: u64, make: impl FnOnce() -> u32) -> u32 {
        let mut i = mix64(key) as usize & self.mask;
        while self.k[i] != !0u64 {
            if self.k[i] == key {
                return self.v[i];
            }
            i = (i + 1) & self.mask;
        }
        let id = make();
        self.k[i] = key;
        self.v[i] = id;
        self.n += 1;
        if self.n * 2 > self.k.len() {
            let ok = std::mem::take(&mut self.k);
            let ov = std::mem::take(&mut self.v);
            self.k = vec![!0u64; ok.len() * 2];
            self.v = vec![0; ok.len() * 2];
            self.mask = self.k.len() - 1;
            for j in 0..ok.len() {
                if ok[j] != !0u64 {
                    let mut x = mix64(ok[j]) as usize & self.mask;
                    while self.k[x] != !0u64 {
                        x = (x + 1) & self.mask;
                    }
                    self.k[x] = ok[j];
                    self.v[x] = ov[j];
                }
            }
        }
        id
    }
}

#[derive(PartialEq, Eq)]
struct Big {
    cnt: i64,
    key: u64,
    idx: u32,
}
impl Ord for Big {
    // max-heap: highest count first, ties on the smallest pair key
    fn cmp(&self, o: &Self) -> Ordering {
        self.cnt.cmp(&o.cnt).then(o.key.cmp(&self.key))
    }
}
impl PartialOrd for Big {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

const NB: i64 = 1 << 16;
struct BucketQueue {
    b: Vec<BinaryHeap<Reverse<(u64, u32)>>>,
    big: BinaryHeap<Big>,
    top: i64,
}

impl BucketQueue {
    fn new() -> Self {
        BucketQueue { b: (0..NB).map(|_| BinaryHeap::new()).collect(), big: BinaryHeap::new(), top: 0 }
    }
    #[inline]
    fn push(&mut self, idx: u32, key: u64, cnt: i64) {
        if cnt <= 0 {
            return;
        }
        if cnt >= NB {
            self.big.push(Big { cnt, key, idx });
            return;
        }
        self.b[cnt as usize].push(Reverse((key, idx)));
        if cnt > self.top {
            self.top = cnt;
        }
    }
    fn pop(&mut self, pcnt: &[i64]) -> Option<(u32, i64)> {
        while let Some(t) = self.big.pop() {
            if pcnt[t.idx as usize] == t.cnt {
                return Some((t.idx, t.cnt));
            }
            self.push(t.idx, t.key, pcnt[t.idx as usize]);
        }
        while self.top > 0 {
            let top = self.top;
            match self.b[top as usize].pop() {
                None => self.top -= 1,
                Some(Reverse((key, idx))) => {
                    if pcnt[idx as usize] == top {
                        return Some((idx, top));
                    }
                    self.push(idx, key, pcnt[idx as usize]);
                }
            }
        }
        None
    }
}

#[derive(Clone, Copy)]
struct Change {
    l: u32,
    r: u32,
    d: i64,
    w: u32,
}

/// Raw pointer that may cross threads: each word's symbols are edited by exactly one thread.
#[derive(Clone, Copy)]
struct Shared<T>(*mut T);
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

/// UTF-8 bytes of a code point (no surrogate check, same as the C++).
fn utf8(cp: u32) -> Vec<u8> {
    if cp < 0x80 {
        vec![cp as u8]
    } else if cp < 0x800 {
        vec![0xC0 | (cp >> 6) as u8, 0x80 | (cp & 63) as u8]
    } else if cp < 0x10000 {
        vec![0xE0 | (cp >> 12) as u8, 0x80 | ((cp >> 6) & 63) as u8, 0x80 | (cp & 63) as u8]
    } else {
        vec![0xF0 | (cp >> 18) as u8, 0x80 | ((cp >> 12) & 63) as u8, 0x80 | ((cp >> 6) & 63) as u8, 0x80 | (cp & 63) as u8]
    }
}

fn main() {
    run(std::env::args().collect());
}

/// Same as the command line: a = [program, args...]; options come from env vars.
pub fn run(a: Vec<String>) {
    if a.len() < 4 {
        eprintln!("usage: {} <input> <n_bytes|0> <vocab_size> [merges_out]", a[0]);
        std::process::exit(1);
    }
    let mut limit: usize = a[2].parse().unwrap();
    let vocab_size: usize = a[3].parse().unwrap();
    let outp = a.get(4);
    let t_n = env_or("BB_THREADS", 1usize).max(1);
    let mode_s = std::env::var("BB_PRETOK").unwrap_or_else(|_| "gpt2".into());
    let block = env_or("BB_BLOCK_MB", 4usize) << 20;
    let kind = pretok::Kind::from_name(&mode_s).unwrap_or_else(|| {
        eprintln!("unknown BB_PRETOK {mode_s}");
        std::process::exit(1)
    });
    let char_level = !kind.byte_level();
    // optional normalizer (spec format: rust_bpe/src/normalize.rs), applied to every line before pre-tokenizing
    let norm = normalize::parse(&std::env::var("BB_NORMALIZER").unwrap_or_default()).unwrap_or_else(|e| {
        eprintln!("BB_NORMALIZER: {e}");
        std::process::exit(1)
    });

    let t0 = now();
    // ---- 1. stream the file in line-aligned blocks; T threads count words per block
    let f = File::open(&a[1]).unwrap_or_else(|e| {
        eprintln!("open: {e}");
        std::process::exit(1)
    });
    if limit == 0 {
        limit = usize::MAX;
    }
    let mut maps: Vec<WordMap> = (0..t_n).map(|_| WordMap::new()).collect();
    // Chunk i = file bytes [lcut(i*B), lcut((i+1)*B)), lcut(x) = first line start >= x (lcut(0) = 0).
    // Each worker reads its chunks itself (pread) and counts words into its own map; chunk indices
    // come from a shared counter. Counting is order-independent, so there are no rounds or barriers.
    let fsize = f.metadata().unwrap().len() as usize;
    let mut size = fsize.min(limit);
    if size < fsize {
        // a byte cap may split a UTF-8 sequence: drop the partial tail (as before)
        let mut tail = [0u8; 4];
        let k = size.min(4);
        f.read_exact_at(&mut tail[..k], (size - k) as u64).unwrap();
        let mut i = k;
        while i > 0 && (tail[i - 1] & 0xC0) == 0x80 {
            i -= 1;
        }
        if i > 0 && tail[i - 1] >= 0xC0 {
            size -= k - (i - 1);
        }
    }
    let lcut = |x: usize, scratch: &mut Vec<u8>| -> usize {
        if x == 0 || x >= size {
            return x.min(size);
        }
        let mut from = x - 1; // a line starts at c if b[c-1] == '\n'
        loop {
            let n = (64usize << 10).min(size - from);
            scratch.resize(n, 0);
            f.read_exact_at(scratch, from as u64).unwrap();
            if let Some(i) = memchr(b'\n', scratch) {
                return from + i + 1;
            }
            if from + n >= size {
                return size;
            }
            from += n;
        }
    };
    let n_chunks = size.div_ceil(block).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let count_chunks = |m: &mut WordMap| {
        let (mut data, mut scratch, mut wbuf, mut nbuf) = (Vec::new(), Vec::new(), Vec::new(), String::new());
        loop {
            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if i >= n_chunks {
                break;
            }
            let (lo, hi) = (lcut(i * block, &mut scratch), lcut((i + 1) * block, &mut scratch));
            if hi <= lo {
                continue;
            }
            data.resize(hi - lo, 0);
            f.read_exact_at(&mut data, lo as u64).unwrap();
            let end = data.len();
            let mut p = 0;
            while p < end {
                // HF trains from files line by line (line endings kept)
                let le = match memchr(b'\n', &data[p..]) {
                    Some(i) => p + i + 1,
                    None => end,
                };
                let mut line = &data[p..le];
                if !norm.is_empty() {
                    normalize::apply(&norm, std::str::from_utf8(line).expect("input is not valid UTF-8"), &mut nbuf);
                    line = nbuf.as_bytes();
                }
                pretok::for_each_word(kind, line, &mut wbuf, |w| {
                    if !w.is_empty() {
                        m.add(w, hbytes(w) | 1, 1);
                    }
                });
                p = le;
            }
        }
    };
    if t_n == 1 {
        count_chunks(&mut maps[0]);
    } else {
        std::thread::scope(|s| {
            for m in maps.iter_mut() {
                let count_chunks = &count_chunks;
                s.spawn(move || count_chunks(m));
            }
        });
    }
    let n_read = size;
    drop(f);
    let (first, rest) = maps.split_at_mut(1);
    let wm = &mut first[0];
    for m in rest.iter_mut() {
        for x in &m.s {
            if x.h != 0 {
                let w = &m.arena[x.off as usize..x.off as usize + x.len as usize];
                wm.add(w, x.h, x.cnt);
            }
        }
        *m = WordMap { s: Vec::new(), arena: Vec::new(), n: 0, mask: 0 };
    }
    let t_count = now();

    // ---- 2. alphabet (HF order) and symbol arrays
    let mut b2u = [0u32; 256];
    {
        let mut k = 0;
        for b in 0..256u32 {
            b2u[b as usize] = if (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174 {
                b
            } else {
                k += 1;
                256 + k - 1
            };
        }
    }
    let mut tok: Vec<Vec<u8>> = Vec::new(); // id -> raw bytes
    let (mut woff, mut wlen, mut wcnt, mut sym) = (Vec::<u32>::new(), Vec::<u32>::new(), Vec::<i64>::new(), Vec::<u32>::new());
    let word = |x: &Slot| &wm.arena[x.off as usize..x.off as usize + x.len as usize];
    if !char_level {
        let mut seen = [false; 256];
        for x in wm.s.iter().filter(|x| x.h != 0) {
            for &c in word(x) {
                seen[c as usize] = true;
            }
        }
        let mut bytes: Vec<usize> = (0..256).filter(|&b| seen[b]).collect();
        bytes.sort_by_key(|&b| b2u[b]);
        let mut byte2id = [0u32; 256];
        for (i, &b) in bytes.iter().enumerate() {
            byte2id[b] = i as u32;
            tok.push(vec![b as u8]);
        }
        for x in wm.s.iter().filter(|x| x.h != 0) {
            woff.push(sym.len() as u32);
            wlen.push(x.len);
            wcnt.push(x.cnt as i64);
            sym.extend(word(x).iter().map(|&c| byte2id[c as usize]));
        }
    } else {
        let mut cps: Vec<u32> = Vec::new();
        for x in wm.s.iter().filter(|x| x.h != 0) {
            let w = word(x);
            let mut p = 0;
            while p < w.len() {
                let (cp, l) = decode(w, p);
                cps.push(cp);
                p += l;
            }
        }
        cps.sort_unstable();
        cps.dedup();
        for &cp in &cps {
            tok.push(utf8(cp));
        }
        for x in wm.s.iter().filter(|x| x.h != 0) {
            woff.push(sym.len() as u32);
            wcnt.push(x.cnt as i64);
            let w = word(x);
            let (mut p, mut n) = (0, 0u32);
            while p < w.len() {
                let (cp, l) = decode(w, p);
                sym.push(cps.binary_search(&cp).unwrap() as u32);
                p += l;
                n += 1;
            }
            wlen.push(n);
        }
    }
    let alphabet = tok.len();
    let nwords = woff.len();
    wm.s = Vec::new();
    wm.arena = Vec::new();

    // ---- 3. initial pair counts + postings
    let mut pm = PairMap::new();
    let mut pcnt: Vec<i64> = Vec::new();
    let mut pkey: Vec<u64> = Vec::new();
    let mut post: Vec<Vec<u32>> = Vec::new();
    let mut plast: Vec<u32> = Vec::new();
    let mut pstamp: Vec<u32> = Vec::new();
    macro_rules! pair_of {
        ($a:expr, $b:expr) => {{
            let key = (($a as u64) << 32) | $b as u64;
            pm.get_or_add(key, || {
                pcnt.push(0);
                pkey.push(key);
                post.push(Vec::new());
                plast.push(!0);
                pstamp.push(0);
                (pcnt.len() - 1) as u32
            }) as usize
        }};
    }
    for w in 0..nwords {
        let o = woff[w] as usize;
        let l = wlen[w] as usize;
        for i in 0..l.saturating_sub(1) {
            // start loading the pair-table slot of the pair 8 positions ahead (hides cache misses)
            if i + 9 < l {
                let slot = mix64(((sym[o + i + 8] as u64) << 32) | sym[o + i + 9] as u64) as usize & pm.mask;
                prefetch(pm.k.as_ptr().wrapping_add(slot));
            }
            let p = pair_of!(sym[o + i], sym[o + i + 1]);
            pcnt[p] += wcnt[w];
            if plast[p] != w as u32 {
                plast[p] = w as u32;
                post[p].push(w as u32);
            }
        }
    }
    let mut q = BucketQueue::new();
    for p in 0..pcnt.len() {
        q.push(p as u32, pkey[p], pcnt[p]);
    }
    let t_init = now();
    if std::env::var("BB_MEM").is_ok() {
        // memory breakdown of the main structures after init (MB)
        let mb = |b: usize| b as f64 / 1e6;
        let post_heap: usize = post.iter().map(|v| v.capacity() * 4).sum();
        eprintln!(
            "mem: sym={:.0} words(off,len,cnt)={:.0} pairs(cnt,key,last,stamp)={:.0} post_headers={:.0} post_heap={:.0} pairmap={:.0} (MB)",
            mb(sym.capacity() * 4), mb(woff.capacity() * 4 + wlen.capacity() * 4 + wcnt.capacity() * 8),
            mb(pcnt.capacity() * 8 + pkey.capacity() * 8 + plast.capacity() * 4 + pstamp.capacity() * 4),
            mb(post.capacity() * 24), mb(post_heap), mb(pm.k.capacity() * 8 + pm.v.capacity() * 4)
        );
    }

    // ---- 4. merge loop
    let mut str_ix: HashMap<Vec<u8>, u32> = HashMap::with_capacity(vocab_size * 2);
    for (i, s) in tok.iter().enumerate() {
        str_ix.entry(s.clone()).or_insert(i as u32);
    }
    let mut tch: Vec<Vec<Change>> = (0..t_n).map(|_| Vec::new()).collect();
    let mut merges: Vec<(u32, u32)> = Vec::new();
    let mut wstamp = vec![0u32; nwords];
    let mut touched: Vec<u32> = Vec::new();
    let (mut round, mut visits, mut par_rounds) = (0u32, 0usize, 0usize);
    let par_min: usize = env_or("BB_PAR_MIN", usize::MAX); // merge words in parallel only if a merge touches this many words (off: it was slower on every test)
    // HF puts special tokens at ids 0..n and counts them in vocab_size; internally we keep ids without them
    let special: Vec<String> = std::env::var("BB_SPECIAL").map(|v| v.split(',').filter(|x| !x.is_empty()).map(String::from).collect()).unwrap_or_default();
    let vocab_size = vocab_size.saturating_sub(special.len());
    while tok.len() < vocab_size {
        let (p, c) = match q.pop(&pcnt) {
            Some(x) => x,
            None => break,
        };
        if c < 1 {
            break;
        }
        round += 1;
        let p = p as usize;
        let (ta, tb) = ((pkey[p] >> 32) as u32, pkey[p] as u32);
        let mut s = tok[ta as usize].clone();
        s.extend_from_slice(&tok[tb as usize]);
        let x_id = match str_ix.get(&s) {
            Some(&id) => id,
            None => {
                let id = tok.len() as u32;
                str_ix.insert(s.clone(), id);
                tok.push(s);
                id
            }
        };
        merges.push((ta, tb));
        let mut ws = std::mem::take(&mut post[p]);
        // dedupe the word list (a word can be listed more than once)
        ws.retain(|&w| {
            if wstamp[w as usize] != round {
                wstamp[w as usize] = round;
                true
            } else {
                false
            }
        });
        visits += ws.len();
        let symp = Shared(sym.as_mut_ptr());
        let wlenp = Shared(wlen.as_mut_ptr());
        let (woff, wcnt) = (&woff, &wcnt);
        let merge_words = move |ws: &[u32], ch: &mut Vec<Change>| {
            let (symp, wlenp) = (symp, wlenp);
            const PF: usize = 8; // prefetch this many words ahead
            for (k, &w) in ws.iter().enumerate() {
                if let Some(&wn) = ws.get(k + PF) {
                    // the per-word arrays now, the word's symbols once its offset is (likely) in cache
                    prefetch(woff.as_ptr().wrapping_add(wn as usize));
                    prefetch(wcnt.as_ptr().wrapping_add(wn as usize));
                    prefetch(unsafe { wlenp.0.add(wn as usize) } as *const u32);
                }
                if let Some(&wn) = ws.get(k + PF / 2) {
                    prefetch(unsafe { symp.0.add(woff[wn as usize] as usize) } as *const u32);
                }
                let wi = w as usize;
                // SAFETY: words are distinct and own disjoint [woff, woff+wlen) ranges
                let l = unsafe { *wlenp.0.add(wi) } as usize;
                let s = unsafe { std::slice::from_raw_parts_mut(symp.0.add(woff[wi] as usize), l) };
                let cw = wcnt[wi];
                // jump to the first occurrence of (ta, tb); symbols before it stay where they are
                let mut r = match simd::find_pair(s, 0, ta, tb) {
                    Some(i) => i,
                    None => continue, // stale posting: this word no longer has the pair
                };
                let mut wr = r;
                loop {
                    // merge s[r], s[r+1] -> x_id; neighbours: the last written symbol and s[r+2]
                    if wr > 0 {
                        ch.push(Change { l: s[wr - 1], r: ta, d: -cw, w });
                        ch.push(Change { l: s[wr - 1], r: x_id, d: cw, w });
                    }
                    if r + 2 < l {
                        ch.push(Change { l: tb, r: s[r + 2], d: -cw, w });
                        ch.push(Change { l: x_id, r: s[r + 2], d: cw, w });
                    }
                    s[wr] = x_id;
                    wr += 1;
                    r += 2;
                    // copy the untouched run up to the next occurrence in one move
                    let nx = simd::find_pair(s, r, ta, tb).unwrap_or(l);
                    s.copy_within(r..nx, wr);
                    wr += nx - r;
                    r = nx;
                    if r >= l {
                        break;
                    }
                }
                unsafe { *wlenp.0.add(wi) = wr as u32 };
            }
        };
        let nt = if t_n > 1 && ws.len() >= par_min { t_n } else { 1 };
        for t in tch.iter_mut().take(nt) {
            t.clear();
        }
        if nt == 1 {
            merge_words(&ws, &mut tch[0]);
        } else {
            par_rounds += 1;
            let n = ws.len();
            std::thread::scope(|sc| {
                for (t, ch) in tch.iter_mut().enumerate() {
                    let part = &ws[n * t / nt..n * (t + 1) / nt];
                    sc.spawn(move || merge_words(part, ch));
                }
            });
        }
        touched.clear();
        for t in 0..nt {
            let list = &tch[t];
            for (k, ch) in list.iter().enumerate() {
                // start loading the pair-table slot of a change further down the list (hides cache misses)
                if let Some(c2) = list.get(k + 16) {
                    let slot = mix64(((c2.l as u64) << 32) | c2.r as u64) as usize & pm.mask;
                    prefetch(pm.k.as_ptr().wrapping_add(slot));
                    prefetch(pm.v.as_ptr().wrapping_add(slot));
                }
                let pp = pair_of!(ch.l, ch.r);
                pcnt[pp] += ch.d;
                if ch.d > 0 {
                    if plast[pp] != ch.w {
                        plast[pp] = ch.w;
                        post[pp].push(ch.w);
                    }
                    if pstamp[pp] != round {
                        pstamp[pp] = round;
                        touched.push(pp as u32);
                    }
                }
            }
        }
        for &pp in &touched {
            q.push(pp, pkey[pp as usize], pcnt[pp as usize]);
        }
    }
    let t_end = now();

    println!(
        "bytes={} threads={} pretok={} unique_words={} alphabet={} merges={} vocab={} pairs={} \
         count_s={:.3} init_s={:.3} merge_s={:.3} total_s={:.3} word_visits={} par_rounds={} peak_rss_mb={:.1}",
        n_read, t_n, mode_s, nwords, alphabet, merges.len(), tok.len(), pcnt.len(),
        t_count - t0, t_init - t_count, t_end - t_init, t_end - t0, visits, par_rounds, peak_rss_mb()
    );
    // BB_VOCAB_OUT: the vocabulary, "id<TAB>token" in id order, tokens in HF's byte-level form
    if let Ok(vp) = std::env::var("BB_VOCAB_OUT") {
        let mut o = BufWriter::new(File::create(vp).unwrap());
        for (i, sp) in special.iter().enumerate() {
            writeln!(o, "{i}\t{sp}").unwrap();
        }
        for (i, t) in tok.iter().enumerate() {
            write!(o, "{}\t", i + special.len()).unwrap();
            if char_level {
                o.write_all(t).unwrap();
            } else {
                for &ch in t {
                    o.write_all(char::from_u32(b2u[ch as usize]).unwrap().to_string().as_bytes()).unwrap();
                }
            }
            o.write_all(b"\n").unwrap();
        }
    }
    // BB_MODEL_OUT: machine-readable model (tokens may contain spaces / newlines):
    //   "<id> <hex of the token's raw bytes>" per token in id order (special tokens first), then a line "merges",
    //   then "<left id> <right id>" per merge step
    if let Ok(mp) = std::env::var("BB_MODEL_OUT") {
        let mut o = BufWriter::new(File::create(mp).unwrap());
        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        for (i, sp) in special.iter().enumerate() {
            writeln!(o, "{i} {}", hex(sp.as_bytes())).unwrap();
        }
        for (i, t) in tok.iter().enumerate() {
            writeln!(o, "{} {}", i + special.len(), hex(t)).unwrap();
        }
        writeln!(o, "merges").unwrap();
        for &(l, r) in &merges {
            writeln!(o, "{} {}", l as usize + special.len(), r as usize + special.len()).unwrap();
        }
    }
    if let Some(op) = outp {
        let mut o = BufWriter::new(File::create(op).unwrap());
        let put = |o: &mut BufWriter<File>, s: &[u8]| {
            if char_level {
                o.write_all(s).unwrap();
                return;
            }
            for &ch in s {
                o.write_all(char::from_u32(b2u[ch as usize]).unwrap().to_string().as_bytes()).unwrap();
            }
        };
        for &(l, r) in &merges {
            put(&mut o, &tok[l as usize]);
            o.write_all(b" ").unwrap();
            put(&mut o, &tok[r as usize]);
            o.write_all(b"\n").unwrap();
        }
    }
}

#[inline]
fn memchr(c: u8, s: &[u8]) -> Option<usize> {
    s.iter().position(|&x| x == c)
}
