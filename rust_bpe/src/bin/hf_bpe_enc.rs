//! Rust port of hf_bpe_enc.cpp: streaming, multi-threaded byte-level BPE
//! encoder that reproduces HuggingFace `tokenizers` ids exactly (models
//! exported with export_hfbp.py).
//!
//!   hf_bpe_enc <m.hfbp> stream <input> <out.ids> [n_bytes]       parallel reads + work stealing (default)
//!   hf_bpe_enc <m.hfbp> stream_old <input> <out.ids> [n_bytes]   previous version: serial read, encode, write rounds
//!   hf_bpe_enc <m.hfbp> lines  <input> <single|batch|concurrent|dump> [reps|out]
//! env: HB_THREADS (1), HB_BLOCK_KB (4096), HB_CACHE (1), HB_U16 (0)
use rust_bpe::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Added {
    s: Vec<u8>,
    id: u32,
    single_word: bool,
}

struct Model {
    pretok: u32,
    kind: pretok::Kind,
    char_level: bool,
    unk: u32,               // id of the unknown-char token (char-level models), !0 if none
    ascii_id: [u32; 128],   // char-level: id of each single ASCII char, !0 if not in vocab
    char_k: Vec<u32>,       // char-level: open-addressing table codepoint -> id (keys; !0 = empty)
    char_v: Vec<u32>,
    char_mask: usize,
    norm: Vec<normalize::Step>, // normalizer applied to each piece before pre-tokenizing (empty = none)
    prefix: bool,
    ignore_merges: bool,
    ntok: usize,
    byte_id: [u32; 256],
    mk: Vec<u64>,
    mv: Vec<u64>, // (l<<32|r) -> (rank<<32 | merged)
    mmask: usize,
    vocab: HashMap<Vec<u8>, u32>,
    added: Vec<Added>,
    added_first: [bool; 256],
}

impl Model {
    /// id of the single-char token for the UTF-8 char c (2-4 bytes), !0 if not in the vocab
    #[inline]
    fn char_id(&self, c: &[u8]) -> u32 {
        let (cp, _) = decode(c, 0);
        let mut i = mix64(cp as u64) as usize & self.char_mask;
        loop {
            let k = self.char_k[i];
            if k == !0 {
                return !0;
            }
            if k == cp {
                return self.char_v[i];
            }
            i = (i + 1) & self.char_mask;
        }
    }

    #[inline]
    fn merge(&self, l: u32, r: u32) -> Option<(u32, u32)> {
        let key = ((l as u64) << 32) | r as u64;
        let mut i = mix64(key) as usize & self.mmask;
        loop {
            let k = self.mk[i];
            if k == !0u64 {
                return None;
            }
            if k == key {
                let v = self.mv[i];
                return Some(((v >> 32) as u32, v as u32));
            }
            i = (i + 1) & self.mmask;
        }
    }

    fn load(path: &str) -> Result<Model, String> {
        let mut d = Vec::new();
        File::open(path).and_then(|mut f| f.read_to_end(&mut d)).map_err(|e| e.to_string())?;
        let mut p = 0usize;
        let mut take = |n: usize| {
            let s = &d[p..p + n];
            p += n;
            s.to_vec()
        };
        let u32_ = |b: Vec<u8>| u32::from_le_bytes(b[..4].try_into().unwrap());
        let u16_ = |b: Vec<u8>| u16::from_le_bytes(b[..2].try_into().unwrap());
        if take(4) != b"HFBP" {
            return Err("bad magic".into());
        }
        let ver = u32_(take(4));
        let pretok = u32_(take(4));
        let kind = pretok::from_code(pretok).ok_or(format!("unknown pre-tokenizer code {pretok}"))?;
        let char_level = !kind.byte_level();
        let pf = take(1)[0];
        let im = take(1)[0];
        take(2);
        let n_vocab = u32_(take(4)) as usize;
        let n_merges = u32_(take(4)) as usize;
        let mut tok = Vec::with_capacity(n_vocab);
        for _ in 0..n_vocab {
            let l = u16_(take(2)) as usize;
            tok.push(take(l));
        }
        let mut byte_id = [!0u32; 256];
        let mut vocab = HashMap::with_capacity(n_vocab);
        for (i, t) in tok.iter().enumerate() {
            if t.len() == 1 && byte_id[t[0] as usize] == !0 {
                byte_id[t[0] as usize] = i as u32;
            }
            if !t.is_empty() {
                vocab.entry(t.clone()).or_insert(i as u32);
            }
        }
        let mut cap = 1;
        while cap < n_merges * 2 + 2 {
            cap <<= 1;
        }
        let mmask = cap - 1;
        let (mut mk, mut mv) = (vec![!0u64; cap], vec![0u64; cap]);
        for r in 0..n_merges {
            let (a, b, m) = (u32_(take(4)), u32_(take(4)), u32_(take(4)));
            let key = ((a as u64) << 32) | b as u64;
            let mut i = mix64(key) as usize & mmask;
            while mk[i] != !0u64 && mk[i] != key {
                i = (i + 1) & mmask;
            }
            if mk[i] == !0u64 {
                mk[i] = key;
                mv[i] = ((r as u64) << 32) | m as u64;
            }
        }
        let na = u32_(take(4));
        let mut added = Vec::new();
        let mut added_first = [false; 256];
        for _ in 0..na {
            let id = u32_(take(4));
            let fl = take(5);
            let l = u16_(take(2)) as usize;
            let s = take(l);
            added_first[s[0] as usize] = true;
            added.push(Added { s, id, single_word: fl[1] != 0 });
        }
        // version 2 appends the unk id
        let unk = if ver >= 2 { u32_(take(4)) } else { !0 };
        let norm = if ver >= 3 {
            let n = u32_(take(4)) as usize;
            normalize::parse(std::str::from_utf8(&take(n)).map_err(|e| e.to_string())?)?
        } else {
            Vec::new()
        };
        if normalize::whole_text_only(&norm) {
            return Err("Prepend / Strip normalizers need the whole text; chunked encoding can't apply them".into());
        }
        let mut ascii_id = [!0u32; 128];
        for c in 0..128u8 {
            if let Some(&id) = vocab.get(&vec![c]) {
                ascii_id[c as usize] = id;
            }
        }
        if char_level {
            byte_id = [!0u32; 256]; // not used: chars are looked up in char_k / char_v
        }
        // single-char tokens: codepoint -> id, for char-level BPE (faster than hashing byte strings)
        let singles: Vec<(u32, u32)> = vocab
            .iter()
            .filter_map(|(t, &id)| {
                let s = std::str::from_utf8(t).ok()?;
                let mut it = s.chars();
                let c = it.next()?;
                if it.next().is_none() { Some((c as u32, id)) } else { None }
            })
            .collect();
        let mut cap = 2;
        while cap < singles.len() * 2 + 2 {
            cap <<= 1;
        }
        let (mut char_k, mut char_v) = (vec![!0u32; cap], vec![0u32; cap]);
        for &(c, id) in &singles {
            let mut i = mix64(c as u64) as usize & (cap - 1);
            while char_k[i] != !0 {
                i = (i + 1) & (cap - 1);
            }
            char_k[i] = c;
            char_v[i] = id;
        }
        Ok(Model { pretok, kind, char_level, unk, ascii_id, char_k, char_v, char_mask: cap - 1, norm, prefix: pf != 0, ignore_merges: im != 0, ntok: n_vocab, byte_id, mk, mv, mmask, vocab, added, added_first })
    }
}

const CW: usize = 24; // cache: words <= 24 bytes with <= 8 ids
const CI: usize = 8;
const CN: usize = 1 << 16;

#[derive(Clone, Copy)]
struct Slot {
    h: u64,
    wl: u8,
    ni: u8,
    w: [u8; CW],
    ids: [u32; CI],
}
const EMPTY: Slot = Slot { h: 0, wl: 0, ni: 0, w: [0; CW], ids: [0; CI] };

#[derive(Clone, Copy)]
struct H {
    rank: u32,
    pos: i32,
    l: u32,
    r: u32,
}

/// One per thread.
// One per thread. Aligned to 128 bytes so two threads' encoders never share a cache line
// (their Vec lengths change on every word; sharing a line made 8 threads only ~2.5x faster than 1).
#[repr(align(128))]
struct Encoder<'m> {
    m: &'m Model,
    use_cache: bool,
    sym: Vec<u32>,
    prv: Vec<i32>,
    nxt: Vec<i32>,
    heap: Vec<H>,
    scratch: Vec<u8>,
    cache: Vec<Slot>,
    wbuf: Vec<u8>,
    wends: Vec<usize>,
    nbuf: String,
}

// min-heap on (rank, pos)
#[inline]
fn less(a: &H, b: &H) -> bool {
    a.rank < b.rank || (a.rank == b.rank && a.pos < b.pos)
}
fn heap_push(h: &mut Vec<H>, x: H) {
    h.push(x);
    let mut i = h.len() - 1;
    while i > 0 {
        let p = (i - 1) / 2;
        if !less(&h[i], &h[p]) {
            break;
        }
        h.swap(i, p);
        i = p;
    }
}
fn heap_pop(h: &mut Vec<H>) -> H {
    let top = h[0];
    let last = h.pop().unwrap();
    if !h.is_empty() {
        h[0] = last;
        let n = h.len();
        let mut i = 0;
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut m = i;
            if l < n && less(&h[l], &h[m]) {
                m = l;
            }
            if r < n && less(&h[r], &h[m]) {
                m = r;
            }
            if m == i {
                break;
            }
            h.swap(i, m);
            i = m;
        }
    }
    top
}

fn is_word(b: &[u8], p: usize) -> bool {
    let (c, _) = cls_at(b, p);
    c == LET || c == NUM || b[p] == b'_'
}

impl<'m> Encoder<'m> {
    fn new(m: &'m Model, use_cache: bool) -> Self {
        Encoder { m, use_cache, sym: vec![], prv: vec![], nxt: vec![], heap: vec![], scratch: vec![], cache: vec![EMPTY; CN], wbuf: vec![], wends: vec![], nbuf: String::new() }
    }

    fn bpe(&mut self, w: &[u8], out: &mut Vec<u32>) {
        let m = self.m;
        let (sym, prv, nxt, heap) = (&mut self.sym, &mut self.prv, &mut self.nxt, &mut self.heap);
        sym.clear();
        prv.clear();
        nxt.clear();
        heap.clear();
        // bytes with no token (tokenizer trained without the full byte alphabet) are dropped,
        // as HF's BPE does with no unk token, and their neighbours may merge across the gap
        if m.char_level {
            // one symbol per char: its token, else [UNK] (HF BPE with unk_token, fuse_unk = false)
            let mut p = 0;
            while p < w.len() {
                let l = utf8_len(w[p]).min(w.len() - p);
                let id = if l == 1 && w[p] < 0x80 { m.ascii_id[w[p] as usize] } else { m.char_id(&w[p..p + l]) };
                let id = if id == !0 { m.unk } else { id };
                if id != !0 {
                    sym.push(id);
                }
                p += l;
            }
        } else {
            sym.extend(w.iter().map(|&b| m.byte_id[b as usize]).filter(|&x| x != !0));
        }
        let l = sym.len();
        if l == 0 {
            return;
        }
        for i in 0..l {
            prv.push(i as i32 - 1);
            nxt.push(if i + 1 < l { i as i32 + 1 } else { -1 });
        }
        macro_rules! push {
            ($p:expr) => {{
                let p = $p as usize;
                let (a, b) = (sym[p], sym[nxt[p] as usize]);
                if let Some((rk, _)) = m.merge(a, b) {
                    heap_push(heap, H { rank: rk, pos: p as i32, l: a, r: b });
                }
            }};
        }
        for i in 0..l.saturating_sub(1) {
            push!(i);
        }
        while !heap.is_empty() {
            let e = heap_pop(heap);
            let p = e.pos as usize;
            let q = nxt[p];
            if q < 0 || sym[p] != e.l || sym[q as usize] != e.r {
                continue;
            }
            let q = q as usize;
            let mg = match m.merge(e.l, e.r) {
                Some((rk, mg)) if rk == e.rank => mg,
                _ => continue,
            };
            sym[p] = mg;
            nxt[p] = nxt[q];
            if nxt[q] >= 0 {
                prv[nxt[q] as usize] = p as i32;
            }
            nxt[q] = -2;
            if prv[p] >= 0 {
                push!(prv[p]);
            }
            if nxt[p] >= 0 {
                push!(p);
            }
        }
        let mut i = 0i32;
        while i >= 0 {
            out.push(sym[i as usize]);
            i = nxt[i as usize];
        }
    }

    fn word(&mut self, w: &[u8], out: &mut Vec<u32>) {
        let l = w.len();
        if l == 1 && !self.m.char_level {
            let id = self.m.byte_id[w[0] as usize];
            if id != !0 {
                out.push(id);
            }
            return;
        }
        let mut slot = usize::MAX;
        let mut h = 0;
        if self.use_cache && l <= CW {
            h = hbytes(w) | 1;
            slot = h as usize & (CN - 1);
            let s = &self.cache[slot];
            if s.h == h && s.wl as usize == l && &s.w[..l] == w {
                out.extend_from_slice(&s.ids[..s.ni as usize]);
                return;
            }
        }
        let o = out.len();
        match if self.m.ignore_merges { self.m.vocab.get(w) } else { None } {
            Some(&id) => out.push(id),
            None => self.bpe(w, out),
        }
        let n = out.len() - o;
        if slot != usize::MAX && n <= CI {
            let s = &mut self.cache[slot];
            s.h = h;
            s.wl = l as u8;
            s.ni = n as u8;
            s.w[..l].copy_from_slice(w);
            s.ids[..n].copy_from_slice(&out[o..]);
        }
    }

    fn pretokenize(&mut self, b: &[u8], out: &mut Vec<u32>) {
        if self.m.kind == pretok::Kind::Gpt4o {
            // GPT-4o never rewrites the text: walk it with the scanner and encode slices (no copies)
            let st = std::str::from_utf8(b).expect("input is not valid UTF-8");
            let t = pretok::table();
            let mut p = 0;
            while p < b.len() {
                let q = pretok::next_gpt4o(st, p, t);
                self.word(&b[p..q], out);
                p = q;
            }
            return;
        }
        if self.m.pretok >= 2 {
            // every other pre-tokenizer: the shared module (rust_bpe/src/pretok.rs). Words are copied out
            // first (Metaspace rewrites spaces, so its words are not slices of b).
            let (mut words, mut ends, mut sc) = (std::mem::take(&mut self.wbuf), std::mem::take(&mut self.wends), Vec::new());
            words.clear();
            ends.clear();
            pretok::for_each_word(self.m.kind, b, &mut sc, |w| {
                words.extend_from_slice(w);
                ends.push(words.len());
            });
            let mut st = 0;
            for &e in &ends {
                self.word(&words[st..e], out);
                st = e;
            }
            self.wbuf = words;
            self.wends = ends;
            return;
        }
        let nx = if self.m.pretok == 1 { next_llama3 } else { next_gpt2 };
        let mut p = 0;
        while p < b.len() {
            let q = nx(b, p);
            self.word(&b[p..q], out);
            p = q;
        }
    }

    fn piece(&mut self, b: &[u8], prefix_ok: bool, out: &mut Vec<u32>) {
        if b.is_empty() {
            return;
        }
        if !self.m.norm.is_empty() {
            let mut nb = std::mem::take(&mut self.nbuf);
            normalize::apply(&self.m.norm, std::str::from_utf8(b).expect("input is not valid UTF-8"), &mut nb);
            let nbytes = std::mem::take(&mut nb).into_bytes();
            self.piece_normalized(&nbytes, prefix_ok, out);
            self.nbuf = String::from_utf8(nbytes).unwrap();
            return;
        }
        self.piece_normalized(b, prefix_ok, out);
    }

    fn piece_normalized(&mut self, b: &[u8], prefix_ok: bool, out: &mut Vec<u32>) {
        if b.is_empty() {
            return;
        }
        if self.m.prefix && prefix_ok && b[0] != b' ' {
            let mut sc = std::mem::take(&mut self.scratch);
            sc.clear();
            sc.push(b' ');
            sc.extend_from_slice(b);
            self.pretokenize(&sc, out);
            self.scratch = sc;
        } else {
            self.pretokenize(b, out);
        }
    }

    /// Encode one unit of text; input_start=false means this unit continues a longer input.
    fn unit(&mut self, s: &[u8], input_start: bool, out: &mut Vec<u32>) {
        let e = s.len();
        let (mut ps, mut first) = (0usize, true);
        let m = self.m;
        if !m.added.is_empty() {
            let mut p = 0;
            while p < e {
                if !m.added_first[s[p] as usize] {
                    p += 1;
                    continue;
                }
                let mut best: Option<&Added> = None;
                for a in &m.added {
                    if a.s.len() <= e - p && s[p..p + a.s.len()] == a.s[..] && best.map_or(true, |b| a.s.len() > b.s.len()) {
                        best = Some(a);
                    }
                }
                let best = match best {
                    Some(b) => b,
                    None => {
                        p += 1;
                        continue;
                    }
                };
                let q = p + best.s.len();
                if best.single_word {
                    let mut left_ok = true;
                    if p > 0 {
                        let mut c = p - 1;
                        while c > 0 && (s[c] & 0xC0) == 0x80 {
                            c -= 1;
                        }
                        left_ok = !is_word(s, c);
                    }
                    let right_ok = q == e || !is_word(s, q);
                    if !left_ok || !right_ok {
                        p = q;
                        continue;
                    }
                }
                self.piece(&s[ps..p], input_start || !first, out);
                first = false;
                out.push(best.id);
                p = q;
                ps = q;
            }
        }
        self.piece(&s[ps..], input_start || !first, out);
    }
}

#[inline]
fn utf8_len(c: u8) -> usize {
    if c < 0x80 { 1 } else if c >= 0xF0 { 4 } else if c >= 0xE0 { 3 } else if c >= 0xC0 { 2 } else { 1 }
}

fn write_ids(out: &mut impl Write, ids: &[u32], u16out: bool) {
    if u16out {
        let v: Vec<u8> = ids.iter().flat_map(|&x| (x as u16).to_le_bytes()).collect();
        out.write_all(&v).unwrap();
    } else {
        let v: Vec<u8> = ids.iter().flat_map(|&x| x.to_le_bytes()).collect();
        out.write_all(&v).unwrap();
    }
}

fn main() {
    run(std::env::args().collect());
}

/// Same as the command line: a = [program, args...]; options come from env vars.
pub fn run(a: Vec<String>) {
    if a.len() < 4 {
        eprintln!("see header for usage");
        std::process::exit(1);
    }
    let t_n = env_or("HB_THREADS", 1usize).max(1);
    let use_cache = env_or("HB_CACHE", 1i32) != 0;
    let u16out = env_or("HB_U16", 0i32) != 0;
    let m = Model::load(&a[1]).unwrap_or_else(|e| {
        eprintln!("cannot load {}: {e}", a[1]);
        std::process::exit(1)
    });
    if u16out && m.ntok > 65536 {
        eprintln!("vocab too large for u16");
        std::process::exit(1);
    }
    let mut enc: Vec<Encoder> = (0..t_n).map(|_| Encoder::new(&m, use_cache)).collect();
    let mut obuf: Vec<Vec<u32>> = vec![Vec::new(); t_n];

    match a[2].as_str() {
        "stream" => {
            // Chunk i covers file bytes [cut(i*B), cut((i+1)*B)), cut(x) = first safe cut point >= x (cut(0) = 0).
            // Every worker can find both edges of its chunk alone, so workers read (pread) and encode their
            // chunks in parallel. env: HB_BLOCK_KB (chunk size, 1024), HB_WINDOW (max chunks ahead of the writer, 4*T)
            let block = env_or("HB_BLOCK_KB", 1024usize) << 10;
            let f = File::open(&a[3]).expect("input");
            let size = (f.metadata().unwrap().len() as usize).min(a.get(5).map(|s| s.parse().unwrap()).unwrap_or(usize::MAX));
            let mut out = BufWriter::with_capacity(1 << 20, File::create(&a[4]).expect("output"));
            // first safe cut >= x, reading forward from x in 64 KB steps
            let cut = |x: usize, scratch: &mut Vec<u8>| -> usize {
                if x == 0 || x >= size {
                    return x.min(size);
                }
                let mut from = x - 2; // safe_cut(c) looks at bytes c-2, c-1, c
                loop {
                    let n = (64usize << 10).min(size - from);
                    scratch.resize(n, 0);
                    f.read_exact_at(scratch, from as u64).unwrap();
                    for c in (x.max(from + 2))..from + n {
                        if pretok::safe_cut(m.kind, scratch, c - from) {
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
            // Workers take chunk indices from `next`; a worker may run at most `window` chunks ahead of the
            // writer (bounded memory). The writer (this thread) writes chunk ids strictly in order while
            // the workers keep encoding, so writing overlaps with encoding.
            let window = env_or("HB_WINDOW", 4 * t_n).max(t_n);
            let next = AtomicUsize::new(0);
            let state = std::sync::Mutex::new((0usize, HashMap::<usize, Vec<u32>>::new())); // (written, finished chunks)
            let cv = std::sync::Condvar::new();
            let mut n_tok = 0usize;
            let t0 = now();
            std::thread::scope(|sc| {
                for e in enc.iter_mut() {
                    let (next, state, cv, cut, f) = (&next, &state, &cv, &cut, &f);
                    sc.spawn(move || {
                        let (mut data, mut scratch, mut free) = (Vec::new(), Vec::new(), Vec::new());
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= n_chunks {
                                break;
                            }
                            {
                                let mut st = state.lock().unwrap();
                                while i >= st.0 + window {
                                    st = cv.wait(st).unwrap();
                                }
                            }
                            let (lo, hi) = (cut(i * block, &mut scratch), cut((i + 1) * block, &mut scratch));
                            let mut o: Vec<u32> = std::mem::take(&mut free);
                            o.clear();
                            if hi > lo {
                                data.resize(hi - lo, 0);
                                f.read_exact_at(&mut data, lo as u64).unwrap();
                                e.unit(&data, lo == 0, &mut o);
                            }
                            state.lock().unwrap().1.insert(i, o);
                            cv.notify_all();
                        }
                    });
                }
                for w in 0..n_chunks {
                    let o = {
                        let mut st = state.lock().unwrap();
                        loop {
                            if let Some(o) = st.1.remove(&w) {
                                break o;
                            }
                            st = cv.wait(st).unwrap();
                        }
                    };
                    n_tok += o.len();
                    write_ids(&mut out, &o, u16out);
                    state.lock().unwrap().0 = w + 1;
                    cv.notify_all();
                }
            });
            out.flush().unwrap();
            let s = now() - t0;
            println!(
                "{{\"mode\":\"stream\",\"bytes\":{},\"tokens\":{},\"bpt\":{:.4},\"threads\":{},\"cache\":{},\"s\":{:.3},\"mbps\":{:.2},\"peak_rss_mb\":{:.1}}}",
                size, n_tok, size as f64 / n_tok as f64, t_n, use_cache as i32, s, size as f64 / s / 1e6, peak_rss_mb()
            );
        }
        "stream_old" => {
            let block = env_or("HB_BLOCK_KB", 4096usize) << 10;
            let mut inp = File::open(&a[3]).expect("input");
            let mut out = BufWriter::with_capacity(1 << 20, File::create(&a[4]).expect("output"));
            let limit: usize = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(usize::MAX);
            let mut buf = vec![0u8; t_n * block + 1];
            let (mut have, mut n_read, mut n_tok, mut eof, mut first_unit) = (0usize, 0usize, 0usize, false, true);
            let t0 = now();
            while !eof || have > 0 {
                if !eof {
                    if have == buf.len() {
                        buf.resize(buf.len() * 2, 0); // no safe cut yet: grow
                    }
                    let want = (buf.len() - have).min(limit - n_read);
                    let mut got = 0;
                    while got < want {
                        let r = inp.read(&mut buf[have + got..have + want]).unwrap();
                        if r == 0 {
                            break;
                        }
                        got += r;
                    }
                    have += got;
                    n_read += got;
                    if got < want || n_read >= limit {
                        eof = true;
                    }
                }
                let mut end = have;
                if !eof {
                    end = 0;
                    let mut k = have - 1;
                    while k >= 2 {
                        if pretok::safe_cut(m.kind, &buf, k) {
                            end = k;
                            break;
                        }
                        k -= 1;
                    }
                    if end == 0 {
                        continue;
                    }
                }
                let mut sp = vec![0usize];
                for t in 1..t_n {
                    let mut c = (*sp.last().unwrap()).max(end * t / t_n);
                    while c < end && !pretok::safe_cut(m.kind, &buf, c) {
                        c += 1;
                    }
                    sp.push(c.min(end));
                }
                sp.push(end);
                let data = &buf[..end];
                let fu = first_unit;
                if t_n == 1 {
                    obuf[0].clear();
                    enc[0].unit(data, fu, &mut obuf[0]);
                } else {
                    std::thread::scope(|s| {
                        for (t, (e, o)) in enc.iter_mut().zip(obuf.iter_mut()).enumerate() {
                            let span = &data[sp[t]..sp[t + 1]];
                            s.spawn(move || {
                                o.clear();
                                if !span.is_empty() {
                                    e.unit(span, fu && t == 0, o);
                                }
                            });
                        }
                    });
                }
                first_unit = false;
                for o in &obuf {
                    n_tok += o.len();
                    write_ids(&mut out, o, u16out);
                }
                buf.copy_within(end..have, 0);
                have -= end;
            }
            out.flush().unwrap();
            let s = now() - t0;
            println!(
                "{{\"mode\":\"stream\",\"bytes\":{},\"tokens\":{},\"bpt\":{:.4},\"threads\":{},\"cache\":{},\"s\":{:.3},\"mbps\":{:.2},\"peak_rss_mb\":{:.1}}}",
                n_read, n_tok, n_read as f64 / n_tok as f64, t_n, use_cache as i32, s, n_read as f64 / s / 1e6, peak_rss_mb()
            );
        }
        "lines" => {
            let mut data = Vec::new();
            File::open(&a[3]).expect("input").read_to_end(&mut data).unwrap();
            // Rust str::lines(): split on \n, strip one trailing \r
            let mut lines: Vec<(usize, usize)> = Vec::new();
            let mut s = 0;
            while s < data.len() {
                let e = data[s..].iter().position(|&c| c == b'\n').map_or(data.len(), |i| s + i);
                let mut le = e;
                if le > s && data[le - 1] == b'\r' {
                    le -= 1;
                }
                lines.push((s, le - s));
                s = e + 1;
            }
            let line = |i: usize| &data[lines[i].0..lines[i].0 + lines[i].1];
            let sub = a[4].as_str();
            if sub == "dump" {
                let mut f = BufWriter::new(File::create(&a[5]).unwrap());
                f.write_all(&(lines.len() as u32).to_le_bytes()).unwrap();
                let mut o = Vec::new();
                for i in 0..lines.len() {
                    o.clear();
                    enc[0].unit(line(i), true, &mut o);
                    f.write_all(&(o.len() as u32).to_le_bytes()).unwrap();
                    write_ids(&mut f, &o, false);
                }
                return;
            }
            let reps: usize = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(10);
            let mut ts: Vec<f64> = Vec::new();
            let (mut n_tok, mut bytes) = (0usize, data.len());
            if sub == "concurrent" {
                let inputs: Vec<Vec<u8>> = (0..t_n)
                    .map(|i| {
                        let mut v = Vec::new();
                        for k in 0..1000 {
                            if k > 0 {
                                v.push(b'\n');
                            }
                            v.extend_from_slice(line(i * 1000 + k));
                        }
                        v
                    })
                    .collect();
                bytes = inputs.iter().map(|v| v.len()).sum();
                for _ in 0..reps {
                    let t0 = now();
                    std::thread::scope(|s| {
                        for ((e, o), inp) in enc.iter_mut().zip(obuf.iter_mut()).zip(inputs.iter()) {
                            s.spawn(move || {
                                o.clear();
                                e.unit(inp, true, o);
                            });
                        }
                    });
                    ts.push(now() - t0);
                    n_tok = obuf.iter().map(|o| o.len()).sum();
                }
            } else {
                for _ in 0..reps {
                    for e in enc.iter_mut() {
                        e.cache.fill(EMPTY); // cold cache each pass
                    }
                    let t0 = now();
                    if sub == "single" {
                        obuf[0].clear();
                        for i in 0..lines.len() {
                            enc[0].unit(line(i), true, &mut obuf[0]);
                        }
                    } else {
                        // batch: batches of 1000 lines spread over T threads
                        for o in obuf.iter_mut() {
                            o.clear();
                        }
                        let mut b = 0;
                        while b < lines.len() {
                            let be = lines.len().min(b + 1000);
                            let next = AtomicUsize::new(b);
                            std::thread::scope(|s| {
                                for (e, o) in enc.iter_mut().zip(obuf.iter_mut()) {
                                    let (next, line) = (&next, &line);
                                    s.spawn(move || loop {
                                        let i = next.fetch_add(1, Ordering::Relaxed);
                                        if i >= be {
                                            break;
                                        }
                                        e.unit(line(i), true, o);
                                    });
                                }
                            });
                            b = be;
                        }
                    }
                    ts.push(now() - t0);
                    n_tok = obuf.iter().map(|o| o.len()).sum();
                }
            }
            ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let med = ts[ts.len() / 2];
            println!(
                "{{\"mode\":\"lines-{}\",\"bytes\":{},\"lines\":{},\"tokens\":{},\"threads\":{},\"cache\":{},\"median_ms\":{:.3},\"min_ms\":{:.3},\"mbps\":{:.2}}}",
                sub, bytes, lines.len(), n_tok, t_n, use_cache as i32, med * 1e3, ts[0] * 1e3, bytes as f64 / med / 1e6
            );
        }
        _ => {
            eprintln!("unknown mode");
            std::process::exit(1);
        }
    }
}
