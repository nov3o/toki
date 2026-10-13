//! union-encode <tokenizer.json> <corpus> <pretok> <ids_out>: HF main's encoder inside our streaming driver (rust_bpe
//! hf_bpe_enc `stream`). Chunk i covers file bytes [cut(i*B), cut((i+1)*B)), cut(x) = first safe cut >= x, so every
//! worker finds its own chunk, reads it with pread, encodes it with HF's `encode_into` (serial, on the worker's
//! thread, no `Encoding`) and writes its ids itself with pwrite at its offset in the output. The only serial step is
//! the offset prefix sum (one addition per chunk); workers run at most HB_WINDOW chunks ahead of it.
//! env: HB_THREADS (1), HB_BLOCK_KB (1024), HB_WINDOW (4*threads). Prints encode_s= tokens=.
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use rust_bpe::{env_or, pretok};
use tk_encode::pipeline::EncodeOptions;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

struct Prefix {
    known: usize,               // chunks 0..known have their output offset in offs
    offs: Vec<usize>,           // offs[i] = output offset (ids) of chunk i, valid for i <= known
    lens: HashMap<usize, usize>, // finished chunks past `known`, waiting for the prefix sum
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (tok, corpus, kind, out) = (&a[1], &a[2], pretok::Kind::from_name(&a[3]).unwrap(), &a[4]);
    let t_n = env_or("HB_THREADS", 1usize);
    // HB_REPLICAS=r: r copies of the model, each built (first-touched) by a thread pinned where its workers run,
    // so every worker reads tables from its own NUMA node. Workers pin to the allowed cpus in order.
    let cpus = allowed_cpus();
    let reps = env_or("HB_REPLICAS", 1usize).clamp(1, t_n);
    let tl = std::time::Instant::now();
    let json = if tok.ends_with(".canon.json") { std::fs::read_to_string(tok).unwrap() } else { tk_convert::canonicalize_file(tok).unwrap() };
    let t_conv = tl.elapsed().as_secs_f64();
    // HB_SAVE_CANON=1: keep the converted (2.0) JSON next to the input as <name>.canon.json, which loads without conversion
    if env_or("HB_SAVE_CANON", 0usize) == 1 && !tok.ends_with(".canon.json") {
        std::fs::write(tok.replace(".json", ".canon.json"), &json).unwrap();
    }
    let models: Vec<_> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..reps)
            .map(|j| {
                let (json, cpus) = (&json, &cpus);
                sc.spawn(move || {
                    pin(cpus[(j * t_n / reps) % cpus.len()]);
                    tk_serialize::from_json(json).unwrap()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    eprintln!("load: canonicalize {:.3} s, build {:.3} s ({reps} replicas)", t_conv, tl.elapsed().as_secs_f64() - t_conv);
    let opts = EncodeOptions::no_specials();
    let block = env_or("HB_BLOCK_KB", 1024usize) << 10;
    let window = env_or("HB_WINDOW", 4 * t_n).max(t_n);
    let t0 = std::time::Instant::now();
    let f = File::open(corpus).unwrap();
    let size = f.metadata().unwrap().len() as usize;
    let w = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(out).unwrap();
    // HB_PREALLOC=1: reserve the output's pages up front (ids <= input bytes for every BPE vocab with all bytes, so
    // 4 * size bounds it; we reserve `size` and let the file grow past it if needed), then trim to the real length
    let prealloc = env_or("HB_PREALLOC", 0usize) == 1;
    if prealloc {
        unsafe { libc::posix_fallocate(std::os::fd::AsRawFd::as_raw_fd(&w), 0, size as i64) };
    }
    // HB_MMAP=1: the output is a sparse file of 4 * size bytes (an id per input byte at most) mapped shared; each
    // worker copies its ids into the mapping (page faults run in parallel, no file lock), the file is trimmed at the end
    let use_mmap = env_or("HB_MMAP", 0usize) == 1;
    // HB_SHARDS=1: map-reduce output. Worker k appends its chunks to <out>.k (no file shared between workers, no
    // ordering wait); <out>.index lists, in chunk order, "worker offset_ids n_ids", so concatenating the pieces in
    // index order gives exactly the single-file output
    let sharded = env_or("HB_SHARDS", 0usize) == 1;
    let map_len = (4 * size).max(4);
    let map = if use_mmap {
        w.set_len(map_len as u64).unwrap();
        let p = unsafe {
            libc::mmap(std::ptr::null_mut(), map_len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED,
                       std::os::fd::AsRawFd::as_raw_fd(&w), 0)
        };
        assert!(p != libc::MAP_FAILED);
        p as usize
    } else {
        0
    };
    // first safe cut >= x, reading forward from x in 64 KB steps
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
    let nowrite = env_or("HB_NOWRITE", 0usize) == 1; // profiling: skip the output write
    let next = AtomicUsize::new(0);
    let state = Mutex::new(Prefix { known: 0, offs: vec![0; n_chunks + 1], lens: HashMap::new() });
    let cv = Condvar::new();
    let index = Mutex::new(vec![(0usize, 0usize, 0usize); n_chunks]);
    std::thread::scope(|sc| {
        for wi in 0..t_n {
            let tk = &models[wi * reps / t_n];
            let (next, state, cv, cut, f, w, opts, cpus) = (&next, &state, &cv, &cut, &f, &w, &opts, &cpus);
            let index = &index;
            sc.spawn(move || {
                let mut shard = if sharded { Some(std::io::BufWriter::with_capacity(8 << 20, File::create(format!("{out}.{wi}")).unwrap())) } else { None };
                let mut shard_off = 0usize;
                if env_or("HB_PIN", 1usize) == 1 {
                    pin(cpus[wi % cpus.len()]);
                }
                let (mut data, mut scratch, mut ids, mut o) = (Vec::new(), Vec::new(), Vec::new(), Vec::<u8>::new());
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n_chunks {
                        break;
                    }
                    if !sharded {
                        let mut st = state.lock().unwrap();
                        while i >= st.known + window {
                            st = cv.wait(st).unwrap();
                        }
                    }
                    let (lo, hi) = (cut(i * block, &mut scratch), cut((i + 1) * block, &mut scratch));
                    ids.clear();
                    if hi > lo {
                        data.resize(hi - lo, 0);
                        f.read_exact_at(&mut data, lo as u64).unwrap();
                        // lossy as HF's own driver (a file cut mid-character ends in U+FFFD); borrows when the text is valid
                        tk.encode_into(&String::from_utf8_lossy(&data), opts, &mut ids).unwrap();
                    }
                    o.clear();
                    o.extend(ids.iter().flat_map(|t| t.id().to_le_bytes()));
                    if let Some(sh) = shard.as_mut() {
                        use std::io::Write;
                        sh.write_all(&o).unwrap();
                        index.lock().unwrap()[i] = (wi, shard_off, ids.len());
                        shard_off += ids.len();
                        continue;
                    }
                    // publish the length, advance the prefix sum as far as it goes, then wait for this chunk's offset
                    let off = {
                        let mut st = state.lock().unwrap();
                        st.lens.insert(i, ids.len());
                        let mut moved = false;
                        while let Some(n) = { let k = st.known; st.lens.remove(&k) } {
                            let k = st.known;
                            st.offs[k + 1] = st.offs[k] + n;
                            st.known += 1;
                            moved = true;
                        }
                        if moved {
                            cv.notify_all();
                        }
                        while st.known < i {
                            st = cv.wait(st).unwrap();
                        }
                        st.offs[i]
                    };
                    if use_mmap {
                        unsafe { std::ptr::copy_nonoverlapping(o.as_ptr(), (map as *mut u8).add(off * 4), o.len()) };
                    } else if !nowrite {
                        w.write_all_at(&o, (off * 4) as u64).unwrap();
                    }
                }
            });
        }
    });
    let mut n_tok = state.into_inner().unwrap().offs[n_chunks];
    if sharded {
        use std::io::Write;
        let idx = index.into_inner().unwrap();
        n_tok = idx.iter().map(|x| x.2).sum();
        let mut g = std::io::BufWriter::new(File::create(format!("{out}.index")).unwrap());
        for (k, off, n) in idx {
            writeln!(g, "{k} {off} {n}").unwrap();
        }
    }
    if use_mmap {
        unsafe { libc::munmap(map as *mut libc::c_void, map_len) };
    }
    if prealloc || use_mmap {
        w.set_len((n_tok * 4) as u64).unwrap();
    }
    println!("encode_s={:.3} tokens={n_tok}", t0.elapsed().as_secs_f64());
}

fn allowed_cpus() -> Vec<usize> {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
    (0..libc::CPU_SETSIZE as usize).filter(|&c| unsafe { libc::CPU_ISSET(c, &set) }).collect()
}

fn pin(cpu: usize) {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}
