//! Shared pieces of the Rust port: Unicode classes, the hand-compiled GPT-2 /
//! Llama-3 / Whitespace pre-tokenizers, and hashing. Mirrors the C++ sources
//! (bucket_bpe_r2.cpp, hf_bpe_enc.cpp) one to one.
mod uclass;
mod hf_scripts;
pub mod pretok;
pub mod normalize;
pub mod simd;

pub const OTH: u8 = 0;
pub const LET: u8 = 1;
pub const NUM: u8 = 2;
pub const WSP: u8 = 3;

#[inline]
pub fn ucls(cp: u32) -> u8 {
    if cp >= 0x110000 {
        return 0;
    }
    uclass::STAGE2[uclass::STAGE1[(cp >> 8) as usize] as usize][(cp & 255) as usize]
}

static ASCII_CLS: [u8; 128] = uclass::ASCII;

/// UTF-8 decode at b[p]; invalid or truncated sequences give (U+FFFD, 1).
#[inline]
pub fn decode(b: &[u8], p: usize) -> (u32, usize) {
    let c = b[p];
    if c < 0x80 {
        return (c as u32, 1);
    }
    let n = if c >= 0xF0 { 4 } else if c >= 0xE0 { 3 } else if c >= 0xC0 { 2 } else { 1 };
    if n == 1 || p + n > b.len() {
        return (0xFFFD, 1);
    }
    let mut cp = (c & (0x7F >> n)) as u32;
    for k in 1..n {
        let x = b[p + k];
        if x & 0xC0 != 0x80 {
            return (0xFFFD, 1);
        }
        cp = (cp << 6) | (x & 0x3F) as u32;
    }
    (cp, n)
}

#[inline]
pub fn cls_at(b: &[u8], p: usize) -> (u8, usize) {
    let c = b[p];
    if c < 0x80 {
        return (ASCII_CLS[c as usize], 1);
    }
    let (cp, l) = decode(b, p);
    (ucls(cp), l)
}

#[inline]
fn run_of(b: &[u8], mut q: usize, c: u8) -> usize {
    // letters / digits: jump over ASCII ones with SIMD, then take one non-ASCII char the exact way
    let k = match c {
        LET => Some(simd::Ascii::Letter),
        NUM => Some(simd::Ascii::Digit),
        _ => None,
    };
    if let Some(k) = k {
        loop {
            q = simd::skip_ascii(b, q, k);
            if q >= b.len() || b[q] < 0x80 {
                return q; // end, or an ASCII byte of another class
            }
            let (kk, l) = cls_at(b, q);
            if kk != c {
                return q;
            }
            q += l;
        }
    }
    while q < b.len() {
        let (k, l) = cls_at(b, q);
        if k != c {
            break;
        }
        q += l;
    }
    q
}

/// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
/// Returns the end of the pre-token starting at p (the slice end is the text end).
#[inline]
pub fn next_gpt2(b: &[u8], p: usize) -> usize {
    let e = b.len();
    if b[p] == b'\'' && p + 1 < e {
        let n = b[p + 1];
        if n == b's' || n == b't' || n == b'm' || n == b'd' {
            return p + 2;
        }
        if p + 2 < e && ((n == b'r' && b[p + 2] == b'e') || (n == b'v' && b[p + 2] == b'e') || (n == b'l' && b[p + 2] == b'l')) {
            return p + 3;
        }
    }
    let mut q = p;
    let (mut c, mut len) = cls_at(b, q);
    if b[p] == b' ' && p + 1 < e {
        let (c2, l2) = cls_at(b, p + 1);
        if c2 != WSP {
            q = p + 1;
            c = c2;
            len = l2;
        }
    }
    if c != WSP {
        return run_of(b, q + len, c);
    }
    let mut last = q;
    q += len;
    while q < e {
        let (c3, l3) = cls_at(b, q);
        if c3 != WSP {
            break;
        }
        last = q;
        q += l3;
    }
    if q == e {
        return q;
    }
    if last > p { last } else { q }
}

/// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
#[inline]
pub fn next_llama3(b: &[u8], p: usize) -> usize {
    let e = b.len();
    if b[p] == b'\'' && p + 1 < e {
        let n = b[p + 1] | 0x20;
        if n == b's' || n == b't' || n == b'm' || n == b'd' {
            return p + 2;
        }
        if p + 2 < e {
            let n2 = b[p + 2] | 0x20;
            if (n == b'r' && n2 == b'e') || (n == b'v' && n2 == b'e') || (n == b'l' && n2 == b'l') {
                return p + 3;
            }
        }
    }
    let (c0, l0) = cls_at(b, p);
    if c0 == LET {
        return run_of(b, p + l0, LET);
    }
    if c0 != NUM && b[p] != b'\r' && b[p] != b'\n' && p + l0 < e {
        let (c1, l1) = cls_at(b, p + l0);
        if c1 == LET {
            return run_of(b, p + l0 + l1, LET);
        }
    }
    if c0 == NUM {
        let mut q = p + l0;
        let mut k = 1;
        while k < 3 && q < e {
            let (c, l) = cls_at(b, q);
            if c != NUM {
                break;
            }
            q += l;
            k += 1;
        }
        return q;
    }
    let (mut s, mut cs, mut ls) = (p, c0, l0);
    if b[p] == b' ' && p + 1 < e {
        let (c1, l1) = cls_at(b, p + 1);
        if c1 == OTH {
            s = p + 1;
            cs = OTH;
            ls = l1;
        }
    }
    if cs == OTH {
        let mut q = run_of(b, s + ls, OTH);
        while q < e && (b[q] == b'\r' || b[q] == b'\n') {
            q += 1;
        }
        return q;
    }
    let mut q = p;
    let mut lastnl = usize::MAX;
    let mut lastc = p;
    while q < e {
        let (c, l) = cls_at(b, q);
        if c != WSP {
            break;
        }
        if b[q] == b'\r' || b[q] == b'\n' {
            lastnl = q;
        }
        lastc = q;
        q += l;
    }
    if lastnl != usize::MAX {
        return lastnl + 1;
    }
    if q == e {
        return q;
    }
    if lastc > p { lastc } else { q }
}

/// \w = Alphabetic | M | Nd | Pc | Join_Control, approximated from the class table.
#[inline]
pub fn is_word_cp(cp: u32) -> bool {
    if cp < 0x80 {
        let l = cp | 0x20;
        return (cp >= b'0' as u32 && cp <= b'9' as u32) || (l >= b'a' as u32 && l <= b'z' as u32) || cp == b'_' as u32;
    }
    let c = ucls(cp);
    if c == LET || c == NUM {
        return true;
    }
    if matches!(cp, 0x200C | 0x200D | 0x203F | 0x2040 | 0x2054 | 0xFE33 | 0xFE34 | 0xFE4D..=0xFE4F | 0xFF3F) {
        return true;
    }
    matches!(cp, 0x0300..=0x036F | 0x0483..=0x0489 | 0x0591..=0x05C7 | 0x0610..=0x061A | 0x064B..=0x065F
        | 0x0900..=0x0903 | 0x093A..=0x094F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F)
}

/// \w+|[^\w\s]+ with whitespace skipped. Returns (word_start, word_end); start == end means nothing left.
#[inline]
pub fn next_whitespace(b: &[u8], mut p: usize) -> (usize, usize) {
    let e = b.len();
    while p < e {
        let (c, l) = cls_at(b, p);
        if c != WSP {
            break;
        }
        p += l;
    }
    let ws = p;
    if p == e {
        return (ws, p);
    }
    let (cp, l) = decode(b, p);
    let w = is_word_cp(cp);
    p += l;
    while p < e {
        let (c2, l2) = decode(b, p);
        if is_word_cp(c2) != w || (!w && ucls(c2) == WSP) {
            break;
        }
        p += l2;
    }
    (ws, p)
}

#[inline]
pub fn hbytes(p: &[u8]) -> u64 {
    let mut n = p.len();
    let mut h = 0x9E3779B97F4A7C15u64 ^ (n as u64).wrapping_mul(0xff51afd7ed558ccd);
    let mut i = 0;
    while n >= 8 {
        let v = u64::from_le_bytes(p[i..i + 8].try_into().unwrap());
        h = (h ^ v).wrapping_mul(0xbf58476d1ce4e5b9);
        h ^= h >> 31;
        i += 8;
        n -= 8;
    }
    let mut t = [0u8; 8];
    t[..n].copy_from_slice(&p[i..]);
    let v = u64::from_le_bytes(t);
    h = (h ^ v).wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 29;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 32;
    h
}

#[inline]
pub fn mix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    k
}

pub fn now() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

/// Peak resident set size in MB (Linux reports ru_maxrss in KB).
pub fn peak_rss_mb() -> f64 {
    #[repr(C)]
    struct Rusage {
        utime: [i64; 2],
        stime: [i64; 2],
        maxrss: i64,
        rest: [i64; 13],
    }
    extern "C" {
        fn getrusage(who: i32, r: *mut Rusage) -> i32;
    }
    let mut r: Rusage = unsafe { std::mem::zeroed() };
    unsafe { getrusage(0, &mut r) };
    r.maxrss as f64 / 1024.0
}

pub fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Ask the CPU to start loading the cache line at p (no effect on results). Used to hide memory latency
/// when the next addresses are known ahead, e.g. the words a merge will visit.
#[inline(always)]
pub fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags));
    }
}
