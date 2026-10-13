//! Hand-written SIMD for the pre-tokenizer: skip over a run of ASCII letters or ASCII digits
//! 32 bytes at a time (AVX2, x86_64) or 16 bytes at a time (NEON, aarch64).
//! Only with the `simd` cargo feature; otherwise (and on other CPUs) a plain byte loop.
//! Callers handle non-ASCII bytes themselves, so results are identical either way.

/// Which ASCII class to skip: letters [A-Za-z] or digits [0-9].
#[derive(Clone, Copy)]
pub enum Ascii {
    Letter,
    Digit,
}

/// First position >= q whose byte is NOT an ASCII byte of class `k` (b.len() if none).
#[inline]
pub fn skip_ascii(b: &[u8], q: usize, k: Ascii) -> usize {
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe { skip_avx2(b, q, k) };
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { skip_neon(b, q, k) };
    }
    #[allow(unreachable_code)]
    skip_scalar(b, q, k)
}

#[inline]
fn is_k(c: u8, k: Ascii) -> bool {
    match k {
        Ascii::Letter => (c | 0x20).wrapping_sub(b'a') < 26,
        Ascii::Digit => c.wrapping_sub(b'0') < 10,
    }
}

#[inline]
pub fn skip_scalar(b: &[u8], mut q: usize, k: Ascii) -> usize {
    while q < b.len() && is_k(b[q], k) {
        q += 1;
    }
    q
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn skip_avx2(b: &[u8], mut q: usize, k: Ascii) -> usize {
    use std::arch::x86_64::*;
    // byte is in class  <=>  (x | or) - base <= lim   (unsigned; max_epu8(t, lim) == lim)
    let (or, base, lim) = match k {
        Ascii::Letter => (0x20u8, b'a', 25u8),
        Ascii::Digit => (0x00u8, b'0', 9u8),
    };
    let (vor, vbase, vlim) = (_mm256_set1_epi8(or as i8), _mm256_set1_epi8(base as i8), _mm256_set1_epi8(lim as i8));
    while q + 32 <= b.len() {
        let x = _mm256_loadu_si256(b.as_ptr().add(q) as *const __m256i);
        let t = _mm256_sub_epi8(_mm256_or_si256(x, vor), vbase);
        let ok = _mm256_cmpeq_epi8(_mm256_max_epu8(t, vlim), vlim);
        let m = !(_mm256_movemask_epi8(ok) as u32);
        if m != 0 {
            return q + m.trailing_zeros() as usize;
        }
        q += 32;
    }
    skip_scalar(b, q, k)
}

#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn skip_neon(b: &[u8], mut q: usize, k: Ascii) -> usize {
    use std::arch::aarch64::*;
    let (or, base, lim) = match k {
        Ascii::Letter => (0x20u8, b'a', 25u8),
        Ascii::Digit => (0x00u8, b'0', 9u8),
    };
    let (vor, vbase, vlim) = (vdupq_n_u8(or), vdupq_n_u8(base), vdupq_n_u8(lim));
    while q + 16 <= b.len() {
        let x = vld1q_u8(b.as_ptr().add(q));
        let t = vsubq_u8(vorrq_u8(x, vor), vbase);
        let bad = vcgtq_u8(t, vlim); // 0xFF where the byte is not in the class
        // 16 bytes -> 64-bit mask with 4 bits per byte (shift-right-narrow trick)
        let m = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(bad), 4)), 0);
        if m != 0 {
            return q + (m.trailing_zeros() / 4) as usize;
        }
        q += 16;
    }
    skip_scalar(b, q, k)
}

/// First i >= from with s[i] == a && s[i+1] == b (the next occurrence of the pair (a, b)), if any.
/// Used by the trainer's merge loop to jump over symbols that cannot be merged.
#[inline]
pub fn find_pair(s: &[u32], from: usize, a: u32, b: u32) -> Option<usize> {
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe { find_pair_avx2(s, from, a, b) };
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { find_pair_neon(s, from, a, b) };
    }
    #[allow(unreachable_code)]
    find_pair_scalar(s, from, a, b)
}

#[inline]
pub fn find_pair_scalar(s: &[u32], from: usize, a: u32, b: u32) -> Option<usize> {
    let mut i = from;
    while i + 1 < s.len() {
        if s[i] == a && s[i + 1] == b {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn find_pair_avx2(s: &[u32], from: usize, a: u32, b: u32) -> Option<usize> {
    use std::arch::x86_64::*;
    let (va, vb) = (_mm256_set1_epi32(a as i32), _mm256_set1_epi32(b as i32));
    let mut i = from;
    // 8 candidate positions i..i+8 need s[i..i+9]
    while i + 9 <= s.len() {
        let x = _mm256_loadu_si256(s.as_ptr().add(i) as *const __m256i);
        let y = _mm256_loadu_si256(s.as_ptr().add(i + 1) as *const __m256i);
        let m = _mm256_and_si256(_mm256_cmpeq_epi32(x, va), _mm256_cmpeq_epi32(y, vb));
        let bits = _mm256_movemask_ps(_mm256_castsi256_ps(m)) as u32;
        if bits != 0 {
            return Some(i + bits.trailing_zeros() as usize);
        }
        i += 8;
    }
    find_pair_scalar(s, i, a, b)
}

#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn find_pair_neon(s: &[u32], from: usize, a: u32, b: u32) -> Option<usize> {
    use std::arch::aarch64::*;
    let (va, vb) = (vdupq_n_u32(a), vdupq_n_u32(b));
    let mut i = from;
    // 4 candidate positions i..i+4 need s[i..i+5]
    while i + 5 <= s.len() {
        let x = vld1q_u32(s.as_ptr().add(i));
        let y = vld1q_u32(s.as_ptr().add(i + 1));
        let m = vandq_u32(vceqq_u32(x, va), vceqq_u32(y, vb));
        if vmaxvq_u32(m) != 0 {
            // 4 lanes -> 4 x 16-bit mask
            let n = vget_lane_u64(vreinterpret_u64_u16(vmovn_u32(m)), 0);
            return Some(i + (n.trailing_zeros() / 16) as usize);
        }
        i += 4;
    }
    find_pair_scalar(s, i, a, b)
}
