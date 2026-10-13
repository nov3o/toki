//! Every HF `tokenizers` pre-tokenizer we support, as one function: `for_each_word(kind, text, f)` calls
//! `f(word)` for each pre-token of `text`, in order, exactly as HF splits it.
//!
//! byte-level (the word's bytes go to byte-level BPE): bytelevel (GPT-2 regex), llama3, gpt4o
//! character-level (BPE over Unicode chars):           whitespace, whitespace_split, metaspace, bert,
//!                                                     punctuation, digits, unicode_scripts, char_delimiter
//!
//! Exactness: the character predicates are the ones HF itself uses — Rust's `char::is_whitespace` /
//! `is_numeric`, `unicode_categories::is_punctuation`, HF's own script table (hf_scripts.rs), and the
//! `regex` crate's \w and \s (via regex-syntax) for `Whitespace`. GPT-4o's \p{..} classes also come
//! from regex-syntax.
use crate::hf_scripts::{get_script, Script};
use std::sync::OnceLock;
use unicode_categories::UnicodeCategories;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Gpt2,
    Llama3,
    Gpt4o,
    Whitespace,
    WhitespaceSplit,
    Metaspace,
    Bert,
    Punctuation,
    Digits,
    UnicodeScripts,
    CharDelimiter,
}

impl Kind {
    /// Names as in bench_server/pretokenizers.py ("gpt2" is accepted for "bytelevel").
    pub fn from_name(s: &str) -> Option<Kind> {
        Some(match s {
            "bytelevel" | "gpt2" => Kind::Gpt2,
            "llama3" => Kind::Llama3,
            "gpt4o" => Kind::Gpt4o,
            "whitespace" => Kind::Whitespace,
            "whitespace_split" => Kind::WhitespaceSplit,
            "metaspace" => Kind::Metaspace,
            "bert" => Kind::Bert,
            "punctuation" => Kind::Punctuation,
            "digits" => Kind::Digits,
            "unicode_scripts" => Kind::UnicodeScripts,
            "char_delimiter" => Kind::CharDelimiter,
            _ => return None,
        })
    }
    pub fn byte_level(self) -> bool {
        matches!(self, Kind::Gpt2 | Kind::Llama3 | Kind::Gpt4o)
    }
}

// ---------------------------------------------------------------- Unicode class bits (built once)
const W: u16 = 1; // regex \w
const S: u16 = 2; // regex \s
const L: u16 = 4; // \p{L}
const N: u16 = 8; // \p{N}
const UP: u16 = 16; // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]   (GPT-4o "upper" set)
const LO: u16 = 32; // [\p{Ll}\p{Lm}\p{Lo}\p{M}]          (GPT-4o "lower" set)

pub fn table() -> &'static [u16] {
    static T: OnceLock<Vec<u16>> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = vec![0u16; 0x110000];
        let mut set = |re: &str, bit: u16| {
            use regex_syntax::hir::{Class, HirKind};
            let h = regex_syntax::parse(re).expect("class");
            match h.kind() {
                HirKind::Class(Class::Unicode(c)) => {
                    for r in c.iter() {
                        for cp in r.start() as u32..=r.end() as u32 {
                            t[cp as usize] |= bit;
                        }
                    }
                }
                _ => panic!("not a class: {re}"),
            }
        };
        set(r"\w", W);
        set(r"\s", S);
        set(r"\p{L}", L);
        set(r"\p{N}", N);
        set(r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]", UP);
        set(r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]", LO);
        t
    })
}

#[inline]
fn has(t: &[u16], c: char, bit: u16) -> bool {
    t[c as usize] & bit != 0
}

// ---------------------------------------------------------------- entry point
/// Calls f(word) for every pre-token of `text`. `scratch` is only used by Metaspace (it rewrites spaces).
pub fn for_each_word(kind: Kind, text: &[u8], scratch: &mut Vec<u8>, mut f: impl FnMut(&[u8])) {
    match kind {
        Kind::Gpt2 | Kind::Llama3 => {
            let nx = if kind == Kind::Gpt2 { crate::next_gpt2 } else { crate::next_llama3 };
            let mut p = 0;
            while p < text.len() {
                let q = nx(text, p);
                f(&text[p..q]);
                p = q;
            }
        }
        Kind::Gpt4o => {
            let s = as_str(text);
            let t = table();
            let mut p = 0;
            while p < s.len() {
                let q = next_gpt4o(s, p, t);
                f(&text[p..q]);
                p = q;
            }
        }
        _ => {
            let s = as_str(text);
            char_level(kind, s, scratch, |w| f(w));
        }
    }
}

fn as_str(b: &[u8]) -> &str {
    // the corpora are valid UTF-8 (prepare_data.py writes them that way)
    std::str::from_utf8(b).expect("input is not valid UTF-8")
}

// ---------------------------------------------------------------- character-level pre-tokenizers
fn char_level(kind: Kind, s: &str, scratch: &mut Vec<u8>, mut f: impl FnMut(&[u8])) {
    let b = s.as_bytes();
    // split helpers on char predicates, mirroring HF's SplitDelimiterBehavior
    let removed = |pred: &dyn Fn(char) -> bool, f: &mut dyn FnMut(&[u8])| {
        let mut start = None;
        for (i, c) in s.char_indices() {
            if pred(c) {
                if let Some(st) = start.take() {
                    f(&b[st..i]);
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(st) = start {
            f(&b[st..]);
        }
    };
    let isolated = |pred: &dyn Fn(char) -> bool, f: &mut dyn FnMut(&[u8])| {
        let mut st = 0;
        for (i, c) in s.char_indices() {
            if pred(c) {
                if i > st {
                    f(&b[st..i]);
                }
                f(&b[i..i + c.len_utf8()]);
                st = i + c.len_utf8();
            }
        }
        if st < b.len() {
            f(&b[st..]);
        }
    };
    match kind {
        Kind::WhitespaceSplit => removed(&|c| c.is_whitespace(), &mut f),
        Kind::CharDelimiter => removed(&|c| c == ' ', &mut f),
        Kind::Punctuation => isolated(&is_punc, &mut f),
        Kind::Digits => isolated(&|c| c.is_numeric(), &mut f),
        Kind::Bert => {
            // split on whitespace (removed), then on punctuation (isolated)
            removed(&|c| c.is_whitespace(), &mut |w: &[u8]| {
                let ws = std::str::from_utf8(w).unwrap();
                let mut st = 0;
                for (i, c) in ws.char_indices() {
                    if is_punc(c) {
                        if i > st {
                            f(&w[st..i]);
                        }
                        f(&w[i..i + c.len_utf8()]);
                        st = i + c.len_utf8();
                    }
                }
                if st < w.len() {
                    f(&w[st..]);
                }
            })
        }
        Kind::Whitespace => {
            // \w+|[^\w\s]+, everything else (whitespace) dropped
            let t = table();
            let mut it = s.char_indices().peekable();
            while let Some(&(i, c)) = it.peek() {
                if has(t, c, W) {
                    let mut e = i;
                    while let Some(&(j, d)) = it.peek() {
                        if !has(t, d, W) {
                            break;
                        }
                        e = j + d.len_utf8();
                        it.next();
                    }
                    f(&b[i..e]);
                } else if !has(t, c, S) {
                    let mut e = i;
                    while let Some(&(j, d)) = it.peek() {
                        if has(t, d, W) || has(t, d, S) {
                            break;
                        }
                        e = j + d.len_utf8();
                        it.next();
                    }
                    f(&b[i..e]);
                } else {
                    it.next();
                }
            }
        }
        Kind::Metaspace => {
            // replacement '▁', prepend_scheme "always", split = true (HF defaults):
            // ' ' -> '▁', prepend '▁' unless the text already starts with it, split before every '▁'
            const R: &str = "\u{2581}";
            if s.is_empty() {
                return;
            }
            scratch.clear();
            if !(s.starts_with(' ') || s.starts_with(R)) {
                scratch.extend_from_slice(R.as_bytes());
            }
            for c in s.chars() {
                if c == ' ' {
                    scratch.extend_from_slice(R.as_bytes());
                } else {
                    let mut buf = [0u8; 4];
                    scratch.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
            let m = &scratch[..];
            let ms = std::str::from_utf8(m).unwrap();
            let mut st = 0;
            for (i, c) in ms.char_indices() {
                if c == '\u{2581}' && i > st {
                    f(&m[st..i]);
                    st = i;
                }
            }
            f(&m[st..]);
        }
        Kind::UnicodeScripts => {
            // HF: a split point wherever a non-space char's script differs from the last non-space char's
            // script (the first non-space char always counts). Pieces are the text between consecutive split
            // points and the end, so anything before the first split point (leading spaces) is dropped.
            let mut last: Option<Script> = None;
            let mut st: Option<usize> = None;
            for (i, c) in s.char_indices() {
                let sc = fixed_script(c);
                if sc != Script::Any && last != Some(sc) {
                    if let Some(p0) = st {
                        f(&b[p0..i]);
                    }
                    st = Some(i);
                }
                if sc != Script::Any {
                    last = Some(sc);
                }
            }
            if let Some(p0) = st {
                f(&b[p0..]);
            }
        }
        _ => unreachable!(),
    }
}

/// HF's is_punc / is_bert_punc
#[inline]
fn is_punc(c: char) -> bool {
    c.is_ascii_punctuation() || c.is_punctuation()
}

/// HF's fixed_script (unicode_scripts/pre_tokenizer.rs)
fn fixed_script(c: char) -> Script {
    if c as u32 == 0x30FC {
        Script::Han
    } else if c == ' ' {
        Script::Any
    } else {
        match get_script(c) {
            Script::Hiragana | Script::Katakana => Script::Han,
            sc => sc,
        }
    }
}

// ---------------------------------------------------------------- GPT-4o (tiktoken o200k_base) regex
// [^\r\n\p{L}\p{N}]?[UP]*[LO]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
// |[^\r\n\p{L}\p{N}]?[UP]+[LO]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
// |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Leftmost alternative wins; each alternative is matched with the regex's greedy-with-backtracking rules.

fn chars_from(s: &str, p: usize) -> impl Iterator<Item = (usize, char)> + '_ {
    s[p..].char_indices().map(move |(i, c)| (p + i, c))
}

/// end of a contraction (?i:'s|'t|'re|'ve|'m|'ll|'d) at p, if any
fn contraction(s: &str, p: usize) -> Option<usize> {
    let b = s.as_bytes();
    if p >= b.len() || b[p] != b'\'' {
        return None;
    }
    let c1 = b.get(p + 1).map(|x| x | 0x20);
    let c2 = b.get(p + 2).map(|x| x | 0x20);
    match c1 {
        Some(b's') | Some(b't') | Some(b'm') | Some(b'd') => Some(p + 2),
        Some(b'r') | Some(b'v') if c2 == Some(b'e') => Some(p + 3),
        Some(b'l') if c2 == Some(b'l') => Some(p + 3),
        _ => None,
    }
}

/// [A]{min_a,}[B]{min_b,} starting at p, with the regex's greedy-with-backtracking result; returns the end
/// after B (and after an optional contraction), or None. No allocation: A* first takes the whole A-run; if B
/// then matches nothing, backtracking gives back A-chars until one of them is also a B-char (B+ then
/// matches exactly that char, since the char after the A-run is not a B-char).
fn star_plus(s: &str, p: usize, t: &[u16], a: u16, min_a: usize, b: u16, min_b: usize) -> Option<usize> {
    let mut na = 0; // A-chars in the run
    let mut a_end = p;
    let mut last_ab: Option<(usize, usize)> = None; // (index among A-chars, end) of the last A-char that is also B
    for (i, c) in chars_from(s, p) {
        if !has(t, c, a) {
            break;
        }
        na += 1;
        a_end = i + c.len_utf8();
        if has(t, c, b) {
            last_ab = Some((na - 1, a_end));
        }
    }
    if na >= min_a {
        let mut e = a_end;
        let mut nb = 0;
        for (i, c) in chars_from(s, a_end) {
            if !has(t, c, b) {
                break;
            }
            e = i + c.len_utf8();
            nb += 1;
        }
        if nb >= min_b {
            return Some(contraction(s, e).unwrap_or(e));
        }
    }
    // backtrack: k A-chars (k < na, k >= min_a) then B+ = the A-char at index k, which must be a B-char
    if min_b >= 1 {
        if let Some((k, e)) = last_ab {
            if k >= min_a {
                return Some(contraction(s, e).unwrap_or(e));
            }
        }
    }
    None
}

/// End of the GPT-4o pre-token starting at p (s[p..] is not empty).
pub fn next_gpt4o(s: &str, p: usize, t: &[u16]) -> usize {
    let c0 = s[p..].chars().next().unwrap();
    let l0 = c0.len_utf8();
    let prefix_ok = c0 != '\r' && c0 != '\n' && !has(t, c0, L) && !has(t, c0, N);
    // alternatives 1 and 2: optional prefix char (tried with it first), then the letter runs
    for (a, min_a, b, min_b) in [(UP, 0, LO, 1), (UP, 1, LO, 0)] {
        if prefix_ok && p + l0 < s.len() {
            if let Some(e) = star_plus(s, p + l0, t, a, min_a, b, min_b) {
                return e;
            }
        }
        if let Some(e) = star_plus(s, p, t, a, min_a, b, min_b) {
            return e;
        }
    }
    // 3: \p{N}{1,3}
    if has(t, c0, N) {
        let mut e = p;
        for (n, (i, c)) in chars_from(s, p).enumerate() {
            if n == 3 || !has(t, c, N) {
                break;
            }
            e = i + c.len_utf8();
        }
        return e;
    }
    // 4: ' ?[^\s\p{L}\p{N}]+[\r\n/]*'
    let other = |c: char| !has(t, c, S) && !has(t, c, L) && !has(t, c, N);
    let q = if c0 == ' ' { p + 1 } else { p };
    if q < s.len() && other(s[q..].chars().next().unwrap()) {
        let mut e = q;
        for (i, c) in chars_from(s, q) {
            if !other(c) {
                break;
            }
            e = i + c.len_utf8();
        }
        for (i, c) in chars_from(s, e) {
            if !(c == '\r' || c == '\n' || c == '/') {
                break;
            }
            e = i + 1;
        }
        return e;
    }
    // 5-7: whitespace. The run of \s from p:
    let (mut run_end, mut last_nl, mut last_start) = (p, None, p);
    for (i, c) in chars_from(s, p) {
        if !has(t, c, S) {
            break;
        }
        if c == '\r' || c == '\n' {
            last_nl = Some(i);
        }
        last_start = i;
        run_end = i + c.len_utf8();
    }
    if run_end == p {
        // not whitespace and nothing else matched (cannot happen for valid text): take one char
        return p + l0;
    }
    if let Some(nl) = last_nl {
        // \s*[\r\n]+ : \s* backtracks to the last newline in the run; [\r\n]+ takes the newlines from there
        let mut e = nl;
        for (i, c) in chars_from(s, nl) {
            if !(c == '\r' || c == '\n') {
                break;
            }
            e = i + 1;
        }
        return e;
    }
    // \s+(?!\S): whole run at the end of the text, else the run minus its last char (if that leaves >= 1)
    if run_end == s.len() {
        return run_end;
    }
    if last_start > p {
        last_start
    } else {
        run_end
    }
}

// ---------------------------------------------------------------- where a long text may be cut
/// True if pre-tokenizing b[..c] and b[c..] separately gives exactly the pre-tokens of b (so a long input can
/// be encoded in independent chunks). Only ASCII bytes are inspected, so c is always a char boundary.
/// bench_server/common.py (pieces) uses the same rules for HF and tiktoken.
#[inline]
pub fn safe_cut(kind: Kind, b: &[u8], c: usize) -> bool {
    // b[c] must be ASCII: an ASCII char never combines with the char before it, so normalizers
    // (NFC / NFKC composition, ...) also give the same result on both sides of the cut
    if c == 0 || c >= b.len() || b[c] >= 0x80 {
        return false;
    }
    let an = |x: u8| x.is_ascii_alphanumeric();
    match kind {
        // [A-Za-z0-9] | '\n' [A-Za-z0-9]: cut before the newline. The regexes restart identically there, also
        // when a normalizer turns the newline into a space (BertNormalizer): " word" starts a pre-token either way
        Kind::Gpt2 | Kind::Llama3 | Kind::Gpt4o => b[c] == b'\n' && an(b[c - 1]) && c + 1 < b.len() && an(b[c + 1]),
        // whitespace is dropped and always ends a word
        Kind::Whitespace | Kind::WhitespaceSplit | Kind::Bert => b[c - 1] == b'\n',
        // a '▁' word starts at every space, and a piece starting with a space gets no extra '▁'
        Kind::Metaspace => b[c] == b' ',
        // the delimiter is dropped and always ends a word
        Kind::CharDelimiter => b[c - 1] == b' ',
        // punctuation / digits are always isolated
        Kind::Punctuation => b[c - 1].is_ascii_punctuation(),
        Kind::Digits => b[c - 1].is_ascii_digit(),
        // Latin letter then ASCII punctuation (script Common): the script changes there anyway, and no
        // normalizer turns punctuation into a space (which has no script and would be dropped at a piece start)
        Kind::UnicodeScripts => b[c - 1].is_ascii_alphabetic() && b[c].is_ascii_punctuation(),
    }
}

/// Code stored in the .hfbp model file (export_hfbp.py) <-> Kind
pub fn from_code(code: u32) -> Option<Kind> {
    Some(match code {
        0 => Kind::Gpt2,
        1 => Kind::Llama3,
        2 => Kind::Gpt4o,
        10 => Kind::Whitespace,
        11 => Kind::WhitespaceSplit,
        12 => Kind::Metaspace,
        13 => Kind::Bert,
        14 => Kind::Punctuation,
        15 => Kind::Digits,
        16 => Kind::UnicodeScripts,
        17 => Kind::CharDelimiter,
        _ => return None,
    })
}
