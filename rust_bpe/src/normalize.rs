//! HF `tokenizers` normalizers, applied to a text before pre-tokenization.
//! Exactness: the same crates as HF — unicode-normalization-alignments for NFC/NFD/NFKC/NFKD and
//! is_combining_mark, unicode_categories for Bert's control / nonspacing-mark tests, std for lowercase.
//!
//! A normalizer is given as a spec string (built by bench_server/normalizers.py from the HF object):
//! steps separated by '\x1e', fields by '\x1f':
//!   nfc | nfd | nfkc | nfkd | lowercase | strip_accents | replace␟<literal>␟<content> | prepend␟<text>
//!   | strip␟<0|1 left>␟<0|1 right> | bert␟<clean_text>␟<handle_chinese_chars>␟<strip_accents>␟<lowercase>
use unicode_categories::UnicodeCategories;
use unicode_normalization_alignments::char::is_combining_mark;
use unicode_normalization_alignments::UnicodeNormalization;

#[derive(Clone, Debug)]
pub enum Step {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
    Lowercase,
    StripAccents,
    Replace(String, String),
    Prepend(String),
    Strip(bool, bool),
    Bert { clean: bool, chinese: bool, strip_accents: bool, lowercase: bool },
}

pub fn parse(spec: &str) -> Result<Vec<Step>, String> {
    let mut v = Vec::new();
    for item in spec.split('\x1e').filter(|s| !s.is_empty()) {
        let f: Vec<&str> = item.split('\x1f').collect();
        let b = |i: usize| f.get(i).map(|x| *x == "1").unwrap_or(false);
        v.push(match f[0] {
            "nfc" => Step::Nfc,
            "nfd" => Step::Nfd,
            "nfkc" => Step::Nfkc,
            "nfkd" => Step::Nfkd,
            "lowercase" => Step::Lowercase,
            "strip_accents" => Step::StripAccents,
            "replace" if f.len() == 3 => Step::Replace(f[1].to_string(), f[2].to_string()),
            "prepend" if f.len() == 2 => Step::Prepend(f[1].to_string()),
            "strip" => Step::Strip(b(1), b(2)),
            "bert" => Step::Bert { clean: b(1), chinese: b(2), strip_accents: b(3), lowercase: b(4) },
            x => return Err(format!("unknown normalizer step {x:?}")),
        });
    }
    Ok(v)
}

/// True if some step only makes sense on a whole text (it looks at the text's start or end), so a long input
/// can't be normalized in independent chunks.
pub fn whole_text_only(steps: &[Step]) -> bool {
    steps.iter().any(|s| matches!(s, Step::Prepend(_) | Step::Strip(..)))
}

fn bert_is_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}
fn bert_is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && c.is_other()
}
fn is_chinese_char(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F | 0x2B920..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

/// Apply the steps to `text` (out is overwritten). Same order and rules as HF's normalizers.
pub fn apply(steps: &[Step], text: &str, out: &mut String) {
    out.clear();
    out.push_str(text);
    let mut tmp = String::new();
    for st in steps {
        tmp.clear();
        let s = out.as_str();
        match st {
            Step::Nfc => tmp.extend(s.nfc().map(|(c, _)| c)),
            Step::Nfd => tmp.extend(s.nfd().map(|(c, _)| c)),
            Step::Nfkc => tmp.extend(s.nfkc().map(|(c, _)| c)),
            Step::Nfkd => tmp.extend(s.nfkd().map(|(c, _)| c)),
            Step::Lowercase => {
                for c in s.chars() {
                    tmp.extend(c.to_lowercase());
                }
            }
            Step::StripAccents => tmp.extend(s.chars().filter(|&c| !is_combining_mark(c))),
            Step::Replace(p, c) => tmp.push_str(&s.replace(p.as_str(), c)),
            Step::Prepend(p) => {
                if !s.is_empty() {
                    tmp.push_str(p);
                }
                tmp.push_str(s);
            }
            Step::Strip(l, r) => {
                let mut t = s;
                if *l {
                    t = t.trim_start_matches(char::is_whitespace);
                }
                if *r {
                    t = t.trim_end_matches(char::is_whitespace);
                }
                tmp.push_str(t);
            }
            Step::Bert { clean, chinese, strip_accents, lowercase } => {
                let mut cur: String = s.to_string();
                if *clean {
                    cur = cur
                        .chars()
                        .filter(|&c| !(c as u32 == 0 || c as u32 == 0xFFFD || bert_is_control(c)))
                        .map(|c| if bert_is_whitespace(c) { ' ' } else { c })
                        .collect();
                }
                if *chinese {
                    let mut n = String::with_capacity(cur.len());
                    for c in cur.chars() {
                        if is_chinese_char(c) {
                            n.push(' ');
                            n.push(c);
                            n.push(' ');
                        } else {
                            n.push(c);
                        }
                    }
                    cur = n;
                }
                if *strip_accents {
                    cur = cur.nfd().map(|(c, _)| c).filter(|c| !c.is_mark_nonspacing()).collect();
                }
                if *lowercase {
                    let mut n = String::with_capacity(cur.len());
                    for c in cur.chars() {
                        n.extend(c.to_lowercase());
                    }
                    cur = n;
                }
                tmp.push_str(&cur);
            }
        }
        std::mem::swap(out, &mut tmp);
    }
}
