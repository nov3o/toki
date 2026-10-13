"""Every normalizer we benchmark, by name.

  make(name)  -> the HF `tokenizers` normalizer object (None for "none")
  spec(name)  -> the same normalizer for our Rust code (BB_NORMALIZER / .hfbp; format: rust_bpe/src/normalize.rs)

A normalizer rewrites the text before pre-tokenization (Unicode normal forms, lowercasing, accents, ...).
"prepend" and "strip" look at the start / end of a text, so they only work when every text is encoded on its
own (training: every line); the file encoders cut long files into pieces, so they are train-only here.
"""
from tokenizers import normalizers as nz

ALL = ["none", "nfc", "nfd", "nfkc", "nfkd", "lowercase", "strip_accents", "nfkc_lowercase",
       "replace", "bert", "prepend", "strip"]
TRAIN_ONLY = {"prepend", "strip"}

US, RS = "\x1f", "\x1e"      # field / step separators of the spec string

_DEF = {
    #  name           HF object                                         our spec steps
    "none":           (lambda: None,                                     []),
    "nfc":            (lambda: nz.NFC(),                                 ["nfc"]),
    "nfd":            (lambda: nz.NFD(),                                 ["nfd"]),
    "nfkc":           (lambda: nz.NFKC(),                                ["nfkc"]),
    "nfkd":           (lambda: nz.NFKD(),                                ["nfkd"]),
    "lowercase":      (lambda: nz.Lowercase(),                           ["lowercase"]),
    "strip_accents":  (lambda: nz.Sequence([nz.NFD(), nz.StripAccents()]), ["nfd", "strip_accents"]),
    "nfkc_lowercase": (lambda: nz.Sequence([nz.NFKC(), nz.Lowercase()]),  ["nfkc", "lowercase"]),
    "replace":        (lambda: nz.Replace("\t", "    "),                 ["replace" + US + "\t" + US + "    "]),
    "bert":           (lambda: nz.BertNormalizer(),                      ["bert" + US + "1" + US + "1" + US + "1" + US + "1"]),
    "prepend":        (lambda: nz.Prepend("▁"),                     ["prepend" + US + "▁"]),
    "strip":          (lambda: nz.Strip(),                               ["strip" + US + "1" + US + "1"]),
}


def make(name):
    return _DEF[name][0]()


def spec(name):
    return RS.join(_DEF[name][1])
